"""
MiCA fine-tuning variant (Minor Component Adaptation, Rüdiger & Raschka,
arXiv:2604.01694) for GPT-Neo 125M.

MiCA is implemented in peft as a LoRA initialization mode
(`LoraConfig(init_lora_weights="mica", ...)`), not a separate adapter class:
`lora_B` is set to the r left singular vectors of the base weight associated
with the *smallest* singular values, `lora_A` is zero-initialized, and
`lora_B` is frozen -- only `lora_A` trains. Merge/save semantics are
identical to vanilla LoRA (delta_W = scaling * B @ A), so this reuses the
same training/merge loop as train_lora.py.

This support landed in peft's main branch after the 0.19.1 PyPI release; it
is not in any released peft version yet (see requirements.txt).

Usage:
    python finetune/train_mica.py                 # all ranks, N=6000, seed 0
    python finetune/train_mica.py --rank 16
    python finetune/train_mica.py --seed 0 --n 6000 --epochs_max 20
"""

import sys
import csv
import copy
import math
import time
import argparse
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "data"))
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    FINETUNE, EPOCH_SWEEP,
    MICA, MICA_LEARNING_RATE,
    PRETRAINED_CKPT, CKPT_OUT_DIR, LOG_DIR,
)
from membership_assignment import load_split
from train import EnronDataset

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model


# MiCA support landed in peft's main branch (unreleased); verify the
# installed peft actually implements it before doing anything else, rather
# than failing deep inside get_peft_model with a confusing error.
_init_field = LoraConfig.__dataclass_fields__["init_lora_weights"]
if "mica" not in _init_field.metadata.get("lora_variants", []):
    import peft as _peft_module
    print(
        f"\n[ERROR] MiCA is not available in the installed peft "
        f"(v{getattr(_peft_module, '__version__', 'unknown')}).\n"
        f"  MiCA (Rüdiger & Raschka, arXiv:2604.01694) is implemented as\n"
        f"  LoraConfig(init_lora_weights='mica'), and this support has not\n"
        f"  shipped in a peft release yet -- it only exists on peft's main\n"
        f"  branch.\n"
        f"\n  To install it:\n"
        f"    pip install --upgrade \"peft @ git+https://github.com/huggingface/peft.git\"\n"
        f"\n  NOTE: after upgrading, verify that LoRA (train_lora.py) and\n"
        f"  DP (train_dp.py) paths still function correctly under the new version.\n"
    )
    sys.exit(1)


ATTENTION_PROJ_NAMES = ("q_proj", "k_proj", "v_proj", "out_proj")


# Checkpoint naming

def mica_ckpt_dir(rank: int, n_members: int, seed: int, epoch: int) -> Path:
    return CKPT_OUT_DIR / f"gpt_neo_ft_mica_r{rank}_N{n_members}_seed{seed}_epoch{epoch}"


# Target-module resolution

def resolve_target_modules(model: AutoModelForCausalLM, configured) -> list[str]:
    """Locate GPT-Neo attention projection modules and resolve MiCA's
    target_modules. GPT-Neo's self-attention exposes q_proj, k_proj, v_proj,
    out_proj (verified by inspecting named_modules())."""
    found = set()
    for name, _ in model.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        if leaf in ATTENTION_PROJ_NAMES:
            found.add(leaf)
    print(f"  GPT-Neo attention projection modules found: {sorted(found)}")

    if configured is not None:
        print(f"  Using configured target_modules: {configured}")
        return configured

    default = ["q_proj", "v_proj"]
    print(f"  MICA['target_modules'] is None -> defaulting to {default}")
    return default


# Checkpoint verification

@torch.no_grad()
def _verify_checkpoint(ckpt_path: Path, device: torch.device) -> None:
    """Reload a merged checkpoint with AutoModelForCausalLM and run a forward
    pass, confirming the merge-and-save round trip produced a plain model
    loadable by signals/compute_signals.py and kl_estimators/compute_kl.py."""
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    model = AutoModelForCausalLM.from_pretrained(ckpt_path).to(device)
    model.eval()
    input_ids = tokenizer("Verification forward pass.", return_tensors="pt")["input_ids"].to(device)
    out = model(input_ids=input_ids)
    assert torch.isfinite(out.logits).all(), (
        f"Non-finite logits in merged checkpoint {ckpt_path.name}!"
    )
    print(f"  [verify] Reloaded {ckpt_path.name} with AutoModelForCausalLM, "
          f"forward pass logits shape: {tuple(out.logits.shape)}  [OK finite]")
    del model


# MiCA training

def train_mica(seed: int, n_members: int, rank: int, epochs: int, device: torch.device) -> dict:
    """
    Train a fresh GPT-Neo wrapped with a rank-`rank` MiCA adapter (LoRA with
    init_lora_weights="mica") for up to `epochs` epochs. `lora_B` is frozen
    at its SVD-derived init; only `lora_A` trains. At each epoch in
    EPOCH_SWEEP (<= epochs), the adapter is merged into a deep-copied base
    model and the merged model is saved -- merge_and_unload() is destructive,
    so the live, still-training `model` is left untouched and training
    continues from where it left off.
    """
    torch.manual_seed(seed)

    final_ckpt = mica_ckpt_dir(rank, n_members, seed, epochs)
    log_path   = LOG_DIR / f"train_mica_r{rank}_N{n_members}_seed{seed}.csv"

    if (final_ckpt / "config.json").exists():
        print(f"  [skip] Final checkpoint exists: {final_ckpt.name}")
        final_loss = float("nan")
        wall_time  = 0.0
        if log_path.exists():
            with log_path.open(encoding="utf-8") as fh:
                rows = list(csv.reader(fh))[1:]
            if rows:
                final_loss = float(rows[-1][1])
                wall_time  = sum(float(r[2]) for r in rows)
        return {
            "rank": rank, "trainable_params": None, "total_params": None,
            "final_loss": final_loss, "wall_time": wall_time,
        }

    # Data
    members, _, _ = load_split(seed, n_members)
    print(f"  Members: {len(members):,}")

    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_CKPT)
    tokenizer.pad_token = tokenizer.eos_token

    dataset  = EnronDataset(members, tokenizer)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    loader   = DataLoader(
        dataset,
        batch_size=FINETUNE["batch_size"],
        shuffle=True,
        collate_fn=collator,
        pin_memory=(device.type == "cuda"),
    )

    # Fresh pretrained base model, wrapped with a MiCA adapter. mica_init
    # runs an SVD of each target weight at adapter-construction time, which
    # requires float32/float16/bfloat16 (not a quantized dtype) -- the
    # default float32 load here satisfies that.
    base_model = AutoModelForCausalLM.from_pretrained(PRETRAINED_CKPT).to(device)

    target_modules = resolve_target_modules(base_model, MICA["target_modules"])
    mica_config = LoraConfig(
        r=rank,
        lora_alpha=MICA["alpha"],
        lora_dropout=MICA["dropout"],
        target_modules=target_modules,
        task_type="CAUSAL_LM",
        init_lora_weights="mica",
    )
    model = get_peft_model(base_model, mica_config)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params     = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable_params:,} / {total_params:,} "
          f"({100 * trainable_params / total_params:.4f}%)")

    # MiCA freezes lora_B internally (peft registers it as a frozen weight
    # when init_lora_weights="mica"); confirm the invariant holds rather
    # than silently trusting it.
    all_trainable  = [n for n, p in model.named_parameters() if p.requires_grad]
    b_trainable    = [n for n in all_trainable if "lora_B" in n]
    non_lora_train = [n for n in all_trainable if "lora_A" not in n and "lora_B" not in n]
    if b_trainable:
        print(f"\n  [WARNING] lora_B params are trainable -- expected frozen "
              f"under MiCA. Flagging: {b_trainable[:5]}")
    else:
        print(f"  [OK] lora_B is frozen (MiCA invariant holds); only lora_A trains.")
    if non_lora_train:
        print(f"\n  [WARNING] Non-adapter params are trainable: {non_lora_train[:5]}")

    optimizer = AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=MICA_LEARNING_RATE,
        weight_decay=FINETUNE["weight_decay"],
    )

    max_epochs = epochs
    epoch_set  = {e for e in EPOCH_SWEEP if e <= max_epochs}
    grad_accum = FINETUNE.get("grad_accum_steps", 1)

    # fp16 AMP scaler -- enabled only on CUDA, no-op on CPU
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_fh = log_path.open("w", newline="", encoding="utf-8")
    writer = csv.writer(log_fh)
    writer.writerow(["epoch", "train_loss", "elapsed_s"])

    eff_batch = FINETUNE["batch_size"] * grad_accum
    print(
        f"  Rank: {rank}  |  MiCA LR: {MICA_LEARNING_RATE}  |  "
        f"Batches/epoch: {len(loader)}  |  Grad accum: {grad_accum}  |  "
        f"Eff. batch: {eff_batch}  |  Max epochs: {max_epochs}  |  Device: {device}"
    )
    print(f"  {'epoch':>5}  {'loss':>9}  {'time':>8}  note")
    print(f"  {'-----':>5}  {'---------':>9}  {'--------':>8}")

    verified = False
    avg_loss = float("nan")
    t_wall0  = time.time()

    try:
        for epoch in range(1, max_epochs + 1):
            model.train()
            running_loss = 0.0
            t0 = time.time()
            optimizer.zero_grad()

            for step, batch in enumerate(loader):
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.autocast(
                    device_type=device.type, dtype=torch.float16,
                    enabled=(device.type == "cuda"),
                ):
                    loss = model(**batch).loss
                scaler.scale(loss / grad_accum).backward()
                running_loss += loss.item()

                if (step + 1) % grad_accum == 0 or (step + 1) == len(loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

            avg_loss = running_loss / len(loader)
            elapsed  = time.time() - t0
            writer.writerow([epoch, f"{avg_loss:.6f}", f"{elapsed:.1f}"])
            log_fh.flush()

            note = ""
            if epoch in epoch_set:
                ckpt = mica_ckpt_dir(rank, n_members, seed, epoch)
                if (ckpt / "config.json").exists():
                    note = f"skip (exists): {ckpt.name}"
                else:
                    # Deep-copy the (small) PEFT model to CPU, merge the
                    # MiCA adapter into that copy, and save the merged
                    # result. merge_and_unload() mutates the model it's
                    # called on, so the original `model` -- still wrapping
                    # the live adapter on `device` -- is left untouched and
                    # training continues unaffected.
                    ckpt.mkdir(parents=True, exist_ok=True)
                    merge_copy = copy.deepcopy(model).to("cpu")
                    merged = merge_copy.merge_and_unload()
                    merged.save_pretrained(ckpt)
                    tokenizer.save_pretrained(ckpt)
                    del merge_copy, merged

                    note = f"saved → {ckpt.name}"

                    if not verified:
                        _verify_checkpoint(ckpt, device)
                        verified = True

            print(f"  {epoch:>5}  {avg_loss:>9.4f}  {elapsed:>7.1f}s  {note}")
    finally:
        log_fh.close()

    wall_time = time.time() - t_wall0

    del base_model, model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "rank": rank, "trainable_params": trainable_params,
        "total_params": total_params, "final_loss": avg_loss, "wall_time": wall_time,
    }


# Entry point

def main() -> None:
    parser = argparse.ArgumentParser(
        description="MiCA fine-tune GPT-Neo on Enron members."
    )
    parser.add_argument("--seed", type=int, default=MICA["seed"],
                        help=f"Seed (default: {MICA['seed']})")
    parser.add_argument("--n", type=int, default=MICA["n"],
                        help=f"N_members (default: {MICA['n']})")
    parser.add_argument("--rank", type=int, default=None,
                        help="Single MiCA rank to run (default: all MICA['ranks'])")
    parser.add_argument("--epochs_max", type=int, default=max(EPOCH_SWEEP),
                        help=f"Max epochs to train to (default: {max(EPOCH_SWEEP)})")
    args = parser.parse_args()

    ranks = [args.rank] if args.rank is not None else MICA["ranks"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device      : {device}")
    print(f"Seed        : {args.seed}")
    print(f"N_members   : {args.n:,}")
    print(f"Ranks       : {ranks}")
    print(f"Epoch sweep : {EPOCH_SWEEP}  (max={args.epochs_max})")
    print(
        f"MiCA alpha  : {MICA['alpha']}  |  dropout: {MICA['dropout']}  |  "
        f"LR: {MICA_LEARNING_RATE}"
    )
    print(
        f"\nNOTE — param budget vs LoRA at equal rank:\n"
        f"  MiCA trains only lora_A (r×d_in); LoRA trains lora_A (r×d_in) + lora_B (d_out×r).\n"
        f"  MiCA therefore has ~half the trainable params of LoRA at the same rank.\n"
        f"  Both counts are printed per-rank below so the caveat is auditable.\n"
    )

    summary: list[dict] = []
    t_wall = time.time()

    for rank in ranks:
        print(f"\n{'='*64}")
        print(f"  MiCA rank={rank}  |  seed={args.seed}  N={args.n:,}")
        print(f"{'='*64}")
        summary.append(
            train_mica(args.seed, args.n, rank, args.epochs_max, device)
        )

    total = time.time() - t_wall

    print(f"\n{'='*64}")
    print(f"Summary — N={args.n:,}  seed={args.seed}")
    print(
        f"  (MiCA trains only lora_A; LoRA would train ~2× params at the same rank)"
    )
    hdr = (
        f"  {'rank':>5} | {'trainable_params':>16} | "
        f"{'total_params':>12} | {'final_loss':>10} | {'wall_time':>9}"
    )
    sep = (
        "  " + "-"*5 + "+" + "-"*18 + "+" + "-"*14 + "+" + "-"*12 + "+" + "-"*11
    )
    print(hdr)
    print(sep)
    for row in summary:
        tp  = f"{row['trainable_params']:,}" if row["trainable_params"] is not None else "n/a"
        tot = f"{row['total_params']:,}"     if row["total_params"]     is not None else "n/a"
        fl  = "n/a" if math.isnan(row["final_loss"]) else f"{row['final_loss']:.4f}"
        print(
            f"  {row['rank']:>5} | {tp:>16} | {tot:>12} | "
            f"{fl:>10} | {row['wall_time']:>8.1f}s"
        )

    print(f"\nTotal wall time: {total:.1f}s  ({total/60:.1f} min)")


if __name__ == "__main__":
    main()
