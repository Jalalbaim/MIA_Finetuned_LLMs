"""
LoRA fine-tuning variant (additive) for GPT-Neo 125M.

Reuses EnronDataset / DataCollatorForLanguageModeling from finetune/train.py
and load_split from data/membership_assignment.py. Does not modify or
interfere with the non-DP (train.py) or DP (train_dp.py) pipelines.

Usage:
    python finetune/train_lora.py                  # all ranks, N=6000, seed 0
    python finetune/train_lora.py --rank 16
    python finetune/train_lora.py --seed 0 --n 6000 --epochs_max 20
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
    LORA, LORA_LEARNING_RATE,
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


ATTENTION_PROJ_NAMES = ("q_proj", "k_proj", "v_proj", "out_proj")


# Checkpoint naming

def lora_ckpt_dir(rank: int, n_members: int, seed: int, epoch: int) -> Path:
    return CKPT_OUT_DIR / f"gpt_neo_ft_lora_r{rank}_N{n_members}_seed{seed}_epoch{epoch}"


# Target-module resolution

def resolve_target_modules(model: AutoModelForCausalLM, configured) -> list[str]:
    """Locate GPT-Neo attention projection modules and resolve LoRA's
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
    print(f"  LORA['target_modules'] is None -> defaulting to {default}")
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
    print(f"  [verify] Reloaded {ckpt_path.name} with AutoModelForCausalLM, "
          f"forward pass logits shape: {tuple(out.logits.shape)}")
    del model


# LoRA training

def train_lora(seed: int, n_members: int, rank: int, epochs: int, device: torch.device) -> dict:
    """
    Train a fresh GPT-Neo wrapped with a rank-`rank` LoRA adapter for up to
    `epochs` epochs. At each epoch in EPOCH_SWEEP (<= epochs), the adapter is
    merged into a deep-copied base model and the merged model is saved --
    merge_and_unload() is destructive, so the live, still-training `model`
    is left untouched and training continues from where it left off.
    """
    torch.manual_seed(seed)

    final_ckpt = lora_ckpt_dir(rank, n_members, seed, epochs)
    log_path = LOG_DIR / f"train_lora_r{rank}_N{n_members}_seed{seed}.csv"

    if (final_ckpt / "config.json").exists():
        print(f"  [skip] Final checkpoint exists: {final_ckpt.name}")
        final_loss = float("nan")
        wall_time = 0.0
        if log_path.exists():
            with log_path.open(encoding="utf-8") as fh:
                rows = list(csv.reader(fh))[1:]
            if rows:
                final_loss = float(rows[-1][1])
                wall_time = sum(float(r[2]) for r in rows)
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

    # Fresh pretrained base model, wrapped with a LoRA adapter
    base_model = AutoModelForCausalLM.from_pretrained(PRETRAINED_CKPT).to(device)

    target_modules = resolve_target_modules(base_model, LORA["target_modules"])
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=LORA["alpha"],
        lora_dropout=LORA["dropout"],
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_config)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params     = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable_params:,} / {total_params:,} "
          f"({100 * trainable_params / total_params:.4f}%)")

    optimizer = AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=LORA_LEARNING_RATE,
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
    print(f"  Rank: {rank}  |  LoRA LR: {LORA_LEARNING_RATE}  |  Batches/epoch: {len(loader)}  |  "
          f"Grad accum: {grad_accum}  |  Eff. batch: {eff_batch}  |  "
          f"Max epochs: {max_epochs}  |  Device: {device}")
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
                with torch.autocast(device_type=device.type, dtype=torch.float16,
                                    enabled=(device.type == "cuda")):
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
                ckpt = lora_ckpt_dir(rank, n_members, seed, epoch)
                if (ckpt / "config.json").exists():
                    note = f"skip (exists): {ckpt.name}"
                else:
                    # Deep-copy the (small) PEFT model to CPU, merge the LoRA
                    # adapter into that copy, and save the merged result.
                    # merge_and_unload() mutates the model it's called on, so
                    # the original `model` -- still wrapping the live adapter
                    # on `device` -- is left untouched and training continues
                    # unaffected.
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
    parser = argparse.ArgumentParser(description="LoRA fine-tune GPT-Neo on Enron members.")
    parser.add_argument("--seed", type=int, default=LORA["seed"],
                        help=f"Seed (default: {LORA['seed']})")
    parser.add_argument("--n", type=int, default=LORA["n"],
                        help=f"N_members (default: {LORA['n']})")
    parser.add_argument("--rank", type=int, default=None,
                        help="Single LoRA rank to run (default: all LORA['ranks'])")
    parser.add_argument("--epochs_max", type=int, default=max(EPOCH_SWEEP),
                        help=f"Max epochs to train to (default: {max(EPOCH_SWEEP)})")
    args = parser.parse_args()

    ranks = [args.rank] if args.rank is not None else LORA["ranks"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device      : {device}")
    print(f"Seed        : {args.seed}")
    print(f"N_members   : {args.n:,}")
    print(f"Ranks       : {ranks}")
    print(f"Epoch sweep : {EPOCH_SWEEP}  (max={args.epochs_max})")
    print(f"LoRA alpha  : {LORA['alpha']}  |  dropout: {LORA['dropout']}  |  "
          f"LR: {LORA_LEARNING_RATE}\n")

    summary: list[dict] = []
    t_wall = time.time()

    for rank in ranks:
        print(f"\n{'='*64}")
        print(f"  LoRA rank={rank}  |  seed={args.seed}  N={args.n:,}")
        print(f"{'='*64}")
        summary.append(train_lora(args.seed, args.n, rank, args.epochs_max, device))

    total = time.time() - t_wall

    print(f"\n{'='*64}")
    print(f"Summary — N={args.n:,}  seed={args.seed}")
    hdr = f"  {'rank':>5} | {'trainable_params':>16} | {'final_loss':>10} | {'wall_time':>9}"
    sep = "  " + "-"*5 + "+" + "-"*18 + "+" + "-"*12 + "+" + "-"*11
    print(hdr)
    print(sep)
    for row in summary:
        tp = f"{row['trainable_params']:,}" if row["trainable_params"] is not None else "n/a"
        fl = "n/a" if math.isnan(row["final_loss"]) else f"{row['final_loss']:.4f}"
        print(f"  {row['rank']:>5} | {tp:>16} | {fl:>10} | {row['wall_time']:>8.1f}s")

    print(f"\nTotal wall time: {total:.1f}s  ({total/60:.1f} min)")


if __name__ == "__main__":
    main()
