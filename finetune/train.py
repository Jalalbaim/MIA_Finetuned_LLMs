"""
Fine-tuning pipeline.

Usage:
    python finetune/train.py                     # all seeds × all N
    python finetune/train.py --seed 0 --n 2000   # single combination
    python finetune/train.py --seed 0            # seed 0, all N values

Null control (same members, different training randomness):
    python finetune/train.py --seed 0 --n 2000 --train_seed 1 --max_epoch 1
    -> checkpoints/gpt_neo_ft_N2000_seed0_trainseed1_epoch1
--seed selects the member split; --train_seed (default: = --seed) seeds the
training RNG (data order; dropout if any). Default behaviour is unchanged.
"""

import sys
import csv
import time
import argparse
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "data"))

from config import (
    MODEL_NAME, MAX_SEQ_LEN,
    SEEDS, CORPUS_SIZES,
    FINETUNE, EPOCH_SWEEP,
    PRETRAINED_CKPT, CKPT_DIR, LOG_DIR,
)
from membership_assignment import load_split

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
)


# Dataset

class EnronDataset(Dataset):
    """
    Causal LM dataset over Enron member records.

    Each item returns only input_ids (no padding).
    DataCollatorForLanguageModeling(mlm=False) handles batch-level padding
    and sets labels = input_ids with -100 at pad positions.
    """

    def __init__(self, records: list[dict], tokenizer) -> None:
        self._samples: list[dict] = []
        for rec in records:
            ids = tokenizer(
                rec["text"],
                truncation=True,
                max_length=MAX_SEQ_LEN,
                return_tensors="pt",
            )["input_ids"].squeeze(0)          # (seq_len,)
            self._samples.append({"input_ids": ids})

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict:
        return self._samples[idx]


# Pretrained reference

def save_pretrained_reference() -> None:
    """Save EleutherAI/gpt-neo-125m weights once to PRETRAINED_CKPT. Skip if present."""
    marker = PRETRAINED_CKPT / "config.json"
    if marker.exists():
        print(f"Pretrained reference already present: {PRETRAINED_CKPT}")
        return

    print(f"Saving pretrained reference → {PRETRAINED_CKPT} ...")
    PRETRAINED_CKPT.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)
    model.save_pretrained(PRETRAINED_CKPT)
    tok.save_pretrained(PRETRAINED_CKPT)
    print(f"  Saved: {PRETRAINED_CKPT}\n")


# Fine-tuning 

def finetune(seed: int, n_members: int, device: torch.device,
             train_seed: int | None = None, max_epoch: int | None = None,
             out_dir: Path | None = None) -> list[Path]:
    """
    Train a fresh GPT-Neo on members for max(EPOCH_SWEEP) epochs.
    Saves full model + tokenizer at each epoch in EPOCH_SWEEP (skips if checkpoint exists).
    Logs per-epoch loss to logs/train_N{n}_seed{seed}.csv.
    Returns list of checkpoint Paths actually written this run.
    """
    torch.manual_seed(seed if train_seed is None else train_seed)
    tag = f"N{n_members}_seed{seed}" + ("" if train_seed is None else f"_trainseed{train_seed}")
    out_dir = CKPT_DIR if out_dir is None else out_dir

    # Data
    members, _, _ = load_split(seed, n_members)
    print(f"  Members: {len(members):,}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
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

    # Model + optimiser (fresh per run)
    # Mixed precision: fp32 master weights + bf16 autocast (below). Loading the
    # weights themselves in bf16 loses updates of size ~lr*grad (< bf16 resolution
    # ~2^-8 relative) and saves rounded BF16 checkpoints. Always keep fp32 weights.
    load_dtype = torch.float32
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=load_dtype).to(device)
    model.gradient_checkpointing_enable()   # trade recompute for activation memory
    optimizer = AdamW(
        model.parameters(),
        lr=FINETUNE["learning_rate"],
        weight_decay=FINETUNE["weight_decay"],
    )

    max_epochs  = max(EPOCH_SWEEP) if max_epoch is None else max_epoch
    epoch_set   = {e for e in EPOCH_SWEEP if e <= max_epochs} | {max_epochs}
    grad_accum  = FINETUNE.get("grad_accum_steps", 1)
    saved: list[Path] = []

    # bfloat16 has fp32 exponent range — no loss scaling needed.
    scaler = torch.cuda.amp.GradScaler(enabled=False)

    # CSV log
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"train_{tag}.csv"
    log_fh   = log_path.open("w", newline="", encoding="utf-8")
    writer   = csv.writer(log_fh)
    writer.writerow(["epoch", "train_loss", "elapsed_s"])

    eff_batch = FINETUNE["batch_size"] * grad_accum
    print(f"  Batches/epoch: {len(loader)}  |  Grad accum: {grad_accum}  |  "
          f"Eff. batch: {eff_batch}  |  Max epochs: {max_epochs}  |  Device: {device}")
    print(f"  {'epoch':>5}  {'loss':>9}  {'time':>8}  note")
    print(f"  {'-----':>5}  {'---------':>9}  {'--------':>8}")

    try:
        for epoch in range(1, max_epochs + 1):
            print("Starting fine-tuning...")
            model.train()
            running_loss = 0.0
            t0 = time.time()
            optimizer.zero_grad()

            for step, batch in enumerate(loader):
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
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
                ckpt = out_dir / f"gpt_neo_ft_{tag}_epoch{epoch}"
                if (ckpt / "config.json").exists():
                    note = f"skip (exists): {ckpt.name}"
                else:
                    ckpt.mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(ckpt)
                    tokenizer.save_pretrained(ckpt)
                    saved.append(ckpt)
                    note = f"saved → {ckpt.name}"

            print(f"  {epoch:>5}  {avg_loss:>9.4f}  {elapsed:>7.1f}s  {note}")
    finally:
        log_fh.close()

    return saved


# Entry point

def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune GPT-Neo on Enron members.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Single seed to run (default: all SEEDS from config)")
    parser.add_argument("--n",   type=int, default=None,
                        help="Single N_members to run (default: all CORPUS_SIZES from config)")
    parser.add_argument("--train_seed", type=int, default=None,
                        help="Training RNG seed (default: = --seed). Requires single --seed and --n.")
    parser.add_argument("--max_epoch", type=int, default=None,
                        help="Stop after this epoch (default: max(EPOCH_SWEEP)); always saved.")
    parser.add_argument("--out_dir", type=Path, default=None,
                        help="Checkpoint output dir (default: CKPT_DIR from config).")
    args = parser.parse_args()
    if args.train_seed is not None and (args.seed is None or args.n is None):
        parser.error("--train_seed requires --seed and --n")

    seeds = [args.seed] if args.seed is not None else SEEDS
    ns    = [args.n]    if args.n    is not None else CORPUS_SIZES

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device     : {device}")
    print(f"Seeds      : {seeds}")
    print(f"N values   : {ns}")
    print(f"Epoch sweep: {EPOCH_SWEEP}  (max={max(EPOCH_SWEEP)})")
    print(f"Batch size : {FINETUNE['batch_size']}  |  LR: {FINETUNE['learning_rate']}\n")

    save_pretrained_reference()

    t_wall   = time.time()
    n_combos = len(seeds) * len(ns)
    all_saved: list[Path] = []

    for i, seed in enumerate(seeds):
        for j, n in enumerate(ns):
            combo = i * len(ns) + j + 1
            print(f"\n{'='*64}")
            print(f"  Run {combo}/{n_combos}  |  seed={seed}  N={n:,}")
            print(f"{'='*64}")
            print(f"Starting fine-tuning for seed={seed} and N={n:,} ...")
            ckpts = finetune(seed, n, device, train_seed=args.train_seed,
                             max_epoch=args.max_epoch, out_dir=args.out_dir)
            all_saved.extend(ckpts)

    total = time.time() - t_wall
    print(f"\n{'='*64}")
    print(f"Checkpoints written this run ({len(all_saved)}):")
    for p in all_saved:
        print(f"  {p.relative_to(_ROOT)}")
    print(f"\nTotal wall time: {total:.1f}s  ({total / 60:.1f} min)")


if __name__ == "__main__":
    main()
