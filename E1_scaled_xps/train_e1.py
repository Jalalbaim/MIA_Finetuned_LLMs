"""
E1a / E1b — full fine-tuning across the model x corpus x N grid.

Differences from finetune/train.py that matter:

  * fp32 master weights + fp16 autocast + a *live* GradScaler. train.py loads
    the model in bfloat16, autocasts to bfloat16 on top, and wraps the step in
    GradScaler(enabled=False). On a P100 (sm_60) bf16 is unsupported, so that
    path does not run at all on E1's target hardware.
  * Resume. train.py trains max(EPOCH_SWEEP) epochs in one process and saves
    no optimizer state, so a Kaggle session killed at 12h loses everything
    after the last grid epoch. Here every epoch writes training_state.pt, and
    the run resumes mid-grid.
  * A session budget. --max-hours exits cleanly before Kaggle's kill so the
    state is guaranteed flushed and pushed.
  * Checkpoints go to the run directory, not to CKPT_DIR (which config.py
    points at a read-only Kaggle *input* mount -- train.py:175 writes there).
  * Inline caching. Right after a grid checkpoint is saved the model is still
    resident, so the log-prob cache is built immediately. This is what makes
    checkpoints disposable: the ~20MB cache, not the 1.6GB checkpoint, is the
    artifact E1c actually needs.

Usage (Kaggle):
    python E1_scaled_xps/train_e1.py --experiment e1a --model pythia-410m --n 2000
    python E1_scaled_xps/train_e1.py --experiment e1a --model pythia-70m      # all N
    python E1_scaled_xps/train_e1.py --run-id <id> --max-hours 11
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import DataCollatorForLanguageModeling, get_linear_schedule_with_warmup

_E1_DIR = Path(__file__).parent.resolve()
_ROOT = _E1_DIR.parent
for _p in (str(_E1_DIR), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import hub
from cache_logprobs import assemble_pools, build_cache
from config_e1 import (
    E1A,
    E1B,
    EFFECTIVE_BATCH,
    EPOCH_GRID,
    LEARNING_RATE,
    SEED,
    WARMUP_STEPS,
    WEIGHT_DECAY,
)
from corpora import load_split
from models import (
    count_params,
    eval_devices,
    get_spec,
    grad_accum_steps,
    load_base_model,
    load_checkpoint,
    load_tokenizer,
    needs_grad_scaler,
    reference_model_key,
    resolve_dtype,
)
from runspec import RunConfig, expand_grid


class MemberDataset(Dataset):
    """Members, tokenized once and truncated to cfg.max_seq_len.

    dup_factor repeats each member in the training set. E1 always uses 1; E2's
    dedup-vs-4x-duplication arm sets 4, which is why it lives here rather than
    in an E2-specific trainer."""

    def __init__(self, records: list[dict], tokenizer, max_seq_len: int, dup_factor: int = 1):
        self._samples = []
        for rec in records:
            ids = tokenizer(
                rec["text"],
                truncation=True,
                max_length=max_seq_len,
                return_tensors="pt",
            )["input_ids"].squeeze(0)
            for _ in range(dup_factor):
                self._samples.append({"input_ids": ids})

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict:
        return self._samples[idx]


# Reproducibility

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


# Resume

def try_resume(cfg: RunConfig, model, optimizer, scaler, scheduler) -> int:
    """Restore from training_state.pt, pulling it back from the Hub first if
    this is a fresh Kaggle session. Returns the number of epochs already done."""
    state_path = cfg.state_path
    if not state_path.exists():
        hub.download_file(hub.remote_state_path(cfg.run_id), state_path)
    if not state_path.exists():
        return 0

    print(f"  Resuming from {state_path}")
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    if state.get("scaler") is not None and scaler is not None:
        scaler.load_state_dict(state["scaler"])
    if state.get("scheduler") is not None and scheduler is not None:
        scheduler.load_state_dict(state["scheduler"])
    _restore_rng(state["rng"])
    done = int(state["epoch"])
    print(f"  Restored through epoch {done}")
    return done


def save_state(cfg: RunConfig, model, optimizer, scaler, scheduler, epoch: int, push: bool) -> None:
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "rng": _rng_state(),
            "run_id": cfg.run_id,
        },
        cfg.state_path,
    )
    if push:
        hub.upload_file(cfg.state_path, hub.remote_state_path(cfg.run_id))


# Training

def train_one(
    cfg: RunConfig,
    device: torch.device,
    max_hours: float,
    ckpt_dtype: torch.dtype,
    cache_inline: bool,
    push_state: bool,
    push_checkpoints: bool,
    cache_batch_size: int,
) -> bool:
    """Train one RunConfig through its epoch grid. Returns True if the grid
    completed, False if the session budget forced a clean early exit."""
    seed_everything(cfg.seed)
    cfg.save()

    spec = get_spec(cfg.model)
    tokenizer = load_tokenizer(cfg.model)
    members, _, _ = load_split(cfg.corpus, cfg.n_members, cfg.seed)

    dataset = MemberDataset(members, tokenizer, cfg.max_seq_len, cfg.dup_factor)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    accum = grad_accum_steps(cfg.model, cfg.effective_batch)
    loader = DataLoader(
        dataset,
        batch_size=spec.per_device_batch,
        shuffle=True,
        collate_fn=collator,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    # fp32 master weights; autocast handles the forward pass in half precision.
    model = load_base_model(cfg.model, device, dtype=None)
    if spec.needs_grad_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=WEIGHT_DECAY)
    autocast_dtype = resolve_dtype(device)
    scaler = torch.amp.GradScaler("cuda") if needs_grad_scaler(autocast_dtype) else None

    steps_per_epoch = max(1, (len(loader) + accum - 1) // accum)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=steps_per_epoch * cfg.max_epochs,
    )

    done_epochs = try_resume(cfg, model, optimizer, scaler, scheduler)

    print(f"  Params      : {count_params(model):,}")
    print(f"  Members     : {len(members):,}  (dataset {len(dataset):,} after dup x{cfg.dup_factor})")
    print(f"  Batch       : {spec.per_device_batch} x {accum} accum = {spec.per_device_batch * accum} effective")
    print(f"  Autocast    : {autocast_dtype}  GradScaler: {scaler is not None}")
    print(f"  Epoch grid  : {list(cfg.epoch_grid)}  (max {cfg.max_epochs}, {done_epochs} done)")

    # Reference model for inline caching. Small relative to the target, and
    # loading it once here avoids a second full pass over every checkpoint later.
    model_pre = None
    pools = None
    dev_ft, dev_pre = eval_devices()
    if cache_inline:
        eval_dtype = resolve_dtype(dev_pre)
        model_pre = load_base_model(reference_model_key(cfg.model), dev_pre, dtype=eval_dtype)
        model_pre.eval()
        pools = assemble_pools(cfg)
        print(f"  Cache pools : " + "  ".join(f"{k}={len(v):,}" for k, v in pools.items()))

    log_new = not cfg.log_path.exists()
    log_fh = cfg.log_path.open("a", newline="", encoding="utf-8")
    log_writer = csv.writer(log_fh)
    if log_new:
        log_writer.writerow(["epoch", "train_loss", "elapsed_s", "lr"])

    t_start = time.time()
    grid = set(cfg.epoch_grid)
    completed = True

    try:
        for epoch in range(done_epochs + 1, cfg.max_epochs + 1):
            model.train()
            running, n_batches = 0.0, 0
            t0 = time.time()
            optimizer.zero_grad(set_to_none=True)

            for step, batch in enumerate(loader):
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                with torch.autocast(
                    device_type=device.type,
                    dtype=autocast_dtype,
                    enabled=(device.type == "cuda"),
                ):
                    loss = model(**batch).loss

                scaled = loss / accum
                if scaler is not None:
                    scaler.scale(scaled).backward()
                else:
                    scaled.backward()

                running += loss.item()
                n_batches += 1

                if (step + 1) % accum == 0 or (step + 1) == len(loader):
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

            avg_loss = running / max(1, n_batches)
            elapsed = time.time() - t0
            log_writer.writerow([epoch, f"{avg_loss:.6f}", f"{elapsed:.1f}", f"{scheduler.get_last_lr()[0]:.3e}"])
            log_fh.flush()

            note = ""
            if epoch in grid:
                ckpt = cfg.ckpt_dir(epoch)
                ckpt.mkdir(parents=True, exist_ok=True)
                model.to(ckpt_dtype).save_pretrained(ckpt)
                model.to(torch.float32)          # restore master weights for the next epoch
                tokenizer.save_pretrained(ckpt)
                note = f"ckpt -> {ckpt.name}"

                if push_checkpoints:
                    hub.upload_dir(ckpt, hub.remote_ckpt_prefix(cfg.run_id, epoch))

                if cache_inline and not cfg.cache_path(epoch).exists():
                    print(f"    caching log-probs for epoch {epoch} ...")
                    model_ft = load_checkpoint(ckpt, dev_ft, dtype=resolve_dtype(dev_ft))
                    out = build_cache(
                        cfg, epoch, pools, model_ft, model_pre, tokenizer,
                        dev_ft, dev_pre, cfg.cache_path(epoch),
                        batch_size=cache_batch_size,
                    )
                    hub.upload_file(out, hub.remote_cache_path(cfg.run_id, epoch, "attack"))
                    del model_ft
                    if dev_ft.type == "cuda":
                        torch.cuda.empty_cache()
                    note += "  + cache"

            save_state(cfg, model, optimizer, scaler, scheduler, epoch, push=False)
            print(f"  epoch {epoch:>3}  loss {avg_loss:>8.4f}  {elapsed:>7.1f}s  {note}")

            hours = (time.time() - t_start) / 3600.0
            if hours >= max_hours and epoch < cfg.max_epochs:
                print(
                    f"\n  Session budget reached ({hours:.2f}h >= {max_hours}h) after epoch {epoch}.\n"
                    f"  Flushing state and exiting cleanly -- rerun the same command to resume."
                )
                save_state(cfg, model, optimizer, scaler, scheduler, epoch, push=push_state)
                completed = False
                break
        else:
            save_state(cfg, model, optimizer, scaler, scheduler, cfg.max_epochs, push=push_state)
    finally:
        log_fh.close()
        del model
        if model_pre is not None:
            del model_pre
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return completed


# Grid entry point

def build_runs(args) -> list[RunConfig]:
    fixed = dict(
        seed=args.seed,
        lr=args.lr,
        epoch_grid=tuple(args.epochs),
        effective_batch=EFFECTIVE_BATCH,
    )
    if args.run_id:
        runs = [r for r in RunConfig.discover() if r.run_id == args.run_id]
        if not runs:
            raise SystemExit(f"No run config found for {args.run_id!r}")
        return runs

    if args.experiment == "e1a":
        grid = E1A
    elif args.experiment == "e1b":
        grid = E1B
    else:
        raise SystemExit("--experiment must be e1a or e1b (or pass --run-id)")

    models = [args.model] if args.model else grid["models"]
    corpora_ = [args.corpus] if args.corpus else grid["corpora"]
    sizes = [args.n] if args.n else grid["corpus_sizes"]
    return expand_grid(models, corpora_, sizes, **fixed)


def main() -> None:
    ap = argparse.ArgumentParser(description="E1a/E1b full fine-tuning.")
    ap.add_argument("--experiment", choices=["e1a", "e1b"], default=None)
    ap.add_argument("--run-id", default=None, help="Resume one specific run")
    ap.add_argument("--model", default=None, help="Restrict to one model key")
    ap.add_argument("--corpus", default=None, help="Restrict to one corpus")
    ap.add_argument("--n", type=int, default=None, help="Restrict to one N_members")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--lr", type=float, default=LEARNING_RATE)
    ap.add_argument("--epochs", type=int, nargs="+", default=EPOCH_GRID)
    ap.add_argument("--max-hours", type=float, default=11.0,
                    help="Exit cleanly after this many hours (Kaggle kills at 12h)")
    ap.add_argument("--ckpt-dtype", choices=["fp16", "fp32"], default="fp16",
                    help="Saved checkpoint precision. fp16 halves disk; master weights stay fp32.")
    ap.add_argument("--no-cache", action="store_true",
                    help="Skip inline log-prob caching (then run cache_logprobs.py separately)")
    ap.add_argument("--cache-batch-size", type=int, default=16)
    ap.add_argument("--push-checkpoints", action="store_true",
                    help="Upload full checkpoints to the Hub. Off by default -- the caches are "
                         "the artifact E1c needs, and the full grid is ~50GB.")
    ap.add_argument("--no-push-state", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dtype = torch.float16 if args.ckpt_dtype == "fp16" else torch.float32

    runs = build_runs(args)
    hub.ensure_repo()

    print(f"Device : {device}  ({torch.cuda.device_count()} CUDA device(s))")
    print(f"Runs   : {len(runs)}")
    for r in runs:
        print(f"  {r.run_id}")

    t_wall = time.time()
    finished, deferred = [], []
    for i, cfg in enumerate(runs, 1):
        print(f"\n{'='*70}\n  Run {i}/{len(runs)}: {cfg.run_id}\n{'='*70}")
        ok = train_one(
            cfg,
            device=device,
            max_hours=args.max_hours,
            ckpt_dtype=ckpt_dtype,
            cache_inline=not args.no_cache,
            push_state=not args.no_push_state,
            push_checkpoints=args.push_checkpoints,
            cache_batch_size=args.cache_batch_size,
        )
        (finished if ok else deferred).append(cfg.run_id)
        if not ok:
            print("  Stopping the queue: the session budget is spent.")
            break

    print(f"\n{'='*70}")
    print(f"Completed ({len(finished)}):")
    for r in finished:
        print(f"  {r}")
    if deferred:
        print(f"Resumable ({len(deferred)}):")
        for r in deferred:
            print(f"  {r}")
    remaining = [r.run_id for r in runs if r.run_id not in finished and r.run_id not in deferred]
    if remaining:
        print(f"Not started ({len(remaining)}):")
        for r in remaining:
            print(f"  {r}")
    print(f"Wall time: {(time.time() - t_wall) / 60:.1f} min")


if __name__ == "__main__":
    main()
