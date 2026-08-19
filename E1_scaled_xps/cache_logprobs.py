"""
Per-token log-prob cache — the single biggest efficiency lever in E1.

The workshop pipeline (signals/compute_signals.py) runs two forward passes per
sequence at batch size 1, derives four scalars, and discards the per-token
log-probs. Every new attack therefore costs a full re-run over every
checkpoint. With seven attacks and ~105 checkpoints that does not fit the
compute budget.

Here each checkpoint is forward-passed exactly once, batched, and everything an
attack could need is written to one .npz:

    token_lp_ft    (N, T) log P_ft(x_t | x_<t)   for the realized token
    token_lp_pre   (N, T) log P_pre(x_t | x_<t)
    mu_ft          (N, T) E_{v ~ P_ft}[log P_ft(v | x_<t)]   = -H(P_ft)
    sigma_ft       (N, T) std_{v ~ P_ft}[log P_ft(v | x_<t)]
    top1_ft        (N, T) bool, argmax P_ft == x_t           (utility proxy)
    n_tokens       (N,)   predicted positions per sequence
    seq_id         (N,)   pool record id
    split_code     (N,)   0 nonmember | 1 member | 2 population | 3 heldout
    zlib_len       (N,)   len(zlib.compress(text))

mu_ft/sigma_ft are what make Min-K%++ possible; they cannot be recovered from
realized-token log-probs after the fact, and they cost nothing to compute while
the logits are already in memory.

After this, all seven attacks, perplexity, next-token accuracy, and every
bootstrap CI run on CPU in minutes. Adding an eighth attack later is free.

Usage:
    python E1_scaled_xps/cache_logprobs.py --run-id pythia410m_enron_N2000_lr5e-5_seed42_a3f2
    python E1_scaled_xps/cache_logprobs.py --all           # every run/epoch missing a cache
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_E1_DIR = Path(__file__).parent.resolve()
_ROOT = _E1_DIR.parent
for _p in (str(_E1_DIR), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import hub
from config_e1 import MAX_SEQ_LEN, RMIA
from corpora import load_split
from models import (
    eval_devices,
    load_base_model,
    load_checkpoint,
    load_tokenizer,
    reference_model_key,
    resolve_dtype,
)
from runspec import RunConfig

SPLIT_CODES = {"nonmember": 0, "member": 1, "population": 2, "heldout": 3}

# Eval-time batch. Larger than the training batch because there are no
# gradients or optimizer states resident.
EVAL_BATCH_DEFAULT = 16


# Batched forward pass

@torch.no_grad()
def _batch_stats(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    want_distribution_stats: bool,
) -> dict[str, torch.Tensor]:
    """One forward pass over a padded batch.

    Returns per-position tensors of shape (B, T) where T = L-1, with positions
    beyond each sequence's true length left as NaN/False by the caller's mask.
    """
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    shift_logits = logits[:, :-1, :].float()      # (B, T, V)
    labels = input_ids[:, 1:]                     # (B, T)

    log_probs = F.log_softmax(shift_logits, dim=-1)
    realized = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)   # (B, T)

    out = {"lp": realized}

    if want_distribution_stats:
        p = log_probs.exp()
        # mu = E_{v~p}[log p(v)] = -H(p);  sigma = sqrt(E[log p^2] - mu^2).
        # Both are over the full vocab under P_ft, which is exactly the
        # normalisation Min-K%++ calibrates each position by.
        mu = (p * log_probs).sum(dim=-1)                                # (B, T)
        second = (p * log_probs.pow(2)).sum(dim=-1)
        sigma = (second - mu.pow(2)).clamp(min=0.0).sqrt()
        out["mu"] = mu
        out["sigma"] = sigma
        out["top1"] = (shift_logits.argmax(dim=-1) == labels)

    return out


def _encode(records, tokenizer, max_seq_len: int) -> list[torch.Tensor]:
    return [
        tokenizer(
            r["text"],
            return_tensors="pt",
            truncation=True,
            max_length=max_seq_len,
        )["input_ids"].squeeze(0)
        for r in records
    ]


def build_cache(
    cfg: RunConfig,
    epoch: int,
    pools: dict[str, list[dict]],
    model_ft,
    model_pre,
    tokenizer,
    dev_ft: torch.device,
    dev_pre: torch.device,
    out_path: Path,
    batch_size: int = EVAL_BATCH_DEFAULT,
    reference_cache: Path | None = None,
) -> Path:
    """Forward-pass the models over every pool once and write the .npz.

    P_ft and P_pre are kept on separate devices when two GPUs are visible (the
    2xT4 trick), so the two passes do not serialise on one card.

    `reference_cache` points at a stored P_pre pass for this run. P_pre is
    identical at every epoch, so it is computed on the first checkpoint and
    reused thereafter -- roughly halving the eval cost, which at 410M dominates
    training (507s of caching against 59s of training per epoch).
    """
    records: list[dict] = []
    codes: list[int] = []
    for split_name, recs in pools.items():
        code = SPLIT_CODES[split_name]
        records.extend(recs)
        codes.extend([code] * len(recs))

    n = len(records)
    if n == 0:
        raise ValueError("No records to cache.")

    encoded = _encode(records, tokenizer, cfg.max_seq_len)
    T = cfg.max_seq_len - 1

    token_lp_ft = np.full((n, T), np.nan, dtype=np.float32)
    token_lp_pre = np.full((n, T), np.nan, dtype=np.float32)
    mu_ft = np.full((n, T), np.nan, dtype=np.float32)
    sigma_ft = np.full((n, T), np.nan, dtype=np.float32)
    top1_ft = np.zeros((n, T), dtype=bool)
    n_tokens = np.zeros(n, dtype=np.int32)

    pad_id = tokenizer.pad_token_id
    t0 = time.time()

    seq_ids = np.array([r["id"] for r in records], dtype=np.int32)

    # Reuse a stored P_pre pass when one exists for exactly these sequences.
    # The identity check is on the realized (id, split) layout: a reference
    # cache built for different pools would silently misalign rows.
    ref_lp: np.ndarray | None = None
    if reference_cache is not None and Path(reference_cache).exists():
        with np.load(reference_cache, allow_pickle=False) as rz:
            if (rz["seq_id"].shape == seq_ids.shape
                    and np.array_equal(rz["seq_id"], seq_ids)
                    and np.array_equal(rz["split_code"], np.array(codes, dtype=np.int8))
                    and rz["token_lp_pre"].shape == (n, T)):
                ref_lp = rz["token_lp_pre"]
                token_lp_pre = ref_lp.copy()
                n_tokens = rz["n_tokens"].copy()
                print("    reusing stored P_pre pass (reference model is epoch-invariant)")
            else:
                print("    [warn] reference cache does not match these pools; recomputing P_pre")

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        chunk = encoded[start:end]
        lengths = [len(ids) for ids in chunk]
        L = max(lengths)

        padded = torch.full((len(chunk), L), pad_id, dtype=torch.long)
        mask = torch.zeros((len(chunk), L), dtype=torch.long)
        for i, ids in enumerate(chunk):
            padded[i, : len(ids)] = ids
            mask[i, : len(ids)] = 1

        ft = _batch_stats(
            model_ft, padded.to(dev_ft), mask.to(dev_ft), want_distribution_stats=True
        )
        pre = None
        if ref_lp is None:
            pre = _batch_stats(
                model_pre, padded.to(dev_pre), mask.to(dev_pre), want_distribution_stats=False
            )

        for i, length in enumerate(lengths):
            t = length - 1                       # predicted positions
            if t <= 0:
                continue
            row = start + i
            n_tokens[row] = t
            token_lp_ft[row, :t] = ft["lp"][i, :t].float().cpu().numpy()
            if pre is not None:
                token_lp_pre[row, :t] = pre["lp"][i, :t].float().cpu().numpy()
            mu_ft[row, :t] = ft["mu"][i, :t].float().cpu().numpy()
            sigma_ft[row, :t] = ft["sigma"][i, :t].float().cpu().numpy()
            top1_ft[row, :t] = ft["top1"][i, :t].cpu().numpy()

        if (end % (batch_size * 20) == 0) or end == n:
            print(f"    {end:>6,}/{n:,}  [{time.time() - t0:.1f}s]")

    meta = {
        "run_id": cfg.run_id,
        "epoch": epoch,
        "model": cfg.model,
        "corpus": cfg.corpus,
        "n_members": cfg.n_members,
        "seed": cfg.seed,
        "lr": cfg.lr,
        "max_seq_len": cfg.max_seq_len,
        "reference_model": reference_model_key(cfg.model),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        token_lp_ft=token_lp_ft,
        token_lp_pre=token_lp_pre,
        mu_ft=mu_ft,
        sigma_ft=sigma_ft,
        top1_ft=top1_ft,
        n_tokens=n_tokens,
        seq_id=seq_ids,
        split_code=np.array(codes, dtype=np.int8),
        zlib_len=np.array(
            [len(zlib.compress(r["text"].encode("utf-8"))) for r in records],
            dtype=np.int32,
        ),
        meta=np.array([json.dumps(meta)]),
    )
    size_mb = out_path.stat().st_size / 1e6
    print(f"    wrote {out_path.name}  ({n:,} sequences, {size_mb:.1f} MB, {time.time() - t0:.1f}s)")

    # First checkpoint of this run pays for the P_pre pass; store it so the
    # remaining six grid epochs skip it entirely.
    if reference_cache is not None and ref_lp is None:
        Path(reference_cache).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            reference_cache,
            token_lp_pre=token_lp_pre,
            n_tokens=n_tokens,
            seq_id=seq_ids,
            split_code=np.array(codes, dtype=np.int8),
        )
        print(f"    stored reference P_pre pass -> {Path(reference_cache).name}")

    return out_path


def load_cache(path: Path) -> dict:
    """Read a cache back. Returns arrays plus the parsed meta dict."""
    with np.load(path, allow_pickle=False) as z:
        out = {k: z[k] for k in z.files if k != "meta"}
        out["meta"] = json.loads(str(z["meta"][0]))
    return out


# Pool assembly

def assemble_pools(cfg: RunConfig, include_population: bool = True) -> dict[str, list[dict]]:
    """members + nonmembers form the attack pool. A disjoint slice of the eval
    split serves as RMIA's population set, and a further disjoint slice is the
    held-out set used for perplexity / next-token accuracy (E5c).

    The three slices never overlap, so the utility measurement is not
    contaminated by the population set used to calibrate RMIA."""
    members, nonmembers, eval_set = load_split(cfg.corpus, cfg.n_members, cfg.seed)

    pools: dict[str, list[dict]] = {"member": members, "nonmember": nonmembers}

    n_pop = RMIA["n_population"] if include_population else 0
    n_pop = min(n_pop, max(0, len(eval_set) // 2))
    if n_pop:
        pools["population"] = eval_set[:n_pop]
    heldout = eval_set[n_pop:]
    if heldout:
        pools["heldout"] = heldout[: RMIA["n_population"]]

    return pools


# Driver

def cache_run(
    cfg: RunConfig,
    epochs: list[int] | None = None,
    batch_size: int = EVAL_BATCH_DEFAULT,
    push: bool = True,
    overwrite: bool = False,
    model_pre=None,
    tokenizer=None,
) -> list[Path]:
    """Build caches for every requested epoch of one run."""
    targets = epochs if epochs is not None else cfg.completed_epochs()
    targets = [e for e in targets if overwrite or not cfg.cache_path(e).exists()]
    if not targets:
        print(f"  [skip] {cfg.run_id}: all caches present")
        return []

    dev_ft, dev_pre = eval_devices()
    eval_dtype = resolve_dtype(dev_ft)

    if tokenizer is None:
        tokenizer = load_tokenizer(cfg.model)
    if model_pre is None:
        print(f"  Loading reference model {reference_model_key(cfg.model)} on {dev_pre} ...")
        model_pre = load_base_model(reference_model_key(cfg.model), dev_pre, dtype=eval_dtype)
        model_pre.eval()

    pools = assemble_pools(cfg)
    print(f"  Pools: " + "  ".join(f"{k}={len(v):,}" for k, v in pools.items()))

    written: list[Path] = []
    for epoch in targets:
        ckpt = cfg.ckpt_dir(epoch)
        if not (ckpt / "config.json").exists():
            print(f"  [skip] epoch {epoch}: no checkpoint at {ckpt}")
            continue
        print(f"\n  epoch {epoch}: loading {ckpt.name} on {dev_ft} ...")
        model_ft = load_checkpoint(ckpt, dev_ft, dtype=eval_dtype)

        out = build_cache(
            cfg, epoch, pools, model_ft, model_pre, tokenizer,
            dev_ft, dev_pre, cfg.cache_path(epoch), batch_size=batch_size,
            reference_cache=cfg.reference_cache_path,
        )
        written.append(out)

        if push:
            hub.upload_file(out, hub.remote_cache_path(cfg.run_id, epoch, "attack"))

        del model_ft
        if dev_ft.type == "cuda":
            torch.cuda.empty_cache()

    return written


def main() -> None:
    ap = argparse.ArgumentParser(description="Build per-token log-prob caches for E1 checkpoints.")
    ap.add_argument("--run-id", default=None, help="Single run to cache")
    ap.add_argument("--all", action="store_true", help="Every run discovered on disk")
    ap.add_argument("--epoch", type=int, default=None, help="Single epoch (default: all completed)")
    ap.add_argument("--batch-size", type=int, default=EVAL_BATCH_DEFAULT)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no-push", action="store_true", help="Skip HF Hub upload")
    args = ap.parse_args()

    runs = RunConfig.discover()
    if args.run_id:
        runs = [r for r in runs if r.run_id == args.run_id]
        if not runs:
            raise SystemExit(f"No run with id {args.run_id!r} under runs/")
    elif not args.all:
        ap.error("Specify --run-id or --all")

    hub.ensure_repo()
    dev_ft, dev_pre = eval_devices()
    print(f"Devices : P_ft={dev_ft}  P_pre={dev_pre}")
    print(f"Runs    : {len(runs)}")

    # Group by model so the reference model is loaded once per family, not
    # once per run -- it is identical across every run of the same size.
    by_model: dict[str, list[RunConfig]] = {}
    for r in runs:
        by_model.setdefault(r.model, []).append(r)

    t0 = time.time()
    for model_key, group in by_model.items():
        ref_key = reference_model_key(model_key)
        eval_dtype = resolve_dtype(dev_ft)
        print(f"\n{'#'*60}\n  {model_key}: loading reference {ref_key} once for {len(group)} run(s)")
        tokenizer = load_tokenizer(model_key)
        model_pre = load_base_model(ref_key, dev_pre, dtype=eval_dtype)
        model_pre.eval()

        for cfg in group:
            print(f"\n{'='*60}\n  {cfg.run_id}")
            cache_run(
                cfg,
                epochs=[args.epoch] if args.epoch else None,
                batch_size=args.batch_size,
                push=not args.no_push,
                overwrite=args.overwrite,
                model_pre=model_pre,
                tokenizer=tokenizer,
            )

        del model_pre
        if dev_pre.type == "cuda":
            torch.cuda.empty_cache()

    print(f"\nTotal: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
