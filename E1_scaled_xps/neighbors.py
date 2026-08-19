"""
Neighbourhood attack — neighbour generation and caching.

The neighbourhood attack (Mattern et al.) compares log P_ft(x) against the
average log P_ft over semantically-similar perturbations of x. It needs no
reference model, which makes it a genuinely different probe from Ref and RMIA:
it measures the *sharpness* of the likelihood around x rather than its height
relative to a baseline.

Cost warning. Each target needs z neighbours, so this multiplies the forward
passes for one checkpoint by (1 + z) -- at the default z=25 that is 26x. Running
it over all ~105 E1 checkpoints does not fit the Kaggle budget. Run it on a
chosen subset (the headline configs and the epoch endpoints) and say so in the
paper; the other six attacks cover the full grid.

Usage:
    python E1_scaled_xps/neighbors.py --run-id <id> --epoch 20
    python E1_scaled_xps/neighbors.py --run-id <id> --epoch 20 --n-neighbors 10
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_E1_DIR = Path(__file__).parent.resolve()
_ROOT = _E1_DIR.parent
for _p in (str(_E1_DIR), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import hub
from cache_logprobs import SPLIT_CODES, _batch_stats, assemble_pools
from config_e1 import NEIGHBORHOOD
from models import eval_devices, load_checkpoint, load_tokenizer, resolve_dtype
from runspec import RunConfig


@torch.no_grad()
def generate_neighbors(
    texts: list[str],
    n_neighbors: int,
    replacement_frac: float,
    mask_model_name: str,
    device: torch.device,
    batch_size: int = 16,
    seed: int = 0,
) -> tuple[list[str], np.ndarray]:
    """Produce n_neighbors perturbations per input text by masked-token
    substitution, and return them flattened with a parent index.

    A random `replacement_frac` of positions is masked at once and every masked
    position is resampled from the mask model's predictive distribution in a
    single pass. Sampling (rather than taking the argmax) is what keeps the z
    neighbours distinct; excluding the original token at each position is what
    keeps them actual perturbations."""
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(mask_model_name)
    mlm = AutoModelForMaskedLM.from_pretrained(mask_model_name).to(device)
    mlm.eval()

    gen = torch.Generator(device="cpu").manual_seed(seed)
    neighbors: list[str] = []
    parents: list[int] = []

    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        enc = tok(chunk, return_tensors="pt", truncation=True,
                  max_length=256, padding=True)
        ids = enc["input_ids"]
        attn = enc["attention_mask"]

        for rep in range(n_neighbors):
            masked = ids.clone()
            # Only interior, non-special positions are eligible for masking.
            special = torch.tensor(
                [[t in tok.all_special_ids for t in row.tolist()] for row in ids],
                dtype=torch.bool,
            )
            eligible = (attn.bool()) & (~special)
            draw = torch.rand(ids.shape, generator=gen)
            mask_pos = eligible & (draw < replacement_frac)
            if not mask_pos.any():
                # replacement_frac too small for short sequences; force one.
                for i in range(ids.shape[0]):
                    idx = eligible[i].nonzero(as_tuple=True)[0]
                    if len(idx):
                        mask_pos[i, idx[torch.randint(len(idx), (1,), generator=gen)]] = True

            masked[mask_pos] = tok.mask_token_id
            logits = mlm(input_ids=masked.to(device),
                         attention_mask=attn.to(device)).logits
            probs = torch.softmax(logits[mask_pos.to(device)].float(), dim=-1)

            # Never resample the original token: a "neighbour" identical to the
            # target would bias the score toward zero.
            originals = ids[mask_pos].to(device)
            probs.scatter_(1, originals.unsqueeze(1), 0.0)
            probs = probs / probs.sum(dim=1, keepdim=True).clamp(min=1e-12)
            sampled = torch.multinomial(probs, 1).squeeze(1).cpu()

            filled = ids.clone()
            filled[mask_pos] = sampled
            for i in range(filled.shape[0]):
                length = int(attn[i].sum())
                neighbors.append(tok.decode(filled[i, :length], skip_special_tokens=True))
                parents.append(start + i)

        print(f"    generated neighbours for {min(start + batch_size, len(texts)):,}/{len(texts):,} targets")

    del mlm
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return neighbors, np.array(parents, dtype=np.int32)


@torch.no_grad()
def cache_neighbors(
    cfg: RunConfig,
    epoch: int,
    n_neighbors: int,
    replacement_frac: float,
    mask_model: str,
    batch_size: int = 8,
    push: bool = True,
) -> Path:
    """Generate neighbours for the attack pool and cache their log-probs under
    P_ft only -- the neighbourhood attack never touches the reference model."""
    out_path = cfg.cache_path(epoch, pool="neighbors")
    if out_path.exists():
        print(f"  [skip] {out_path.name} already exists")
        return out_path

    ckpt = cfg.ckpt_dir(epoch)
    if not (ckpt / "config.json").exists():
        raise FileNotFoundError(
            f"No checkpoint at {ckpt}. The neighbourhood attack needs the model "
            f"itself, so it cannot run from the log-prob cache alone -- keep the "
            f"checkpoints for the subset of configs you want it on."
        )

    dev_ft, _ = eval_devices()
    tokenizer = load_tokenizer(cfg.model)

    # Only member + nonmember rows are attack targets; population and heldout
    # rows are not scored, so perturbing them would be wasted compute.
    pools = assemble_pools(cfg)
    targets = pools["member"] + pools["nonmember"]
    texts = [r["text"] for r in targets]
    print(f"  Targets: {len(texts):,}  x {n_neighbors} neighbours = {len(texts) * n_neighbors:,} sequences")

    t0 = time.time()
    neighbor_texts, parents = generate_neighbors(
        texts, n_neighbors, replacement_frac, mask_model, dev_ft, seed=cfg.seed
    )
    print(f"  Generation: {time.time() - t0:.1f}s")

    model_ft = load_checkpoint(ckpt, dev_ft, dtype=resolve_dtype(dev_ft))

    T = cfg.max_seq_len - 1
    n = len(neighbor_texts)
    token_lp = np.full((n, T), np.nan, dtype=np.float32)
    n_tokens = np.zeros(n, dtype=np.int32)
    pad_id = tokenizer.pad_token_id

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        chunk = [
            tokenizer(t, return_tensors="pt", truncation=True,
                      max_length=cfg.max_seq_len)["input_ids"].squeeze(0)
            for t in neighbor_texts[start:end]
        ]
        lengths = [len(ids) for ids in chunk]
        L = max(lengths)
        padded = torch.full((len(chunk), L), pad_id, dtype=torch.long)
        mask = torch.zeros((len(chunk), L), dtype=torch.long)
        for i, ids in enumerate(chunk):
            padded[i, : len(ids)] = ids
            mask[i, : len(ids)] = 1

        out = _batch_stats(model_ft, padded.to(dev_ft), mask.to(dev_ft),
                           want_distribution_stats=False)
        for i, length in enumerate(lengths):
            t = length - 1
            if t <= 0:
                continue
            token_lp[start + i, :t] = out["lp"][i, :t].float().cpu().numpy()
            n_tokens[start + i] = t

        if end % (batch_size * 50) == 0 or end == n:
            print(f"    scored {end:,}/{n:,}  [{time.time() - t0:.1f}s]")

    # parent_index must address rows of the *base* cache. The base cache is
    # built from assemble_pools() in dict order (member, nonmember, ...), and
    # targets above is member+nonmember in that same order, so parents already
    # aligns with base-cache row numbers 0..len(targets)-1.
    meta = {
        "run_id": cfg.run_id, "epoch": epoch, "n_neighbors": n_neighbors,
        "replacement_frac": replacement_frac, "mask_model": mask_model,
        "n_targets": len(texts),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        token_lp_ft=token_lp,
        n_tokens=n_tokens,
        parent_index=parents,
        split_code=np.full(n, SPLIT_CODES["member"], dtype=np.int8),  # unused; kept for shape parity
        meta=np.array([json.dumps(meta)]),
    )
    print(f"  wrote {out_path.name} ({out_path.stat().st_size / 1e6:.1f} MB, {time.time() - t0:.1f}s)")

    if push:
        hub.upload_file(out_path, hub.remote_cache_path(cfg.run_id, epoch, "neighbors"))

    del model_ft
    if dev_ft.type == "cuda":
        torch.cuda.empty_cache()
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate and cache neighbours for the neighbourhood attack.")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--epoch", type=int, required=True)
    ap.add_argument("--n-neighbors", type=int, default=NEIGHBORHOOD["n_neighbors"])
    ap.add_argument("--replacement-frac", type=float, default=NEIGHBORHOOD["replacement_frac"])
    ap.add_argument("--mask-model", default=NEIGHBORHOOD["mask_model"])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--no-push", action="store_true")
    args = ap.parse_args()

    runs = [r for r in RunConfig.discover() if r.run_id == args.run_id]
    if not runs:
        raise SystemExit(f"No run config for {args.run_id!r}")

    hub.ensure_repo()
    cache_neighbors(
        runs[0], args.epoch, args.n_neighbors, args.replacement_frac,
        args.mask_model, batch_size=args.batch_size, push=not args.no_push,
    )


if __name__ == "__main__":
    main()
