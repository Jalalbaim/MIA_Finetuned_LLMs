"""
E1c — attack evaluation over every cached checkpoint.

Pure CPU. Reads the .npz caches, runs every attack, computes AUROC /
TPR@{1%,0.1%} / empirical advantage / empirical TV / score-space KL and the
Pinsker & BH ceilings, attaches bootstrap CIs to all of it, folds in E5c's
held-out perplexity and next-token accuracy, and writes one tidy table.

The RQ1 claim is checked here: no attack's empirical advantage may exceed
min(Pinsker, BH). A violation is a fire alarm about the estimator, not a
result -- it is reported loudly and per-row rather than buried in a summary.

Usage:
    python E1_scaled_xps/eval_e1.py                       # everything cached
    python E1_scaled_xps/eval_e1.py --run-id <id>
    python E1_scaled_xps/eval_e1.py --attacks loss ref rmia --bootstrap 200
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

_E1_DIR = Path(__file__).parent.resolve()
_ROOT = _E1_DIR.parent
for _p in (str(_E1_DIR), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import attacks as attack_lib
import stats
from cache_logprobs import SPLIT_CODES, load_cache
from config_e1 import (
    BOOTSTRAP_N,
    E1C_ATTACKS,
    E1_RESULTS_DIR,
    PRIMARY_SIGNAL,
)
from runspec import RunConfig


def neighbor_cache_path(cfg: RunConfig, epoch: int) -> Path:
    return cfg.cache_path(epoch, pool="neighbors")


def utility_metrics(cache: dict) -> dict[str, float]:
    """Held-out perplexity and next-token accuracy (E5c), from the same forward
    pass the attacks use. Measured on the held-out slice, which is disjoint
    from both the attack pool and RMIA's population set."""
    mask = cache["split_code"] == SPLIT_CODES["heldout"]
    if not mask.any():
        return {"perplexity": float("nan"), "next_token_acc": float("nan")}
    return {
        "perplexity": stats.perplexity_from_logprobs(
            cache["token_lp_ft"][mask], cache["n_tokens"][mask]
        ),
        "next_token_acc": stats.next_token_accuracy(
            cache["top1_ft"][mask], cache["n_tokens"][mask]
        ),
    }


def evaluate_checkpoint(
    cfg: RunConfig,
    epoch: int,
    attack_names: list[str],
    n_bootstrap: int,
) -> list[dict]:
    cache_path = cfg.cache_path(epoch)
    if not cache_path.exists():
        print(f"  [skip] no cache: {cache_path.name}")
        return []

    cache = load_cache(cache_path)
    util = utility_metrics(cache)

    nb_cache = None
    nb_path = neighbor_cache_path(cfg, epoch)
    if nb_path.exists():
        nb_cache = load_cache(nb_path)

    rows = []
    for attack in attack_names:
        if attack in attack_lib.NEEDS_NEIGHBORS and nb_cache is None:
            print(f"  [skip] {attack}: no neighbour cache at {nb_path.name}")
            continue
        try:
            scores = attack_lib.score(attack, cache, neighbor_cache=nb_cache)
        except ValueError as exc:
            print(f"  [skip] {attack}: {exc}")
            continue

        m, nm = attack_lib.split_scores(scores, cache)
        result = stats.evaluate(m, nm, n_resamples=n_bootstrap, seed=cfg.seed)
        if "auroc" not in result:
            print(f"  [warn] {attack}: too few finite scores "
                  f"(members={result.get('n_members_scored')}, "
                  f"nonmembers={result.get('n_nonmembers_scored')})")
            continue

        rows.append({
            "run_id": cfg.run_id,
            "model": cfg.model,
            "corpus": cfg.corpus,
            "n_members": cfg.n_members,
            "epochs": epoch,
            "lr": cfg.lr,
            "seed": cfg.seed,
            "attack": attack,
            **result,
            **util,
        })

    if rows:
        primary = next((r for r in rows if r["attack"] == PRIMARY_SIGNAL), rows[0])
        flag = "  <-- VIOLATION" if primary["adv"] > primary["bound"] + 1e-6 else ""
        print(
            f"  epoch {epoch:>3}  [{primary['attack']}]  "
            f"auroc={primary['auroc']:.4f}  adv={primary['adv']:.4f}  "
            f"KL={primary['kl_score_space']:.4f}  bound={primary['bound']:.4f}  "
            f"gap={primary['tightness_gap']:+.4f}  ppl={primary['perplexity']:.2f}{flag}"
        )
    return rows


def rq1_check(df: pd.DataFrame) -> None:
    """Falsifiable claim: no attack exceeds min(Pinsker, BH), anywhere."""
    print(f"\n{'='*70}")
    print("RQ1 -- does any attack exceed the ceiling?")

    viol = df[df["adv"] > df["bound"] + 1e-6]
    if viol.empty:
        print(f"  PASS: 0 violations across {len(df):,} (checkpoint x attack) rows.")
    else:
        print(f"  FAIL: {len(viol)} violation(s). This is a fire alarm about the KL")
        print("  estimator or the member-distribution assumption, not a finding.")
        cols = ["run_id", "epochs", "attack", "adv", "bound", "pinsker", "bh", "kl_score_space"]
        print(viol.sort_values("adv", ascending=False)[cols].head(15).to_string(index=False))

    print("\nRQ1 -- what fraction of the ceiling does the best attack recover?")
    summary = (
        df.loc[df.groupby(["model", "corpus", "n_members", "epochs"])["adv"].idxmax()]
        [["model", "corpus", "n_members", "epochs", "attack", "kl_score_space",
          "adv", "bound", "adv_over_bound", "tightness_gap"]]
        .sort_values(["model", "corpus", "n_members", "epochs"])
    )
    if not summary.empty:
        print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    crossover = stats.pinsker_bh_crossover()
    n_r1 = int((df["kl_score_space"] < crossover).sum())
    print(f"\nRegime split at the exact Pinsker/BH crossover KL = {crossover:.6f}:")
    print(f"  R1 (Pinsker tighter): {n_r1:,} rows   R2 (BH tighter): {len(df) - n_r1:,} rows")


def main() -> None:
    ap = argparse.ArgumentParser(description="E1c attack evaluation over cached checkpoints.")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--epoch", type=int, default=None)
    ap.add_argument("--attacks", nargs="+", default=E1C_ATTACKS)
    ap.add_argument("--bootstrap", type=int, default=BOOTSTRAP_N,
                    help="Bootstrap resamples per statistic; 0 disables CIs")
    ap.add_argument("--out", default=None, help="Output CSV (default results/e1_metrics.csv)")
    args = ap.parse_args()

    runs = RunConfig.discover()
    if args.run_id:
        runs = [r for r in runs if r.run_id == args.run_id]
        if not runs:
            raise SystemExit(f"No run config for {args.run_id!r}")
    if not runs:
        raise SystemExit("No runs found under runs/. Train something first.")

    print(f"Runs      : {len(runs)}")
    print(f"Attacks   : {args.attacks}")
    print(f"Bootstrap : {args.bootstrap} resamples")

    t0 = time.time()
    all_rows: list[dict] = []
    for cfg in runs:
        epochs = [args.epoch] if args.epoch else sorted(cfg.epoch_grid)
        available = [e for e in epochs if cfg.cache_path(e).exists()]
        if not available:
            continue
        print(f"\n{'='*70}\n  {cfg.run_id}  ({len(available)} cached epoch(s))")
        for epoch in available:
            all_rows.extend(evaluate_checkpoint(cfg, epoch, args.attacks, args.bootstrap))

    if not all_rows:
        raise SystemExit("No caches found. Run cache_logprobs.py (or train with inline caching).")

    df = pd.DataFrame(all_rows)
    out_path = Path(args.out) if args.out else E1_RESULTS_DIR / "e1_metrics.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nWrote {len(df):,} rows x {len(df.columns)} columns -> {out_path}")

    rq1_check(df)
    print(f"\nElapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
