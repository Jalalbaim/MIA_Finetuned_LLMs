"""
Step 6 — Compute MIA metrics for every (seed, N, epoch, signal) combination.

Outputs: results/metrics_all.csv

Usage:
    python metrics/compute_metrics.py
"""

import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from config import (
    ALL_SIGNALS,
    CORPUS_SIZES,
    EPOCH_SWEEP,
    FPR_THRESHOLDS,
    PRIMARY_SIGNAL,
    RESULTS_COLUMNS,
    RESULTS_DIR,
    SEEDS,
    TV_N_BINS,
)

_SIGNAL_KEY = {s: f"s_{s}" for s in ALL_SIGNALS}


def load_scores(signals_path: Path) -> dict[str, tuple[list[float], list[float]]]:
    members:    dict[str, list[float]] = {s: [] for s in ALL_SIGNALS}
    nonmembers: dict[str, list[float]] = {s: [] for s in ALL_SIGNALS}

    with signals_path.open(encoding="utf-8") as fh:
        for line in fh:
            rec    = json.loads(line)
            target = members if rec["split"] == "member" else nonmembers
            for sig in ALL_SIGNALS:
                val = rec.get(_SIGNAL_KEY[sig])
                if val is not None and not math.isnan(val):
                    target[sig].append(val)

    return {sig: (members[sig], nonmembers[sig]) for sig in ALL_SIGNALS}


def compute_auroc(scores_members: list[float], scores_nonmembers: list[float]) -> float:
    labels = [1] * len(scores_members) + [0] * len(scores_nonmembers)
    scores = scores_members + scores_nonmembers
    return float(roc_auc_score(labels, scores))


def compute_tpr_at_fpr(
    scores_members: list[float],
    scores_nonmembers: list[float],
    fpr_threshold: float,
) -> float:
    labels   = [1] * len(scores_members) + [0] * len(scores_nonmembers)
    scores   = scores_members + scores_nonmembers
    fpr_arr, tpr_arr, _ = roc_curve(labels, scores)
    eligible = fpr_arr <= fpr_threshold
    if not eligible.any():
        return 0.0
    return float(tpr_arr[eligible][-1])


def compute_advantage(
    scores_members: list[float], scores_nonmembers: list[float]
) -> float:
    labels  = [1] * len(scores_members) + [0] * len(scores_nonmembers)
    scores  = scores_members + scores_nonmembers
    fpr_arr, tpr_arr, _ = roc_curve(labels, scores)
    return float(np.max(tpr_arr - fpr_arr))


def compute_tv_empirical(
    scores_members: list[float],
    scores_nonmembers: list[float],
    n_bins: int = TV_N_BINS,
) -> float:
    m  = np.array(scores_members,    dtype=float)
    nm = np.array(scores_nonmembers, dtype=float)
    lo = min(m.min(), nm.min())
    hi = max(m.max(), nm.max())
    if hi == lo:
        return 0.0
    bins      = np.linspace(lo, hi, n_bins + 1)
    bin_width = bins[1] - bins[0]
    hist_m,  _ = np.histogram(m,  bins=bins, density=True)
    hist_nm, _ = np.histogram(nm, bins=bins, density=True)
    return float(0.5 * np.sum(np.abs(hist_m - hist_nm)) * bin_width)


def compute_metrics_for_checkpoint(
    seed: int, n_members: int, epoch: int
) -> list[dict] | None:
    signals_path = RESULTS_DIR / f"signals_N{n_members}_seed{seed}_epoch{epoch}.jsonl"
    kl_path      = RESULTS_DIR / f"kl_N{n_members}_seed{seed}_epoch{epoch}.json"

    if not signals_path.exists():
        print(f"  [skip] Missing signals: {signals_path.name}")
        return None
    if not kl_path.exists():
        print(f"  [skip] Missing KL:      {kl_path.name}")
        return None

    with kl_path.open(encoding="utf-8") as fh:
        kl = json.load(fh)

    all_scores = load_scores(signals_path)
    results: list[dict] = []

    for sig in ALL_SIGNALS:
        sc_m, sc_nm = all_scores[sig]

        if len(sc_m) < 2 or len(sc_nm) < 2:
            print(f"  [warn] Too few scores for signal={sig} — skipping")
            continue

        auroc     = compute_auroc(sc_m, sc_nm)
        tpr_1pct  = compute_tpr_at_fpr(sc_m, sc_nm, FPR_THRESHOLDS[0])
        tpr_01pct = compute_tpr_at_fpr(sc_m, sc_nm, FPR_THRESHOLDS[1])
        adv       = compute_advantage(sc_m, sc_nm)
        tv_emp    = compute_tv_empirical(sc_m, sc_nm)

        bh_seq      = kl["bh_seq"]
        pinsker_seq  = kl["pinsker_seq"]
        adv_over_bound = adv / max(bh_seq, pinsker_seq) if bh_seq != 0.0 else float("nan")

        row = {
            "seed":             seed,
            "n_members":        n_members,
            "epochs":           epoch,
            "signal":           sig,
            "auroc":            auroc,
            "tpr_at_fpr_1pct":  tpr_1pct,
            "tpr_at_fpr_01pct": tpr_01pct,
            "adv":              adv,
            "tv_empirical":     tv_emp,
            "kl_seq":           kl["kl_seq"],
            "kl_tok":           kl["kl_tok"],
            "pinsker_seq":      kl["pinsker_seq"],
            "bh_seq":           bh_seq,
            "pinsker_tok":      kl["pinsker_tok"],
            "bh_tok":           kl["bh_tok"],
            "dp_epsilon":       None,
            "perplexity":       None,
            "adv_over_bound_seq":  adv_over_bound,
        }
        results.append(row)

        if sig == PRIMARY_SIGNAL:
            checks = {
                "auroc > 0.5":                   auroc > 0.5,
                "adv > 0":                       adv > 0,
                "adv <= bh_seq + 1e-6":          adv <= bh_seq + 1e-6,
                "adv <= tv_empirical + 1e-6":    adv <= tv_emp + 1e-6,
                "tv_empirical <= bh_seq + 1e-6": tv_emp <= bh_seq + 1e-6,
            }
            print(
                f"\n  seed={seed} N={n_members} epoch={epoch} | signal={sig}  "
                f"auroc={auroc:.4f}  adv={adv:.4f}  bh_seq={bh_seq:.4f}  "
                f"tv_emp={tv_emp:.4f}  adv/bh={'NaN' if math.isnan(adv_over_bound) else f'{adv_over_bound:.4f}'}"
            )
            print("  Sanity checks:")
            for name, passed in checks.items():
                print(f"    {'PASS' if passed else 'FAIL'}  {name}")

    return results or None


def main() -> None:
    all_rows: list[dict] = []

    for seed in SEEDS:
        for n in CORPUS_SIZES:
            for epoch in EPOCH_SWEEP:
                rows = compute_metrics_for_checkpoint(seed, n, epoch)
                if rows:
                    all_rows.extend(rows)

    if not all_rows:
        print("No results found — nothing to write.")
        return

    out_cols = RESULTS_COLUMNS + ["adv_over_bound_seq"]
    df = pd.DataFrame(all_rows, columns=out_cols)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "metrics_all.csv"
    df.to_csv(out_path, index=False)
    print(f"\nWrote {len(df):,} rows → {out_path}")

    # ── Summary table: PRIMARY_SIGNAL, seed=0, N=2000 ────────────────────────
    sub = df[
        (df["signal"]    == PRIMARY_SIGNAL) &
        (df["seed"]      == 0)              &
        (df["n_members"] == 2000)
    ].sort_values("epochs")

    if not sub.empty:
        print(f"\n  Summary — signal={PRIMARY_SIGNAL}  seed=0  N=2,000")
        hdr = (
            f"  {'epoch':>5} | {'auroc':>6} | {'adv':>6} | {'bh_seq':>6} | "
            f"{'adv/bh':>6} | {'tv_emp':>6} | {'pinsker_seq':>11} | vacuous"
        )
        sep = (
            "  " + "-"*5 + "+" + "-"*7 + "+" + "-"*7 + "+" + "-"*7 + "+"
            + "-"*7 + "+" + "-"*7 + "+" + "-"*12 + "+" + "-"*9
        )
        print(hdr)
        print(sep)
        for _, r in sub.iterrows():
            vac_str = "YES  ←" if r["pinsker_seq"] > 1.0 else "NO"
            adv_bh  = " NaN" if math.isnan(r["adv_over_bound_seq"]) else f"{r['adv_over_bound_seq']:.3f}"
            print(
                f"  {int(r['epochs']):>5} | {r['auroc']:>6.3f} | {r['adv']:>6.3f} | "
                f"{r['bh_seq']:>6.3f} | {adv_bh:>6} | {r['tv_empirical']:>6.3f} | "
                f"  {r['pinsker_seq']:>9.3f}  | {vac_str}"
            )
    else:
        print("\n  (No seed=0 N=2000 rows found for summary table)")

    # ── Global RQ1 checks ────────────────────────────────────────────────────
    print(f"\n{'='*60}")

    violations_bh = df[df["adv"] > df["bh_seq"] + 1e-6]
    print("RQ1 CHECK: Adv̂ <= BH_seq at every point?")
    if violations_bh.empty:
        print(f"  PASS — no violations across {len(df):,} rows")
    else:
        print(f"  FAIL — {len(violations_bh)} violation(s) found!")
        worst = violations_bh.loc[
            (violations_bh["adv"] - violations_bh["bh_seq"]).idxmax()
        ]
        print(
            f"  Worst: seed={worst['seed']} N={worst['n_members']} "
            f"epoch={worst['epochs']} signal={worst['signal']}  "
            f"adv={worst['adv']:.4f}  bh_seq={worst['bh_seq']:.4f}"
        )

    violations_tv = df[df["adv"] > df["tv_empirical"] + 1e-6]
    print("\nRQ1 CHECK: Adv̂ <= TV_empirical at every point?")
    if violations_tv.empty:
        print(f"  PASS — no violations across {len(df):,} rows")
    else:
        print(f"  FAIL — {len(violations_tv)} violation(s) found!")
        worst = violations_tv.loc[
            (violations_tv["adv"] - violations_tv["tv_empirical"]).idxmax()
        ]
        print(
            f"  Worst: seed={worst['seed']} N={worst['n_members']} "
            f"epoch={worst['epochs']} signal={worst['signal']}  "
            f"adv={worst['adv']:.4f}  tv_empirical={worst['tv_empirical']:.4f}"
        )


if __name__ == "__main__":
    main()
