"""
Metrics and bootstrap confidence intervals.

extension.md correction #1: with a single seed, the LoRA-vs-MiCA style pairwise
gap is unclaimable. The zero-GPU-hour mitigation is bootstrap CIs over
*evaluation examples* -- resample the member and non-member pools 1000x and
report the resulting band. That is an honest statement of evaluation-sampling
uncertainty, and it is what every number in E1 gets.

This matters most for TPR@0.1% FPR: with 2000 non-members, an FPR of 0.1% is
two negatives, so the point estimate is quantised to multiples of 1/2000 and is
close to meaningless without a band around it.

Point estimates delegate to metrics/compute_metrics.py so E1 reports exactly
the same estimators as the workshop pipeline. The bootstrap loop uses
equivalent fast numpy/scipy implementations (test_stats.py asserts they agree
with the sklearn versions) because sklearn's roc_curve is too slow to run
1000x per checkpoint per attack.
"""

from __future__ import annotations

import math
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.stats import ks_2samp, rankdata

_E1_DIR = Path(__file__).parent.resolve()
_ROOT = _E1_DIR.parent
for _p in (str(_E1_DIR), str(_ROOT), str(_ROOT / "metrics")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config_e1 import (
    BOOTSTRAP_CI,
    BOOTSTRAP_N,
    FPR_THRESHOLDS,
    KL_BIN_SENSITIVITY,
    TV_N_BINS,
)

# Reused verbatim from the workshop pipeline -- same estimators, same numbers.
from compute_metrics import (  # noqa: E402
    compute_advantage,
    compute_auroc,
    compute_bounds_from_kl,
    compute_kl_score_space,
    compute_tpr_at_fpr,
    compute_tv_empirical,
)


# Regime boundary

def pinsker_bh_crossover() -> float:
    """The KL at which Pinsker sqrt(KL/2) and Bretagnolle-Huber
    sqrt(1-exp(-KL)) cross: the nonzero root of KL/2 = 1 - exp(-KL).

    Below it Pinsker is the tighter bound (regime R1); above it BH is (R2), and
    Pinsker goes vacuous entirely at KL = 2. RQ2 asks whether this crossover
    marks the regime boundary regardless of which knob produced the KL, so the
    value needs to be exact rather than eyeballed."""
    return brentq(lambda k: k / 2.0 - (1.0 - math.exp(-k)), 1e-6, 10.0)


def tv_bound(kl: float) -> tuple[float, float, float]:
    """(pinsker, bh, bound) where bound = min(pinsker, bh), the usable ceiling.

    Taking BH alone overstates the ceiling below the crossover; taking Pinsker
    alone makes it vacuous above KL=2."""
    pinsker, bh = compute_bounds_from_kl(kl)
    return pinsker, bh, min(pinsker, bh)


# Fast estimators for the bootstrap inner loop

def _auroc_fast(m: np.ndarray, nm: np.ndarray) -> float:
    """Mann-Whitney U form of AUROC, with average ranks for ties."""
    n_m, n_nm = len(m), len(nm)
    if n_m == 0 or n_nm == 0:
        return float("nan")
    ranks = rankdata(np.concatenate([m, nm]))
    return float((ranks[:n_m].sum() - n_m * (n_m + 1) / 2.0) / (n_m * n_nm))


def _tpr_at_fpr_fast(m: np.ndarray, nm: np.ndarray, fpr: float) -> float:
    """Highest TPR achievable at FPR <= `fpr`.

    k = floor(fpr * n_nm) non-members may be admitted. The threshold is the
    k-th largest non-member score; k = 0 means the threshold sits strictly
    above every non-member."""
    n_m, n_nm = len(m), len(nm)
    if n_m == 0 or n_nm == 0:
        return float("nan")
    k = int(math.floor(fpr * n_nm))
    nm_sorted = np.sort(nm)[::-1]
    if k == 0:
        return float(np.mean(m > nm_sorted[0]))
    thresh = nm_sorted[k - 1]
    return float(np.mean(m >= thresh))


def _advantage_fast(m: np.ndarray, nm: np.ndarray) -> float:
    """max(TPR - FPR) over the ROC curve, which is exactly the two-sample
    Kolmogorov-Smirnov statistic between the two score distributions.

    Computed directly from the empirical ROC and therefore independent of any
    KL or TV estimate -- that independence is what makes 'Adv <= bound' a real
    check rather than an identity."""
    if len(m) == 0 or len(nm) == 0:
        return float("nan")
    return float(ks_2samp(m, nm).statistic)


# Point estimates + bootstrap

_STAT_KEYS = [
    "auroc",
    "tpr_at_fpr_1pct",
    "tpr_at_fpr_01pct",
    "adv",
    "tv_empirical",
    "kl_score_space",
    "pinsker",
    "bh",
    "bound",
    "tightness_gap",
    "adv_over_bound",
]


def _all_stats_fast(m: np.ndarray, nm: np.ndarray) -> dict[str, float]:
    kl = compute_kl_score_space(list(m), list(nm), n_bins=TV_N_BINS)
    pinsker, bh, bound = tv_bound(kl)
    adv = _advantage_fast(m, nm)
    return {
        "auroc": _auroc_fast(m, nm),
        "tpr_at_fpr_1pct": _tpr_at_fpr_fast(m, nm, FPR_THRESHOLDS[0]),
        "tpr_at_fpr_01pct": _tpr_at_fpr_fast(m, nm, FPR_THRESHOLDS[1]),
        "adv": adv,
        "tv_empirical": compute_tv_empirical(list(m), list(nm), n_bins=TV_N_BINS),
        "kl_score_space": kl,
        "pinsker": pinsker,
        "bh": bh,
        "bound": bound,
        "tightness_gap": bound - adv,
        "adv_over_bound": adv / bound if bound > 0 else float("nan"),
    }


def point_estimates(m: np.ndarray, nm: np.ndarray) -> dict[str, float]:
    """Headline numbers, via the workshop's sklearn-based estimators."""
    m_l, nm_l = list(map(float, m)), list(map(float, nm))
    kl = compute_kl_score_space(m_l, nm_l, n_bins=TV_N_BINS)
    pinsker, bh, bound = tv_bound(kl)
    adv = compute_advantage(m_l, nm_l)
    return {
        "auroc": compute_auroc(m_l, nm_l),
        "tpr_at_fpr_1pct": compute_tpr_at_fpr(m_l, nm_l, FPR_THRESHOLDS[0]),
        "tpr_at_fpr_01pct": compute_tpr_at_fpr(m_l, nm_l, FPR_THRESHOLDS[1]),
        "adv": adv,
        "tv_empirical": compute_tv_empirical(m_l, nm_l, n_bins=TV_N_BINS),
        "kl_score_space": kl,
        "pinsker": pinsker,
        "bh": bh,
        "bound": bound,
        "tightness_gap": bound - adv,
        "adv_over_bound": adv / bound if bound > 0 else float("nan"),
    }


def kl_bin_sensitivity(
    m: np.ndarray,
    nm: np.ndarray,
    bin_grid: list[int] | None = None,
) -> dict[str, float]:
    """Score-space KL and the resulting ceiling recomputed at several bin
    counts.

    The plug-in histogram KL is bin-count dependent (see the table in
    config_e1.py), so the paper has to show that the RQ1 conclusion is not an
    artifact of one arbitrary choice. These columns let a reviewer see the
    whole sensitivity curve rather than take TV_N_BINS on trust."""
    grid = bin_grid if bin_grid is not None else KL_BIN_SENSITIVITY
    out: dict[str, float] = {}
    for b in grid:
        kl = compute_kl_score_space(list(m), list(nm), n_bins=b)
        out[f"kl_bins_{b}"] = kl
        out[f"bound_bins_{b}"] = tv_bound(kl)[2]
    return out


def bootstrap(
    m: np.ndarray,
    nm: np.ndarray,
    n_resamples: int = BOOTSTRAP_N,
    ci: float = BOOTSTRAP_CI,
    seed: int = 0,
) -> dict[str, tuple[float, float]]:
    """Percentile CIs over evaluation examples.

    Members and non-members are resampled independently with replacement,
    preserving both pool sizes. This is uncertainty from *which examples were
    evaluated*, not from training randomness -- the paper must say so, since
    with one seed there is no training-variance estimate at all."""
    rng = np.random.default_rng(seed)
    n_m, n_nm = len(m), len(nm)
    draws: dict[str, list[float]] = {k: [] for k in _STAT_KEYS}

    for _ in range(n_resamples):
        bm = m[rng.integers(0, n_m, n_m)]
        bnm = nm[rng.integers(0, n_nm, n_nm)]
        for k, v in _all_stats_fast(bm, bnm).items():
            draws[k].append(v)

    lo_q, hi_q = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    out = {}
    for k, vals in draws.items():
        arr = np.asarray(vals, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            out[k] = (float("nan"), float("nan"))
        else:
            out[k] = (float(np.percentile(arr, lo_q)), float(np.percentile(arr, hi_q)))
    return out


def evaluate(
    member_scores: np.ndarray,
    nonmember_scores: np.ndarray,
    n_resamples: int = BOOTSTRAP_N,
    seed: int = 0,
) -> dict[str, float]:
    """Point estimates plus <stat>_ci_lo / <stat>_ci_hi for every statistic.

    Drops NaN scores first (zlib returns NaN on empty compression, Min-K% on
    sequences shorter than 5 tokens) so the two pools stay comparable."""
    m = np.asarray(member_scores, dtype=float)
    nm = np.asarray(nonmember_scores, dtype=float)
    m = m[np.isfinite(m)]
    nm = nm[np.isfinite(nm)]

    if len(m) < 2 or len(nm) < 2:
        return {"n_members_scored": len(m), "n_nonmembers_scored": len(nm)}

    # The plug-in histogram KL needs enough samples per bin to mean anything.
    # Below ~20 per bin it is dominated by discretisation noise and reports
    # implausibly large values (a 24-vs-24 smoke run reads KL ~ 17), which
    # would silently inflate the ceiling. E1's real pools give 100+ per bin.
    per_bin = min(len(m), len(nm)) / TV_N_BINS
    if per_bin < 20:
        warnings.warn(
            f"Only {per_bin:.1f} samples per histogram bin "
            f"(min pool {min(len(m), len(nm))}, TV_N_BINS={TV_N_BINS}). "
            f"kl_score_space and every bound derived from it are unreliable "
            f"below ~20 per bin; treat this row as diagnostic only.",
            RuntimeWarning,
            stacklevel=2,
        )

    out = point_estimates(m, nm)
    out["samples_per_bin"] = per_bin
    out.update(kl_bin_sensitivity(m, nm))
    if n_resamples > 0:
        for k, (lo, hi) in bootstrap(m, nm, n_resamples=n_resamples, seed=seed).items():
            out[f"{k}_ci_lo"] = lo
            out[f"{k}_ci_hi"] = hi
    out["n_members_scored"] = len(m)
    out["n_nonmembers_scored"] = len(nm)
    return out


# Utility metrics (E5c, folded into the E1c eval pass)

def perplexity_from_logprobs(token_lp: np.ndarray, n_tokens: np.ndarray) -> float:
    """Corpus-level held-out perplexity: exp of the total negative log-likelihood
    divided by the total token count. Weighting by tokens rather than averaging
    per-sequence perplexities is what makes it comparable across corpora with
    different length distributions."""
    total_lp, total_tok = 0.0, 0
    for row, t in zip(token_lp, n_tokens):
        if t <= 0:
            continue
        total_lp += float(np.nansum(row[:t]))
        total_tok += int(t)
    if total_tok == 0:
        return float("nan")
    return float(math.exp(-total_lp / total_tok))


def next_token_accuracy(top1: np.ndarray, n_tokens: np.ndarray) -> float:
    """Fraction of held-out positions where argmax P_ft is the true next token.
    The downstream proxy E5c asks for, on the same forward pass as perplexity."""
    correct, total = 0, 0
    for row, t in zip(top1, n_tokens):
        if t <= 0:
            continue
        correct += int(row[:t].sum())
        total += int(t)
    return float(correct / total) if total else float("nan")
