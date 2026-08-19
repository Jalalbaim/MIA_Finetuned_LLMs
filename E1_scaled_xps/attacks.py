"""
Attack signals, computed from the log-prob cache.

Every function here is pure: cache arrays in, one score per sequence out, with
the convention that **higher means more member-like**. No GPU, no model
loading. Adding an eighth attack costs a function, not a re-run.

Roster:
    loss          mean log P_ft(x)                                  [Yeom et al.]
    ref           mean log P_ft(x) - mean log P_pre(x)              [Carlini et al., LiRA-style calibration]
    zlib          mean log P_ft(x) / len(zlib.compress(text))       [Carlini et al.]
    mink          mean of the lowest k% token log-probs             [Shi et al., Min-K%]
    minkpp        same, on per-position z-scores                    [Zhang et al., Min-K%++]
    rmia          pairwise likelihood-ratio vs a population set     [Zarifzadeh et al.]
    neighborhood  log-prob gap to perturbed neighbours              [Mattern et al.]

The first four reproduce signals/s_{loss,ref,zlib,mink}.py exactly (asserted in
tests/test_attacks.py); they are vectorised here only because looping 4000
sequences x 105 checkpoints x 1000 bootstrap resamples in torch is needlessly
slow.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

_E1_DIR = Path(__file__).parent.resolve()
if str(_E1_DIR) not in sys.path:
    sys.path.insert(0, str(_E1_DIR))

from cache_logprobs import SPLIT_CODES
from config_e1 import MIN_K_FRACTION, RMIA


def _nanmean_rows(arr: np.ndarray) -> np.ndarray:
    """Row means over valid (non-pad) positions. Padding is NaN by
    construction, so nanmean is the mask."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN rows -> NaN, intended
        return np.nanmean(arr, axis=1)


def _nansum_rows(arr: np.ndarray) -> np.ndarray:
    return np.nansum(arr, axis=1)


def _lowest_k_mean(arr: np.ndarray, n_tokens: np.ndarray, k_fraction: float) -> np.ndarray:
    """Mean of the lowest ceil(k_fraction * T) valid values per row.

    Matches signals/s_mink.py, including its guard: rows with fewer than 4
    predicted positions return NaN rather than a score derived from one or two
    tokens."""
    out = np.full(len(arr), np.nan, dtype=float)
    for i, t in enumerate(n_tokens):
        t = int(t)
        if t < 4:
            continue
        vals = arr[i, :t]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        k = max(1, int(np.ceil(k_fraction * vals.size)))
        out[i] = float(np.sort(vals)[:k].mean())
    return out


# Individual signals

def s_loss(cache: dict) -> np.ndarray:
    return _nanmean_rows(cache["token_lp_ft"])


def s_ref(cache: dict) -> np.ndarray:
    """Calibrated log-likelihood ratio against the pretrained reference. The
    primary signal: subtracting log P_pre removes the 'this text is just
    intrinsically easy' component that makes raw loss a weak attack."""
    return _nanmean_rows(cache["token_lp_ft"]) - _nanmean_rows(cache["token_lp_pre"])


def s_zlib(cache: dict) -> np.ndarray:
    """mean log P_ft(x) / len(zlib.compress(x)), reproducing signals/s_zlib.py.

    Note the sign convention: the numerator is negative, so a larger compressed
    length pushes the score *up*. This is the workshop implementation and is
    kept identical for continuity; if the intended Carlini ratio is
    perplexity/entropy (where the ordering is inverted), that is a change to
    make deliberately and in both places, not silently here."""
    lens = cache["zlib_len"].astype(float)
    lens[lens == 0] = np.nan
    return _nanmean_rows(cache["token_lp_ft"]) / lens


def s_mink(cache: dict, k_fraction: float = MIN_K_FRACTION) -> np.ndarray:
    """Min-K%: members are recognised by having *no* very-low-probability
    tokens, so the mean of the worst k% separates better than the overall mean."""
    return _lowest_k_mean(cache["token_lp_ft"], cache["n_tokens"], k_fraction)


def s_minkpp(cache: dict, k_fraction: float = MIN_K_FRACTION) -> np.ndarray:
    """Min-K%++: calibrate each position by the mean and standard deviation of
    log P_ft over the *whole vocabulary* at that position before taking the
    lowest k%.

        z_t = (log p(x_t) - mu_t) / sigma_t,   mu_t = E_{v~p}[log p(v)]

    This removes the per-position difficulty confound that plain Min-K% still
    carries. mu_t and sigma_t are cached at forward-pass time precisely because
    they cannot be reconstructed from realized-token log-probs afterwards."""
    sigma = cache["sigma_ft"].copy()
    sigma[sigma <= 0] = np.nan
    z = (cache["token_lp_ft"] - cache["mu_ft"]) / sigma
    return _lowest_k_mean(z, cache["n_tokens"], k_fraction)


def s_rmia(cache: dict, gamma: float | None = None) -> np.ndarray:
    """RMIA: score a target x by how often its likelihood ratio dominates that
    of a random population sequence z.

        score(x) = Pr_{z ~ population} [ (P_ft(x)/P_pre(x)) / (P_ft(z)/P_pre(z)) >= gamma ]

    Rather than asking 'is x's loss low in absolute terms' (LOSS) or 'low
    relative to a reference model' (Ref), RMIA asks whether x is *unusually*
    favoured by fine-tuning compared with the population. That pairwise
    comparison is what gives it its strength at low FPR.

    Log-ratios are length-normalised (mean rather than sum) so that variable
    document lengths do not dominate the comparison -- the same normalisation
    every other signal here uses."""
    gamma = RMIA["gamma"] if gamma is None else gamma
    log_ratio = _nanmean_rows(cache["token_lp_ft"]) - _nanmean_rows(cache["token_lp_pre"])
    pop_mask = cache["split_code"] == SPLIT_CODES["population"]
    if not pop_mask.any():
        raise ValueError(
            "RMIA needs a population pool in the cache. Rebuild it with "
            "cache_logprobs.assemble_pools(include_population=True)."
        )
    pop = log_ratio[pop_mask]
    pop = np.sort(pop[np.isfinite(pop)])
    if pop.size == 0:
        raise ValueError("Population pool contains no finite log-ratios.")

    # ratio_x / ratio_z >= gamma  <=>  log_ratio_x - log_ratio_z >= log(gamma)
    thresholds = log_ratio - np.log(gamma)
    # Fraction of population strictly below each threshold, via binary search.
    counts = np.searchsorted(pop, thresholds, side="right")
    scores = counts.astype(float) / pop.size
    scores[~np.isfinite(log_ratio)] = np.nan
    return scores


def s_neighborhood(cache: dict, neighbor_cache: dict) -> np.ndarray:
    """Neighbourhood attack: compare log P_ft(x) with the average log P_ft over
    semantically-similar perturbations of x.

        score(x) = mean log P_ft(x) - (1/z) * sum_j mean log P_ft(x~_j)

    A memorised sequence sits on a sharp local maximum of the fine-tuned
    likelihood, so perturbing it costs a lot; a merely fluent non-member sits
    on a plateau. Unlike Ref and RMIA this needs no reference model at all,
    which is why it is worth carrying: it probes a different failure mode.

    `neighbor_cache` is produced by neighbors.py and stores z neighbours per
    target, laid out contiguously with a `parent_index` array."""
    base = _nanmean_rows(cache["token_lp_ft"])
    nb_scores = _nanmean_rows(neighbor_cache["token_lp_ft"])
    parent = neighbor_cache["parent_index"]

    sums = np.zeros(len(base))
    counts = np.zeros(len(base))
    finite = np.isfinite(nb_scores)
    np.add.at(sums, parent[finite], nb_scores[finite])
    np.add.at(counts, parent[finite], 1.0)

    with np.errstate(invalid="ignore", divide="ignore"):
        neighbor_mean = np.where(counts > 0, sums / counts, np.nan)
    return base - neighbor_mean


# Dispatch

ATTACKS = {
    "loss": s_loss,
    "ref": s_ref,
    "zlib": s_zlib,
    "mink": s_mink,
    "minkpp": s_minkpp,
    "rmia": s_rmia,
    "neighborhood": s_neighborhood,
}

# Attacks needing artifacts beyond the base cache.
NEEDS_NEIGHBORS = {"neighborhood"}
NEEDS_POPULATION = {"rmia"}


def score(attack: str, cache: dict, neighbor_cache: dict | None = None) -> np.ndarray:
    """Per-sequence scores for one attack, aligned with the cache row order."""
    if attack not in ATTACKS:
        raise KeyError(f"Unknown attack {attack!r}. Known: {sorted(ATTACKS)}")
    if attack in NEEDS_NEIGHBORS:
        if neighbor_cache is None:
            raise ValueError(f"Attack {attack!r} requires a neighbour cache (see neighbors.py).")
        return ATTACKS[attack](cache, neighbor_cache)
    return ATTACKS[attack](cache)


def split_scores(scores: np.ndarray, cache: dict) -> tuple[np.ndarray, np.ndarray]:
    """Partition scores into (members, non-members). Population and held-out
    rows are excluded: they are inputs to the attack, not targets of it."""
    codes = cache["split_code"]
    return (
        scores[codes == SPLIT_CODES["member"]],
        scores[codes == SPLIT_CODES["nonmember"]],
    )
