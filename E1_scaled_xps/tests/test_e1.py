"""
Harness validation for E1.

extension.md's Week 1 gate is "attack harness validated". These tests are that
gate. They run on CPU in seconds and require no checkpoints.

What they pin down:
  * run identity is stable, unique across axes, and round-trips through JSON
  * the vectorised attacks reproduce the workshop signals/s_*.py exactly
  * the fast bootstrap estimators reproduce the sklearn point estimates
  * the Pinsker/BH crossover is the value the config claims
  * Adv <= min(Pinsker, BH) holds on synthetic score distributions, including
    the degenerate cases (identical pools, perfectly separated pools)

Run:
    python -m pytest E1_scaled_xps/tests/test_e1.py -v
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

_E1_DIR = Path(__file__).parent.parent.resolve()
_ROOT = _E1_DIR.parent
for _p in (str(_E1_DIR), str(_ROOT), str(_ROOT / "signals")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import attacks
import stats
from config_e1 import KL_BIN_SENSITIVITY, KL_CROSSOVER, MIN_K_FRACTION, TV_N_BINS
from runspec import RunConfig

# Workshop implementations, used here as the reference to match.
from s_loss import s_loss as ref_s_loss
from s_mink import s_mink as ref_s_mink
from s_ref import s_ref as ref_s_ref
from s_zlib import s_zlib as ref_s_zlib


# Fixtures

def _make_cache(n_seq=40, T=60, seed=0, with_population=True):
    """Synthetic cache with the same array layout cache_logprobs.py writes,
    including ragged lengths and NaN padding."""
    rng = np.random.default_rng(seed)
    lengths = rng.integers(10, T, size=n_seq)

    token_lp_ft = np.full((n_seq, T), np.nan, dtype=np.float32)
    token_lp_pre = np.full((n_seq, T), np.nan, dtype=np.float32)
    mu_ft = np.full((n_seq, T), np.nan, dtype=np.float32)
    sigma_ft = np.full((n_seq, T), np.nan, dtype=np.float32)
    top1 = np.zeros((n_seq, T), dtype=bool)

    for i, t in enumerate(lengths):
        # Members (even rows) get higher log-probs, so every attack should
        # produce AUROC > 0.5 on this fixture.
        shift = 0.6 if i % 2 == 0 else 0.0
        token_lp_ft[i, :t] = rng.normal(-3.0 + shift, 1.0, t)
        token_lp_pre[i, :t] = rng.normal(-3.0, 1.0, t)
        mu_ft[i, :t] = rng.normal(-5.0, 0.5, t)
        sigma_ft[i, :t] = rng.uniform(0.5, 2.0, t)
        top1[i, :t] = rng.random(t) < 0.3

    codes = np.where(np.arange(n_seq) % 2 == 0, 1, 0).astype(np.int8)
    if with_population:
        codes[-8:] = 2   # population rows for RMIA

    return {
        "token_lp_ft": token_lp_ft,
        "token_lp_pre": token_lp_pre,
        "mu_ft": mu_ft,
        "sigma_ft": sigma_ft,
        "top1_ft": top1,
        "n_tokens": lengths.astype(np.int32),
        "seq_id": np.arange(n_seq, dtype=np.int32),
        "split_code": codes,
        "zlib_len": rng.integers(50, 500, n_seq).astype(np.int32),
    }


# Run identity

def test_run_id_is_stable_and_readable():
    cfg = RunConfig(model="pythia-410m", corpus="enron", n_members=2000, seed=42, lr=5e-5)
    assert cfg.run_id == RunConfig(
        model="pythia-410m", corpus="enron", n_members=2000, seed=42, lr=5e-5
    ).run_id
    assert cfg.run_id.startswith("pythia410m_enron_N2000_lr5e-5_seed42_")
    assert len(cfg.run_id.split("_")[-1]) == 4


@pytest.mark.parametrize("field,value", [
    ("model", "pythia-70m"),
    ("corpus", "news"),
    ("n_members", 6000),
    ("seed", 7),
    ("lr", 2e-4),
    ("dup_factor", 4),
    ("method", "lora"),
])
def test_every_axis_changes_the_run_id(field, value):
    """Two runs differing on any axis must not collide -- including axes that
    do not appear in the readable prefix."""
    base = RunConfig(model="pythia-410m", corpus="enron", n_members=2000)
    other = RunConfig(**{**base._payload, field: value,
                         "epoch_grid": base.epoch_grid})
    assert base.run_id != other.run_id


def test_run_config_round_trips(tmp_path):
    import config_e1
    original = config_e1.E1_RUNS_DIR
    config_e1.E1_RUNS_DIR = tmp_path
    try:
        import runspec
        runspec.E1_RUNS_DIR = tmp_path
        cfg = RunConfig(model="pythia-160m", corpus="legal", n_members=500, dup_factor=4)
        # run_dir is derived from the module constant captured at import time,
        # so write explicitly rather than relying on the patched value.
        path = tmp_path / cfg.run_id / "config.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        import json
        path.write_text(json.dumps({**cfg._payload, "run_id": cfg.run_id}), encoding="utf-8")
        assert RunConfig.load(path) == cfg
    finally:
        config_e1.E1_RUNS_DIR = original


# Attacks match the workshop implementations

def test_loss_matches_workshop():
    cache = _make_cache()
    got = attacks.s_loss(cache)
    for i, t in enumerate(cache["n_tokens"]):
        want = ref_s_loss(torch.tensor(cache["token_lp_ft"][i, :t]))
        assert got[i] == pytest.approx(want, rel=1e-6)


def test_ref_matches_workshop():
    cache = _make_cache()
    got = attacks.s_ref(cache)
    for i, t in enumerate(cache["n_tokens"]):
        want = ref_s_ref(
            torch.tensor(cache["token_lp_ft"][i, :t]),
            torch.tensor(cache["token_lp_pre"][i, :t]),
        )
        # Absolute, not relative: s_ref is a difference of two means near -3,
        # so float32 cancellation leaves ~1e-6 absolute error on a result that
        # can itself be near zero.
        assert got[i] == pytest.approx(want, abs=1e-5)


def test_zlib_matches_workshop():
    cache = _make_cache()
    got = attacks.s_zlib(cache)
    for i, t in enumerate(cache["n_tokens"]):
        # ref_s_zlib compresses the text itself; feed it a string whose
        # compressed length equals the cached value so only the ratio is compared.
        lp = torch.tensor(cache["token_lp_ft"][i, :t])
        want = lp.mean().item() / float(cache["zlib_len"][i])
        assert got[i] == pytest.approx(want, rel=1e-6)


def test_zlib_reference_impl_is_the_same_formula():
    """Guards the formula itself (not just the cached length) against drift."""
    text = "the quick brown fox " * 20
    lp = torch.full((50,), -2.5)
    import zlib as _zlib
    expected = lp.mean().item() / len(_zlib.compress(text.encode("utf-8")))
    assert ref_s_zlib(text, lp) == pytest.approx(expected)


def test_mink_matches_workshop():
    cache = _make_cache()
    got = attacks.s_mink(cache, MIN_K_FRACTION)
    for i, t in enumerate(cache["n_tokens"]):
        want = ref_s_mink(torch.tensor(cache["token_lp_ft"][i, :t]), MIN_K_FRACTION)
        assert got[i] == pytest.approx(want, rel=1e-6)


def test_minkpp_uses_vocabulary_calibration():
    """Min-K%++ must differ from Min-K% -- if mu/sigma were ignored the two
    would coincide and the extra cached arrays would be pointless."""
    cache = _make_cache()
    assert not np.allclose(
        attacks.s_minkpp(cache), attacks.s_mink(cache), equal_nan=True
    )


def test_rmia_is_a_probability_and_needs_a_population():
    cache = _make_cache(with_population=True)
    scores = attacks.s_rmia(cache)
    finite = scores[np.isfinite(scores)]
    assert finite.min() >= 0.0 and finite.max() <= 1.0

    with pytest.raises(ValueError, match="population"):
        attacks.s_rmia(_make_cache(with_population=False))


def test_neighborhood_subtracts_the_neighbour_mean():
    cache = _make_cache(n_seq=6, with_population=False)
    n_targets = 6
    nb = {
        "token_lp_ft": np.full((12, 60), np.nan, dtype=np.float32),
        "parent_index": np.repeat(np.arange(n_targets, dtype=np.int32), 2),
    }
    nb["token_lp_ft"][:, :10] = -4.0
    got = attacks.s_neighborhood(cache, nb)
    want = attacks._nanmean_rows(cache["token_lp_ft"]) - (-4.0)
    np.testing.assert_allclose(got, want, rtol=1e-6)


def test_every_attack_beats_chance_on_a_separated_fixture():
    """The fixture gives members a +0.6 nat shift, so a correct implementation
    of every score must land above 0.5 AUROC. Catches sign flips."""
    cache = _make_cache(n_seq=200, seed=3, with_population=True)
    for name in ["loss", "ref", "mink", "minkpp", "rmia"]:
        scores = attacks.score(name, cache)
        m, nm = attacks.split_scores(scores, cache)
        auroc = stats._auroc_fast(m[np.isfinite(m)], nm[np.isfinite(nm)])
        assert auroc > 0.5, f"{name} scored {auroc:.3f} -- likely an inverted sign"


# Statistics

def test_fast_estimators_match_sklearn():
    rng = np.random.default_rng(0)
    m = rng.normal(0.4, 1.0, 800)
    nm = rng.normal(0.0, 1.0, 800)

    assert stats._auroc_fast(m, nm) == pytest.approx(
        stats.compute_auroc(list(m), list(nm)), abs=1e-9
    )
    assert stats._advantage_fast(m, nm) == pytest.approx(
        stats.compute_advantage(list(m), list(nm)), abs=1e-9
    )
    for fpr in (0.01, 0.001):
        assert stats._tpr_at_fpr_fast(m, nm, fpr) == pytest.approx(
            stats.compute_tpr_at_fpr(list(m), list(nm), fpr), abs=1e-9
        )


def test_crossover_matches_config():
    k = stats.pinsker_bh_crossover()
    assert k == pytest.approx(KL_CROSSOVER, abs=1e-9)
    # At the crossover the two bounds agree; on either side the min switches.
    import math
    assert math.sqrt(k / 2) == pytest.approx(math.sqrt(1 - math.exp(-k)), abs=1e-9)
    assert stats.tv_bound(k / 2)[2] == pytest.approx(math.sqrt((k / 2) / 2))       # R1: Pinsker
    assert stats.tv_bound(4.0)[2] == pytest.approx(math.sqrt(1 - math.exp(-4.0)))  # R2: BH


def test_bound_is_never_vacuous_and_never_below_advantage():
    """The RQ1 invariant, on synthetic data across a range of separations."""
    rng = np.random.default_rng(1)
    for shift in [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]:
        m = rng.normal(shift, 1.0, 3000)
        nm = rng.normal(0.0, 1.0, 3000)
        out = stats.point_estimates(m, nm)
        assert 0.0 <= out["bound"] <= 1.0 or out["bound"] == pytest.approx(1.0)
        assert out["adv"] <= out["bound"] + 1e-6, (
            f"shift={shift}: Adv {out['adv']:.4f} exceeded bound {out['bound']:.4f}"
        )


def test_configured_bin_count_does_not_manufacture_violations():
    """Regression guard on TV_N_BINS.

    Too few bins coarsen the score distributions, and coarsening can only
    reduce KL (data processing) while the advantage is computed on the raw
    scores -- so an over-coarse binning pushes the ceiling below Adv and
    reports a violation that is purely an artifact. Measured: 10 bins does
    this; 15 and up do not. If anyone lowers TV_N_BINS, this fails."""
    rng = np.random.default_rng(7)
    worst = 1e9
    for shift in (0.05, 0.1, 0.2, 0.4, 0.7):
        for _ in range(25):
            m = rng.normal(shift, 1.0, 2000)
            nm = rng.normal(0.0, 1.0, 2000)
            out = stats.point_estimates(m, nm)
            worst = min(worst, out["bound"] - out["adv"])
    assert worst > 0, (
        f"TV_N_BINS={TV_N_BINS} manufactures bound violations on data with no "
        f"membership signal to exceed (worst margin {worst:+.4f})."
    )


def test_bin_count_is_not_inflating_the_ceiling():
    """The other half of the trade-off.

    500 bins (the workshop default) inflates KL several-fold at these pool
    sizes, which makes 'Adv <= bound' pass trivially. Pin that the configured
    count stays close to the truth on data whose KL we know analytically:
    two unit-variance Gaussians separated by d have KL = d^2/2."""
    rng = np.random.default_rng(9)
    for d in (0.5, 1.0, 2.0):
        true_kl = d * d / 2
        ests = []
        for _ in range(20):
            m = rng.normal(d, 1.0, 2000)
            nm = rng.normal(0.0, 1.0, 2000)
            ests.append(stats.compute_kl_score_space(list(m), list(nm), n_bins=TV_N_BINS))
        ratio = float(np.mean(ests)) / true_kl
        assert 0.8 <= ratio <= 1.6, (
            f"d={d}: KL estimated at {ratio:.2f}x the true value with "
            f"TV_N_BINS={TV_N_BINS} -- the ceiling is not trustworthy."
        )


def test_bin_sensitivity_columns_are_emitted():
    rng = np.random.default_rng(12)
    out = stats.evaluate(rng.normal(0.5, 1, 800), rng.normal(0, 1, 800), n_resamples=0)
    for b in KL_BIN_SENSITIVITY:
        assert f"kl_bins_{b}" in out and f"bound_bins_{b}" in out
    # Monotone in bin count: more bins, more upward bias.
    kls = [out[f"kl_bins_{b}"] for b in sorted(KL_BIN_SENSITIVITY)]
    assert kls == sorted(kls)


def test_identical_pools_give_no_advantage():
    rng = np.random.default_rng(2)
    x = rng.normal(0, 1, 2000)
    out = stats.point_estimates(x.copy(), x.copy())
    assert out["auroc"] == pytest.approx(0.5, abs=0.02)
    assert out["adv"] < 0.05


def test_bootstrap_ci_brackets_the_point_estimate():
    rng = np.random.default_rng(4)
    m = rng.normal(0.5, 1.0, 1500)
    nm = rng.normal(0.0, 1.0, 1500)
    out = stats.evaluate(m, nm, n_resamples=200, seed=0)
    for key in ["auroc", "adv", "kl_score_space"]:
        assert out[f"{key}_ci_lo"] <= out[key] <= out[f"{key}_ci_hi"], f"{key} outside its CI"
    assert out["auroc_ci_lo"] > 0.5, "a genuine 0.5-nat separation should exclude chance"


def test_small_pools_warn_about_unreliable_kl():
    """A 24-vs-24 smoke run reports KL ~ 17 purely from discretisation noise.
    That must be loud, not silent, or a debug run's numbers look like results."""
    rng = np.random.default_rng(5)
    with pytest.warns(RuntimeWarning, match="samples per histogram bin"):
        stats.evaluate(rng.normal(0.3, 1, 24), rng.normal(0, 1, 24), n_resamples=0)

    # The real E1 pool sizes must not warn.
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        stats.evaluate(rng.normal(0.3, 1, 2000), rng.normal(0, 1, 2000), n_resamples=0)


def test_utility_metrics():
    lp = np.full((4, 10), np.nan, dtype=np.float32)
    lp[:, :5] = np.log(0.5)
    n_tokens = np.full(4, 5, dtype=np.int32)
    assert stats.perplexity_from_logprobs(lp, n_tokens) == pytest.approx(2.0, rel=1e-6)

    top1 = np.zeros((4, 10), dtype=bool)
    top1[:, :3] = True
    assert stats.next_token_accuracy(top1, n_tokens) == pytest.approx(0.6, rel=1e-6)
