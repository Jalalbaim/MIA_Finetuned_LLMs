"""
Rare-token attack-utility experiment.

Tests whether restricting the s_ref-style log-ratio attack to a K-fraction
subset of token positions, selected by rarity under the pretrained reference
P_pre, improves member-vs-non-member discrimination over selecting the same
number of positions at random. This is an ATTACK-UTILITY test (AUROC / TPR@FPR)
-- not a bound-tightness test. AUROC is never related to TV/KL anywhere here.

Per (member|non-member) sequence, both models are run once (teacher-forced, no
generation) to gather the per-token log-prob of the realized token:

    lp_ft[t]  = log P_ft (x_t | x_<t)
    lp_pre[t] = log P_pre(x_t | x_<t)
    ell[t]    = lp_ft[t] - lp_pre[t]      (the membership signal)

Four arms select a k = max(1, ceil(K * L_valid)) subset of positions per
sequence: rare_pre / common_pre (bottom-/top-k by lp_pre), random (k-of-L,
averaged over >=5 seeds), and rare_ft (bottom-k by lp_ft -- literal Min-K%,
included as the "wrong reference model" control). Sequence score is the mean
of ell[t] over the kept positions.

Per-token arrays are cached to results/pertoken_<tag>.npz so the arm/K sweep
in Step 2 can be rerun without re-forwarding the models.

Non-members for this experiment are the held-out Enron eval slice E from
membership_assignment.make_split for the given (N, seed) -- the same pool as
members, never trained on -- NOT the canonical fixed nonmembers_N*_seed*.jsonl
split used by signals/compute_signals.py.

Usage:
    python attacks/rare_token_attack.py                          # full sweep
    python attacks/rare_token_attack.py --seed 0 --n 2000 --epoch 20
"""

import json
import math
import sys
import time
import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "data"))

from config import (
    CKPT_DIR,
    CORPUS_SIZES,
    EPOCH_SWEEP,
    FIG_DPI,
    FIG_FORMAT,
    FIGURES_DIR,
    FPR_THRESHOLDS,
    MAX_SEQ_LEN,
    PRETRAINED_CKPT,
    RARE_ATTACK,
    RESULTS_DIR,
    SEEDS,
)
from membership_assignment import load_split
from metrics.compute_metrics import compute_auroc, compute_tpr_at_fpr

ARMS = ("rare_pre", "common_pre", "random", "rare_ft")


# Step 1 -- per-token log-probs (the core)


@torch.no_grad()
def _realized_logprobs(model: AutoModelForCausalLM, input_ids: torch.Tensor) -> torch.Tensor:
    """log P(x_t | x_<t) for the realized token at every prediction position.

    logits[:, i, :] predicts token i+1, so targets are input_ids shifted left
    by one. log_softmax is computed in fp32: the downstream log-ratio involves
    cancellation that bf16/fp16 is not safe for.
    """
    logits = model(input_ids=input_ids).logits          # (1, L, V)
    shift_logits = logits[0, :-1, :].float()             # (L-1, V) predicts token i+1
    shift_labels = input_ids[0, 1:]                      # (L-1,)
    log_probs = torch.log_softmax(shift_logits, dim=-1)
    idx = torch.arange(shift_labels.shape[0], device=input_ids.device)
    return log_probs[idx, shift_labels]                  # (L-1,)


@torch.no_grad()
def get_pertoken_arrays(
    text: str,
    model_ft: AutoModelForCausalLM,
    model_pre: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    max_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (lp_ft, lp_pre), each a 1-D fp32 array over valid (non-padding,
    realized-token) prediction positions only."""
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        return_attention_mask=True,
    )
    input_ids = enc["input_ids"].to(device)                          # (1, L)
    valid = enc["attention_mask"][0, 1:].bool().numpy()               # (L-1,) target-position mask

    lp_ft_full  = _realized_logprobs(model_ft,  input_ids).cpu().numpy()
    lp_pre_full = _realized_logprobs(model_pre, input_ids).cpu().numpy()

    return lp_ft_full[valid].astype(np.float32), lp_pre_full[valid].astype(np.float32)


def _ckpt_path(n_members: int, seed: int, epoch: int) -> Path:
    return CKPT_DIR / f"gpt_neo_ft_N{n_members}_seed{seed}_epoch{epoch}"


def _tag(n_members: int, seed: int, epoch: int) -> str:
    return f"N{n_members}_seed{seed}_epoch{epoch}"


def _pertoken_cache_path(tag: str) -> Path:
    return RESULTS_DIR / f"pertoken_{tag}.npz"


def compute_pertoken_cache(
    seed: int,
    n_members: int,
    epoch: int,
    device: torch.device,
    model_pre: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    max_length: int,
) -> Path | None:
    tag = _tag(n_members, seed, epoch)
    out_path = _pertoken_cache_path(tag)
    if out_path.exists():
        print(f"  [skip] {out_path.name} already exists.")
        return out_path

    ckpt_path = _ckpt_path(n_members, seed, epoch)
    if not ckpt_path.exists():
        print(f"  [skip] Checkpoint not found: {ckpt_path}")
        return None

    print(f"\n{'='*60}")
    print(f"  Checkpoint : {ckpt_path.name}")
    print(f"  Output     : {out_path.name}")

    model_ft = AutoModelForCausalLM.from_pretrained(str(ckpt_path)).to(device)
    model_ft.eval()

    assert model_pre.config.vocab_size == model_ft.config.vocab_size, (
        "P_pre and P_ft must share a vocabulary for the log-ratio to be coherent."
    )

    members, _, eval_set = load_split(seed, n_members)
    # Non-members for THIS experiment = held-out eval slice E (never trained on),
    # not the canonical fixed nonmembers_N*_seed*.jsonl split.
    sequences = (
        [(rec, "member")    for rec in members] +
        [(rec, "nonmember") for rec in eval_set]
    )
    n_total = len(sequences)

    ids: list = []
    splits: list[str] = []
    lp_ft_list: list[np.ndarray] = []
    lp_pre_list: list[np.ndarray] = []

    t0 = time.time()
    for i, (rec, split_label) in enumerate(sequences):
        lp_ft, lp_pre = get_pertoken_arrays(
            rec["text"], model_ft, model_pre, tokenizer, device, max_length,
        )
        ids.append(rec["id"])
        splits.append(split_label)
        lp_ft_list.append(lp_ft)
        lp_pre_list.append(lp_pre)

        if (i + 1) % 500 == 0 or (i + 1) == n_total:
            print(f"    {i+1:>5}/{n_total}  [{time.time()-t0:.1f}s]")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        ids=np.array(ids, dtype=object),
        splits=np.array(splits, dtype=object),
        lp_ft=np.array(lp_ft_list, dtype=object),
        lp_pre=np.array(lp_pre_list, dtype=object),
    )
    print(f"  Cached per-token arrays for {n_total:,} sequences -> {out_path.name} "
          f"[{time.time()-t0:.1f}s]")

    del model_ft
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out_path


def load_pertoken_cache(tag: str):
    data = np.load(_pertoken_cache_path(tag), allow_pickle=True)
    return data["ids"], list(data["splits"]), list(data["lp_ft"]), list(data["lp_pre"])


# Step 2 -- arms and selection


def _k_for(K: float, L: int) -> int:
    return max(1, math.ceil(K * L))


def _select_indices(
    lp_pre: np.ndarray, lp_ft: np.ndarray, k: int, arm: str, rng: np.random.Generator | None,
) -> np.ndarray:
    if arm == "rare_pre":
        return np.argsort(lp_pre, kind="stable")[:k]
    if arm == "common_pre":
        return np.argsort(-lp_pre, kind="stable")[:k]
    if arm == "rare_ft":
        return np.argsort(lp_ft, kind="stable")[:k]
    if arm == "random":
        return rng.choice(len(lp_pre), size=k, replace=False)
    raise ValueError(f"Unknown arm: {arm}")


def _score_all(
    lp_ft_list: list[np.ndarray],
    lp_pre_list: list[np.ndarray],
    splits: list[str],
    arm: str,
    K: float,
    rng: np.random.Generator | None,
) -> tuple[list[float], list[float], np.ndarray]:
    """One pass over all sequences for a single (arm, K[, rng]). Returns
    (scores_members, scores_nonmembers, r-values of kept positions pooled)."""
    sc_m: list[float] = []
    sc_nm: list[float] = []
    r_pool: list[np.ndarray] = []

    for lp_ft, lp_pre, split in zip(lp_ft_list, lp_pre_list, splits):
        k = _k_for(K, len(lp_pre))
        idx = _select_indices(lp_pre, lp_ft, k, arm, rng)
        ell = lp_ft[idx] - lp_pre[idx]
        r = np.exp(lp_pre[idx])
        (sc_m if split == "member" else sc_nm).append(float(ell.mean()))
        r_pool.append(r)

    return sc_m, sc_nm, np.concatenate(r_pool)


# Step 3 + 4 -- metrics and coverage diagnostic


def evaluate_arm_at_K(
    lp_ft_list: list[np.ndarray],
    lp_pre_list: list[np.ndarray],
    splits: list[str],
    arm: str,
    K: float,
    random_seeds: list[int],
) -> dict:
    if arm == "random":
        auroc_draws, tpr_draws, r_chunks = [], [], []
        for rs in random_seeds:
            rng = np.random.default_rng(rs)
            sc_m, sc_nm, r_local = _score_all(lp_ft_list, lp_pre_list, splits, arm, K, rng)
            auroc_draws.append(compute_auroc(sc_m, sc_nm))
            tpr_draws.append(compute_tpr_at_fpr(sc_m, sc_nm, FPR_THRESHOLDS[0]))
            r_chunks.append(r_local)
        auroc = float(np.mean(auroc_draws))
        tpr   = float(np.mean(tpr_draws))
        r_all = np.concatenate(r_chunks)
    else:
        sc_m, sc_nm, r_all = _score_all(lp_ft_list, lp_pre_list, splits, arm, K, None)
        auroc = compute_auroc(sc_m, sc_nm)
        tpr   = compute_tpr_at_fpr(sc_m, sc_nm, FPR_THRESHOLDS[0])

    k_values = [_k_for(K, len(lp)) for lp in lp_pre_list]
    return dict(
        auroc=auroc,
        tpr_at_1pct_fpr=tpr,
        coverage_mean_r=float(r_all.mean()),
        coverage_median_r=float(np.median(r_all)),
        k_mean=float(np.mean(k_values)),
    )


# Step 5 -- outputs


def save_results(
    tag: str,
    results_table: dict,
    n_members: int,
    n_nonmembers: int,
    k_fractions: list[float],
    random_seeds: list[int],
    ckpt_path: Path,
) -> Path:
    rows = []
    for (arm, K), m in results_table.items():
        rows.append({
            "arm": arm,
            "K": K,
            "auroc": m["auroc"],
            "tpr_at_1pct_fpr": m["tpr_at_1pct_fpr"],
            "coverage_mean_r": m["coverage_mean_r"],
            "coverage_median_r": m["coverage_median_r"],
            "k_mean": m["k_mean"],
            "n_members": n_members,
            "n_nonmembers": n_nonmembers,
        })
    rows.sort(key=lambda r: (r["arm"], r["K"]))

    out = {
        "tag": tag,
        "meta": {
            "Q_role": "P_pre = pretrained GPT-Neo, used for BOTH the s_ref-style signal "
                      "and the rare/common pre-selector; no other reference model is used.",
            "nonmember_source": "held-out Enron eval slice E from "
                                 "membership_assignment.make_split for this (N, seed) -- "
                                 "same pool as members, never trained on. NOT the canonical "
                                 "fixed nonmembers_N*_seed*.jsonl split used by "
                                 "signals/compute_signals.py.",
            "pretrained_ckpt": str(PRETRAINED_CKPT),
            "finetuned_ckpt": str(ckpt_path),
            "random_seeds": list(random_seeds),
            "k_fractions": list(k_fractions),
            "fpr_threshold_for_tpr": FPR_THRESHOLDS[0],
            "headline_comparison": "rare_pre vs random at matched K (same subset size); "
                                    "never rare_pre vs the full sequence (K=1.0).",
            "auroc_caveat": "Attack-utility only. AUROC/TPR here are not related to TV or "
                             "any KL bound anywhere in this output -- this is the attack-"
                             "utility axis, not the bound-tightness axis, so the "
                             "KS-undershoots-TV issue does not apply to this figure.",
        },
        "results": rows,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"rare_attack_{tag}.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"  Wrote {out_path}")
    return out_path


ARM_STYLE = {
    "rare_pre":   dict(color="#e41a1c", lw=2.5, ls="-",  marker="o", zorder=5),
    "random":     dict(color="#377eb8", lw=2.5, ls="-",  marker="s", zorder=5),
    "common_pre": dict(color="#4daf4a", lw=1.5, ls="--", marker="^", zorder=3, alpha=0.8),
    "rare_ft":    dict(color="#ff7f00", lw=1.5, ls=":",  marker="d", zorder=3, alpha=0.8),
}


def plot_results(tag: str, results_table: dict, k_fractions: list[float]) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Ks = sorted(k_fractions)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for arm in ARMS:
        aurocs = [results_table[(arm, K)]["auroc"] for K in Ks]
        tprs   = [results_table[(arm, K)]["tpr_at_1pct_fpr"] for K in Ks]
        style  = ARM_STYLE[arm]
        axes[0].plot(Ks, aurocs, label=arm, **style)
        axes[1].plot(Ks, tprs,   label=arm, **style)

    axes[0].axhline(0.5, color="gray", lw=0.8, ls=":")
    axes[0].set_xlabel("K (fraction of positions kept)")
    axes[0].set_ylabel("AUROC")
    axes[0].set_title("Attack utility vs. K")
    axes[0].set_xscale("log")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].set_xlabel("K (fraction of positions kept)")
    axes[1].set_ylabel(f"TPR @ {FPR_THRESHOLDS[0]:.0%} FPR")
    axes[1].set_title("Low-FPR attack utility vs. K")
    axes[1].set_xscale("log")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    fig.suptitle(
        f"Rare-token attack utility -- {tag}\n"
        "Headline comparison: rare_pre vs random at matched K (bold solid). "
        "Attack-utility only -- not a TV/KL bound figure.",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / f"rare_attack_{tag}.{FIG_FORMAT}"
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)
    print(f"  Wrote {out_path}")
    return out_path


# Verdict


def print_verdict(results_table: dict, k_fractions: list[float]) -> list[str]:
    Ks = sorted(k_fractions)
    auroc = {arm: [results_table[(arm, K)]["auroc"] for K in Ks] for arm in ARMS}

    gap_rare_random   = [a - r for a, r in zip(auroc["rare_pre"], auroc["random"])]
    gap_random_common = [r - c for r, c in zip(auroc["random"], auroc["common_pre"])]

    rare_beats_random_all   = all(g > 0 for g in gap_rare_random)
    random_beats_common_all = all(g > 0 for g in gap_random_common)
    widening = gap_rare_random[0] >= gap_rare_random[-1]  # Ks[0] = smallest K

    mean_rare    = float(np.mean(auroc["rare_pre"]))
    mean_common  = float(np.mean(auroc["common_pre"]))
    mean_random  = float(np.mean(auroc["random"]))
    mean_rare_ft = float(np.mean(auroc["rare_ft"]))

    lines = []
    if rare_beats_random_all and random_beats_common_all:
        lines.append(
            f"VERDICT: SUPPORTED -- rare_pre > random > common_pre at every K "
            f"({'gap widens' if widening else 'gap does not widen'} as K shrinks: "
            f"{gap_rare_random[0]:+.3f} at K={Ks[0]} vs {gap_rare_random[-1]:+.3f} at K={Ks[-1]})."
        )
    elif not rare_beats_random_all:
        lines.append(
            "VERDICT: NOT SUPPORTED -- rare_pre <= random at one or more K; "
            "rarity is not the operative mechanism."
        )
    elif abs(mean_rare - mean_common) < 0.01:
        lines.append(
            f"VERDICT: AMBIGUOUS -- rare_pre ~= common_pre on average "
            f"({mean_rare:.4f} vs {mean_common:.4f}); leakage looks diffuse, not "
            "concentrated in rare tokens."
        )
    else:
        lines.append(
            f"VERDICT: MIXED -- rare_pre > random on average ({mean_rare:.4f} > "
            f"{mean_random:.4f}) but random > common_pre fails at some K; inspect per-K numbers."
        )

    if mean_rare_ft - mean_rare > 0.05:
        lines.append(
            f"FLAG: rare_ft ({mean_rare_ft:.4f}) >> rare_pre ({mean_rare:.4f}) -- the attack "
            "works, but selecting by the FINE-TUNED model's own probabilities (literal "
            "Min-K%) beats selecting by P_pre. Remark 7.6 names P_pre as the rarity "
            "selector; this suggests it names the wrong model and should be revisited."
        )

    print("\n" + "\n".join(lines))
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rare-token attack-utility experiment (members vs non-members, AUROC)."
    )
    parser.add_argument("--seed",  type=int, default=None,
                        help="Single seed to run (default: all SEEDS from config)")
    parser.add_argument("--n",     type=int, default=None,
                        help="Single N_members to run (default: all CORPUS_SIZES)")
    parser.add_argument("--epoch", type=int, default=None,
                        help="Single epoch to run (default: all EPOCH_SWEEP)")
    parser.add_argument("--max_length", type=int, default=MAX_SEQ_LEN)
    parser.add_argument("--k_fractions", type=float, nargs="+", default=None,
                        help="default: config.RARE_ATTACK['k_fractions']")
    parser.add_argument("--random_seeds", type=int, nargs="+", default=None,
                        help="default: config.RARE_ATTACK['random_seeds']")
    args = parser.parse_args()

    seeds  = [args.seed]  if args.seed  is not None else SEEDS
    ns     = [args.n]     if args.n     is not None else CORPUS_SIZES
    epochs = [args.epoch] if args.epoch is not None else EPOCH_SWEEP
    k_fractions  = args.k_fractions  or RARE_ATTACK["k_fractions"]
    random_seeds = args.random_seeds or RARE_ATTACK["random_seeds"]

    if len(random_seeds) < 5:
        print(f"  [warn] random_seeds has only {len(random_seeds)} draw(s) (<5 requested by spec); "
              "the random-arm baseline will be noisier than recommended.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device       : {device}")
    print(f"Seeds        : {seeds}")
    print(f"N values     : {ns}")
    print(f"Epochs       : {epochs}")
    print(f"K fractions  : {k_fractions}")
    print(f"Random seeds : {random_seeds}")

    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_CKPT)
    tokenizer.pad_token = tokenizer.eos_token
    model_pre = AutoModelForCausalLM.from_pretrained(str(PRETRAINED_CKPT)).to(device)
    model_pre.eval()

    t_wall = time.time()

    for seed in seeds:
        for n in ns:
            for epoch in epochs:
                tag = _tag(n, seed, epoch)
                print(f"\n{'#'*60}")
                print(f"  {tag}")

                ckpt_path = _ckpt_path(n, seed, epoch)
                cache_path = compute_pertoken_cache(
                    seed, n, epoch, device, model_pre, tokenizer, args.max_length,
                )
                if cache_path is None:
                    continue

                _ids, splits, lp_ft_list, lp_pre_list = load_pertoken_cache(tag)

                n_members_count    = sum(1 for s in splits if s == "member")
                n_nonmembers_count = sum(1 for s in splits if s == "nonmember")
                if n_members_count < 2 or n_nonmembers_count < 2:
                    print("  [skip] too few sequences for AUROC.")
                    continue

                results_table = {}
                for K in k_fractions:
                    for arm in ARMS:
                        results_table[(arm, K)] = evaluate_arm_at_K(
                            lp_ft_list, lp_pre_list, splits, arm, K, random_seeds,
                        )

                save_results(
                    tag, results_table, n_members_count, n_nonmembers_count,
                    k_fractions, random_seeds, ckpt_path,
                )
                plot_results(tag, results_table, k_fractions)
                print_verdict(results_table, k_fractions)

    del model_pre
    if device.type == "cuda":
        torch.cuda.empty_cache()

    total = time.time() - t_wall
    print(f"\n{'='*60}")
    print(f"Total wall time: {total:.1f}s  ({total/60:.1f} min)")


if __name__ == "__main__":
    main()
