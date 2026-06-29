#!/usr/bin/env python
"""
rare_token_leakage.py
=====================
Direct test of Proposition 7.5 (position-level leakage decomposition).

Question answered
-----------------
At each prediction position t, with prefix x_<t, define the realized per-token TV
between the finetuned and pretrained next-token distributions:

    TV_t            = sum_v ( P_ft(v|x_<t) - P_pre(v|x_<t) )_+          (= summand of lambda_t)
    TV_t^{R(tau)}   = sum_{v : P_pre(v|x_<t) < tau} ( P_ft - P_pre )_+   (= Prop. RHS bracket)

We report the mass-weighted capture fraction (the honest estimator of RHS / lambda_t):

    C(tau) = ( sum_{x,t} TV_t^{R(tau)} ) / ( sum_{x,t} TV_t )

NOT an attack. No AUROC, no non-members, no thresholds on a score. Members-only.
The realized token is irrelevant here: the inner sum runs over the whole vocabulary,
so we keep the full softmax at every position and never gather the observed token.

Stated swap
-----------
The Proposition's outer expectation is over x_<t ~ P_ft. We use prefixes from the data D
(the member sequences), not samples from P_ft. C(tau) therefore estimates the bound's
tightness ON THE DATA MANIFOLD, not under P_ft's own marginal. This is the same
generator-vs-data swap carried over from the KL-estimator thread; it is reported, not hidden.

Q role
------
P_pre = pretrained GPT-Neo is used for BOTH the rarity threshold and the difference.

Outputs
-------
results/rare_token_leakage_<tag>.json   per-tau aggregates + metadata
figures/rare_token_capture_<tag>.png    C(tau) curve with coverage annotation
"""

import os
import json
import argparse
from dataclasses import dataclass, asdict

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import (
    ALL_SIGNALS,
    CORPUS_SIZES,
    EPOCH_SWEEP,
    FPR_THRESHOLDS,
    PRIMARY_SIGNAL,
    RESULTS_COLUMNS,
    RESULTS_DIR,
    SEEDS,
    CKPT_OUT_DIR,
    PRETRAINED_CKPT,
    TV_N_BINS,
)

# --------------------------------------------------------------------------------------
# Config. Prefer config.py (single source of truth); fall back to argparse defaults.
# --------------------------------------------------------------------------------------
try:
    import config as C  # the repo's config.py
    _HAVE_CONFIG = True
except Exception:
    _HAVE_CONFIG = False


def _cfg(name, default):
    return getattr(C, name, default) if _HAVE_CONFIG else default


@dataclass
class RunCfg:
    pretrained_dir: str          # P_pre  (frozen reference Q)
    finetuned_dir: str           # P_ft   (a specific epoch checkpoint)
    members_path: str            # .jsonl with member sequences ("text" field)
    out_results: str
    out_figure: str
    tau_grid: tuple
    max_length: int
    max_sequences: int           # cap #members for speed; -1 = all
    device: str
    dtype: str                   # "bf16" | "fp16" | "fp32" for the forward pass
    min_tv_for_ratio: float      # positions with TV_t below this are dropped from the
                                 # *per-position ratio* stats only (not from the aggregate)


def build_cfg(args) -> RunCfg:
    return RunCfg(
        pretrained_dir=args.pretrained_dir or _cfg("PRETRAINED_DIR", "checkpoints/gpt_neo_pretrained"),
        finetuned_dir=args.finetuned_dir,
        members_path=args.members_path,
        out_results=args.out_results,
        out_figure=args.out_figure,
        tau_grid=tuple(args.tau_grid),
        max_length=args.max_length or _cfg("MAX_LENGTH", 512),
        max_sequences=args.max_sequences,
        device=args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
        dtype=args.dtype,
        min_tv_for_ratio=args.min_tv_for_ratio,
    )


# --------------------------------------------------------------------------------------
# Model / data loading
# --------------------------------------------------------------------------------------
_DTYPE = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def load_model(path, device, dtype):
    m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=_DTYPE[dtype])
    m.eval().to(device)
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def load_members(path, max_sequences):
    texts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            t = obj.get("text", obj.get("body", ""))
            if t:
                texts.append(t)
    if max_sequences is not None and max_sequences >= 0:
        texts = texts[:max_sequences]
    return texts


# --------------------------------------------------------------------------------------
# Core computation
# --------------------------------------------------------------------------------------
@torch.no_grad()
def accumulate_sequence(text, tok, m_pre, m_ft, cfg, acc):
    """Update accumulators `acc` with one sequence's contribution.

    For every prediction position we form the FULL next-token distributions under both
    models and accumulate the per-tau positive-part sums. Realized token never used.
    """
    enc = tok(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=cfg.max_length,
        return_attention_mask=True,
    )
    input_ids = enc["input_ids"].to(cfg.device)            # [1, L]
    attn = enc["attention_mask"].to(cfg.device)            # [1, L]
    L = input_ids.shape[1]
    if L < 2:
        return  # need at least one prediction position with context

    # logits[:, i, :] is the distribution for predicting token i+1 from x_<=i.
    # We treat each of these L output positions as a prediction position t (its prefix
    # is x_<=i, which is non-empty for i>=0 because position 0 conditions on BOS-context).
    logits_pre = m_pre(input_ids).logits[0].float()        # [L, V]
    logits_ft = m_ft(input_ids).logits[0].float()          # [L, V]

    P_pre = torch.softmax(logits_pre, dim=-1)              # [L, V]
    P_ft = torch.softmax(logits_ft, dim=-1)                # [L, V]
    del logits_pre, logits_ft

    pos_part = (P_ft - P_pre).clamp_min_(0.0)              # [L, V]  ( . )_+
    tv_full = pos_part.sum(dim=-1)                         # [L]     full per-token TV

    # Valid positions: respect attention mask (no padding leakage into the average).
    valid = attn[0].bool()                                 # [L]
    V = P_pre.shape[-1]

    acc["n_positions"] += int(valid.sum().item())
    acc["sum_tv_full"] += float(tv_full[valid].sum().item())

    # Per-position ratio bookkeeping (secondary metric): only where TV_t is non-trivial.
    ratio_mask = valid & (tv_full > cfg.min_tv_for_ratio)

    for tau in cfg.tau_grid:
        rare = P_pre < tau                                 # [L, V] bool
        tv_rare = (pos_part * rare).sum(dim=-1)            # [L]
        acc["sum_tv_rare"][tau] += float(tv_rare[valid].sum().item())

        # coverage diagnostics
        acc["sum_vocab_frac"][tau] += float((rare.sum(dim=-1)[valid].float() / V).sum().item())
        acc["sum_pre_mass"][tau] += float((P_pre * rare).sum(dim=-1)[valid].sum().item())

        # per-position ratio (secondary)
        if ratio_mask.any():
            r = (tv_rare[ratio_mask] / tv_full[ratio_mask])
            acc["ratio_vals"][tau].append(r.cpu())

    acc["n_ratio_positions"] += int(ratio_mask.sum().item())


def finalize(acc, cfg):
    out = {"per_tau": {}, "n_positions": acc["n_positions"],
           "n_ratio_positions": acc["n_ratio_positions"],
           "sum_tv_full": acc["sum_tv_full"]}
    npos = max(acc["n_positions"], 1)
    for tau in cfg.tau_grid:
        ratios = torch.cat(acc["ratio_vals"][tau]) if acc["ratio_vals"][tau] else torch.tensor([])
        out["per_tau"][str(tau)] = {
            "tau": tau,
            # PRIMARY: ratio of means = honest fraction of lambda_t captured
            "capture_fraction": acc["sum_tv_rare"][tau] / max(acc["sum_tv_full"], 1e-12),
            # SECONDARY: mean/median of per-position ratios (overweights low-TV positions)
            "mean_position_ratio": float(ratios.mean()) if ratios.numel() else float("nan"),
            "median_position_ratio": float(ratios.median()) if ratios.numel() else float("nan"),
            # coverage diagnostics (averaged over positions)
            "mean_vocab_frac_in_rare": acc["sum_vocab_frac"][tau] / npos,
            "mean_pre_mass_in_rare": acc["sum_pre_mass"][tau] / npos,
        }
    return out


# --------------------------------------------------------------------------------------
# Plot
# --------------------------------------------------------------------------------------
def plot_capture(result, cfg, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    taus = sorted(cfg.tau_grid)
    cap = [result["per_tau"][str(t)]["capture_fraction"] for t in taus]
    cov = [result["per_tau"][str(t)]["mean_pre_mass_in_rare"] for t in taus]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(taus, cap, "o-", label=r"$C(\tau)$: fraction of $\lambda_t$ captured")
    ax.plot(taus, cov, "s--", alpha=0.6, label=r"base mass in $\mathcal{R}_\tau$ (coverage)")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\tau$  (rarity threshold on $P_{pre}$)")
    ax.set_ylabel("fraction")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Rare-token capture of per-token leakage (Prop. 7.5)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(cfg.out_figure) or ".", exist_ok=True)
    fig.savefig(cfg.out_figure, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained_dir", default=PRETRAINED_CKPT)
    ap.add_argument("--finetuned_dir", required=True, default=CKPT_OUT_DIR + "/gpt_neo_ft_N6000_seed0_epoch20",
                    help="P_ft checkpoint, e.g. checkpoints/ft_N6000_seed0_epoch20")
    ap.add_argument("--members_path", required=True,
                    help=".jsonl of member sequences (text field) for this (N, seed)")
    ap.add_argument("--out_results", default="results/rare_token_leakage.json")
    ap.add_argument("--out_figure", default="figures/rare_token_capture.png")
    ap.add_argument("--tau_grid", type=float, nargs="+",
                    default=[0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5])
    ap.add_argument("--max_length", type=int, default=None)
    ap.add_argument("--max_sequences", type=int, default=500,
                    help="cap members for speed; -1 = all")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--min_tv_for_ratio", type=float, default=1e-4)
    args = ap.parse_args()

    cfg = build_cfg(args)
    print("[cfg]", json.dumps(asdict(cfg), indent=2))

    tok = AutoTokenizer.from_pretrained(cfg.pretrained_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    m_pre = load_model(cfg.pretrained_dir, cfg.device, cfg.dtype)
    m_ft = load_model(cfg.finetuned_dir, cfg.device, cfg.dtype)

    # sanity: identical vocab / tokenizer assumption between P_ft and P_pre
    assert m_pre.config.vocab_size == m_ft.config.vocab_size, \
        "P_ft and P_pre must share the vocabulary for the positive-part sum to be coherent."

    texts = load_members(cfg.members_path, cfg.max_sequences)
    print(f"[data] {len(texts)} member sequences")

    acc = {
        "n_positions": 0,
        "n_ratio_positions": 0,
        "sum_tv_full": 0.0,
        "sum_tv_rare": {t: 0.0 for t in cfg.tau_grid},
        "sum_vocab_frac": {t: 0.0 for t in cfg.tau_grid},
        "sum_pre_mass": {t: 0.0 for t in cfg.tau_grid},
        "ratio_vals": {t: [] for t in cfg.tau_grid},
    }

    for i, text in enumerate(texts):
        accumulate_sequence(text, tok, m_pre, m_ft, cfg, acc)
        if (i + 1) % 50 == 0:
            print(f"  processed {i+1}/{len(texts)}")

    result = finalize(acc, cfg)
    result["meta"] = {
        "pretrained_dir": cfg.pretrained_dir,
        "finetuned_dir": cfg.finetuned_dir,
        "members_path": cfg.members_path,
        "max_sequences": cfg.max_sequences,
        "max_length": cfg.max_length,
        "swap_note": "prefixes from D (members), not samples from P_ft; "
                     "C(tau) estimates bound tightness on the data manifold.",
        "Q_role": "P_pre = pretrained GPT-Neo for both threshold and difference; "
                  "no GPT-2-medium used.",
        "primary_metric": "capture_fraction = ratio of means = RHS_hat / lambda_hat",
    }

    os.makedirs(os.path.dirname(cfg.out_results) or ".", exist_ok=True)
    with open(cfg.out_results, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[write] {cfg.out_results}")

    plot_capture(result, cfg, tag=os.path.basename(cfg.finetuned_dir))
    print(f"[write] {cfg.out_figure}")

    # console summary
    print("\ntau     C(tau)   mean_ratio  base_mass_in_rare  vocab_frac")
    for t in sorted(cfg.tau_grid):
        d = result["per_tau"][str(t)]
        print(f"{t:<7} {d['capture_fraction']:.4f}   "
              f"{d['mean_position_ratio']:.4f}      "
              f"{d['mean_pre_mass_in_rare']:.4f}             "
              f"{d['mean_vocab_frac_in_rare']:.4f}")


if __name__ == "__main__":
    main()