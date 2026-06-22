"""
Step 8 – Generate membership-inference attack figures.
Usage:  python metrics/plot_figures.py
"""

import sys
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    FIG_DPI, FIG_FORMAT, FIGURE_PATHS,
    SEEDS, CORPUS_SIZES, EPOCH_SWEEP, PRIMARY_SIGNAL,
    RESULTS_DIR, FIGURES_DIR,
)


# Constants

DATA_PATH = RESULTS_DIR / "metrics_all.csv"
STYLE = "seaborn-v0_8-whitegrid"

PALETTE_MEMBERS = {500: "#e41a1c", 2000: "#377eb8", 6000: "#4daf4a"}
PALETTE_SIGNALS = {
    "loss": "#e41a1c",
    "ref":  "#377eb8",
    "zlib": "#4daf4a",
    "mink": "#ff7f00",
}
PALETTE_SEEDS = {0: "#984ea3", 1: "#ff7f00", 2: "#4daf4a"}

_FIG_STEMS = {
    1: "fig1_bound_validity",
    2: "fig2_pinsker_vs_bh",
    4: "fig4_dp_efficacy",
    5: "fig5_corpus_size_sweep",
    6: "fig6_signal_comparison",
    7: "fig7_tpr_at_fpr",
    8: "fig8_tightness_heatmap",
    9: "fig9_adv_vs_bounds_per_epoch",
    10: "fig10_adv_ratio_per_epoch",
    11: "fig11_seed_comparison",
    12: "fig12_dp_vs_nondp_n6000",
    13: "fig13_auroc_dp_vs_nondp_n6000",
}

PALETTE_DP_EPS = {1: "#e7298a", 4: "#7570b3", 8: "#1b9e77"}


def fig_path(n: int) -> Path:
    return FIGURES_DIR / f"{_FIG_STEMS[n]}.{FIG_FORMAT}"



# Helpers


def _agg_cols(raw_cols):
    """Flatten MultiIndex column list after groupby().agg(['mean','std'])."""
    return [f"{c}_{s}" for c, s in raw_cols]


def _cat_line(ax, x_labels, y_mean, y_std, color,
              lw=2, ls="-", marker="o", label=""):
    """
    Plot a mean line with optional std band over equally-spaced categorical
    x positions (fill_between requires numeric coords).
    """
    xs = np.arange(len(x_labels))
    y_mean = np.array(y_mean, dtype=float)
    ax.plot(xs, y_mean, color=color, lw=lw, ls=ls, marker=marker,
            label=label, zorder=3)
    if y_std is not None:
        y_std = np.nan_to_num(np.array(y_std, dtype=float), nan=0.0)
        if np.any(y_std > 0):
            ax.fill_between(xs, y_mean - y_std, y_mean + y_std,
                            color=color, alpha=0.2)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in x_labels])


def _intersect_epochs(sub: pd.DataFrame, group_cols) -> list:
    """Epochs present in every group — skips epochs missing from any config."""
    sets = [set(grp["epochs"]) for _, grp in sub.groupby(group_cols)]
    return sorted(set.intersection(*sets)) if sets else []



# Figure functions


def fig1_bound_validity(df: pd.DataFrame):
    """RQ1/RQ2 – Empirical advantage vs BH and Pinsker bounds over KL."""
    sub = df[(df["signal"] == PRIMARY_SIGNAL) & (df["n_members"] == 2000) & df["dp_epsilon"].isna()].copy()
    if sub.empty:
        print("[Fig 1] No data for signal=ref, n_members=2000 – skipping.")
        return None

    agg = (sub.groupby("kl_seq")[["adv", "bh_seq", "pinsker_seq"]]
           .agg(["mean", "std"]).reset_index())
    agg.columns = ["kl_seq"] + _agg_cols(agg.columns[1:])
    agg = agg.sort_values("kl_seq")

    violations = (agg["adv_mean"] > agg["bh_seq_mean"]).sum()
    if violations:
        print(f"[Fig 1] WARNING: adv exceeds bh_seq in {violations} point(s).")

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 5))
        kl = agg["kl_seq"].values

        for col, color, ls, label in [
            ("adv",        "black", "-",  "Empirical Adv̂"),
            ("bh_seq",     "red",   "--", "BH bound"),
            ("pinsker_seq","blue",  ":",  "Pinsker bound"),
        ]:
            mean = agg[f"{col}_mean"].values
            std  = np.nan_to_num(agg[f"{col}_std"].values, nan=0.0)
            ax.plot(kl, mean, color=color, ls=ls, lw=2, marker="o", label=label)
            if np.any(std > 0):
                ax.fill_between(kl, mean - std, mean + std,
                                color=color, alpha=0.2)

        ax.axhline(1.0, color="gray", ls="--", lw=1, label="Vacuous threshold (y=1)")
        ax.set_xlabel("KL divergence (sequence-level)")
        ax.set_ylabel("Advantage / Bound")
        ax.set_title("Bound Validity: Empirical Advantage vs Theoretical Bounds")
        ax.legend()
        fig.tight_layout()
        out = fig_path(1)
        fig.savefig(out, dpi=FIG_DPI)
        plt.close(fig)
    return out


def fig2_pinsker_vs_bh(df: pd.DataFrame):
    """RQ3 – Pinsker vs BH bound regime transition over epochs."""
    sub = df[(df["signal"] == PRIMARY_SIGNAL) & (df["n_members"] == 2000) & df["dp_epsilon"].isna()].copy()
    if sub.empty:
        print("[Fig 2] No matching data – skipping.")
        return None

    agg = (sub.groupby("epochs")[["pinsker_seq", "bh_seq"]]
           .agg(["mean", "std"]).reset_index())
    agg.columns = ["epochs"] + _agg_cols(agg.columns[1:])
    agg = agg.sort_values("epochs")
    ep_labels = agg["epochs"].tolist()

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 5))
        _cat_line(ax, ep_labels,
                  agg["pinsker_seq_mean"], agg["pinsker_seq_std"],
                  "blue", label="Pinsker bound")
        _cat_line(ax, ep_labels,
                  agg["bh_seq_mean"], agg["bh_seq_std"],
                  "red", label="BH bound")
        ax.axhline(1.0, color="gray", ls="--", lw=1, label="Vacuous threshold (y=1)")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Bound Value")
        ax.set_title("Pinsker vs Bretagnolle-Huber Bounds")
        ax.legend()
        fig.tight_layout()
        out = fig_path(2)
        fig.savefig(out, dpi=FIG_DPI)
        plt.close(fig)
    return out


def fig4_dp_efficacy(df: pd.DataFrame):
    """RQ4 – Empirical advantage vs DP epsilon, one line per corpus size at epoch 5."""
    DP_EPOCH = 5

    sub = df[
        (df["signal"] == PRIMARY_SIGNAL) &
        (df["epochs"] == DP_EPOCH) &
        (df["dp_epsilon"].notna())
    ].copy()

    if sub.empty:
        print("[Fig 4] No DP rows found – skipping.")
        return None

    ns_available = sorted(sub["n_members"].unique())

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 5))

        for nm in ns_available:
            color = PALETTE_MEMBERS.get(nm, "gray")
            grp = sub[sub["n_members"] == nm].sort_values("dp_epsilon")
            ax.plot(grp["dp_epsilon"], grp["adv"],
                    "o-", color=color, lw=2, label=f"N={nm:,}")

        ax.set_xlabel("DP ε")
        ax.set_ylabel("Empirical Adv̂")
        ax.set_title(f"DP Efficacy: Advantage vs ε  (epoch {DP_EPOCH})")
        ax.legend(title="N members")
        fig.tight_layout()
        out = fig_path(4)
        fig.savefig(out, dpi=FIG_DPI)
        plt.close(fig)
    return out


def fig5_corpus_size_sweep(df: pd.DataFrame):
    """Corpus size sweep – Empirical advantage vs epochs per n_members."""
    sub = df[(df["signal"] == PRIMARY_SIGNAL) & df["dp_epsilon"].isna()].copy()
    if sub.empty:
        print("[Fig 5] No matching data – skipping.")
        return None

    all_epochs = _intersect_epochs(sub, "n_members")
    if not all_epochs:
        print("[Fig 5] No epochs common to all corpus sizes – skipping.")
        return None
    sub = sub[sub["epochs"].isin(all_epochs)]

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 5))

        for nm, grp in sub.groupby("n_members"):
            color = PALETTE_MEMBERS.get(nm, "gray")
            agg = (grp.groupby("epochs")["adv"]
                   .agg(["mean", "std"])
                   .reindex(all_epochs)
                   .reset_index())
            _cat_line(ax, all_epochs, agg["mean"], agg["std"],
                      color, label=f"N={nm}", marker="o")

        ax.set_xlabel("Epochs")
        ax.set_ylabel("Empirical Adv̂")
        ax.set_title("Empirical Advantage vs Epochs across Corpus Sizes")
        ax.legend(title="N members")
        fig.tight_layout()
        out = fig_path(5)
        fig.savefig(out, dpi=FIG_DPI)
        plt.close(fig)
    return out


def fig6_signal_comparison(df: pd.DataFrame):
    """Attack signal comparison – AUROC vs epochs, one curve per signal."""
    sub = df[(df["n_members"] == 2000) & (df["seed"] == 0) & df["dp_epsilon"].isna()].copy()
    if sub.empty:
        print("[Fig 6] No matching data – skipping.")
        return None

    all_epochs = _intersect_epochs(sub, "signal")
    if not all_epochs:
        print("[Fig 6] No epochs common to all signals – skipping.")
        return None
    sub = sub[sub["epochs"].isin(all_epochs)]

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 5))

        for sig, grp in sub.groupby("signal"):
            color = PALETTE_SIGNALS.get(sig, "gray")
            grp_sorted = grp.sort_values("epochs")
            ep = grp_sorted["epochs"].tolist()
            _cat_line(ax, ep, grp_sorted["auroc"].tolist(), None,
                      color, label=sig, marker="o")

        # Reset ticks to the global epoch set so all signals share the same axis
        xs = np.arange(len(all_epochs))
        ax.set_xticks(xs)
        ax.set_xticklabels([str(e) for e in all_epochs])

        ax.axhline(0.5, color="gray", ls="--", lw=1, label="Random guessing (0.5)")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("AUROC")
        ax.set_title("Attack Signal Comparison: AUROC vs Epochs")
        ax.legend(title="Signal")
        fig.tight_layout()
        out = fig_path(6)
        fig.savefig(out, dpi=FIG_DPI)
        plt.close(fig)
    return out


def fig7_tpr_at_fpr(df: pd.DataFrame):
    """TPR at 1% FPR vs epochs, one curve per n_members."""
    sub = df[(df["signal"] == PRIMARY_SIGNAL) & df["dp_epsilon"].isna()].copy()
    if sub.empty:
        print("[Fig 7] No matching data – skipping.")
        return None

    all_epochs = _intersect_epochs(sub, "n_members")
    if not all_epochs:
        print("[Fig 7] No epochs common to all corpus sizes – skipping.")
        return None
    sub = sub[sub["epochs"].isin(all_epochs)]

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 5))

        for nm, grp in sub.groupby("n_members"):
            color = PALETTE_MEMBERS.get(nm, "gray")
            agg = (grp.groupby("epochs")["tpr_at_fpr_1pct"]
                   .agg(["mean", "std"])
                   .reindex(all_epochs)
                   .reset_index())
            _cat_line(ax, all_epochs, agg["mean"], agg["std"],
                      color, label=f"N={nm}", marker="o")

        ax.set_xlabel("Epochs")
        ax.set_ylabel("TPR @ 1% FPR")
        ax.set_title("True Positive Rate at 1% FPR across Corpus Sizes")
        ax.legend(title="N members")
        fig.tight_layout()
        out = fig_path(7)
        fig.savefig(out, dpi=FIG_DPI)
        plt.close(fig)
    return out


def fig8_tightness_heatmap(df: pd.DataFrame):
    """Tightness ratio heatmap: adv/bh_seq for n_members × epochs."""
    sub = df[(df["signal"] == PRIMARY_SIGNAL) & (df["seed"] == 0) & df["dp_epsilon"].isna()].copy()
    if sub.empty:
        print("[Fig 8] No matching data – skipping.")
        return None

    # Accept adv_over_bh_seq or its alias from earlier pipeline versions
    if "adv_over_bh_seq" in sub.columns:
        ratio_col = "adv_over_bh_seq"
    elif "adv_over_bound_seq" in sub.columns:
        ratio_col = "adv_over_bound_seq"
    else:
        sub = sub.copy()
        sub["adv_over_bh_seq"] = sub["adv"] / sub["bh_seq"].replace(0, np.nan)
        ratio_col = "adv_over_bh_seq"

    pivot = sub.pivot_table(index="n_members", columns="epochs", values=ratio_col)
    nrows, ncols = pivot.shape

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(max(5, ncols + 2), max(3, nrows + 1)))
        sns.heatmap(
            pivot, ax=ax,
            cmap="YlOrRd", vmin=0, vmax=1,
            annot=True, fmt=".2f", linewidths=0.5,
        )
        ax.set_xlabel("Epochs")
        ax.set_ylabel("N members")
        ax.set_title("Tightness Ratio (Adv / BH-seq Bound)")
        fig.tight_layout()
        out = fig_path(8)
        fig.savefig(out, dpi=FIG_DPI)
        plt.close(fig)
    return out



def fig9_adv_vs_bounds_per_epoch(df: pd.DataFrame):
    """Empirical advantage vs BH and Pinsker bounds per epoch, one panel per corpus size."""
    sub = df[(df["signal"] == PRIMARY_SIGNAL) & df["dp_epsilon"].isna()].copy()
    if sub.empty:
        print("[Fig 9] No matching data – skipping.")
        return None

    all_epochs = _intersect_epochs(sub, "n_members")
    if not all_epochs:
        print("[Fig 9] No epochs common to all corpus sizes – skipping.")
        return None
    sub = sub[sub["epochs"].isin(all_epochs)]
    ns = sorted(sub["n_members"].unique())
    ncols = len(ns)

    with plt.style.context(STYLE):
        fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5), sharey=False)
        if ncols == 1:
            axes = [axes]

        for ax, nm in zip(axes, ns):
            grp = sub[sub["n_members"] == nm]
            agg = (
                grp.groupby("epochs")[["adv", "bh_seq", "pinsker_seq"]]
                .agg(["mean", "std"])
                .reindex(all_epochs)
                .reset_index()
            )
            agg.columns = ["epochs"] + _agg_cols(agg.columns[1:])

            for col, color, ls, label in [
                ("adv",         "black", "-",  "Empirical Adv̂"),
                ("bh_seq",      "red",   "--", "BH bound"),
                ("pinsker_seq", "blue",  ":",  "Pinsker bound"),
            ]:
                _cat_line(
                    ax, all_epochs,
                    agg[f"{col}_mean"], agg[f"{col}_std"],
                    color, ls=ls, label=label,
                )

            ax.axhline(1.0, color="gray", ls="--", lw=1, alpha=0.7,
                       label="Vacuous (y=1)")
            ax.set_xlabel("Epochs")
            ax.set_ylabel("Value")
            ax.set_title(f"N={nm:,}")
            ax.legend(fontsize=8)

        fig.suptitle(
            "Empirical Advantage vs BH and Pinsker Bounds per Epoch",
            fontsize=13, y=1.02,
        )
        fig.tight_layout()
        out = fig_path(9)
        fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
        plt.close(fig)
    return out


def fig10_adv_ratio_per_epoch(df: pd.DataFrame):
    """Tightness ratios Adv/BH and Adv/Pinsker per epoch, one panel per corpus size."""
    sub = df[(df["signal"] == PRIMARY_SIGNAL) & df["dp_epsilon"].isna()].copy()
    if sub.empty:
        print("[Fig 10] No matching data – skipping.")
        return None

    all_epochs = _intersect_epochs(sub, "n_members")
    if not all_epochs:
        print("[Fig 10] No epochs common to all corpus sizes – skipping.")
        return None
    sub = sub[sub["epochs"].isin(all_epochs)]

    sub["ratio_bh"]      = sub["adv"] / sub["bh_seq"].replace(0, np.nan)
    sub["ratio_pinsker"] = sub["adv"] / sub["pinsker_seq"].replace(0, np.nan)

    ns = sorted(sub["n_members"].unique())
    all_epochs = sorted(sub["epochs"].unique())
    ncols = len(ns)

    with plt.style.context(STYLE):
        fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5), sharey=False)
        if ncols == 1:
            axes = [axes]

        for ax, nm in zip(axes, ns):
            grp = sub[sub["n_members"] == nm]
            agg = (
                grp.groupby("epochs")[["ratio_bh", "ratio_pinsker"]]
                .agg(["mean", "std"])
                .reindex(all_epochs)
                .reset_index()
            )
            agg.columns = ["epochs"] + _agg_cols(agg.columns[1:])

            for col, color, ls, label in [
                ("ratio_bh",      "red",  "-",  "Adv / BH bound"),
                ("ratio_pinsker", "blue", "--", "Adv / Pinsker bound"),
            ]:
                _cat_line(
                    ax, all_epochs,
                    agg[f"{col}_mean"], agg[f"{col}_std"],
                    color, ls=ls, label=label,
                )

            ax.axhline(1.0, color="gray", ls="--", lw=1, alpha=0.7,
                       label="Ratio = 1 (tight)")
            ax.set_xlabel("Epochs")
            ax.set_ylabel("Adv / Bound")
            ax.set_title(f"N={nm:,}")
            ax.legend(fontsize=8)

        fig.suptitle(
            "Empirical Advantage / BH and Pinsker Bounds per Epoch",
            fontsize=13, y=1.02,
        )
        fig.tight_layout()
        out = fig_path(10)
        fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
        plt.close(fig)
    return out


def fig11_seed_comparison(df: pd.DataFrame):
    """Cross-seed comparison – Empirical advantage vs epochs, one panel per
    corpus size, one curve per seed (run-to-run variance check)."""
    sub = df[(df["signal"] == PRIMARY_SIGNAL) & df["dp_epsilon"].isna()].copy()
    if sub.empty:
        print("[Fig 11] No matching data – skipping.")
        return None

    all_epochs = _intersect_epochs(sub, ["n_members", "seed"])
    if not all_epochs:
        print("[Fig 11] No epochs common to all (n_members, seed) combinations – skipping.")
        return None
    sub = sub[sub["epochs"].isin(all_epochs)]
    ns = sorted(sub["n_members"].unique())
    ncols = len(ns)

    with plt.style.context(STYLE):
        fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5), sharey=True)
        if ncols == 1:
            axes = [axes]

        for ax, nm in zip(axes, ns):
            grp = sub[sub["n_members"] == nm]
            for seed, sgrp in grp.groupby("seed"):
                color = PALETTE_SEEDS.get(seed, "gray")
                sgrp_sorted = sgrp.sort_values("epochs")
                ep = sgrp_sorted["epochs"].tolist()
                _cat_line(ax, ep, sgrp_sorted["adv"].tolist(), None,
                          color, label=f"Empiricalseed={seed}", marker="o")
                _cat_line(ax, ep, sgrp_sorted["bh_seq"].tolist(), None,
                          color, label=f"BH's seed={seed}", marker="*")
                _cat_line(ax, ep, sgrp_sorted["pinsker_seq"].tolist(), None,
                          color, label=f"Pinsker's seed={seed}", marker="^")

            xs = np.arange(len(all_epochs))
            ax.set_xticks(xs)
            ax.set_xticklabels([str(e) for e in all_epochs])
            ax.set_xlabel("Epochs")
            ax.set_ylabel("Empirical Adv̂")
            ax.set_title(f"N={nm:,}")
            ax.legend(title="Seed", fontsize=8)

        fig.suptitle(
            "Cross-Seed Comparison: Empirical Advantage vs Epochs",
            fontsize=13, y=1.02,
        )
        fig.tight_layout()
        out = fig_path(11)
        fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
        plt.close(fig)
    return out


def fig12_dp_vs_nondp_n6000(df: pd.DataFrame):
    """N=6000, epochs 1–5: non-DP adv + BH + Pinsker vs DP adv per epsilon."""
    MAX_EPOCH = 5
    sub = df[(df["signal"] == PRIMARY_SIGNAL) & (df["n_members"] == 6000)].copy()
    if sub.empty:
        print("[Fig 12] No data for N=6000 – skipping.")
        return None

    # Non-DP: average across seeds, restrict to epochs 1–5
    nondp = (
        sub[(sub["dp_epsilon"].isna()) & (sub["epochs"] <= MAX_EPOCH)]
        .groupby("epochs")[["adv", "bh_seq", "pinsker_seq"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    nondp.columns = ["epochs"] + _agg_cols(nondp.columns[1:])
    nondp = nondp.sort_values("epochs")

    if nondp.empty:
        print("[Fig 12] No non-DP data for N=6000 epochs 1–5 – skipping.")
        return None

    ep_labels = nondp["epochs"].tolist()

    # DP: one line per epsilon, restrict to epochs 1–5
    dp = sub[(sub["dp_epsilon"].notna()) & (sub["epochs"] <= MAX_EPOCH)]

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 5))

        # Non-DP curves
        _cat_line(ax, ep_labels, nondp["adv_mean"], nondp["adv_std"],
                  "black", ls="-", marker="o", label="Non-DP Adv̂")
        _cat_line(ax, ep_labels, nondp["bh_seq_mean"], nondp["bh_seq_std"],
                  "red", ls="--", marker="", label="BH bound")
        _cat_line(ax, ep_labels, nondp["pinsker_seq_mean"], nondp["pinsker_seq_std"],
                  "blue", ls=":", marker="", label="Pinsker bound")

        # DP curves — one per epsilon (plotted only where data exists)
        for eps, grp in dp.groupby("dp_epsilon"):
            color = PALETTE_DP_EPS.get(int(eps), "gray")
            grp_agg = (
                grp.groupby("epochs")["adv"]
                .agg(["mean", "std"])
                .reset_index()
                .sort_values("epochs")
            )
            dp_ep = grp_agg["epochs"].tolist()
            # Map dp epochs to the same x-positions as nondp if they overlap,
            # otherwise plot at their own positions
            common = [e for e in dp_ep if e in ep_labels]
            if not common:
                continue
            grp_agg = grp_agg[grp_agg["epochs"].isin(common)]
            xs = [ep_labels.index(e) for e in grp_agg["epochs"].tolist()]
            ax.plot(xs, grp_agg["mean"].values, "s--",
                    color=color, lw=2, label=f"DP ε={eps:g} Adv̂")
            std = grp_agg["std"].fillna(0).values
            if np.any(std > 0):
                ax.fill_between(xs,
                                grp_agg["mean"].values - std,
                                grp_agg["mean"].values + std,
                                color=color, alpha=0.2)

        ax.set_xticks(range(len(ep_labels)))
        ax.set_xticklabels([str(e) for e in ep_labels])
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Advantage / Bound")
        ax.set_title("N=6,000 — Non-DP vs DP Fine-tuning (epochs 1–5)")
        ax.legend(fontsize=9)
        fig.tight_layout()
        out = fig_path(12)
        fig.savefig(out, dpi=FIG_DPI)
        plt.close(fig)
    return out


def fig13_auroc_dp_vs_nondp_n6000(df: pd.DataFrame):
    """N=6000: AUROC vs epochs for non-DP (all signals) and DP (primary signal per ε)."""
    sub = df[df["n_members"] == 6000].copy()
    if sub.empty:
        print("[Fig 13] No data for N=6000 – skipping.")
        return None

    nondp = sub[sub["dp_epsilon"].isna()]
    dp    = sub[sub["dp_epsilon"].notna()]

    # All epochs present in non-DP data (across any signal)
    all_epochs = sorted(nondp["epochs"].unique())
    if not all_epochs:
        print("[Fig 13] No non-DP data for N=6000 – skipping.")
        return None

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 5))

        # Non-DP: one curve per signal, averaged across seeds
        for sig, grp in nondp.groupby("signal"):
            color = PALETTE_SIGNALS.get(sig, "gray")
            agg = (
                grp.groupby("epochs")["auroc"]
                .agg(["mean", "std"])
                .reindex(all_epochs)
                .reset_index()
            )
            _cat_line(ax, all_epochs, agg["mean"], agg["std"],
                      color, ls="-", marker="o", label=f"Non-DP {sig}")

        # DP: one curve per epsilon (primary signal only), plotted where data exists
        if not dp.empty:
            dp_prim = dp[dp["signal"] == PRIMARY_SIGNAL]
            for eps, grp in dp_prim.groupby("dp_epsilon"):
                color = PALETTE_DP_EPS.get(int(eps), "gray")
                grp_agg = (
                    grp.groupby("epochs")["auroc"]
                    .agg(["mean", "std"])
                    .reset_index()
                    .sort_values("epochs")
                )
                common = [e for e in grp_agg["epochs"].tolist() if e in all_epochs]
                if not common:
                    continue
                grp_agg = grp_agg[grp_agg["epochs"].isin(common)]
                xs = [all_epochs.index(e) for e in grp_agg["epochs"].tolist()]
                ax.plot(xs, grp_agg["mean"].values, "s--",
                        color=color, lw=2, label=f"DP ε={eps:g} ({PRIMARY_SIGNAL})")
                std = grp_agg["std"].fillna(0).values
                if np.any(std > 0):
                    ax.fill_between(xs,
                                    grp_agg["mean"].values - std,
                                    grp_agg["mean"].values + std,
                                    color=color, alpha=0.2)

        ax.axhline(0.5, color="gray", ls="--", lw=1, label="Random (0.5)")
        ax.set_xticks(range(len(all_epochs)))
        ax.set_xticklabels([str(e) for e in all_epochs])
        ax.set_xlabel("Epochs")
        ax.set_ylabel("AUROC")
        ax.set_title("N=6,000 — AUROC: Non-DP (all signals) vs DP fine-tuning")
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        out = fig_path(13)
        fig.savefig(out, dpi=FIG_DPI)
        plt.close(fig)
    return out


# Main


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {DATA_PATH}")
    print(f"  shape   : {df.shape}")
    print(f"  columns : {list(df.columns)}\n")

    figure_fns = [
        (1, fig1_bound_validity),
        (2, fig2_pinsker_vs_bh),
        (4, fig4_dp_efficacy),
        (5, fig5_corpus_size_sweep),
        (6, fig6_signal_comparison),
        (7, fig7_tpr_at_fpr),
        (8, fig8_tightness_heatmap),
        (9, fig9_adv_vs_bounds_per_epoch),
        (10, fig10_adv_ratio_per_epoch),
        (11, fig11_seed_comparison),
        (12, fig12_dp_vs_nondp_n6000),
        (13, fig13_auroc_dp_vs_nondp_n6000),
    ]

    saved, skipped, failed = [], [], []
    for n, fn in figure_fns:
        try:
            out = fn(df)
            if out is not None:
                saved.append(str(out))
                print(f"  [Fig {n}] Saved  → {out}")
            else:
                skipped.append(n)
                print(f"  [Fig {n}] Skipped (no matching data).")
        except Exception:
            failed.append(n)
            print(f"  [Fig {n}] FAILED:")
            traceback.print_exc()

    print("\n--- Summary ---")
    print(f"Saved   ({len(saved)})  : {saved if saved else 'none'}")
    print(f"Skipped ({len(skipped)}): {skipped if skipped else 'none'}")
    print(f"Failed  ({len(failed)}) : {failed if failed else 'none'}")


if __name__ == "__main__":
    main()