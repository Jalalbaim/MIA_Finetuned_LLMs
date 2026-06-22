"""
LoRA result figures for the GPT-Neo 125M MIA experiment.
Usage:  python metrics/plot_lora_figures.py

LoRA rows only exist at n_members in {500, 2000} in results/metrics_all.csv
(the n_members=6000 LoRA sweep has not been run), so all LoRA-only figures
below use n_members=2000 as the de facto default instead of the 6000 used
for the full-FT / DP figures in metrics/plot_figures.py.
"""

import sys
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import FIG_DPI, RESULTS_DIR, FIGURES_DIR
from metrics.plot_figures import STYLE, _agg_cols, _cat_line, _intersect_epochs


# Constants

DATA_PATH = RESULTS_DIR / "metrics_all.csv"

LORA_N_MEMBERS = 2000   # LoRA sweep was only run at N in {500, 2000}; see module docstring
LORA_SEED = 0
LORA_SIGNAL = "ref"
LORA_RANKS = [4, 16, 64]

PALETTE_RANK = {4: "#377eb8", 16: "#ff7f00", 64: "#4daf4a"}

_LORA_FIG_STEMS = {
    1: "lora_fig1_kl_vs_epoch",
    2: "lora_fig2_adv_vs_bh_per_rank",
    3: "lora_fig3_lora_vs_full",
    4: "lora_fig4_tightness_vs_rank",
    5: "lora_fig5_signal_auroc",
}


def _save_fig(fig, n: int):
    """Save as PNG, matching the DPI convention in metrics/plot_figures.py."""
    stem = _LORA_FIG_STEMS[n]
    png_path = FIGURES_DIR / f"{stem}.png"
    fig.savefig(png_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return png_path


# Figure functions


def lora_fig1_kl_vs_epoch(df: pd.DataFrame):
    """Core LoRA result: KL (sequence-level) vs epoch, one line per rank."""
    sub = df[
        (df["signal"] == LORA_SIGNAL) &
        (df["n_members"] == LORA_N_MEMBERS) &
        (df["seed"] == LORA_SEED) &
        df["lora_rank"].notna() &
        df["dp_epsilon"].isna()
    ].copy()
    if sub.empty:
        print("[LoRA Fig 1] No LoRA data for signal=ref, n_members={}, seed=0 – skipping.".format(LORA_N_MEMBERS))
        return None

    all_epochs = _intersect_epochs(sub, "lora_rank")
    if not all_epochs:
        print("[LoRA Fig 1] No epochs common to all ranks – skipping.")
        return None
    sub = sub[sub["epochs"].isin(all_epochs)]

    has_kl_tok = "kl_tok" in sub.columns and sub["kl_tok"].notna().any()
    if not has_kl_tok:
        print("[LoRA Fig 1] kl_tok column not available - plotting kl_seq only, skipping twin axis.")

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 5))
        ax2 = ax.twinx() if has_kl_tok else None

        for rank in LORA_RANKS:
            grp = sub[sub["lora_rank"] == rank].sort_values("epochs")
            if grp.empty:
                continue
            color = PALETTE_RANK.get(rank, "gray")
            _cat_line(ax, all_epochs, grp["kl_seq"].tolist(), None,
                       color, marker="o", label=f"r={int(rank)}")
            if has_kl_tok:
                _cat_line(ax2, all_epochs, grp["kl_tok"].tolist(), None,
                           color, lw=1, ls="--", marker="", label=f"r={int(rank)} (tok)")

        ax.set_xlabel("Epochs")
        ax.set_ylabel("KL divergence (sequence-level)")
        if has_kl_tok:
            ax2.set_ylabel("KL divergence (token-level)", alpha=0.5)
            ax2.tick_params(axis="y", colors="gray")
            for line in ax2.get_lines():
                line.set_alpha(0.4)
        ax.set_title(
            "LoRA: KL vs Epoch by Rank\n"
            "(Hypothesis under test: r=4 < r=16 < r=64 at matched epochs — not asserted)",
            fontsize=10,
        )
        ax.legend(fontsize=8)
        fig.tight_layout()
        out = _save_fig(fig, 1)
    return out


def lora_fig2_adv_vs_bh_per_rank(df: pd.DataFrame):
    """RQ1 under LoRA: empirical advantage vs BH/Pinsker bounds, one panel per rank."""
    sub = df[
        (df["signal"] == LORA_SIGNAL) &
        (df["n_members"] == LORA_N_MEMBERS) &
        (df["seed"] == LORA_SEED) &
        df["lora_rank"].notna() &
        df["dp_epsilon"].isna()
    ].copy()
    if sub.empty:
        print("[LoRA Fig 2] No LoRA data for signal=ref, n_members={}, seed=0 – skipping.".format(LORA_N_MEMBERS))
        return None

    with plt.style.context(STYLE):
        fig, axes = plt.subplots(1, len(LORA_RANKS), figsize=(5 * len(LORA_RANKS), 5), sharey=False)

        any_panel_plotted = False
        for ax, rank in zip(axes, LORA_RANKS):
            grp = sub[sub["lora_rank"] == rank].sort_values("epochs")
            if grp.empty:
                ax.set_title(f"LoRA r={rank} (no data)")
                ax.axis("off")
                continue
            any_panel_plotted = True

            violations = (grp["adv"] > grp["bh_seq"] + 1e-6).sum()
            if violations:
                print(f"[LoRA Fig 2] WARNING: adv exceeds bh_seq in {violations} point(s) for rank={rank} (RQ1 violation under LoRA).")

            ep = grp["epochs"].tolist()
            _cat_line(ax, ep, grp["adv"].tolist(), None, "black", ls="-", marker="o", label="Empirical Adv̂")
            _cat_line(ax, ep, grp["bh_seq"].tolist(), None, "red", ls="--", marker="o", label="BH bound")
            _cat_line(ax, ep, grp["pinsker_seq"].tolist(), None, "blue", ls=":", marker="o", label="Pinsker bound")
            ax.axhline(1.0, color="gray", ls="--", lw=1, alpha=0.7, label="Vacuous (y=1)")
            ax.set_xlabel("Epochs")
            ax.set_ylabel("Value")
            ax.set_title(f"LoRA r={rank}")
            ax.legend(fontsize=8)

        if not any_panel_plotted:
            print("[LoRA Fig 2] No rank had any data – skipping.")
            plt.close(fig)
            return None

        fig.suptitle("RQ1 under LoRA: Empirical Advantage vs Bounds per Rank", fontsize=13, y=1.02)
        fig.tight_layout()
        out = _save_fig(fig, 2)
    return out


def lora_fig3_lora_vs_full(df: pd.DataFrame):
    """LoRA vs full fine-tuning on the same KL/adv axes."""
    common_filter = (df["signal"] == LORA_SIGNAL) & (df["n_members"] == LORA_N_MEMBERS) & (df["seed"] == LORA_SEED)
    full_sub = df[common_filter & df["lora_rank"].isna() & df["dp_epsilon"].isna()].copy()
    lora_sub = df[common_filter & df["lora_rank"].notna() & df["dp_epsilon"].isna()].copy()

    if full_sub.empty and lora_sub.empty:
        print("[LoRA Fig 3] No data for either family – skipping.")
        return None
    if full_sub.empty:
        print("[LoRA Fig 3] No full-FT data at n_members={} – skipping.".format(LORA_N_MEMBERS))
        return None
    if lora_sub.empty:
        print("[LoRA Fig 3] No LoRA data at n_members={} – skipping.".format(LORA_N_MEMBERS))
        return None

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 5))

        full_sorted = full_sub.sort_values("epochs")
        ax.plot(full_sorted["kl_seq"], full_sorted["adv"], "o-", color="black", lw=2, label="Full-FT")

        for rank in LORA_RANKS:
            grp = lora_sub[lora_sub["lora_rank"] == rank]
            if grp.empty:
                continue
            color = PALETTE_RANK.get(rank, "gray")
            ax.scatter(grp["kl_seq"], grp["adv"], color=color, marker="o", label=f"LoRA r={int(rank)}", zorder=3)

        bh_ref = (
            pd.concat([full_sub[["kl_seq", "bh_seq"]], lora_sub[["kl_seq", "bh_seq"]]])
            .drop_duplicates()
            .sort_values("kl_seq")
        )
        ax.plot(bh_ref["kl_seq"], bh_ref["bh_seq"], color="red", lw=1.5, ls="-", label="BH bound (fn of KL only)")

        ax.axhline(1.0, color="gray", ls="--", lw=1)
        ax.annotate("Saturation ceiling (y=1)", xy=(ax.get_xlim()[1], 1.0),
                    xytext=(0, 5), textcoords="offset points", ha="right",
                    color="gray", fontsize=8)

        ax.set_xlabel("KL divergence (sequence-level)")
        ax.set_ylabel("Empirical Adv̂")
        ax.set_title("LoRA vs Full Fine-Tuning: Advantage vs KL")
        ax.legend(fontsize=8)
        fig.tight_layout()
        out = _save_fig(fig, 3)
    return out


def lora_fig4_tightness_vs_rank(df: pd.DataFrame):
    """Bound tightness (adv / bh_seq) vs epoch, one line per rank."""
    sub = df[
        (df["signal"] == LORA_SIGNAL) &
        (df["n_members"] == LORA_N_MEMBERS) &
        (df["seed"] == LORA_SEED) &
        df["lora_rank"].notna() &
        df["dp_epsilon"].isna()
    ].copy()
    if sub.empty:
        print("[LoRA Fig 4] No LoRA data for signal=ref, n_members={}, seed=0 – skipping.".format(LORA_N_MEMBERS))
        return None

    if "adv_over_bh_seq" in sub.columns:
        ratio_col = "adv_over_bh_seq"
    elif "adv_over_bound_seq" in sub.columns:
        ratio_col = "adv_over_bound_seq"
    else:
        sub["adv_over_bh_seq"] = sub["adv"] / sub["bh_seq"].replace(0, np.nan)
        ratio_col = "adv_over_bh_seq"

    all_epochs = _intersect_epochs(sub, "lora_rank")
    if not all_epochs:
        print("[LoRA Fig 4] No epochs common to all ranks – skipping.")
        return None
    sub = sub[sub["epochs"].isin(all_epochs)]

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 5))

        for rank in LORA_RANKS:
            grp = sub[sub["lora_rank"] == rank].sort_values("epochs")
            if grp.empty:
                continue
            color = PALETTE_RANK.get(rank, "gray")
            _cat_line(ax, all_epochs, grp[ratio_col].tolist(), None,
                       color, marker="o", label=f"r={int(rank)}")

        ax.axhline(1.0, color="gray", ls="--", lw=1, alpha=0.7, label="Ratio = 1 (tight)")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Adv / BH bound")
        ax.set_title("LoRA: Bound Tightness vs Rank")
        ax.legend(fontsize=8)
        fig.text(0.5, -0.02,
                  "Note: high adv/BH near saturation is an artifact (both approach their ceiling) — read together with LoRA Fig 1.",
                  ha="center", fontsize=8, style="italic")
        fig.tight_layout()
        out = _save_fig(fig, 4)
    return out


def lora_fig5_signal_auroc(df: pd.DataFrame):
    """Signal robustness under LoRA: AUROC vs epoch for rank=16, all signals."""
    sub = df[
        (df["n_members"] == LORA_N_MEMBERS) &
        (df["seed"] == LORA_SEED) &
        (df["lora_rank"] == 16) &
        df["dp_epsilon"].isna()
    ].copy()
    if sub.empty:
        print("[LoRA Fig 5] No LoRA r=16 data at n_members={}, seed=0 – skipping.".format(LORA_N_MEMBERS))
        return None

    all_epochs = _intersect_epochs(sub, "signal")
    if not all_epochs:
        print("[LoRA Fig 5] No epochs common to all signals – skipping.")
        return None
    sub = sub[sub["epochs"].isin(all_epochs)]

    from metrics.plot_figures import PALETTE_SIGNALS

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 5))

        for sig, grp in sub.groupby("signal"):
            color = PALETTE_SIGNALS.get(sig, "gray")
            grp_sorted = grp.sort_values("epochs")
            _cat_line(ax, all_epochs, grp_sorted["auroc"].tolist(), None,
                       color, marker="o", label=sig)

        ax.axhline(0.5, color="gray", ls="--", lw=1, label="Random guessing (0.5)")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("AUROC")
        ax.set_title("LoRA r=16: Signal Robustness")
        ax.legend(fontsize=8)
        fig.tight_layout()
        out = _save_fig(fig, 5)
    return out


# Main


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {DATA_PATH}")
    print(f"  shape   : {df.shape}\n")

    print("LoRA rows per rank:")
    lora_rows = df[df["lora_rank"].notna()]
    for rank in LORA_RANKS:
        n = (lora_rows["lora_rank"] == rank).sum()
        print(f"  r={rank}: {n} row(s)")
    print()

    figure_fns = [
        (1, lora_fig1_kl_vs_epoch),
        (2, lora_fig2_adv_vs_bh_per_rank),
        (3, lora_fig3_lora_vs_full),
        (4, lora_fig4_tightness_vs_rank),
        (5, lora_fig5_signal_auroc),
    ]

    saved, skipped, failed = [], [], []
    for n, fn in figure_fns:
        try:
            out = fn(df)
            if out is not None:
                saved.append(str(out))
                print(f"  [LoRA Fig {n}] Saved  -> {out}")
            else:
                skipped.append(n)
                print(f"  [LoRA Fig {n}] Skipped (no matching data).")
        except Exception as exc:
            failed.append(n)
            print(f"  [LoRA Fig {n}] FAILED: {exc}")
            traceback.print_exc()

    print("\n--- Summary ---")
    print(f"Saved   ({len(saved)})  : {saved if saved else 'none'}")
    print(f"Skipped ({len(skipped)}): {skipped if skipped else 'none'}")
    print(f"Failed  ({len(failed)}) : {failed if failed else 'none'}")


if __name__ == "__main__":
    main()
