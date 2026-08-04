from pathlib import Path

import matplotlib.pyplot as plt

COLOR_FULL_FT = "#0072B2"   # blue
COLOR_LORA    = "#E69F00"   # orange
COLOR_MICA    = "#009E73"   # green
COLOR_DP      = "#D55E00"   # vermillion (red accent) -- base DP color, epsilon shades below
COLOR_BH      = "#CC79A7"   # reddish purple -- BH bound
COLOR_PINSKER = "#56B4E9"   # sky blue -- Pinsker bound
COLOR_ADV     = "#000000"   # black -- empirical advantage / TV

PALETTE_N_MEMBERS = {500: "#56B4E9", 2000: "#0072B2", 6000: "#D55E00"}
PALETTE_DP_EPS    = {1: "#F0E442", 4: "#E69F00", 8: "#D55E00"}  # light->dark = weak->strong privacy budget
SEED_MARKERS      = {0: "o", 1: "s", 2: "^"}

SAVE_DPI = 300


def apply_style() -> None:
    """Rcparams shared by every figure notebook -- legible at ~3.3in NeurIPS column width."""
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7,
        "figure.dpi": 100,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
    })


def save_fig(fig, name: str, out_dir: Path) -> tuple[Path, Path]:
    """Save fig as both vector PDF (for LaTeX \\includegraphics) and 300dpi PNG (quick preview)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{name}.pdf"
    png_path = out_dir / f"{name}.png"
    fig.savefig(pdf_path, dpi=SAVE_DPI)
    fig.savefig(png_path, dpi=SAVE_DPI)
    print(f"  saved -> {pdf_path}")
    print(f"  saved -> {png_path}")
    return pdf_path, png_path
