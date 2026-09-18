"""House figure style (IEEE single-column) + organized, timestamped saving.

Every figure:
  - imports apply_style() before plotting
  - saves via save(name, category) -> results/figures/<category>/<name>_<timestamp>.{pdf,png}
Both vector PDF (for the paper) and PNG (quick preview) are written.
"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
from pylab import rcParams

FONT_SIZE = 9

# house palette
RED   = "#B85450"   # dead / worst
GREEN = "#82B366"   # covered / best
BLUE  = "#6C8EBF"   # neutral / third series
RED2  = "#EA6B66"
LIGHTBLUE = "#DAE8FC"
GREY  = "#666666"
PALETTE = [RED, GREEN, BLUE, RED2, LIGHTBLUE, GREY]

FIGDIR = Path("results/figures")
PNG_DPI = 300


def apply_style(usetex: bool = True):
    params = {
        "axes.labelsize": FONT_SIZE,
        "axes.linewidth": 1,
        "font.size": FONT_SIZE,
        "legend.fontsize": FONT_SIZE - 2,
        "xtick.labelsize": FONT_SIZE,
        "xtick.major.size": 2,
        "ytick.labelsize": FONT_SIZE,
        "ytick.major.size": 2,
        "text.usetex": usetex,
        "figure.figsize": [4 * 0.9, 3 * 0.9],
    }
    rcParams.update(params)


def grid_box():
    plt.grid(alpha=0.35, linewidth=1, zorder=0)
    plt.box(on=True)


def save(name: str, category: str, pad: float = 0.01, fig=None):
    """Save current (or given) figure as PDF + PNG into a per-category folder,
    with a timestamp in the filename.

    Output: results/figures/<category>/<name>_<YYYYmmdd_HHMMSS>.pdf  (and .png)
    Returns (pdf_path, png_path).
    """
    outdir = FIGDIR / category
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{name}_{stamp}"

    f = fig if fig is not None else plt.gcf()
    f.tight_layout()

    pdf_path = outdir / f"{stem}.pdf"
    png_path = outdir / f"{stem}.png"
    f.savefig(pdf_path, format="pdf", bbox_inches="tight", pad_inches=pad)
    f.savefig(png_path, format="png", dpi=PNG_DPI, bbox_inches="tight", pad_inches=pad)
    print(f"saved {pdf_path}")
    print(f"saved {png_path}")
    return pdf_path, png_path