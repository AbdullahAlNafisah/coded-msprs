"""Render the §III amplitude-multiplicity figure: a compact 2x4 grid showing,
for each rate-2 MS-PRS design, the enumerated 1-D output-amplitude levels with
their multiplicities (stems) overlaid on the fitted Gaussian envelope.

Layout (mirrors the Fig. 9 I/Q-grid arrangement):

    columns : L0 = 3, 4, 5, 6  (left to right)
    row 1   : balanced family    (eta_0 = eta_1 = 5/2)
    row 2   : unbalanced family   (eta_0 != eta_1)

For each (L0, family) the 2^(L0+1) trellis-branch output amplitudes
(precompute()["branch_labels"]) are enumerated; the stem heights are the level
multiplicities (peak-normalised), and the smooth curve is the unit Gaussian
N(0,1) -- the power normalisation fixes mean 0 and variance 1 for every design,
so the controlled-ISI memory makes the amplitude alphabet trace a Gaussian
envelope (a structural shaping gain) rather than the flat M-ASK distribution.

Output:
    figures/amplitude_multiplicity_grid.{pdf,png}

Run from repo root in the venv:
    python scripts/render_amplitude_grid.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nsm.modem.msprs import precompute
from nsm.utils.plotting_style import apply_style
apply_style()

L0_COLUMNS = [3, 4, 5, 6]
FAMILIES = ["balanced", "unbalanced"]
N_BITS = 64                      # only needed to drive precompute(); unused here
XLIM = 3.6                       # symbol-amplitude axis limit (sigma = 1)

OUT_PDF = ROOT / "figures" / "amplitude_multiplicity_grid.pdf"
OUT_PNG = ROOT / "figures" / "amplitude_multiplicity_grid.png"

C_STEM = "#1f77b4"   # level-multiplicity stems
C_GAUSS = "#111111"  # fitted Gaussian envelope


def _levels(L0: int, family: str):
    """Unique output-amplitude levels and their integer multiplicities."""
    bl = np.asarray(precompute(L0, N_BITS, family)["branch_labels"])
    lev, cnt = np.unique(np.round(bl, 6), return_counts=True)
    return lev, cnt, len(lev)


def _plot_panel(ax, L0, family):
    lev, cnt, n_lev = _levels(L0, family)
    cmax = int(cnt.max())
    ax.vlines(lev, 0.0, cnt, color=C_STEM, lw=1.3, zorder=2)   # raw multiplicity
    ax.plot(lev, cnt, "o", ms=2.4, color=C_STEM, zorder=3)

    ax.axvline(0, color="0.6", lw=0.6, ls="--", alpha=0.6, zorder=1)
    ax.set_xlim(-XLIM, XLIM)
    ax.set_ylim(0, cmax + max(1, round(0.28 * cmax)))
    ax.set_xticks([-2, 0, 2])
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))
    ax.tick_params(labelsize=6)
    ax.text(0.96, 0.93, rf"$M={n_lev}$", transform=ax.transAxes,
            ha="right", va="top", fontsize=7,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="0.8", lw=0.4,
                      alpha=0.9))
    return n_lev


def main():
    ncols = len(L0_COLUMNS)
    fig, axes = plt.subplots(2, ncols, figsize=(7.1, 3.45),
                             sharex=True, sharey=False)
    for r, fam in enumerate(FAMILIES):
        for c, L0 in enumerate(L0_COLUMNS):
            n = _plot_panel(axes[r, c], L0, fam)
            print(f"  {fam:<10} L{L0}: M={n} levels", flush=True)

    for c, L0 in enumerate(L0_COLUMNS):
        axes[0, c].set_title(rf"$L_0={L0}$", fontsize=9, pad=4)
    for r in range(2):
        axes[r, 0].set_ylabel("Multiplicity", fontsize=7)
    for c in range(ncols):
        axes[1, c].set_xlabel(r"Symbol amplitude $s_k$", fontsize=7)

    fig.tight_layout(rect=[0.03, 0, 1, 1], h_pad=0.6, w_pad=0.5)

    # Family group labels down the far-left margin (one per row).
    fig.text(0.008, 0.70, "Balanced", rotation=90, va="center", ha="left",
             fontsize=9, fontweight="bold")
    fig.text(0.008, 0.30, "Unbalanced", rotation=90, va="center", ha="left",
             fontsize=9, fontweight="bold")

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF)
    fig.savefig(OUT_PNG, dpi=200)
    plt.close(fig)
    print(f"saved {OUT_PDF.name}, {OUT_PNG.name}")


if __name__ == "__main__":
    main()
