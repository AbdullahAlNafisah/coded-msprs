"""Regenerate figures/ber_convergence.{pdf,png} as a 2-row grid of
per-iteration BER convergence panels, one inner modem per column.

Layout (mirrors the EXIT grid of Fig.~4):

    columns : 4-ASK | L0 = 3, 4, 5, 6   (left to right)
    top row : 4-ASK Gray     /  balanced MS-PRS family   (eta_0 = eta_1 = 5/2)
    bottom  : 4-ASK natural   /  unbalanced MS-PRS family  (eta_0 != eta_1)

The leftmost column is the coded 4-ASK reference (Gray top, natural bottom),
matching the benchmark set of Fig.~5; the remaining columns are rate-2 MS-PRS
for filter lengths L0 = 3, 4, 5, 6.

Each panel shows BER versus turbo-iteration index ell for a fixed set of
representative Eb/N0 values; a single shared Eb/N0 legend sits below the grid.

A column is rendered only once BOTH its rows' caches contain every picked Eb/N0
point, so the figure can be produced before every modem has finished simulating
(e.g. L6 still running). Missing columns are filled in automatically on the next
render once their caches complete.

Input: per-iteration error counts from
    results/ber/<dir>/snr_*.json
        field ``ers_per_iter`` (length = turbo iters + 1, index 0 = one-shot),
        normalised by ``bits_cnt``.
MS-PRS dirs: nsm_L{L0}_{balanced,unbalanced}_conv_K3_7iters.
4-ASK dirs:  ask4_{gray,natural}_conv_K3_7iters_periter -- the per-iteration
variant; the plain ask4_*_conv_K3_7iters caches store only the final-iteration
BER and are consumed by make_ber_overview.py instead.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nsm.utils.plotting_style import apply_style
from nsm.curves import load_curve
apply_style()

FIG_DIR = ROOT / "figures"
SIM_DIR = ROOT / "results" / "ber"

# MS-PRS filter lengths, one column each (after the 4-ASK reference column).
L0_COLUMNS = [3, 4, 5, 6]
# Representative Eb/N0 points (dB); shared across all panels so the single
# legend is meaningful. A column needs all of these in both rows to render.
EB_NO_PICK_DB = [3.0, 4.0, 5.0, 6.0, 7.0]


def _msprs_column(L0: int):
    """Column spec for an MS-PRS filter length: [(top_dir, title), (bottom, title)]."""
    return [(f"nsm_L{L0}_balanced_conv_K3_7iters",   rf"$L_0={L0}$ (bal.)"),
            (f"nsm_L{L0}_unbalanced_conv_K3_7iters", rf"$L_0={L0}$ (unbal.)")]


# Leftmost reference column: coded 4-ASK (Gray top, natural bottom).
ASK4_COLUMN = [("ask4_gray_conv_K3_7iters_periter",    "4-ASK (Gray)"),
               ("ask4_natural_conv_K3_7iters_periter", "4-ASK (natural)")]

# Full column order: 4-ASK reference, then MS-PRS L0 = 3..6.
COLUMNS = [ASK4_COLUMN] + [_msprs_column(L0) for L0 in L0_COLUMNS]


def _load_curves(subdir: str):
    """Return {eb_no_db: ber_per_iter[]} for the picked Eb/N0 points."""
    curve = load_curve(subdir)
    if curve is None or curve.ers_per_iter is None:
        return {}
    out = {}
    for eb, per_iter, bits in zip(curve.eb_no_db, curve.ers_per_iter, curve.bit_count):
        eb = float(eb)
        if eb not in EB_NO_PICK_DB:
            continue
        ers = np.asarray(per_iter, dtype=float)
        # Half-error floor so a zero count stays on the log axis.
        out[eb] = np.where(ers > 0, ers / float(bits), 0.5 / float(bits))
    return out


def _column_ready(column) -> bool:
    """True iff both rows have every picked Eb/N0 point."""
    for subdir, _ in column:
        have = _load_curves(subdir)
        if not all(eb in have for eb in EB_NO_PICK_DB):
            return False
    return True


def _plot_panel(ax, curves, title, cmap):
    handles, labels = [], []
    for color, eb in zip(cmap, EB_NO_PICK_DB):
        ber = curves[eb]
        iters = np.arange(len(ber))
        (line,) = ax.semilogy(iters, ber, "o-", color=color, lw=1.1, ms=3.5,
                              label=rf"$E_b/N_0 = {eb:+.0f}$ dB")
        handles.append(line); labels.append(line.get_label())
    ax.axhline(1e-5, color="0.35", ls="--", lw=0.7)
    ax.set_xticks(range(0, 8))
    ax.set_xlim(-0.2, 7.2)
    ax.set_ylim(1e-7, 1.0)
    ax.grid(True, which="both", lw=0.3, alpha=0.4)
    ax.set_title(title, fontsize=8, pad=3)
    return handles, labels


def main():
    cols = [col for col in COLUMNS if _column_ready(col)]
    for col in COLUMNS:
        if col not in cols:
            print(f"  skip column {col[0][1]} (cache incomplete for picked Eb/N0)")
    if not cols:
        print("  no complete columns yet; nothing rendered")
        return

    ncols = len(cols)
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(EB_NO_PICK_DB)))
    fig, axes = plt.subplots(2, ncols, figsize=(2.0 * ncols + 0.4, 4.6),
                             sharex=True, sharey=True)
    axes = np.atleast_2d(axes)

    handles = labels = None
    for c, col in enumerate(cols):
        (top_dir, top_title), (bot_dir, bot_title) = col
        h, lab = _plot_panel(axes[0, c], _load_curves(top_dir), top_title, cmap)
        _plot_panel(axes[1, c], _load_curves(bot_dir), bot_title, cmap)
        handles, labels = h, lab

    for r in range(2):
        axes[r, 0].set_ylabel("BER")
    for c in range(ncols):
        axes[1, c].set_xlabel(r"Turbo iteration index $\ell$")

    # Single shared Eb/N0 legend below the grid.
    fig.legend(handles, labels, loc="lower center", ncol=len(EB_NO_PICK_DB),
               fontsize=9, framealpha=0.95, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(pad=0.4, rect=[0, 0.08, 1, 1])

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out = FIG_DIR / f"ber_convergence.{ext}"
        fig.savefig(out, dpi=160 if ext == "png" else None, bbox_inches="tight")
        print(f"  wrote {out.name}")
    plt.close(fig)


if __name__ == "__main__":
    main()
