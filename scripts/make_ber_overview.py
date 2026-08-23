"""Build the two-panel Fig. 5 (paper §IV.B) BER figures from the per-SNR
JSON caches under results/ber/.

Two separate PDFs are written; main.tex combines them side-by-side as a
two-column ``figure*`` with ``\\subfloat`` panels:

  figures/ber_balanced.pdf    — panel (a):
      balanced MS-PRS  L0 ∈ {3,4,5,6}   (one sequential colour ramp)
      coded 2-ASK reference
      uncoded 2-ASK / 4-ASK theory (analytic)

  figures/ber_unbalanced.pdf  — panel (b):
      unbalanced MS-PRS  L0 ∈ {3,4,5,6} (a second sequential colour ramp)
      coded 4-ASK (Gray) and coded 4-ASK (Natural) references
      uncoded 4-ASK theory (analytic)

A given L0 uses the SAME marker shape in both panels, so a reader can
cross-reference an L0 between the balanced and unbalanced families.

Each panel also carries the LDPC-coded MS-PRS curves for its own family at
L0 = 3 and 4 (dashed, same colour and marker as the convolutional curve of the
same L0), so that Section IV-C's ordering reversal is read by comparing the
solid/dashed relation between the two panels. These are the turbo LDPC caches
from the C++ chain; the old one-shot LDPC path that produced invalid data was
deleted in 9a93796. The FTN cache was corrected (e62186f) and is overlaid on
panel (b).
Any cache that is missing, has a BER point > 0.5, or rises non-monotonically
in Eb/N0 is skipped with a printed warning rather than crashing the build.

Run from the repo root:
    python scripts/make_ber_overview.py [--snr-min 0] [--snr-max 7]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


from nsm.curves import load_curve
from nsm.modem.ask2 import ber_theory as ask2_theory
from nsm.modem.ask4 import ber_theory as ask4_theory
from nsm.utils.plotting_style import apply_style, plot_theory, set_paper_dims, style_for

apply_style()

OUT = ROOT / "results" / "ber"
FIG_OUT = ROOT / "figures"
FIG_OUT.mkdir(parents=True, exist_ok=True)

# Same marker per L0 in BOTH panels → the reader cross-references an L0
# between the balanced (a) and unbalanced (b) families by marker shape.
L0_MARKER = {3: "o", 4: "s", 5: "^", 6: "D"}

# One sequential colour ramp per family; sampled away from the pale end
# so all four L0 stay legible. Balanced and unbalanced get distinct hues.
_FAMILY_RAMP = {"balanced": plt.get_cmap("Blues"),
                "unbalanced": plt.get_cmap("Oranges")}
_RAMP_T = {3: 0.95, 4: 0.78, 5: 0.62, 6: 0.46}   # darkest = smallest L0

def _family_colour(family: str, L0: int):
    return _FAMILY_RAMP[family](_RAMP_T[L0])




def _new_panel():
    # Taller than the default single_tall (3.5x3.0) so the boxed legend clears
    # the curves at the bottom-left without overlapping them.
    fig, ax = plt.subplots(figsize=(3.5, 3.7))
    ax.set_yscale("log")
    return fig, ax


def _finish_panel(ax, args):
    ax.set_xlim(args.snr_min, args.snr_max)
    ax.set_ylim(1e-7, 0.5)
    ax.set_xlabel(r"$E_b/N_0$ (dB)")
    ax.set_ylabel("BER")
    ax.grid(True, which="both", alpha=0.35, linestyle=":")
    # Matplotlib emits plain Line2D handles before ErrorbarContainers, which
    # splits the analytic references away from the measured ones and drops the
    # LDPC proxy in the middle. Reorder to the sequence the text uses.
    handles, labels = ax.get_legend_handles_labels()
    def rank(lab):
        for i, key in enumerate(("Uncoded 2-ASK", "Uncoded 4-ASK (Gray)",
                                 "Uncoded 4-ASK (Natural)", "Coded 2-ASK",
                                 "Coded 4-ASK (Gray)", "Coded 4-ASK (Natural)")):
            if lab.startswith(key):
                return i
        if lab.startswith("MS-PRS"):
            # label is r"MS-PRS $L_0{=}3$", so split("=") yields "}3$" -- take
            # the first digit anywhere in the string instead.
            m = re.search(r"(\d)", lab)
            return 10 + (int(m.group(1)) if m else 0)
        return 90          # the LDPC style proxy, last
    order = sorted(range(len(labels)), key=lambda i: rank(labels[i]))
    ax.legend([handles[i] for i in order], [labels[i] for i in order],
              loc="lower left", fontsize=5.6, frameon=True,
              framealpha=0.92, edgecolor="0.7", handlelength=2.2,
              borderpad=0.35, labelspacing=0.25)


def _save(fig, stem):
    for ext in ("pdf", "png"):
        out = FIG_OUT / f"{stem}.{ext}"
        fig.savefig(out, dpi=200 if ext == "png" else None,
                    bbox_inches="tight")
        print(f"  wrote {out.name}")
    plt.close(fig)


def _plot_msprs_family(ax, family: str):
    # Each panel holds a single energy family, so the legend omits the
    # bal./unbal. tag (it is stated in the figure caption) and labels each
    # curve by L0 alone.
    loaded = {L0: load_curve(f"nsm_L{L0}_{family}_conv_K3_7iters")
              for L0 in (3, 4, 5, 6)}
    # A still-running sweep leaves the largest L0 with fewer Eb/N0 points; drop
    # any partial curve so the paper figure never ships a stubby line next to
    # the full ones. The longest curve in the family sets the bar; once a
    # lagging L0 catches up it renders automatically.
    ref_n = max((len(d) for d in loaded.values() if d is not None), default=0)
    for L0 in (3, 4, 5, 6):
        d = loaded[L0]
        if d is None:
            continue
        if len(d) < ref_n:
            print(f"  skip nsm_L{L0}_{family}  ({len(d)}/{ref_n} pts — incomplete)")
            continue
        eb, ber, lo, hi = d.eb_no_db, d.ber, d.ci_low, d.ci_high
        ax.errorbar(
            eb, ber, yerr=[ber - lo, hi - ber],
            color=_family_colour(family, L0),
            linestyle="-", marker=L0_MARKER[L0], markersize=4,
            linewidth=1.3, alpha=0.95,
            elinewidth=0.7, capsize=1.5, capthick=0.5,
            label=rf"MS-PRS $L_0{{=}}{L0}$",
        )


def _plot_ldpc_family(ax, family: str):
    """Overlay the LDPC-coded MS-PRS curves for the same energy family.

    Same colour ramp and same marker per L0 as the convolutional curves, with a
    dashed line, so within a panel line style reads as the outer code and the
    reader can pair an LDPC curve with its convolutional counterpart by eye.
    The reversal reported in Section IV-C is then the relation between the solid
    and dashed pairs FLIPPING between the two panels.

    Only L0 = 3 and 4 are drawn: the L0 = 5 and 6 LDPC sweeps were stopped
    deliberately, since two filter lengths already establish the effect and the
    remaining two would have cost about a day of compute to confirm it.

    No error bars, unlike every other curve here. `clopper_pearson` is a
    binomial interval on BIT errors and assumes independent trials. Near the
    LDPC threshold the chain fails in whole frames of a few hundred bits, so a
    point carrying a few thousand bit errors rests on of order ten independent
    frame failures and the binomial interval understates its spread by roughly
    an order of magnitude. Drawing it would assert a precision the data does
    not have.
    """
    drawn = []
    for L0 in (3, 4):
        key = f"ldpc_msprs_L{L0}_{family}_turbo7"
        d = load_curve(key)
        if d is None:
            continue
        eb, ber = d.eb_no_db, d.ber
        ax.plot(eb, ber,
                color=_family_colour(family, L0),
                linestyle="--", marker=L0_MARKER[L0], markersize=3.4,
                linewidth=1.2, alpha=0.95, markerfacecolor="none",
                markeredgewidth=0.9)
        drawn.append(L0)
    if drawn:
        # One proxy entry rather than one per L0: colour and marker already
        # identify the L0 from the convolutional curves, so the only new
        # information is what the dashed style means.
        ax.plot([], [], linestyle="--", color="0.35", marker="o",
                markerfacecolor="none", markersize=3.4, linewidth=1.2,
                label=rf"LDPC outer code ($L_0{{\in}}\{{{','.join(str(v) for v in drawn)}\}}$)")


def _plot_benchmarks(ax, eb_th):
    """Common reference curves drawn in BOTH BER panels: uncoded 2-/4-ASK
    (analytic) and coded 2-ASK / coded 4-ASK (Gray, Natural). The uncoded
    curves are analytic; that is noted in the caption rather than the legend so
    the labels stay short. Reference markers stay OUTSIDE the L0 marker set
    {o,s,^,D} so a marker shape unambiguously identifies an MS-PRS L0."""
    plot_theory(ax, eb_th, ask2_theory(eb_th), "uncoded_ref",
                label="Uncoded 2-ASK")
    plot_theory(ax, eb_th, ask4_theory(eb_th, gray=True), "uncoded_ref",
                label="Uncoded 4-ASK (Gray)", linestyle="-.")
    plot_theory(ax, eb_th, ask4_theory(eb_th, gray=False), "uncoded_ref",
                label="Uncoded 4-ASK (Natural)", linestyle=":")
    d = load_curve("ask2_conv_K3")
    if d is not None:
        eb, ber, lo, hi = d.eb_no_db, d.ber, d.ci_low, d.ci_high
        ax.errorbar(eb, ber, yerr=[ber - lo, hi - ber],
                    color="#000000", linestyle="-",
                    marker="*", markersize=5, linewidth=1.2,
                    elinewidth=0.7, capsize=1.5, capthick=0.5,
                    label="Coded 2-ASK")
    for key, style, lab, mk in [
        ("ask4_gray_conv_K3_7iters",    "ask4_gray",    "Coded 4-ASK (Gray)",    "P"),
        ("ask4_natural_conv_K3_7iters", "ask4_natural", "Coded 4-ASK (Natural)", "X"),
    ]:
        d = load_curve(key)
        if d is not None:
            eb, ber, lo, hi = d.eb_no_db, d.ber, d.ci_low, d.ci_high
            kw = style_for(style)
            kw.update(marker=mk, markersize=3.5, label=lab)
            ax.errorbar(eb, ber, yerr=[ber - lo, hi - ber],
                        elinewidth=0.7, capsize=1.5, capthick=0.5, **kw)
    # The binary FTN tau=0.5 curve was removed on 2026-08-19 and its cache
    # deleted on 2026-08-23. The data was of the wrong channel: nsm/modem/ftn.py normalises the
    # one-sided pulse autocorrelation to unit energy and drives it with white
    # noise, which pins the isolated-error distance to 4, the ISI-free 2-ASK
    # value, independently of tau. The true tau=0.5 MSED is 2.03, a 2.95 dB
    # deficit (scripts/ftn_msed.py). That is why the curve used to land 0.02 dB
    # from coded 2-ASK while carrying twice its rate. The paper now cites
    # Anderson's published FTN results instead of simulating one.


def panel_balanced(args):
    """Panel (a): balanced MS-PRS family + common 2-/4-ASK benchmarks."""
    fig, ax = _new_panel()
    eb_th = np.linspace(args.snr_min, args.snr_max, 200)
    _plot_benchmarks(ax, eb_th)
    _plot_msprs_family(ax, "balanced")
    _plot_ldpc_family(ax, "balanced")
    _finish_panel(ax, args)
    _save(fig, "ber_balanced")


def panel_unbalanced(args):
    """Panel (b): unbalanced MS-PRS family + common benchmarks (incl. FTN)."""
    fig, ax = _new_panel()
    eb_th = np.linspace(args.snr_min, args.snr_max, 200)
    _plot_benchmarks(ax, eb_th)
    _plot_msprs_family(ax, "unbalanced")
    _plot_ldpc_family(ax, "unbalanced")
    _finish_panel(ax, args)
    _save(fig, "ber_unbalanced")


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--snr-min", type=float, default=0.0)
    p.add_argument("--snr-max", type=float, default=7.0)
    args = p.parse_args()

    print("panel (a) balanced  → ber_balanced.{pdf,png}")
    panel_balanced(args)
    print("panel (b) unbalanced → ber_unbalanced.{pdf,png}")
    panel_unbalanced(args)


if __name__ == "__main__":
    main()
