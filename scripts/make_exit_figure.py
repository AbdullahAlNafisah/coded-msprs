"""Render the §IV.A EXIT figure: a 5x2 grid of modem EXIT families.

Layout (matches the requested figure):

    row 1 : 4-ASK Gray            | 4-ASK Natural
    row 2 : MS-PRS L0=3 balanced  | MS-PRS L0=3 unbalanced
    row 3 : MS-PRS L0=4 balanced  | MS-PRS L0=4 unbalanced
    row 4 : MS-PRS L0=5 balanced  | MS-PRS L0=5 unbalanced
    row 5 : MS-PRS L0=6 balanced  | MS-PRS L0=6 unbalanced

i.e. the left column is the balanced family (plus 4-ASK Gray), the right
column is the unbalanced family (plus 4-ASK Natural). Each panel shows the
modem EXIT family gradient-shaded across Eb/N0 (single shared viridis
colorbar on the right) with the rate-1/2, K=3 convolutional decoder curve
overlaid (axes swapped). Reads the JSON caches under
``results/exit/`` produced by ``scripts/make_exit_data.py`` and the
sibling EXIT notebooks.

Run from repo root in the venv:
    python scripts/make_exit_figure.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nsm.utils.plotting_style import apply_style
from nsm.provenance import verify
apply_style()

OUT = ROOT / "results" / "exit"
FIG_OUT = ROOT / "figures"

# Grid is 2 rows x N columns: each column is one modem, the top row its first
# variant and the bottom row its second. Column 1 is the 4-ASK reference
# (Gray top / natural bottom); the remaining columns are MS-PRS L0=3,4,5,6
# (left to right) with the balanced family on the top row and the unbalanced
# family on the bottom row. A column is rendered only once both its caches
# exist, so the figure can be produced before every L0 has finished generating.
COLUMNS = [
    ("ask4_gray",        "ask4_natural"),
    ("nsm_L3_balanced",  "nsm_L3_unbalanced"),
    ("nsm_L4_balanced",  "nsm_L4_unbalanced"),
    ("nsm_L5_balanced",  "nsm_L5_unbalanced"),
    ("nsm_L6_balanced",  "nsm_L6_unbalanced"),
]
DECODER = "coder_K3"
#: Second decoder characteristic, overlaid so that Section IV-C's claim about
#: which END of the modem curve governs is visible rather than asserted. This is
#: the same MacKay matrix and the same 20 BP iterations as the ldpc-msprs BER
#: caches, produced by scripts/make_ldpc_exit_data.py. Set to None to omit.
DECODER_LDPC = "ldpc_mackay"
CMAP = "viridis"
MODEM_LS = "-"
DECODER_LS = "--"
DECODER_LDPC_LS = "-."
# Only a few standout Eb/N0 values are drawn per panel (with a legend) so the
# modem EXIT characteristics read clearly instead of as a dense swept band.
STANDOUT_SNRS = (0.0, 3.0, 6.0, 9.0)


_SEEN_CONVENTIONS = {}


def _load(key: str) -> dict:
    p = OUT / f"{key}.json"
    if not p.exists():
        raise FileNotFoundError(f"missing EXIT cache: {p.relative_to(ROOT)} "
                                f"(generate with scripts/make_exit_data.py)")
    d = json.loads(p.read_text())
    # Record the metric convention of every cache this figure touches, then
    # refuse the whole figure if any affected one is stale or if they disagree.
    _SEEN_CONVENTIONS[key] = d.get("meta")
    verify(_SEEN_CONVENTIONS)
    return d


def get_modem_curve(key: str, eb_no_db: float, metric: str = "IE_avg"):
    d = _load(key)
    snr_str = next((k for k in d["results"] if abs(float(k) - eb_no_db) < 1e-6), None)
    if snr_str is None:
        raise KeyError(f"{key} has no Eb/N0={eb_no_db} dB")
    res = d["results"][snr_str]
    return np.asarray(d["IA"]), np.asarray(res.get(metric, res["IE_avg"]))


def get_decoder_curve(key: str, metric: str = "IE_avg"):
    d = _load(key)
    return np.asarray(d["IA"]), np.asarray(d[metric])


def modem_title(key: str) -> str:
    if key == "ask4_gray":
        return r"$4$-ASK (Gray)"
    if key == "ask4_natural":
        return r"$4$-ASK (Natural)"
    if key.startswith("nsm_L"):
        L0 = key.split("_")[1][1:]
        fam = "unbal." if key.endswith("unbalanced") else "bal."
        return rf"MS-PRS $L_0={L0}$ ({fam})"
    return key


def converged_ie(IA, IE, IA_d, IE_d, n_iter=5000, tol=1e-6):
    """The mutual information the turbo iteration converges to from zero
    a-priori: the first fixed point of the modem->decoder staircase (i.e. the
    inner/outer EXIT intersection), or ~1 when the tunnel stays open. With no
    decoder curve, fall back to the modem extrinsic at full a-priori."""
    if IA_d is None:
        return float(np.interp(1.0, IA, IE))
    x = 0.0
    for _ in range(n_iter):
        y = float(np.interp(x, IA, IE))           # modem: I_E^modem
        x_new = float(np.interp(y, IA_d, IE_d))   # decoder (axes swapped)
        if x_new > 0.999 or x_new - x < tol:
            x = x_new
            break
        x = x_new
    return float(np.interp(min(x, 1.0), IA, IE))


def plot_family(ax, modem_key, eb_no_db_list, decoder=None, show_ldpc=True,
                metric="IE_avg", cmap_name=CMAP):
    """Render one modem's EXIT family: the swept-Eb/N0 viridis colour map, a
    few emphasised standout curves on top, and a lower-right legend giving each
    standout Eb/N0 and the I_E the turbo iteration converges to."""
    ax.plot([0, 1], [0, 1], ":", color="0.5", linewidth=0.9, zorder=1)

    snrs = np.sort(np.asarray(eb_no_db_list, dtype=float))
    cmap = plt.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=snrs[0], vmax=snrs[-1])
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)

    # Swept colour map: shade the band between consecutive Eb/N0 EXIT curves.
    IA_common = np.linspace(0, 1, 400)
    base = np.array([np.interp(IA_common, *get_modem_curve(modem_key, s, metric))
                     for s in snrs])
    dense = np.linspace(snrs[0], snrs[-1], 200)
    grid = np.empty((dense.size, IA_common.size))
    for i in range(IA_common.size):
        grid[:, i] = np.interp(dense, snrs, base[:, i])
    for i in range(dense.size - 1):
        ax.fill_between(IA_common, grid[i], grid[i + 1],
                        color=mappable.to_rgba(dense[i]), edgecolor="none",
                        antialiased=True, zorder=2, rasterized=True)

    IA_d = IE_d = None
    if decoder is not None:
        IA_d, IE_d = get_decoder_curve(decoder)
        ax.plot(IE_d, IA_d, DECODER_LS, color="0.0", linewidth=1.5, zorder=6,
                path_effects=[pe.Stroke(linewidth=3.0, foreground="white"),
                              pe.Normal()])

    # The LDPC decoder characteristic, axes swapped like the convolutional one.
    # Its transfer is far steeper (max slope ~22 against ~3), so it meets the
    # modem curves near the bootstrap end while the K=3 curve meets them near
    # the terminal end. That contrast is the mechanism Section IV-C reports.
    # The converged-I_E annotations below stay tied to the convolutional
    # decoder, which is the one every plotted BER curve in Fig. 5 used.
    if DECODER_LDPC is not None and show_ldpc:
        try:
            IA_l, IE_l = get_decoder_curve(DECODER_LDPC)
            # Crimson with a white casing, drawn above the convolutional curve.
            # A grey line was illegible against the dark end of viridis, which
            # is exactly the low-Eb/N0 region where the contrast between the two
            # decoders matters most.
            ax.plot(IE_l, IA_l, DECODER_LDPC_LS, color="#d62728", linewidth=1.9,
                    zorder=7,
                    path_effects=[pe.Stroke(linewidth=3.4, foreground="white"),
                                  pe.Normal()])
        except FileNotFoundError:
            pass

    # Emphasise a few standout Eb/N0 curves (black-outlined) and report, in the
    # legend, the I_E each one converges to under iterative decoding.
    outline = [pe.Stroke(linewidth=3.0, foreground="black"), pe.Normal()]
    handles = []
    for s in STANDOUT_SNRS:
        IA, IE = get_modem_curve(modem_key, s, metric)
        ie_conv = converged_ie(IA, IE, IA_d, IE_d)
        ln, = ax.plot(IA_common, np.interp(IA_common, IA, IE),
                      color=cmap(norm(s)), linewidth=1.8, zorder=6,
                      path_effects=outline,
                      label=rf"${s:.0f}$ dB $\to {ie_conv:.2f}$")
        handles.append(ln)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_title(modem_title(modem_key), pad=3, fontsize=8)
    leg = ax.legend(handles=handles, loc="lower right", fontsize=6.5, frameon=True,
                    framealpha=1.0, handlelength=1.1, handletextpad=0.35,
                    borderpad=0.28, labelspacing=0.22)
    # draw the legend on top of every curve (standout curves use zorder up to 6)
    leg.set_zorder(20)
    leg.get_frame().set_edgecolor("0.7")
    leg.get_frame().set_facecolor("white")
    return mappable


def main():
    # Render only rows whose data exists; missing rows are filled in once their
    # caches appear (same partial-render logic as before).
    rows = [row for row in COLUMNS
            if all((OUT / f"{k}.json").exists() for k in row)]
    for row in COLUMNS:
        if row not in rows:
            print(f"  skip row {row} (cache not present yet)")

    # SNR grid is whatever the caches were generated on; read it from one file.
    eb = np.asarray(_load(rows[0][0])["eb_no_db"])

    nrows = len(rows)
    # Single-column, tall 5x2 grid of square panels with the swept-Eb/N0 colour
    # map (shared colorbar at right). A few standout curves are emphasised per
    # panel, each panel's lower-right legend giving the standout Eb/N0 values
    # and the I_E the turbo iteration converges to.
    panel = 1.5
    fig_w = 2 * panel + 1.0   # panels + y-label + colorbar
    fig_h = nrows * panel + 0.8  # panels + x-label + titles
    fig, axes = plt.subplots(nrows, 2, figsize=(fig_w, fig_h),
                             sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    mappable = None
    for r, (left, right) in enumerate(rows):
        for c, key in enumerate((left, right)):
            mappable = plot_family(axes[r, c], key, eb, decoder=DECODER,
                                   show_ldpc=key.startswith("nsm_"))
            axes[r, c].set_box_aspect(1)   # perfect square

    # y-label on leftmost column; x-label on bottom row only.
    for r in range(nrows):
        axes[r, 0].set_ylabel(r"$I_E^\mathrm{modem}\!=\!I_A^\mathrm{dec}$",
                              fontsize=7)
    for c in range(2):
        axes[nrows - 1, c].set_xlabel(r"$I_A^\mathrm{modem}\!=\!I_E^\mathrm{dec}$",
                                      fontsize=7)

    fig.tight_layout(rect=[0, 0, 0.89, 1], h_pad=0.5, w_pad=0.4)
    cbar_ax = fig.add_axes([0.91, 0.06, 0.022, 0.88])
    cbar = fig.colorbar(mappable, cax=cbar_ax,
                        ticks=np.arange(eb[0], eb[-1] + 0.1, 2.0))
    cbar.set_label(r"$E_b/N_0$ (dB)", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    FIG_OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out = FIG_OUT / f"exit_modems_grid.{ext}"
        fig.savefig(out, dpi=200 if ext == "png" else 400, bbox_inches="tight")
        print(f"  wrote {out.relative_to(ROOT)}")
    plt.close(fig)


if __name__ == "__main__":
    main()
