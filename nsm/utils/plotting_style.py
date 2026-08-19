"""Project-wide plotting style for the codednsm paper figures.

A single source of truth for colors, line styles, legend labels, and IEEE
double-column figure dimensions. Notebooks should call ``apply_style()``
once at the top, then build curves with ``plot_theory(ax, ...)`` and
``plot_measured(ax, ...)`` (or look up styles directly via
``style_for(name)`` / ``LABEL[name]``).
"""
from __future__ import annotations

from typing import Optional

import matplotlib as mpl


# ── Communication-systems palette ──────────────────────────────────────
#
# Roles (not literal scheme names) so the same key works across BER,
# EXIT, and SDR plots. Resolve a scheme name to a role via STYLE below.
PALETTE = {
    "baseline":          {"color": "#1f77b4", "linestyle": "-",  "marker": None, "linewidth": 1.2, "alpha": 1.0},
    "compare_gray":      {"color": "#d62728", "linestyle": "--", "marker": None, "linewidth": 1.2, "alpha": 1.0},
    "compare_natural":   {"color": "#8b0000", "linestyle": "-.", "marker": None, "linewidth": 1.2, "alpha": 1.0},
    "compare_mftn":      {"color": "#ff7f0e", "linestyle": "--", "marker": None, "linewidth": 1.2, "alpha": 1.0},
    "hero":              {"color": "#000000", "linestyle": "-",  "marker": None, "linewidth": 1.5, "alpha": 1.0},
    "hero_sibling":      {"color": "#404040", "linestyle": ":",  "marker": None, "linewidth": 1.5, "alpha": 1.0},
    "hero_ldpc":         {"color": "#000000", "linestyle": "-",  "marker": "D",  "linewidth": 1.5, "alpha": 1.0, "markevery": 5},
    "measured_bpsk":     {"color": "#1f77b4", "linestyle": "none", "marker": "o", "alpha": 0.7},
    "measured_qpsk":     {"color": "#1f77b4", "linestyle": "none", "marker": "s", "alpha": 0.7},
    "measured_msprs":    {"color": "#000000", "linestyle": "none", "marker": "x", "alpha": 0.7},
    "measured_msprs_coded": {"color": "#000000", "linestyle": "none", "marker": "v", "alpha": 0.7},
    "uncoded_ref":       {"color": "#808080", "linestyle": "--", "marker": None, "linewidth": 0.8, "alpha": 0.6},
    "outer_decoder":     {"color": "#9467bd", "linestyle": "--", "marker": None, "linewidth": 1.2, "alpha": 1.0},
}


# Map scheme names → palette role + canonical legend label.
# Scheme names are the keys you pass to ``plot_theory`` / ``plot_measured``.
STYLE = {
    "ask2":          ("baseline",        "Theory: 2-ASK"),
    "bpsk":          ("baseline",        "Theory: BPSK"),
    "qpsk":          ("baseline",        "Theory: QPSK"),
    "ask4_gray":     ("compare_gray",    "Theory: 4-ASK (Gray)"),
    "ask4_natural":  ("compare_natural", "Theory: 4-ASK (Natural)"),
    "mftn":          ("compare_mftn",    r"Theory: MFTN ($\tau_\phi{=}0.5$)"),
    "msprs_unbal":   ("hero",            r"Sim: MS-PRS ($L_0{=}3$, unbal.)"),
    "msprs_bal":     ("hero_sibling",    r"Sim: MS-PRS ($L_0{=}3$, bal.)"),
    "msprs_coded":   ("hero",            r"Sim: Coded MS-PRS (turbo, $\ell{=}7$)"),
    "msprs_ldpc":    ("hero_ldpc",       "Sim: MS-PRS + LDPC"),
    "uncoded_ref":   ("uncoded_ref",     "Uncoded reference"),
    "outer_decoder": ("outer_decoder",   r"Outer decoder (rate-$1/2$, $K{=}3$)"),

    # Measured (SDR) — share theory color, distinct markers.
    "measured_bpsk":      ("measured_bpsk",       "Measured: BPSK"),
    "measured_qpsk":      ("measured_qpsk",       "Measured: QPSK"),
    "measured_msprs":     ("measured_msprs",      "Measured: MS-PRS (uncoded)"),
    "measured_msprs_coded": ("measured_msprs_coded", "Measured: Coded MS-PRS"),
}


# Convenience exports
LABEL = {name: label for name, (_, label) in STYLE.items()}


# ── IEEE double-column figure dimensions ───────────────────────────────
#
# IEEEtran two-column body: \columnwidth ≈ 3.5 in, \textwidth ≈ 7.16 in.
# Use these so figures embed at 1:1 and rcParam font sizes survive into print.
_PAPER_DIMS = {
    "single":      (3.5, 2.6),   # column-width, 4:3
    "single_tall": (3.5, 3.0),   # column-width, 7:6 (EXIT charts, 2-D maps)
    "double":      (7.16, 2.7),  # full text-width (figure*)
}


def set_paper_dims(span: str = "single") -> tuple[float, float]:
    """Return ``figsize`` for an IEEEtran ``\\columnwidth`` (``span="single"``)
    or ``\\textwidth`` (``span="double"``) figure. ``"single_tall"`` keeps
    column width but a slightly taller aspect ratio for square data."""
    try:
        return _PAPER_DIMS[span]
    except KeyError as e:
        raise ValueError(
            f"unknown span {span!r}; expected one of {list(_PAPER_DIMS)}"
        ) from e


# ── rcParams ───────────────────────────────────────────────────────────
_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.2,
    "lines.markersize": 4,
    "axes.grid": True,
    "grid.alpha": 0.4,
    "grid.linestyle": ":",
    "legend.framealpha": 0.9,
    "figure.figsize": _PAPER_DIMS["single"],
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,   # TrueType — required by IEEE submission
    "ps.fonttype": 42,
}


def apply_style() -> None:
    """Apply project-wide rcParams. Idempotent; safe to call repeatedly."""
    mpl.rcParams.update(_RC)


# ── Style resolution ───────────────────────────────────────────────────
def style_for(name: str, **overrides) -> dict:
    """Return a ``**kwargs`` dict ready to splat into ``ax.plot``.

    Unknown keys raise ``KeyError`` rather than silently falling back, so
    typos surface immediately.
    """
    if name not in STYLE:
        raise KeyError(
            f"unknown style {name!r}; expected one of {sorted(STYLE)}"
        )
    role, _label = STYLE[name]
    kw = dict(PALETTE[role])
    kw.update(overrides)
    return kw


# ── High-level plot helpers ────────────────────────────────────────────
def plot_theory(ax, x, y, name: str, *, label: Optional[str] = None,
                **overrides):
    """``ax.plot`` with the project's theory-curve style for ``name``.

    ``label=None`` (default) uses the canonical legend label. Pass
    ``label=""`` to suppress it, or any string to override.
    """
    kw = style_for(name, **overrides)
    legend_label = LABEL[name] if label is None else label
    if legend_label != "":
        kw["label"] = legend_label
    return ax.plot(x, y, **kw)


def plot_measured(ax, x, y, name: str, *, yerr=None, xerr=None,
                  label: Optional[str] = None, **overrides):
    """``ax.errorbar`` for measured (SDR) points using the canonical style.

    Errorbars inherit the curve color at ``alpha=0.4``, ``capsize=2``.
    """
    kw = style_for(name, **overrides)
    legend_label = LABEL[name] if label is None else label
    if legend_label != "":
        kw["label"] = legend_label

    if yerr is None and xerr is None:
        return ax.plot(x, y, **kw)

    ecolor = kw.get("color", "k")
    return ax.errorbar(
        x, y, yerr=yerr, xerr=xerr,
        ecolor=ecolor, elinewidth=0.8, capsize=2, capthick=0.6,
        # `alpha` on ax.errorbar applies to bars too; let the marker
        # carry the alpha already set in `kw` and dim only the bars.
        **kw,
    )


__all__ = [
    "PALETTE", "STYLE", "LABEL",
    "apply_style", "set_paper_dims", "style_for",
    "plot_theory", "plot_measured",
]
