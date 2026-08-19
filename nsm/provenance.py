"""Metric-convention stamping for simulation caches.

Background. Until 2026-08-10 the MS-PRS and FTN BCJR branch metrics divided the
squared error by ``sigma^2`` rather than ``2 sigma^2`` (``nsm/modem/msprs.py``,
``nsm/modem/ftn.py``), which made every extrinsic LLR exactly twice too large.

This docstring used to add that uncoded BER never saw it, on the grounds that
hard decisions are invariant to a positive LLR scaling. That reasoning is
wrong and was corrected on 2026-08-16. The invariance holds for scaling the
*output* LLR, but the bug scaled the *branch metrics*, and the BCJR's
log-sum-exp marginalisation is not homogeneous, so the posterior is not merely
rescaled. Measured on L0=3 balanced uncoded, the two metrics give 2.12e-1 vs
2.03e-1 at 1 dB, a 4.6 % gap at roughly 60 sigma. Uncoded caches for MS-PRS and
FTN are therefore affected too, which is why ``nsm_L3_balanced_uncoded`` is
listed as stale.

Consequence for cached results: a cache produced before that fix is NOT
convertible to a post-fix one. Do not attempt to rescale it. The 2x factor
holds only in the regime where the a priori dominates; in general a mismatched
metric is a BCJR running at half the true noise variance, and its effect on the
extrinsic output is nonlinear in I_A. Affected caches must be regenerated or
left marked stale.

Usage:

    from nsm.provenance import stamp, verify

    meta = stamp({"L0": 3, "filter": "balanced"})     # writer
    verify(loaded_metas)                              # reader, raises on mixed
"""
from __future__ import annotations

__all__ = ["METRIC_CONVENTION", "AFFECTED", "stamp", "convention_of", "verify"]

#: Identifier for the current soft-output convention. Bump this whenever a
#: change alters the numerical value of any LLR, so old caches stop validating.
METRIC_CONVENTION = "llr-2sigma2-2026-08-10"

#: The convention in force before the factor-2 fix. Caches carrying no stamp at
#: all are assumed to predate stamping and are treated as this.
LEGACY_CONVENTION = "llr-sigma2-legacy"

#: Substrings identifying schemes whose soft output changed. 2-ASK, 4-ASK and
#: the QAM benchmarks were always correct and are unaffected.
AFFECTED = ("nsm_", "nsm-", "msprs", "ftn")

_KEY = "metric_convention"


def stamp(meta: dict | None = None) -> dict:
    """Return ``meta`` with the current convention recorded. Mutates and returns."""
    meta = {} if meta is None else meta
    meta[_KEY] = METRIC_CONVENTION
    return meta


def convention_of(meta: dict | None) -> str:
    """Convention a cache was produced under. Unstamped means pre-fix."""
    if not isinstance(meta, dict):
        return LEGACY_CONVENTION
    return meta.get(_KEY, LEGACY_CONVENTION)


def _affected(name: str) -> bool:
    low = str(name).lower()
    return any(tag in low for tag in AFFECTED)


def verify(caches, *, strict: bool = True) -> list[str]:
    """Check a set of caches about to be combined into one figure.

    ``caches`` maps a label (scheme name or path) to its ``meta`` dict, or is an
    iterable of ``(label, meta)`` pairs.

    Rejects two situations, both of which silently corrupt a figure:

    * an affected scheme (MS-PRS, FTN) carrying a stale or absent stamp;
    * a mix of conventions across the affected schemes in one figure, which
      would plot pre-fix and post-fix curves on the same axes.

    Returns the list of problems. Raises ``RuntimeError`` when ``strict``.
    """
    items = caches.items() if isinstance(caches, dict) else list(caches)

    problems: list[str] = []
    seen: dict[str, list[str]] = {}
    for label, meta in items:
        conv = convention_of(meta)
        if _affected(label):
            seen.setdefault(conv, []).append(str(label))
            if conv != METRIC_CONVENTION:
                problems.append(
                    f"{label}: produced under {conv!r}, current is "
                    f"{METRIC_CONVENTION!r} - regenerate it"
                )

    if len(seen) > 1:
        joined = "; ".join(f"{c}: {sorted(v)}" for c, v in sorted(seen.items()))
        problems.append(f"figure mixes metric conventions across schemes - {joined}")

    if problems and strict:
        raise RuntimeError(
            "Refusing to render from inconsistent simulation caches.\n  "
            + "\n  ".join(problems)
            + "\n\nCaches predating the 2026-08-10 factor-2 branch-metric fix cannot be "
            "rescaled analytically; regenerate them (see nsm/provenance.py)."
        )
    return problems
