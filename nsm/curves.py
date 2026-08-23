"""Reading simulated curves off disk.

A *simulated curve* is the bit error rate of one *scheme* across increasing
Eb/N0, stored as one JSON record per point under ``results/ber/``.
See ``CONTEXT.md`` for the vocabulary.

This module owns everything a caller would otherwise re-derive: locating the
records, tolerating six vintages of the record shape, checking the *metric
convention*, applying the *reliability floor*, recomputing the confidence
interval, and rejecting curves that are visibly broken. Callers get arrays and
a convention; they do not get a schema to interpret.

    from nsm.curves import load_curve
    curve = load_curve("nsm_L3_balanced_conv_K3_7iters")
    if curve is not None:
        ax.plot(curve.eb_no_db, curve.ber)

Why this exists: before it, five readers each re-derived glob, parse, verify,
sort and filter, and disagreed. Two used different reliability floors, three
names for the confidence interval were in circulation, and the provenance gate
was blind to 12 of 30 directories because it keyed on a substring of a label
the caller built.

Scope is deliberately simulated curves only. A measured curve is a different
domain object with an operating point and an error-vector-magnitude, stored one
file per curve; it is not this module's business.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from nsm.bounds import clopper_pearson
from nsm.provenance import LEGACY_CONVENTION, convention_of

__all__ = ["RELIABILITY_FLOOR", "SimulatedCurve", "load_curve", "assert_one_convention"]

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "results" / "ber"

#: Minimum observed error events for a point to be plotted or quoted. Below it
#: a point is Poisson noise; at zero errors it bounds the BER rather than
#: measuring it, and drawing it would put a fake plateau on the axis floor.
RELIABILITY_FLOOR = 30

#: A curve is rejected if the BER rises by more than this factor between
#: adjacent points, but only where both points are above _RELIABLE_BER. Below
#: that the curve is dominated by single-error granularity and small rises mean
#: nothing.
_MONO_TOL = 1.5
_RELIABLE_BER = 1e-5


@dataclass(frozen=True)
class SimulatedCurve:
    """One scheme's simulated BER against Eb/N0, already gated and filtered.

    ``eb_no_db``, ``ber``, ``ci_low`` and ``ci_high`` are equal-length arrays in
    ascending Eb/N0. The confidence interval is always recomputed from the error
    and bit counts, never read from the record: the stored fields go by three
    different names across vintages, and only 16 of 414 records carry one.

    ``convention`` is the metric convention the points were produced under.
    Curves with different conventions must not share a figure; pass them through
    :func:`assert_one_convention` to enforce that.
    """

    scheme: str
    eb_no_db: np.ndarray
    ber: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray
    #: error events and bits each point rests on; the BER is their ratio
    error_events: np.ndarray
    bit_count: np.ndarray
    #: cumulative error count after each turbo iteration, per point, where the
    #: record carries it. ``None`` for the vintages that predate it.
    ers_per_iter: list | None
    convention: str
    n_points_dropped: int

    def __len__(self) -> int:
        return len(self.eb_no_db)


def _records(scheme: str) -> list[dict]:
    out = []
    for path in sorted((CACHE_DIR / scheme).glob("snr_*.json")):
        try:
            rec = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if "eb_no_db" in rec and "ber" in rec:
            out.append(rec)
    return out


def _convention(records: list[dict]) -> str:
    """The single convention these records share, or raise if they disagree.

    Unlike the old gate this looks at every record and does not consult the
    scheme name, so a directory cannot escape the check by being called
    something the gate does not recognise.
    """
    seen = {convention_of(rec.get("config")) for rec in records}
    if len(seen) > 1:
        raise ValueError(
            f"metric conventions disagree within one curve: {sorted(seen)}. "
            "Regenerate the directory; points from two conventions are not comparable."
        )
    return seen.pop() if seen else LEGACY_CONVENTION


def load_curve(scheme: str) -> SimulatedCurve | None:
    """Load one scheme's simulated curve, or ``None`` if it cannot be trusted.

    Returns ``None``, having printed why, when the cache is missing, when no
    point survives the reliability floor, when any BER exceeds 0.5, or when the
    curve is non-monotone in Eb/N0 within the reliable region. Raises
    ``ValueError`` if the directory mixes metric conventions.
    """
    records = _records(scheme)
    if not records:
        print(f"  skip {scheme}  (no cache)")
        return None

    convention = _convention(records)
    records.sort(key=lambda r: r["eb_no_db"])

    eb = np.array([r["eb_no_db"] for r in records], dtype=float)
    ber = np.array([r["ber"] for r in records], dtype=float)
    errs = np.array([r.get("ers_cnt", r.get("n_errs", 10**9)) for r in records], dtype=float)
    bits = np.array([r.get("bits_cnt", r.get("n_bits", 0)) for r in records], dtype=float)

    per_iter = [r.get("ers_per_iter") for r in records]

    keep = np.isfinite(ber) & (ber > 0.0) & (errs >= RELIABILITY_FLOOR)
    dropped = int((~keep).sum())
    eb, ber, errs, bits = eb[keep], ber[keep], errs[keep], bits[keep]
    per_iter = [v for v, k in zip(per_iter, keep) if k]
    if any(v is None for v in per_iter):
        per_iter = None

    if len(eb) == 0:
        print(f"  skip {scheme}  (no measured nonzero-BER points)")
        return None
    if np.any(ber > 0.5):
        print(f"  skip {scheme}  (BER > 0.5 - looks broken)")
        return None
    rise = (ber[1:] > ber[:-1] * _MONO_TOL) & (ber[1:] > _RELIABLE_BER) & (ber[:-1] > _RELIABLE_BER)
    if np.any(rise):
        print(f"  skip {scheme}  (BER non-monotone in Eb/N0 - looks broken)")
        return None

    if convention == LEGACY_CONVENTION:
        print(f"  note {scheme}  (legacy metric convention; cannot share a figure "
              f"with current-convention curves)")

    low, high = clopper_pearson(errs, bits)
    return SimulatedCurve(scheme, eb, ber, low, high, errs, bits, per_iter,
                          convention, dropped)


def assert_one_convention(curves) -> str:
    """Raise unless every curve shares one metric convention; return it.

    This is the invariant the provenance stamp exists to protect. Call it before
    drawing several curves on one axis.
    """
    present = {c.convention for c in curves if c is not None}
    if len(present) > 1:
        raise ValueError(
            "refusing to mix metric conventions in one figure: "
            + ", ".join(
                f"{c.scheme}={c.convention}" for c in curves if c is not None
            )
        )
    return present.pop() if present else LEGACY_CONVENTION
