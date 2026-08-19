"""Multiplicity of minimum-distance error events, per L0 and energy family.

Answers reviewer comment 179 directly. He observes that coded BER does not
improve clearly as L0 grows, and proposes a mechanism:

    "probably the multiplicity of the error events with lowest MSED increases
     somehow, even though the MSED increases"

That is a checkable claim, not a matter of opinion. The union bound is

    BER ~= (k_e / 2) * erfc( sqrt( d2_min * Rc * m * Eb/N0 ) )

so d2_min sets the exponent and k_e the prefactor. If k_e grows with L0 fast
enough it can cancel a rising d2_min over the plotted SNR range, which is
exactly the effect he describes.

Method. MS-PRS is linear in the bipolar domain,

    s[k] = sum_j h0[j] x0[k-j] + h1 x1[k],    x in {-1,+1},

so the difference between two transmitted sequences depends only on the
difference of their inputs, never on the data itself. Enumerating ERROR
SEQUENCES is therefore exact and far cheaper than enumerating path pairs: each
position carries e0[k], e1[k] in {-2, 0, +2}, and

    d2 = sum_k ( sum_j h0[j] e0[k-j] + h1 e1[k] )^2.

An error event starts with e0 != 0, ends when the last L0-1 values of e0 are
zero again, and is counted only at the SHORTEST length attaining d2_min: the
amplitude alphabet has multiplicities (that is what Fig 3 shows), so zero-cost
divergent steps exist and the raw count over unbounded length is infinite. The
shortest events are also the dominant union-bound term.

An earlier version of this script held stream 1 tied on the argument that a
stream-1 disagreement costs (2*h1)^2 and would always be pruned. That is wrong,
and the balanced family is the counterexample: its stream-0-tied minimum at
L0=4 is 1.804 while the true minimum is 1.268, because the h0 difference at the
same step partly cancels the 2*h1 term. Stream 1 is free here.

Distances are on the Es=1 scale; multiply by 5 for the paper's Table I scale.

Usage:

    python scripts/msed_multiplicity.py
    python scripts/msed_multiplicity.py --max-len 12
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nsm import bounds
from nsm.modem.msprs import load_coefficients, precompute

ERRS = (-2.0, 0.0, 2.0)   # difference of two bipolar symbols


def enumerate_events(h0, h1, d2_target: float, max_len: int, tol: float = 1e-9):
    """(n_events, k_e, length) for the shortest events attaining ``d2_target``.

    ``k_e`` counts differing input bits summed over those events, i.e. the
    union-bound coefficient before averaging.
    """
    L0 = len(h0)
    best = {"len": max_len + 1, "n": 0, "ke": 0}

    def walk(hist, depth, dist, nbits):
        """hist holds the last L0 values of e0, newest first."""
        if dist > d2_target + tol or depth > max_len or depth > best["len"]:
            return
        # Event closes when the FIR memory has flushed: the last L0-1 error
        # values are zero, so no further symbol is disturbed.
        if depth > 0 and all(v == 0.0 for v in hist[:L0 - 1]):
            if abs(dist - d2_target) <= tol:
                if depth < best["len"]:
                    best.update(len=depth, n=1, ke=nbits)
                elif depth == best["len"]:
                    best["n"] += 1
                    best["ke"] += nbits
            return
        for e0 in ERRS:
            nh = (e0,) + hist[:L0 - 1]
            base = sum(h0[j] * nh[j] for j in range(L0))
            for e1 in ERRS:
                nd = dist + (base + h1 * e1) ** 2
                if nd > d2_target + tol:
                    continue
                walk(nh, depth + 1, nd,
                     nbits + (e0 != 0.0) + (e1 != 0.0))

    # Time invariance: every event is a shift of one starting at position 0,
    # so force e0[0] != 0 rather than allowing a zero-cost undiverged prefix.
    zero = (0.0,) * L0
    for e0 in (-2.0, 2.0):
        nh = (e0,) + zero[:L0 - 1]
        base = sum(h0[j] * nh[j] for j in range(L0))
        for e1 in ERRS:
            d = (base + h1 * e1) ** 2
            if d > d2_target + tol:
                continue
            walk(nh, 1, d, 1 + (e1 != 0.0))
    return best["n"], best["ke"], best["len"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-len", type=int, default=12)
    args = ap.parse_args()

    print("Shortest minimum-distance error events (Es = 1 scale)")
    print(f"{'family':<12}{'L0':>3}{'d2_min':>9}{'x5':>9}{'len':>6}"
          f"{'events':>9}{'k_e':>7}", flush=True)
    rows = {}
    for family in ("balanced", "unbalanced"):
        for L0 in (3, 4, 5, 6):
            h0, h1 = load_coefficients(L0, family)
            d2 = bounds.min_squared_distance(precompute(L0, 1000, family))
            n, ke, ln = enumerate_events(h0, float(h1), d2, args.max_len)
            rows[(family, L0)] = (d2, n, ke, ln)
            print(f"{family:<12}{L0:>3}{d2:9.4f}{d2 * 5:9.4f}{ln:6d}"
                  f"{n:9d}{ke:7d}", flush=True)

    print("\nComment 179: does the multiplicity grow with L0?")
    for family in ("balanced", "unbalanced"):
        d2 = [rows[(family, L)][0] for L in (3, 4, 5, 6)]
        n  = [rows[(family, L)][1] for L in (3, 4, 5, 6)]
        gain_db = 10.0 * np.log10(d2[-1] / d2[0])
        print(f"  {family:<11} d2_min {d2[0]:.3f} -> {d2[-1]:.3f} "
              f"({gain_db:+.2f} dB)   events {n[0]} -> {n[-1]} "
              f"(x{n[-1] / max(n[0], 1):.0f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
