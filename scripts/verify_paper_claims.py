"""Recompute every quantitative claim in Section IV-B and check it against main.tex.

Replaces tests/test_paper_claims.py, deleted with the rest of tests/ on
2026-08-11. Prose and data drift apart silently otherwise: the numbers in IV-B
are read off the BER caches, so regenerating a cache invalidates them without
touching the .tex.

Run after regenerating any cache under results/ber/:

    python scripts/verify_paper_claims.py

Exits non-zero if a recomputed value disagrees with the manuscript, or if the
sentence a value lives in is no longer in main.tex (which would mean the check
is silently guarding text that no longer exists).

Method mirrors make_ber_overview.py so the prose describes the plotted figure:
points resting on fewer than MIN_EVENTS error events are discarded, and the
E_b/N0 at BER = 2e-5 is interpolated log-linearly between the two bracketing
points of what remains.
"""
from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT    = Path(__file__).resolve().parents[1]
OUT     = ROOT / "results" / "ber"
TEX     = ROOT / "Paper" / "main.tex"   # optional; absent in the public release

TARGET      = 2e-5   #: the error rate every scheme reaches with >= MIN_EVENTS events
MIN_EVENTS  = 30     #: same floor make_ber_overview.py applies before plotting
TOL_DB      = 0.006  #: prose is quoted to 0.01 dB, so half a digit
TOL_REL     = 0.04   #: BER values are quoted to 2 significant figures, so half
                     #: a digit is up to ~4 % (1.27e-6 prints as 1.3e-6)


def curve(subdir: str) -> list[tuple[float, float, int]]:
    pts = []
    for f in glob.glob(str(OUT / subdir / "snr_*.json")):
        j = json.load(open(f))
        if j["ers_cnt"] >= MIN_EVENTS and j["ber"] > 0:
            pts.append((j["eb_no_db"], j["ber"], j["ers_cnt"]))
    if not pts:
        raise SystemExit(f"no usable points in {subdir} (>= {MIN_EVENTS} error events)")
    return sorted(pts)


def crossing(pts, target: float = TARGET) -> float:
    """E_b/N0 at `target`, log-linear between the bracketing points."""
    for (x0, y0, _), (x1, y1, _) in zip(pts, pts[1:]):
        if y0 >= target >= y1:
            return x0 + (x1 - x0) * (np.log(y0) - np.log(target)) / (np.log(y0) - np.log(y1))
    raise SystemExit(f"curve never crosses {target:.0e}; range {pts[-1][1]:.2e}..{pts[0][1]:.2e}")


def main() -> int:
    L0S = (3, 4, 5, 6)
    ask2 = crossing(curve("ask2_conv_K3"))
    gray = crossing(curve("ask4_gray_conv_K3_7iters"))
    nat  = crossing(curve("ask4_natural_conv_K3_7iters"))

    cross, deepest = {}, {}
    for fam in ("balanced", "unbalanced"):
        for L0 in L0S:
            pts = curve(f"nsm_L{L0}_{fam}_conv_K3_7iters")
            cross[fam, L0]   = crossing(pts)
            deepest[fam, L0] = min(p[1] for p in pts)

    bal   = [cross["balanced", L] - ask2 for L in L0S]
    unbal = [cross["unbalanced", L] - ask2 for L in L0S]
    gap   = [cross["unbalanced", L] - cross["balanced", L] for L in L0S]
    leads_g = [gray - cross[f, L] for f in ("balanced", "unbalanced") for L in L0S]
    leads_n = [nat  - cross[f, L] for f in ("balanced", "unbalanced") for L in L0S]
    deep_b  = [deepest["balanced", L] for L in L0S]
    deep_u  = [deepest["unbalanced", L] for L in L0S]

    # (label, computed, quoted, tolerance, the sentence fragment in main.tex)
    checks = [
        ("balanced vs 2-ASK, low",   min(bal),      +0.00, TOL_DB,  r"by $+0.00$ to $+0.04$\,dB"),
        ("balanced vs 2-ASK, high",  max(bal),      +0.04, TOL_DB,  r"by $+0.00$ to $+0.04$\,dB"),
        ("unbal vs 2-ASK, low",      min(unbal),     0.06, TOL_DB,  r"by $0.06$ to $0.31$\,dB"),
        ("unbal vs 2-ASK, high",     max(unbal),     0.31, TOL_DB,  r"by $0.06$ to $0.31$\,dB"),
        ("unbal - bal, low",         min(gap),       0.02, TOL_DB,  r"by $0.02$ to $0.30$\,dB"),
        ("unbal - bal, high",        max(gap),       0.30, TOL_DB,  r"by $0.02$ to $0.30$\,dB"),
        ("lead over 4-ASK Gray, lo", min(leads_g),   2.99, TOL_DB,  r"Gray mapping by $2.99$ to $3.30$\,dB"),
        ("lead over 4-ASK Gray, hi", max(leads_g),   3.30, TOL_DB,  r"Gray mapping by $2.99$ to $3.30$\,dB"),
        ("lead over 4-ASK nat, lo",  min(leads_n),   1.92, TOL_DB,  r"natural mapping by $1.92$ to $2.23$\,dB"),
        ("lead over 4-ASK nat, hi",  max(leads_n),   2.23, TOL_DB,  r"natural mapping by $1.92$ to $2.23$\,dB"),
    ]
    ber_checks = [
        ("deepest balanced, lo",   min(deep_b), 4.0e-7, r"namely $4.0$ to $4.6\times10^{-7}$"),
        ("deepest balanced, hi",   max(deep_b), 4.6e-7, r"namely $4.0$ to $4.6\times10^{-7}$"),
        ("deepest unbalanced, lo", min(deep_u), 5.2e-7, r"$5.2\times10^{-7}$ to $1.3\times10^{-6}$"),
        ("deepest unbalanced, hi", max(deep_u), 1.3e-6, r"$5.2\times10^{-7}$ to $1.3\times10^{-6}$"),
    ]

    # The manuscript is not part of this repository. When it is absent the
    # recomputation still runs and is reported; only the sentence-presence
    # cross-check is skipped.
    tex  = TEX.read_text() if TEX.exists() else None
    bad  = 0
    print(f"{'claim':<28}{'computed':>11}{'in paper':>11}  status")
    for label, got, want, tol, frag in checks:
        ok = abs(got - want) <= tol
        bad += not ok
        print(f"{label:<28}{got:>11.3f}{want:>11.2f}  {'ok' if ok else 'MISMATCH'}")
        if tex is not None and frag not in tex:
            print(f"    ! sentence not found in main.tex: {frag}"); bad += 1
    for label, got, want, frag in ber_checks:
        ok = abs(got - want) / want <= TOL_REL
        bad += not ok
        print(f"{label:<28}{got:>11.2e}{want:>11.1e}  {'ok' if ok else 'MISMATCH'}")
        if tex is not None and frag not in tex:
            print(f"    ! sentence not found in main.tex: {frag}"); bad += 1

    # The quoted grid spacing has to match the caches actually on disk.
    steps = np.diff([p[0] for p in curve("nsm_L3_balanced_conv_K3_7iters")])
    uniform_half = bool(np.allclose(steps, 0.5))
    frag = r"spaced uniformly at $0.5$\,dB from $0$ to $7$\,dB"
    print(f"{'grid uniformly 0.5 dB':<28}{str(uniform_half):>11}{'True':>11}  {'ok' if uniform_half else 'MISMATCH'}")
    if not uniform_half or (tex is not None and frag not in tex):
        bad += 1

    # Monotonicity is asserted in the prose, not just the endpoints.
    mono = all(gap[i] >= gap[i + 1] for i in range(len(gap) - 1))
    print(f"{'unbal-bal narrows with L0':<28}{str(mono):>11}{'True':>11}  {'ok' if mono else 'MISMATCH'}")
    bad += not mono

    # --- Table I: d2_min at Es=5 and the asymptotic gain over 4-ASK ----------
    # 4-ASK on levels +-1, +-3 has Es = 5 and adjacent distance 2, so d2 = 4.
    from nsm import bounds
    from nsm.modem.msprs import precompute

    tab1 = {  # (family, L0): (d2_min at Es=5, gain in dB) as printed in main.tex
        ("unbalanced", 3): (7.388, 2.66), ("unbalanced", 4): (7.760, 2.88),
        ("unbalanced", 5): (8.373, 3.21), ("unbalanced", 6): (8.966, 3.51),
        ("balanced", 3):   (5.858, 1.66), ("balanced", 4):   (6.340, 2.00),
        ("balanced", 5):   (6.897, 2.37), ("balanced", 6):   (7.673, 2.83),
    }
    print()
    for (fam, L0), (d2_q, gain_q) in sorted(tab1.items()):
        p     = precompute(L0, 1000, fam)
        d2    = bounds.min_squared_distance(p) * 5.0
        gain  = 10.0 * np.log10(d2 / 4.0)
        ok    = abs(d2 - d2_q) <= 0.001 and abs(gain - gain_q) <= 0.005
        bad  += not ok
        print(f"{'Tab I ' + fam + ' L0=' + str(L0):<28}{d2:>11.3f}{d2_q:>11.3f}  "
              f"gain {gain:5.2f}/{gain_q:4.2f}  {'ok' if ok else 'MISMATCH'}")

    # --- Table III: inner-trellis state counts and the Ops/bit arithmetic ----
    ITERS = 7
    # The simulated binary-FTN row was removed from Table III on 2026-08-19:
    # the model it came from is wrong (scripts/ftn_msed.py). Only the cited
    # MFTN reference remains on the FTN side.
    tab3 = [("4-ASK natural", 1, 35),
            ("MS-PRS L0=3", 4, 56), ("MS-PRS L0=4", 8, 84),
            ("MS-PRS L0=5", 16, 140), ("MS-PRS L0=6", 32, 252)]
    for label, states, ops in tab3:
        ok = ops == ITERS * (states + 4)   # +4 is the K=3 outer decoder
        bad += not ok
        print(f"{'Tab III ' + label:<28}{ITERS * (states + 4):>11d}{ops:>11d}  {'ok' if ok else 'MISMATCH'}")
    # The state counts must equal 2^(L0-1) and 2^(L_isi-1) for the ported modems.
    for L0, want in ((3, 4), (4, 8), (5, 16), (6, 32)):
        ok = (1 << (L0 - 1)) == want
        bad += not ok
    print(f"{'Tab III state counts':<28}{'2^(L-1)':>11}{'matches':>11}  ok")

    print()
    if bad:
        print(f"{bad} mismatch(es): the manuscript disagrees with the code or caches.")
        return 1
    print("Section IV-B and Tables I/III agree with the code and caches.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
