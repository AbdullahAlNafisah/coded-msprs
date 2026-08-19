"""Minimum squared Euclidean distance of the binary FTN benchmark, measured.

Answers the second author's objection (2026-08-19):

    "such an FTN scheme must have a very low MSED. So the bootstrap of the
     iterative algorithm should suffer compared to our schemes ... I doubt the
     fact that you have correctly accounted for the noise variance."

The physical quantity. For s(t) = sqrt(Es) sum_l x_l p(t - l tau T) with p of
unit energy and x_l = +-1, two distinct sequences differing by e_l = x_l - x'_l
in {0, +-2} are separated by

    d^2(e) = Es * sum_k sum_l e_k e_l g[k-l],    g[m] = int p(t) p(t - m tau T) dt

with g[0] = 1. This is a quadratic form in the TRUE pulse autocorrelation, and
it is what governs the high-SNR error probability and the EXIT bootstrap. For
ISI-free BPSK a single error gives d^2 = 4 Es, so 4 is the reference.

What nsm/modem/ftn.py does instead. `precompute` takes the ONE-SIDED
autocorrelation h = g[0..L_isi-1], normalises it to ||h||^2 = 1, and treats it
as a Forney channel response driven by WHITE noise:

    r[k] = sum_j h[j] x[k-j] + n[k],    n white, variance sigma^2

Two things are wrong with that as a model of FTN, and they compound:

  1. The one-sided autocorrelation is not the Forney response. The Forney model
     needs the minimum-phase spectral factor f with f * reverse(f) = g. Using g
     itself squares the channel in the distance metric.
  2. Normalising to ||h||^2 = 1 forces the isolated-error distance to
     ||2h||^2 = 4 ||h||^2 = 4, i.e. exactly the ISI-free BPSK value, by
     construction and independently of tau. The scheme cannot lose against BPSK
     on an isolated error no matter how hard it packs.

scripts/ftn_whiten.py already showed that no valid Forney model exists for this
channel at tau = 0.5: four of the six roots of G(z) sit on the unit circle and
the folded spectrum is null over 21 % of the band, so no stable causal whitener
exists. The white-noise-plus-FIR model is therefore not a mild simplification
here, it is a different channel.

Run from the repo root:
    python scripts/ftn_msed.py
    python scripts/ftn_msed.py --tau 0.7 --depth 12
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nsm.modem.ftn import _rrc_taps, precompute


def true_autocorr(tau: float, rolloff: float, n_lags: int,
                  sps: int = 32, span: int = 24) -> np.ndarray:
    """g[0..n_lags] at lags k*tau*T, normalised to g[0] = 1 (the physical scale)."""
    p = _rrc_taps(rolloff, sps, span)
    step = int(round(sps * tau))
    g = np.zeros(n_lags + 1)
    for k in range(n_lags + 1):
        lag = k * step
        n = len(p) - lag
        g[k] = float(np.dot(p[:n], p[lag:lag + n])) if n > 0 else 0.0
    return g / g[0]


def msed_quadratic(g: np.ndarray, depth: int) -> tuple[float, tuple]:
    """min over error sequences of e^T G e, the TRUE distance.

    e_k in {0, +-2}. The form is invariant under global sign and under a time
    shift, so the search fixes e_0 = +2 and enumerates the remaining depth-1
    positions. That is exhaustive over error EVENTS of length <= depth.
    """
    best, arg = np.inf, None
    for tail in itertools.product((0, 2, -2), repeat=depth - 1):
        e = np.array((2,) + tail, dtype=float)
        d2 = 0.0
        for k in range(len(e)):
            for l in range(len(e)):
                m = abs(k - l)
                if m < len(g):
                    d2 += e[k] * e[l] * g[m]
        if d2 < best:
            best, arg = d2, tuple(int(v) for v in e)
    return best, arg


def msed_fir_white(h: np.ndarray, depth: int) -> tuple[float, tuple]:
    """min over error sequences of ||e * h||^2, the distance the CODE's model sees."""
    best, arg = np.inf, None
    for tail in itertools.product((0, 2, -2), repeat=depth - 1):
        e = np.array((2,) + tail, dtype=float)
        d2 = float(np.sum(np.convolve(e, h) ** 2))
        if d2 < best:
            best, arg = d2, tuple(int(v) for v in e)
    return best, arg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--rolloff", type=float, default=0.3)
    ap.add_argument("--L-isi", type=int, default=5)
    ap.add_argument("--depth", type=int, default=10,
                    help="error-event length searched (3^(depth-1) sequences)")
    args = ap.parse_args()

    print(f"binary FTN, tau = {args.tau}, RRC roll-off = {args.rolloff}, "
          f"L_isi = {args.L_isi}, search depth = {args.depth}")
    print(f"reference: ISI-free BPSK has d^2_min = 4 at E_s = 1\n")

    g = true_autocorr(args.tau, args.rolloff, n_lags=args.L_isi + 3)
    print("TRUE pulse autocorrelation g[k] (g[0] = 1, the physical scale):")
    print("  " + "  ".join(f"{v:+.5f}" for v in g))

    h = precompute(64, tau=args.tau, rolloff=args.rolloff, L_isi=args.L_isi)["isi"]
    print(f"\nwhat precompute() feeds the BCJR, h = g[0..L_isi-1] / ||.||  "
          f"(||h||^2 = {np.sum(h**2):.6f}):")
    print("  " + "  ".join(f"{v:+.5f}" for v in h))

    d2_true, e_true = msed_quadratic(g, args.depth)
    d2_code, e_code = msed_fir_white(h, args.depth)

    print(f"\n{'quantity':<44}{'d^2_min':>10}{'vs BPSK':>12}")
    print(f"{'TRUE FTN (quadratic form in g, E_s=1)':<44}{d2_true:>10.4f}"
          f"{10 * np.log10(d2_true / 4.0):>+9.2f} dB")
    print(f"{'MODEL SIMULATED (FIR h + white noise)':<44}{d2_code:>10.4f}"
          f"{10 * np.log10(d2_code / 4.0):>+9.2f} dB")
    print(f"{'ISI-free BPSK':<44}{4.0:>10.4f}{0.0:>+9.2f} dB")
    print(f"\nminimising error events: true {e_true}\n"
          f"                         model {e_code}")

    gap = 10 * np.log10(d2_code / d2_true)
    print(f"\n=> The simulated model is optimistic by {gap:.2f} dB in MSED.")
    print("   The isolated-error distance of the simulated model is "
          f"||2h||^2 = {4*np.sum(h**2):.4f}, pinned to the BPSK value of 4 by the")
    print("   ||h||^2 = 1 normalisation, independently of tau. That is why the")
    print("   benchmark tracks coded 2-ASK in Fig. 5.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
