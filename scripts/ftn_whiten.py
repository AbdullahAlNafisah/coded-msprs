"""Forney whitened-matched-filter taps for the binary FTN channel.

Answers reviewer comment 137:

    "not clear! how the noise be considered as white after matched filtering at
     higher than the Nyquist rate?"

He is right that it is not white. Sampling the matched-filter output at
1/(tau*T) > 1/T leaves noise correlated with autocorrelation sigma^2 * g[k],
where g is the same pulse autocorrelation that forms the ISI. The received
sequence is

    r[k] = sum_j g[j] x[k-j] + n_c[k],     E{n_c[k] n_c[k-m]} = sigma^2 g[m].

`nsm/modem/ftn.py` and `Modem_FTN` both treat n_c as white, which is the
standard tutorial simplification and is disclosed in Section IV-B, but it is a
simplification and he is entitled to a number for what it costs.

Forney's fix: factor the folded spectrum G(z) = F(z) F*(1/z*) and filter by
1/F*(1/z*). What comes out is a causal minimum-phase channel F(z) driven by
genuinely WHITE noise of the same variance, so the BCJR model becomes exact:

    y[k] = sum_j f[j] x[k-j] + n_w[k],     E{n_w[k] n_w[k-m]} = sigma^2 delta[m].

RESULT: that fix does not apply to this channel. Rooting the z-transform of g
shows FOUR of its six roots sitting exactly on the unit circle, and the folded
spectrum G(e^jw) is zero over about 21 % of the band. The reason is structural,
not numerical: an RRC with roll-off 0.3 occupies (1+alpha)/2T = 0.65/T, but
tau = 0.5 samples at 2/T, so the sequence is oversampled with respect to its own
bandwidth and the folded spectrum has a null BAND rather than isolated nulls.
A spectral factor with roots on the unit circle has no stable causal inverse,
so no whitening filter exists.

A second, separate defect shows up in the same check: the truncated g is not a
valid autocorrelation at all. Its spectrum reaches -0.132, and a genuine
autocorrelation has a non-negative spectrum everywhere. Truncating to L_isi taps
destroys positive semi-definiteness.

So the honest answer to 137 is that he is right the noise is not white, and also
that the textbook remedy is unavailable here. The options are to sample at the
pulse's own Nyquist rate rather than the symbol rate, or to keep the white-noise
approximation and cite it, which is the second thing his comment asks for.

Usage:

    python scripts/ftn_whiten.py                 # print f and the check
    python scripts/ftn_whiten.py --L-isi 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nsm.modem.ftn import _rrc_taps


def autocorr(tau: float, rolloff: float, L_isi: int, sps: int = 32, span: int = 12):
    """Two-sided pulse autocorrelation at lags k*tau*T, k = -(L-1)..(L-1)."""
    p    = _rrc_taps(rolloff, sps, span)
    step = int(round(sps * tau))
    g    = np.zeros(2 * L_isi - 1)
    for k in range(-(L_isi - 1), L_isi):
        lag = k * step
        n   = len(p) - abs(lag)
        if n <= 0:
            continue
        g[k + L_isi - 1] = float(np.dot(p[:n], p[lag:lag + n]) if lag >= 0
                                 else np.dot(p[-lag:-lag + n], p[:n]))
    return g / g[L_isi - 1]          # normalise so g[0] = 1


def spectral_factor(g: np.ndarray, tol: float = 1e-4):
    """Minimum-phase f with f (*) reverse(f) == g.

    The outer taps of g are the raised-cosine Nyquist zeros, order 1e-5. Left
    in place they make the polynomial numerically degenerate: np.roots sees a
    vanishing leading coefficient and returns spurious roots at huge magnitude,
    which silently drops the factor's degree. Trim them first.
    """
    nz = np.nonzero(np.abs(g) > tol)[0]
    gt = g[nz[0]:nz[-1] + 1]                 # trimmed, still symmetric

    roots  = np.roots(gt[::-1])              # highest power first
    inside = [r for r in roots if abs(r) < 1.0 - 1e-9]
    f = np.real(np.poly(inside)) if inside else np.array([1.0])

    # f (*) reverse(f) has peak sum(f^2); g is normalised to g[0] = 1.
    return f / np.sqrt(np.sum(f ** 2)), gt


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--rolloff", type=float, default=0.3)
    ap.add_argument("--L-isi", type=int, default=5)
    args = ap.parse_args()

    g = autocorr(args.tau, args.rolloff, args.L_isi)
    L = args.L_isi
    print(f"two-sided autocorrelation g (tau={args.tau}, alpha={args.rolloff}):")
    print("  " + "  ".join(f"{v:+.5f}" for v in g))

    f, gt = spectral_factor(g)
    roots = np.roots(gt[::-1])
    on    = int(np.sum(np.abs(np.abs(roots) - 1.0) < 1e-6))
    w     = np.linspace(0.0, np.pi, 4001)
    c     = (len(gt) - 1) // 2
    G     = np.real(sum(gt[i] * np.exp(-1j * w * (i - c)) for i in range(len(gt))))
    null  = float(np.mean(G < 1e-3))

    print(f"\nroots of G(z): {len(roots)} total, {on} ON the unit circle")
    print(f"folded spectrum: min {G.min():+.4f}, zero over {null * 100:.1f} % of the band")
    if on > 0 or null > 0.05:
        print("\n=> No stable whitening filter exists for this channel.")
        print("   RRC alpha=0.3 occupies 0.65/T but tau=0.5 samples at 2/T, so the")
        print("   sequence is oversampled relative to its own bandwidth and the")
        print("   folded spectrum has a null band. Roots on the unit circle have no")
        print("   stable causal inverse, so Forney whitening is degenerate here.")
    if G.min() < 0:
        print(f"\n=> The TRUNCATED g is not a valid autocorrelation: its spectrum")
        print(f"   reaches {G.min():+.4f}, and a true autocorrelation is non-negative")
        print("   everywhere. Truncation to L_isi taps breaks positive semi-definiteness.")
    print(f"\nminimum-phase factor from the {len(roots) - on} off-circle roots "
          f"({len(f)} taps, NOT a valid whitener):")
    print("  " + "  ".join(f"{v:+.5f}" for v in f))

    # Verification: f convolved with its own reverse must reproduce the
    # trimmed g. Both are symmetric of length 2*len(f)-1, so they align.
    rec = np.convolve(f, f[::-1])
    err = np.max(np.abs(rec - gt)) if len(rec) == len(gt) else float("nan")
    print(f"\ncheck  max |f (*) reverse(f) - g| = {err:.2e}"
          f"   (len f={len(f)}, rec={len(rec)}, g trimmed={len(gt)})")

    cur = np.array([g[L - 1 + k] for k in range(L)])
    cur = cur / np.sqrt(np.sum(cur ** 2))
    print(f"\ncurrent model uses the one-sided autocorrelation, renormalised:")
    print("  " + "  ".join(f"{v:+.5f}" for v in cur))
    print("\nConclusion for comment 137: the noise is indeed not white, and the")
    print("standard remedy does not apply to this channel. Keep the white-noise")
    print("model, state the reason, and cite it as he asks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
