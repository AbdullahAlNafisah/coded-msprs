"""
Numba JIT helpers shared across EXIT chart kernels.

Kept in a leaf module (imports only numpy + numba) so that exit.py,
ldpc.py, msprs.py, and ask4.py can all import from here without
creating circular dependencies.
"""

import numpy as np
from numba import njit


@njit(cache=True)
def _i_inv(ia):
    """Scalar J^{-1}(I_A) — Numba-compatible (ten Brink 2001, eq. 13-14)."""
    if ia <= 0.3646:
        return 1.09542 * ia**2 + 0.214217 * ia + 2.33737 * ia**0.5
    arg = 0.386013 * (1.0 - ia)
    if arg < 1e-300:
        arg = 1e-300
    return -0.706692 * np.log(arg) + 1.75017 * ia


@njit(cache=True)
def _gen_llrs_jit(true_bits, sigma_a):
    """Gaussian a-priori LLRs — @njit version of gen_llrs (ten Brink 2001, eq. 6-8)."""
    half_var = sigma_a * sigma_a * 0.5
    mean = (2.0 * true_bits - 1.0) * half_var
    return mean + sigma_a * np.random.randn(len(true_bits))


@njit(cache=True)
def _mi_avg_jit(llr_values, true_bits):
    """Averaging MI estimator — @njit version (ten Brink 2001, eq. 14)."""
    total = 0.0
    count = 0
    for i in range(len(llr_values)):
        s = (2 * true_bits[i] - 1) * llr_values[i]
        h = np.log2(1.0 + np.exp(-s))
        if np.isfinite(h):
            total += h
            count += 1
    if count == 0:
        return 0.0
    ie = 1.0 - total / count
    if ie < 0.0:
        return 0.0
    if ie > 1.0:
        return 1.0
    return ie


@njit(cache=True)
def _mi_hist_jit(llr_values, true_bits):
    """Histogram MI estimator — @njit version with manual histogram (ten Brink 2001, eq. 15).

    np.histogram is not available in Numba nopython mode; the histogram is
    built with a hand-written binning loop using Scott's bandwidth rule.
    """
    n0 = 0
    n1 = 0
    for i in range(len(true_bits)):
        if true_bits[i] == 0:
            n0 += 1
        else:
            n1 += 1
    if n0 == 0 or n1 == 0:
        return 0.0

    s0 = 0.0; s1 = 0.0; q0 = 0.0; q1 = 0.0; cnt = 0
    lo = np.inf; hi = -np.inf
    for i in range(len(llr_values)):
        v = llr_values[i]
        if not np.isfinite(v):
            continue
        cnt += 1
        if v < lo:
            lo = v
        if v > hi:
            hi = v
        if true_bits[i] == 0:
            s0 += v; q0 += v * v
        else:
            s1 += v; q1 += v * v
    if cnt < 2:
        return 0.0

    m0 = s0 / n0; m1 = s1 / n1
    std0 = np.sqrt(max(q0 / n0 - m0 * m0, 0.0))
    std1 = np.sqrt(max(q1 / n1 - m1 * m1, 0.0))
    sp = 0.5 * (std0 + std1)
    if sp < 1e-10:
        return 0.0 if abs(m0 - m1) < 1e-10 else 1.0

    bw = 3.49 * sp * cnt ** (-1.0 / 3.0)
    if bw == 0.0:
        return 0.0

    lo_bin = int(np.floor(lo / bw)) - 1
    hi_bin = int(np.ceil(hi / bw)) + 2
    n_bins = hi_bin - lo_bin

    h0 = np.zeros(n_bins, dtype=np.int64)
    h1 = np.zeros(n_bins, dtype=np.int64)
    for i in range(len(llr_values)):
        if np.isfinite(llr_values[i]):
            idx = int(np.floor(llr_values[i] / bw)) - lo_bin
            if 0 <= idx < n_bins:
                if true_bits[i] == 0:
                    h0[idx] += 1
                else:
                    h1[idx] += 1

    ie = 0.0
    for b in range(n_bins):
        p0b = h0[b] / n0
        p1b = h1[b] / n1
        pt  = p0b + p1b
        if pt > 0.0:
            if p0b > 0.0:
                ie += 0.5 * p0b * np.log2(2.0 * p0b / pt)
            if p1b > 0.0:
                ie += 0.5 * p1b * np.log2(2.0 * p1b / pt)

    if ie < 0.0:
        return 0.0
    if ie > 1.0:
        return 1.0
    return ie


@njit(cache=True)
def _mi_mag_jit(llr_values):
    """Magnitude MI estimator — @njit version (Hagenauer 2004, eq. 9).

    Sign-convention-agnostic; no true bits required.
    """
    total = 0.0
    count = 0
    for i in range(len(llr_values)):
        v = llr_values[i]
        if not np.isfinite(v):
            continue
        absL    = abs(v)
        log1pex = np.logaddexp(0.0, absL)
        sig     = 1.0 / (1.0 + np.exp(-absL))
        h       = (log1pex - absL * sig) / np.log(2.0)
        if np.isfinite(h):
            total += h
            count += 1
    if count == 0:
        return 0.0
    ie = 1.0 - total / count
    if ie < 0.0:
        return 0.0
    if ie > 1.0:
        return 1.0
    return ie
