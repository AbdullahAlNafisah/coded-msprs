"""
EXIT chart utilities — Extrinsic Information Transfer analysis.

References
----------
S. ten Brink, "Convergence behavior of iteratively decoded parallel concatenated
codes," IEEE Trans. Commun., vol. 49, no. 10, pp. 1727–1737, Oct. 2001.

J. Hagenauer, "The EXIT chart — introduction to extrinsic information transfer
in iterative processing," in Proc. EUSIPCO, Vienna, 2004.
"""

import numpy as np
from numba import njit, prange
from nsm.codec.conv import decode as _decode
from nsm._exit_jit import _i_inv, _gen_llrs_jit, _mi_avg_jit, _mi_hist_jit


# ── J-function inverse: σ_A = J^{-1}(I_A) ────────────────────────────────────

def i_inverse(ia):
    """
    Map mutual information I_A ∈ [0, 1) to Gaussian LLR std σ_A = J^{-1}(I_A).

    Piecewise polynomial approximation of J^{-1} from ten Brink (2001),
    eq. (13)–(14). Accepts scalar or NumPy array.

    The J-function J(σ) = I(X; L) for L|X=x ~ N(x·σ²/2, σ²), i.e. the mutual
    information of an AWGN-consistent soft channel. J^{-1} is used to set σ_A
    such that the generated a-priori LLRs carry exactly I_A bits of information.
    """
    ia = np.asarray(ia, dtype=np.float64)
    scalar = ia.ndim == 0
    ia = np.atleast_1d(ia)

    low  = 1.09542 * ia**2 + 0.214217 * ia + 2.33737 * ia**0.5
    with np.errstate(divide="ignore", invalid="ignore"):
        # Clamp argument to avoid log(0) when ia→1; result is +inf there (σ→∞)
        high = -0.706692 * np.log(np.maximum(0.386013 * (1.0 - ia), 1e-300)) + 1.75017 * ia

    out = np.where(ia <= 0.3646, low, high)
    return float(out[0]) if scalar else out


# ── A-priori LLR generation ──────────────────────────────────────────────────

def gen_llrs(true_bits, sigma_a):
    """
    Generate Gaussian a-priori LLRs carrying I_A = J(σ_A) bits of information.

    Model: L | b ~ N((2b−1)·σ_A²/2, σ_A²)                 [ten Brink eq. (6)–(8)]

    This is the AWGN-consistent LLR distribution: the mean is (2b−1)·σ²/2
    and the variance is σ², so that a matched receiver would see exactly
    L_A = log P(b=1|y) / P(b=0|y) with the Gaussian parameterisation of J.

    Parameters
    ----------
    true_bits : int array, shape (N,), values in {0, 1}
    sigma_a   : float — σ_A = J^{-1}(I_A)

    Returns
    -------
    float array, shape (N,)
    """
    mean = (2 * true_bits - 1) * (sigma_a**2 / 2)
    return mean + sigma_a * np.random.randn(len(true_bits))


# ── Mutual information estimators ────────────────────────────────────────────

def mutual_info_averaging(llr_values, true_bits):
    """
    Estimate I(X; L) via the J-function averaging method.

    Computes  I_E = 1 − E_{b,L}[ log2(1 + exp(−(2b−1)·L)) ]
                                                              [ten Brink eq. (14)]

    The factor (2b−1) sign-normalises so that positive LLRs always correspond
    to the correct bit, making the estimator valid for mixed 0/1 bit sequences
    and for any LLR sign convention (provided true_bits are supplied).

    This is algebraically equivalent to ten Brink eq. (11), conditioned on
    X=+1 with σ_E-distributed samples, when evaluated on sign-flipped 0-bit
    samples.  It matches the ``compute_mutual_information_avg`` function in
    the reference MATLAB EXIT.m implementation.

    Parameters
    ----------
    llr_values : float array, shape (N,)
    true_bits  : int array, shape (N,), values in {0, 1}

    Returns
    -------
    float in [0, 1]
    """
    llr_values = np.asarray(llr_values, dtype=np.float64)
    true_bits  = np.asarray(true_bits,  dtype=np.int32)
    # Sign-normalise: signed_L > 0 means "correct" regardless of the bit value
    signed = (2 * true_bits - 1) * llr_values
    with np.errstate(over="ignore", invalid="ignore"):
        h = np.log2(1.0 + np.exp(-signed))   # binary entropy term, clipped at 0 for large signed
    valid = np.isfinite(h)
    if not np.any(valid):
        return 0.0
    return float(np.clip(1.0 - np.mean(h[valid]), 0.0, 1.0))


def mutual_info_histogram(llr_values, true_bits):
    """
    Estimate I(X; L) via the class-conditional histogram method.

    Estimates the densities p(l|x=0) and p(l|x=1) from histograms with
    Scott's bandwidth rule, then computes:

        I_E = ½ Σ_k [ p(l_k|0)·log2(2p(l_k|0)/(p(l_k|0)+p(l_k|1)))
                     + p(l_k|1)·log2(2p(l_k|1)/(p(l_k|0)+p(l_k|1))) ]
                                                              [ten Brink eq. (15)]

    Sign-convention-agnostic: gives the same result for L and −L, since it
    only measures the separation between the two class-conditional densities.

    Parameters
    ----------
    llr_values : float array, shape (N,)
    true_bits  : int array, shape (N,), values in {0, 1}

    Returns
    -------
    float in [0, 1]
    """
    llr_values = np.asarray(llr_values, dtype=np.float64)
    true_bits  = np.asarray(true_bits,  dtype=np.int32)

    n0 = int(np.sum(true_bits == 0))
    n1 = int(np.sum(true_bits == 1))
    if n0 == 0 or n1 == 0:
        return 0.0

    valid = np.isfinite(llr_values)
    llr_v = llr_values[valid]
    if len(llr_v) < 2:
        return 0.0

    std0 = np.std(llr_values[true_bits == 0])
    std1 = np.std(llr_values[true_bits == 1])
    std_pool = 0.5 * (std0 + std1)
    if std_pool < 1e-10:
        # Degenerate: zero-spread distributions — MI is 1 if means differ, 0 if equal
        mean0 = float(np.mean(llr_values[true_bits == 0]))
        mean1 = float(np.mean(llr_values[true_bits == 1]))
        return 0.0 if np.isclose(mean0, mean1, atol=1e-10) else 1.0

    # Scott's bandwidth: h = 3.49·σ·N^{-1/3}
    bw = 3.49 * std_pool * len(llr_v) ** (-1.0 / 3.0)
    if bw == 0.0:
        return 0.0

    lo   = np.floor(llr_v.min() / bw) - 1
    hi   = np.ceil(llr_v.max()  / bw) + 2
    bins = np.arange(lo, hi) * bw

    h0, _ = np.histogram(llr_values[true_bits == 0], bins=bins)
    h1, _ = np.histogram(llr_values[true_bits == 1], bins=bins)

    p0 = h0 / n0   # normalised class-conditional histograms
    p1 = h1 / n1
    pt = p0 + p1   # mixture (unnormalised by 2)

    ie = 0.0
    for b in range(len(p0)):
        if pt[b] > 0:
            if p0[b] > 0:
                ie += 0.5 * p0[b] * np.log2(2.0 * p0[b] / pt[b])
            if p1[b] > 0:
                ie += 0.5 * p1[b] * np.log2(2.0 * p1[b] / pt[b])

    return float(np.clip(ie, 0.0, 1.0))



# ── Parallel EXIT curve kernel ───────────────────────────────────────────────

@njit(parallel=True, cache=True)
def exit_curve(IA, coded, code_length, source_length, n_outputs, memory,
               total_states, next_states, outputs, N_TRIALS):
    """
    Compute the decoder EXIT curve, parallelised across IA points via prange.

    Each of the len(IA) points runs in its own thread.  All temporary arrays
    inside _decode (Gamma, Alpha, Beta, Lambda, Sigma) and the LLR buffers
    are allocated per-thread, so there are no data races.  Numba uses
    thread-local RNG state, so _gen_llrs_jit is also thread-safe.

    Parameters
    ----------
    IA           : float64 array (n_pts,)
    coded        : int array — encoded codeword, shared read-only
    code_length  : int
    source_length: int
    n_outputs    : int
    memory       : int
    total_states : int
    next_states  : int array (states, 2)
    outputs      : int array (states, 2)
    N_TRIALS     : int

    Returns
    -------
    ie_avg  : float64 array (n_pts,)
    ie_hist : float64 array (n_pts,)
    ia_meas : float64 array (n_pts,)  — measured IA (first-trial sanity check)
    """
    n_pts = len(IA)
    ie_avg  = np.zeros(n_pts)
    ie_hist = np.zeros(n_pts)
    ia_meas = np.zeros(n_pts)

    for k in prange(n_pts):
        sigma_a  = _i_inv(IA[k])
        sum_avg  = 0.0
        sum_hist = 0.0

        for t in range(N_TRIALS):
            l_a = _gen_llrs_jit(coded, sigma_a)

            if t == 0:
                ia_meas[k] = _mi_avg_jit(l_a, coded)

            l_ext, _ = _decode(l_a, code_length, source_length, n_outputs,
                               memory, total_states, next_states, outputs)

            sum_avg  += _mi_avg_jit(l_ext, coded)
            sum_hist += _mi_hist_jit(l_ext, coded)

        ie_avg[k]  = sum_avg  / N_TRIALS
        ie_hist[k] = sum_hist / N_TRIALS

    return ie_avg, ie_hist, ia_meas
