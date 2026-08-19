"""
MS-PRS (Multi-Stream Partial Response Signaling) modulation.

filter_type accepts "unbalanced" / "unconstraint" (same file) or
"balanced" / "constraint_equal_energy" (same file).
"""

import json
import os
import numpy as np
from numba import njit, prange

from nsm._math import llr_to_log_prob, log_prob_to_llr, log_add
from nsm.codec.conv import encode as conv_encode, decode as conv_decode
from nsm.interleaver import interleave, deinterleave
from nsm.channel.awgn import transmit
from nsm._exit_jit import _i_inv, _gen_llrs_jit, _mi_avg_jit, _mi_hist_jit, _mi_mag_jit

_FILTERS_DIR = os.path.join(os.path.dirname(__file__), "filters")

_FILTER_ALIAS = {
    "unconstraint": "unbalanced",
    "constraint_equal_energy": "balanced",
}


def load_coefficients(L0, filter_type="unbalanced"):
    """
    Load and energy-normalise MSPRS filter coefficients.

    Returns h0 (ndarray) and h1 (float).
    """
    ft = _FILTER_ALIAS.get(filter_type, filter_type)
    filename = os.path.join(_FILTERS_DIR, f"{ft}.json")
    with open(filename) as f:
        raw = json.load(f)[str(L0)]

    h0 = np.array(raw["h0"])
    h1_vec = np.array(raw["h1"])
    eta = raw["eta"]

    # Defensive: the JSON convention stores unit-norm h0 and h1. Trip loudly
    # if a bad entry slips in (e.g. the balanced L0=9, L0=10 anomalies
    # surfaced in Paper/FILTER_TABLE_VERIFICATION.md §7). atol=1e-6 is
    # wide enough that 7-digit JSON truncation noise passes; only genuine
    # transcription / re-export bugs trip the assertion.
    n0_sq = float(np.sum(h0 ** 2))
    assert np.isclose(n0_sq, 1.0, atol=1e-6), (
        f"Filter h0 for L0={L0} in {filename} is not unit-norm: "
        f"||h0||^2 = {n0_sq:.6f}"
    )
    n1_sq = float(np.sum(h1_vec ** 2))
    assert np.isclose(n1_sq, 1.0, atol=1e-6), (
        f"Filter h1 for L0={L0} in {filename} is not unit-norm: "
        f"||h1||^2 = {n1_sq:.6f}"
    )

    h1 = h1_vec[0]
    h0 = np.sqrt(eta / 5.0) * h0
    h1 = np.sqrt((5.0 - eta) / 5.0) * h1

    if round(np.linalg.norm(h0) ** 2 + h1 ** 2) != 1.0:
        raise ValueError(f"Energy normalisation failed for L0={L0}, filter_type={filter_type}")

    return h0, h1


def precompute(L0, input_length, filter_type="unbalanced"):
    """
    Pre-compute all trellis quantities needed by modulate / demodulate.

    Returns a dict with keys: h0, h1, branch_labels, memory, total_states,
    next_states, branch_indices, modulation_length.
    """
    h0, h1 = load_coefficients(L0, filter_type)
 
    S = np.arange(2 ** L0)
    R = np.zeros((len(S), L0))
    for i, s in enumerate(S):
        for j in range(L0):
            R[i, L0 - 1 - j] = s % 2
            s //= 2

    branch_labels = np.hstack([
        (2 * R - 1) @ h0 - h1,
        (2 * R - 1) @ h0 + h1,
    ])

    memory = L0 - 1
    M = 2 ** memory
    next_states = np.empty((M, 2), dtype=np.uint16)
    branch_indices = np.empty((M, 2), dtype=np.uint16)
    for i in range(M):
        for j in range(2):
            next_states[i, j] = (i >> 1) + j * 2 ** (L0 - 2)
            branch_indices[i, j] = i + j * M

    return {
        "h0": h0,
        "h1": h1,
        "branch_labels": branch_labels,
        "memory": memory,
        "total_states": M,
        "next_states": next_states,
        "branch_indices": branch_indices,
        "modulation_length": int(np.ceil((input_length + L0 - 1) / 2)),
    }


@njit(cache=True)
def modulate(data, L0, h0, h1):
    """MSPRS partial-response modulator."""
    modSig0 = 2 * data[0:-L0+1:2] - 1
    modSig0 = np.convolve(modSig0, h0)

    modSig1 = data[1:-L0+1:2]
    modSig1 = h1 * (2 * np.hstack((modSig1, data[-L0+1:])) - 1)

    if L0 % 2 == 0:
        modSig1 = np.hstack((modSig1, -h1 * np.ones(1)))
        sig = modSig1 + modSig0
        for i in range(L0):
            modSig0[i] += np.sum(-1.0 * h0[i+1:])
            sig[i] = modSig1[i] + modSig0[i]
            modSig0[-L0+1+i] += np.sum(-1.0 * h0[:i+1])
            sig[-L0+1+i] = modSig1[-L0+1+i] + modSig0[-L0+1+i]
    else:
        sig = modSig1 + modSig0
        for i in range(L0 - 1):
            modSig0[i] += np.sum(-1.0 * h0[i+1:])
            sig[i] = modSig1[i] + modSig0[i]
            modSig0[-L0+1+i] += np.sum(-1.0 * h0[:i+1])
            sig[-L0+1+i] = modSig1[-L0+1+i] + modSig0[-L0+1+i]

    return sig


def modulate_iq(bits_i, bits_q, L0, h0, h1):
    """Independent MS-PRS on I and Q, returned as a complex baseband stream.

    Both streams use the same filter (`h0`, `h1`); ``bits_i`` drives the real
    axis, ``bits_q`` the imaginary axis.  The two MS-PRS outputs share length
    only when ``len(bits_i) == len(bits_q)`` (truncated to the shorter on
    mismatch).  The resulting symbol-level constellation is the Cartesian
    product of the 1-D MS-PRS amplitude set with itself.
    """
    s_i = modulate(np.asarray(bits_i, dtype=np.int32), L0, h0, h1)
    s_q = modulate(np.asarray(bits_q, dtype=np.int32), L0, h0, h1)
    n   = min(len(s_i), len(s_q))
    return (s_i[:n] + 1j * s_q[:n]).astype(np.complex64)


@njit(cache=True)
def demodulate(
    received,
    noise_variance,
    signal_length,
    branch_labels,
    a_priori_llr,
    llr_length,
    memory,
    total_states,
    next_states,
    branch_indices,
):
    """Soft BCJR demodulator for MSPRS. Returns extrinsic LLRs, clipped to ±50.

    ``noise_variance`` is sigma^2 = N0/2 per real dimension. The branch metric
    is the Gaussian log-likelihood -(Y - s)^2 / (2 sigma^2). The factor 2 in
    that denominator is load-bearing: without it every extrinsic LLR comes out
    exactly twice too large, making the soft output overconfident, degrading
    turbo convergence and depressing the EXIT transfer. The check that catches
    a regression here is the I_A = 1 EXIT endpoint, which must equal the closed
    form J(4 Eb/N0).

    This docstring used to claim uncoded BER was untouched by the error. That
    is wrong, and was corrected on 2026-08-16: scaling every branch metric is
    not a monotone map through the log-sum-exp marginalisation, so the hard
    decisions move as well (4.6 % on L0=3 balanced at 1-4 dB). Invariance would
    hold only for a memoryless demodulator.
    """
    M = total_states
    L = signal_length
    a_priori_llr = np.clip(a_priori_llr, -50.0, 50.0)

    # Trellis-static reverse-edge tables, built once per call. prev_states[ns]
    # holds the 2 source states leading to ns, and is_state_from_one[ns] is
    # True iff ns is reachable via a j=1 (bit=1) branch. The ascending ps
    # order is load-bearing: the alpha recursion log_adds along it and must
    # match the original ps=0..M-1 scan order to stay bit-exact.
    prev_states = np.zeros((M, 2), dtype=np.uint16)
    _prev_count = np.zeros(M, dtype=np.uint16)
    for ps in range(M):
        for j in range(2):
            ns_ = next_states[ps, j]
            slot = _prev_count[ns_]
            prev_states[ns_, slot] = ps
            _prev_count[ns_] = slot + 1

    is_state_from_one = np.zeros(M, dtype=np.bool_)
    for ps in range(M):
        is_state_from_one[next_states[ps, 1]] = True

    Gamma0 = np.full((L, M, M), -np.inf)
    Gamma1 = np.full((L, M, M), -np.inf)
    Alpha  = np.full((L + 1, M), -np.inf)
    Beta   = np.full((L + 1, M), -np.inf)
    Lambda = np.full((L + 1, M), -np.inf)
    Sigma0 = np.full((L, M, M), -np.inf)
    Sigma1 = np.full((L, M, M), -np.inf)

    valid = np.zeros(M)
    llr_out = np.zeros(llr_length)

    k = 0
    valid[0] = 1
    for t in range(L):
        prev = np.where(valid == 1)[0]
        valid.fill(0)
        Y = received[t]
        if t < L - memory:
            for i in prev:
                ns = next_states[i]
                valid[ns] = 1
                for j in range(2):
                    bi = branch_indices[i, j]
                    p0 = llr_to_log_prob(-a_priori_llr[k]) if j == 0 else llr_to_log_prob(a_priori_llr[k])
                    p1 = llr_to_log_prob(-a_priori_llr[k + 1])
                    lch = -(Y - branch_labels[bi]) ** 2 / (2.0 * noise_variance)
                    Gamma0[t, i, ns[j]] = lch + p0 + p1
                    p1 = llr_to_log_prob(a_priori_llr[k + 1])
                    lch = -(Y - branch_labels[bi + 2 * M]) ** 2 / (2.0 * noise_variance)
                    Gamma1[t, i, ns[j]] = lch + p0 + p1
            k += 2
        else:
            if t == L - 1 and (memory + 1) % 2 == 0:
                for i in prev:
                    Gamma0[t, i, next_states[i, 0]] = 0.0
            else:
                for i in prev:
                    ns0 = next_states[i, 0]
                    valid[ns0] = 1
                    bi = branch_indices[i, 0]
                    lch = -(Y - branch_labels[bi]) ** 2 / (2.0 * noise_variance)
                    Gamma0[t, i, ns0] = lch + llr_to_log_prob(-a_priori_llr[k])
                    lch = -(Y - branch_labels[bi + 2 * M]) ** 2 / (2.0 * noise_variance)
                    Gamma1[t, i, ns0] = lch + llr_to_log_prob(a_priori_llr[k])
                k += 1

    # α-forward. Iterates the 2 valid incoming edges per ns (from
    # prev_states) instead of the M×M (ps, ns) grid with sentinel filter.
    # log_add(x, -inf) = x exactly (nsm._math.log_add) so any temporally
    # invalid edge (γ = -inf during warm-up) contributes nothing — order
    # of finite log_adds is preserved because prev_states[ns] holds the
    # two ps values in ascending order, matching the original ps=0..M-1
    # scan with sentinel filtering.
    Alpha[0, 0] = 0.0
    for t in range(L):
        norm = -np.inf
        for ns in range(M):
            Alpha[t + 1, ns] = -np.inf
            for kk in range(2):
                ps = prev_states[ns, kk]
                Alpha[t + 1, ns] = log_add(
                    Alpha[t + 1, ns],
                    Alpha[t, ps] + log_add(Gamma0[t, ps, ns], Gamma1[t, ps, ns])
                )
            norm = log_add(Alpha[t + 1, ns], norm)
        for ns in range(M):
            Alpha[t + 1, ns] -= norm

    # β-backward. Symmetric to α: iterates the 2 valid outgoing edges per
    # ps via the existing next_states table (no new structure needed).
    Beta[-1, 0] = 0.0
    for t in range(L, 0, -1):
        norm = -np.inf
        for ps in range(M):
            Beta[t - 1, ps] = -np.inf
            for j in range(2):
                ns = next_states[ps, j]
                Beta[t - 1, ps] = log_add(
                    Beta[t - 1, ps],
                    Beta[t, ns] + log_add(Gamma0[t - 1, ps, ns], Gamma1[t - 1, ps, ns])
                )
            norm = log_add(Beta[t - 1, ps], norm)
        for ps in range(M):
            Beta[t - 1, ps] -= norm

    valid.fill(0)
    valid[0] = 1
    for t in range(L + 1):
        Lambda[t] = Alpha[t] + Beta[t]
        if t < L:
            prev = np.where(valid == 1)[0]
            valid.fill(0)
            for i in prev:
                if t < L - memory:
                    ns = next_states[i]
                    valid[ns] = 1
                    for j in range(2):
                        Sigma0[t, i, ns[j]] = Alpha[t, i] + Gamma0[t, i, ns[j]] + Beta[t + 1, ns[j]]
                        Sigma1[t, i, ns[j]] = Alpha[t, i] + Gamma1[t, i, ns[j]] + Beta[t + 1, ns[j]]
                else:
                    ns0 = next_states[i, 0]
                    valid[ns0] = 1
                    Sigma0[t, i, ns0] = Alpha[t, i] + Gamma0[t, i, ns0] + Beta[t + 1, ns0]
                    Sigma1[t, i, ns0] = Alpha[t, i] + Gamma1[t, i, ns0] + Beta[t + 1, ns0]

    # LLR extraction. Two changes vs the pre-#1 form:
    #   - `ps in next_states[:, 1]` (O(M) Python `in` scan, hit 440k×/packet
    #     in profile) → `is_state_from_one[ps]`, a precomputed O(1) lookup.
    #   - Sigma0/1 marginalisation `for ns in range(M)` (M×M with sentinel
    #     -inf entries) → `for j in range(2): ns = next_states[ps, j]`,
    #     visiting only the 2M trellis-valid edges. Same ascending-ns
    #     visit order as the original, so log_add accumulation is
    #     bit-exact (-inf entries from the original visit contribute
    #     nothing under log_add).
    k = 0
    for t in range(L):
        lp1_b1 = -np.inf
        lp0_b1 = -np.inf
        if t < L - memory:
            lp1_b0 = -np.inf
            lp0_b0 = -np.inf
            for ps in range(M):
                if is_state_from_one[ps]:
                    lp1_b0 = log_add(lp1_b0, Lambda[t + 1, ps])
                else:
                    lp0_b0 = log_add(lp0_b0, Lambda[t + 1, ps])
                for j in range(2):
                    ns = next_states[ps, j]
                    lp1_b1 = log_add(lp1_b1, Sigma1[t, ps, ns])
                    lp0_b1 = log_add(lp0_b1, Sigma0[t, ps, ns])
            llr_out[k]     = log_prob_to_llr(lp1_b0, lp0_b0)
            llr_out[k + 1] = log_prob_to_llr(lp1_b1, lp0_b1)
            k += 2
        else:
            if t == L - 1 and (memory + 1) % 2 == 0:
                continue
            for ps in range(M):
                for j in range(2):
                    ns = next_states[ps, j]
                    lp1_b1 = log_add(lp1_b1, Sigma1[t, ps, ns])
                    lp0_b1 = log_add(lp0_b1, Sigma0[t, ps, ns])
            llr_out[k] = log_prob_to_llr(lp1_b1, lp0_b1)
            k += 1

    return np.clip(llr_out - a_priori_llr, -50.0, 50.0)


@njit(cache=True)
def coded_ber(
    src_bits,
    code_length,
    code_polynomials,
    n_outputs,
    code_memory,
    code_total_states,
    code_next_states,
    code_outputs,
    mod_length,
    mod_L0,
    mod_h0,
    mod_h1,
    mod_branch_labels,
    mod_memory,
    mod_total_states,
    mod_next_states,
    mod_branch_indices,
    interleaver_indices,
    max_iters,
    noise_var,
    noise_std,
):
    """
    Simulate one packet through the coded MSPRS turbo loop.

    Returns error_counts : ndarray (max_iters+1,) — bit errors per turbo iteration.
    """
    source = np.random.randint(0, 2, src_bits)
    coded  = conv_encode(source, code_length, code_polynomials)

    tx = modulate(interleave(coded, interleaver_indices), mod_L0, mod_h0, mod_h1)
    rx = transmit(tx, noise_std)

    # Stay in float64 throughout the turbo loop: demodulate returns float64
    # (channel LLR), conv_decode returns float32 (extrinsic), and interleave
    # preserves dtype. Casting both endpoints to float64 keeps numba happy
    # by avoiding unification of float32 / float64 along the SSA chain.
    mod_llr_in   = np.zeros(code_length, dtype=np.float64)
    error_counts = np.zeros(max_iters + 1, dtype=np.uint32)

    for it in range(max_iters + 1):
        mod_llr_ext = demodulate(
            rx, noise_var, mod_length,
            mod_branch_labels, mod_llr_in, code_length,
            mod_memory, mod_total_states, mod_next_states, mod_branch_indices,
        )

        code_llr_in = deinterleave(mod_llr_ext, interleaver_indices)

        code_llr_ext, estimated = conv_decode(
            code_llr_in, code_length, src_bits,
            n_outputs, code_memory, code_total_states, code_next_states, code_outputs,
        )

        mod_llr_in = interleave(code_llr_ext.astype(np.float64), interleaver_indices)

        error_counts[it] = np.sum(source != estimated)

    return error_counts


@njit(cache=True)
def uncoded_ber(
    src_bits,
    mod_L0,
    mod_h0,
    mod_h1,
    mod_branch_labels,
    mod_memory,
    mod_total_states,
    mod_next_states,
    mod_branch_indices,
    noise_var,
    noise_std,
):
    """Simulate one packet through uncoded MSPRS. Returns bit error count."""
    mod_length = int(np.ceil((src_bits + mod_L0 - 1) / 2))
    source = np.random.randint(0, 2, src_bits)
    rx = transmit(modulate(source, mod_L0, mod_h0, mod_h1), noise_std)
    llr_ext = demodulate(
        rx, noise_var, mod_length,
        mod_branch_labels, np.zeros(src_bits, dtype=np.float32), src_bits,
        mod_memory, mod_total_states, mod_next_states, mod_branch_indices,
    )
    return np.sum(source != (llr_ext > 0).astype(np.int32))


@njit(parallel=True, cache=True)
def exit_curve_nsm(IA, bits, rx, noise_var, mod_length, branch_labels,
                   source_length, memory, total_states, next_states,
                   branch_indices, N_TRIALS):
    """
    Compute the NSM modem EXIT curve, parallelised across IA points.

    For each IA point, N_TRIALS independent a-priori LLR draws are fed into
    the BCJR demodulator; the three IE estimates are averaged.  The received
    signal rx and the true source bits are shared read-only across threads.

    Parameters
    ----------
    IA            : float64 array (n_pts,)
    bits          : int array (source_length,) — true transmitted bits
    rx            : float64 array — received AWGN signal
    noise_var     : float — σ² of the AWGN channel
    mod_length    : int — number of modulated symbols
    branch_labels : float64 array — MSPRS trellis branch metrics
    source_length : int — number of source bits
    memory        : int
    total_states  : int
    next_states   : int array (states, 2)
    branch_indices: int array (states, 2)
    N_TRIALS      : int

    Returns
    -------
    ie_avg  : float64 array (n_pts,)
    ie_hist : float64 array (n_pts,)
    ie_mag  : float64 array (n_pts,)
    ia_meas : float64 array (n_pts,)
    """
    n_pts = len(IA)
    ie_avg  = np.zeros(n_pts)
    ie_hist = np.zeros(n_pts)
    ie_mag  = np.zeros(n_pts)
    ia_meas = np.zeros(n_pts)

    for k in prange(n_pts):
        sigma_a  = _i_inv(IA[k])
        sum_avg  = 0.0
        sum_hist = 0.0
        sum_mag  = 0.0

        for t in range(N_TRIALS):
            l_a = _gen_llrs_jit(bits, sigma_a)

            if t == 0:
                ia_meas[k] = _mi_avg_jit(l_a, bits)

            # demodulate() returns L_E directly (no sign flip needed)
            l_ext = demodulate(rx, noise_var, mod_length, branch_labels,
                               l_a, source_length, memory, total_states,
                               next_states, branch_indices)

            sum_avg  += _mi_avg_jit(l_ext, bits)
            sum_hist += _mi_hist_jit(l_ext, bits)
            sum_mag  += _mi_mag_jit(l_ext)

        ie_avg[k]  = sum_avg  / N_TRIALS
        ie_hist[k] = sum_hist / N_TRIALS
        ie_mag[k]  = sum_mag  / N_TRIALS

    return ie_avg, ie_hist, ie_mag, ia_meas
