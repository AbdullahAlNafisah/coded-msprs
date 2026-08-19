import numpy as np
from numba import njit
import commpy.channelcoding as cc
from nsm._math import llr_to_log_prob, log_prob_to_llr, log_add


# ── Trellis precomputation ────────────────────────────────────────────────────

def precompute(coder, packet_length):
    """
    Build trellis tables for the convolutional code.

    Parameters
    ----------
    coder : dict with keys 'K' (constraint length) and 'octal_code' (generator polynomials)
    packet_length : int — number of information bits

    Returns
    -------
    dict with keys: coding_length, rate, memory, polynomials,
                    total_states, next_states, outputs
    """
    memory = np.array([coder["K"] - 1])
    generators = np.array([coder["octal_code"]])
    trellis = cc.Trellis(memory, generators)
    n_outputs = len(generators[0])
    rate = 1/n_outputs

    # Generator polynomials in binary
    width = trellis.total_memory + 1

    def oct2bin(val):
        dec = int(str(val), 8)
        out = np.zeros(width, dtype=bool)
        for i in range(width):
            out[i] = dec % 2
            dec //= 2
        return out

    G = np.stack([oct2bin(generators[0, 0]), oct2bin(generators[0, 1])])

    return {
        "coding_length": (packet_length + trellis.total_memory) * n_outputs,
        "n_outputs": n_outputs,
        "rate": rate,
        "memory": trellis.total_memory,
        "polynomials": G,
        "total_states": trellis.number_states,
        "next_states": trellis.next_state_table,
        "outputs": trellis.output_table,
    }


# ── Encoder ───────────────────────────────────────────────────────────────────

@njit(cache=True)
def encode(data, coding_length, polynomials):
    """
    Rate-1/2 convolutional encoder (binary polynomial convolution).

    Parameters
    ----------
    data         : binary input sequence (0/1), including tail zeros
    coding_length: output length (= (len(data) + memory) * rate)
    polynomials  : (rate, memory+1) bool array from precompute()
    """
    N = len(polynomials)
    rows = coding_length // N
    result = np.zeros((rows, N), dtype=data.dtype)
    for n in range(N):
        conv = np.convolve(data, polynomials[n])
        result[:, n] = conv[:rows] % 2
    out = np.empty(coding_length, dtype=data.dtype)
    for i in range(rows):
        for n in range(N):
            out[N * i + n] = result[i, n]
    return out


# ── BCJR decoder ─────────────────────────────────────────────────────────────

@njit(cache=True)
def decode(llr_in, coding_length, source_length, n_outputs, memory, total_states, next_states, outputs):
    """
    Soft-output BCJR decoder for rate-1/2 convolutional code.

    Sign convention: input/output LLR = ln P(b=1)/P(b=0).

    Returns
    -------
    llr_ext : extrinsic LLRs L_post − L_a (length = coding_length), clipped to ±50
    est     : hard-decision bit estimates (length = source_length)
    """
    llr_in = np.clip(llr_in, -50.0, 50.0)
    L = source_length + memory
    M = total_states

    # Precomputed lookups (trellis-static, invariant across t). Built once at
    # the top rather than recomputed inside the γ-build and soft-decision
    # loops, which together cost 8.2% of decode time when profiled.
    # MSB-first ordering matches the decimal->binary expansion of outputs.
    bits_lookup = np.zeros((M, 2, n_outputs), dtype=np.int32)
    for i in range(M):
        for j in range(2):
            v = outputs[i, j]
            for r in range(n_outputs):
                bits_lookup[i, j, n_outputs - 1 - r] = v % 2
                v //= 2

    # is_one_for_ns[ns] == True iff ns is reachable from some ps via the
    # j=1 (bit=1) branch. Replaces the O(M) `for ps: if next_states[ps,1]
    # == ns` scan in the hard-decision branch (was 4.1% of decode time).
    is_one_for_ns = np.zeros(M, dtype=np.bool_)
    for ps in range(M):
        is_one_for_ns[next_states[ps, 1]] = True

    Gamma  = np.full((L, M, M), -np.inf, dtype=np.float32)
    Alpha  = np.full((L + 1, M), -np.inf, dtype=np.float32)
    Beta   = np.full((L + 1, M), -np.inf, dtype=np.float32)
    Lambda = np.full((L + 1, M), -np.inf, dtype=np.float32)
    Sigma  = np.full((L, M, M), -np.inf, dtype=np.float32)

    valid = np.zeros(M, dtype=np.uint16)
    llr_out = np.zeros(coding_length, dtype=np.float32)
    est = np.zeros(source_length, dtype=np.uint16)

    # ── Gamma ─────────────────────────────────────────────────────────────────
    valid[0] = 1
    for t in range(L):
        mask = valid == 1
        valid.fill(0)
        for i in range(M):
            if not mask[i]:
                continue
            ns = next_states[i]
            valid[ns] = 1
            for j in range(2):
                X = bits_lookup[i, j]
                lch = 0.0
                for r in range(n_outputs):
                    idx = n_outputs * t + r
                    lch += llr_to_log_prob(llr_in[idx] if X[r] == 1 else -llr_in[idx])
                Gamma[t, i, ns[j]] = lch

    # ── Alpha (forward) ───────────────────────────────────────────────────────
    Alpha[0, 0] = 0.0
    for t in range(L):
        norm = -np.inf
        for ns in range(M):
            Alpha[t + 1, ns] = -np.inf
            for ps in range(M):
                g = Gamma[t, ps, ns]
                if g != -np.inf:
                    Alpha[t + 1, ns] = log_add(Alpha[t + 1, ns], Alpha[t, ps] + g)
            norm = log_add(Alpha[t + 1, ns], norm)
        for ns in range(M):
            Alpha[t + 1, ns] -= norm

    # ── Beta (backward) ───────────────────────────────────────────────────────
    Beta[-1, 0] = 0.0
    for t in range(L, 0, -1):
        norm = -np.inf
        for ps in range(M):
            Beta[t - 1, ps] = -np.inf
            for ns in range(M):
                g = Gamma[t - 1, ps, ns]
                if g != -np.inf:
                    Beta[t - 1, ps] = log_add(Beta[t - 1, ps], Beta[t, ns] + g)
            norm = log_add(Beta[t - 1, ps], norm)
        for ps in range(M):
            Beta[t - 1, ps] -= norm

    # ── Lambda & Sigma ────────────────────────────────────────────────────────
    valid.fill(0)
    valid[0] = 1
    for t in range(L + 1):
        Lambda[t] = Alpha[t] + Beta[t]
        if t < L:
            mask = valid == 1
            valid.fill(0)
            for i in range(M):
                if not mask[i]:
                    continue
                if t < L - memory:
                    ns = next_states[i]
                    valid[ns] = 1
                    for j in range(2):
                        Sigma[t, i, ns[j]] = Alpha[t, i] + Gamma[t, i, ns[j]] + Beta[t + 1, ns[j]]
                else:
                    ns0 = next_states[i, 0]
                    valid[ns0] = 1
                    Sigma[t, i, ns0] = Alpha[t, i] + Gamma[t, i, ns0] + Beta[t + 1, ns0]

    # ── Soft & hard decisions ─────────────────────────────────────────────────
    valid.fill(0)
    valid[0] = 1
    lp1 = np.empty(n_outputs, dtype=np.float32)
    lp0 = np.empty(n_outputs, dtype=np.float32)

    for t in range(L):
        mask = valid == 1
        valid.fill(0)
        lp1.fill(-np.inf)
        lp0.fill(-np.inf)

        for i in range(M):
            if not mask[i]:
                continue
            ns = next_states[i]
            valid[ns] = 1
            for j in range(2):
                sv = Sigma[t, i, ns[j]]
                if sv != -np.inf:
                    X = bits_lookup[i, j]
                    for r in range(n_outputs):
                        if X[r] == 1:
                            lp1[r] = log_add(sv, lp1[r])
                        else:
                            lp0[r] = log_add(sv, lp0[r])

        for r in range(n_outputs):
            llr_out[n_outputs * t + r] = log_prob_to_llr(lp1[r], lp0[r])

        if t < source_length:
            lp1_bit = -np.inf
            lp0_bit = -np.inf
            for ns in range(M):
                lv = Lambda[t + 1, ns]
                if lv != -np.inf:
                    if is_one_for_ns[ns]:
                        lp1_bit = log_add(lp1_bit, lv)
                    else:
                        lp0_bit = log_add(lp0_bit, lv)
            est[t] = 1 if lp1_bit > lp0_bit else 0

    llr_ext = np.clip(llr_out - llr_in, -50.0, 50.0)
    return llr_ext, est
