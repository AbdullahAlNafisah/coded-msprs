import numpy as np
from numba import njit, prange
from nsm._exit_jit import _i_inv, _gen_llrs_jit, _mi_avg_jit, _mi_hist_jit
from scipy.special import erfc

# Gray mapping:  b0 b1  →  level
#                 0  0  → -3
#                 0  1  → -1
#                 1  1  →  1   (gray)
#                 1  0  →  3   (natural: 1 1 → 1)

def ber_theory(eb_no_db, gray):
    """Theoretical BER vs Eb/No."""
    M = 4
    r = 10 ** (np.asarray(eb_no_db) / 10)   # linear Eb/No
    if gray:
        # 4-ASK with Gray coding
        ber = ((M - 1) / (M * np.log2(M))) * erfc(
            np.sqrt((3 * np.log2(M) / (M**2 - 1)) * r)
        )
    else:
        # 4-ASK without Gray coding
        ber = 0.5 * erfc( 
            np.sqrt((3 * np.log2(M) / (M**2 - 1)) * r)
        )
    return ber
    

@njit(cache=True)
def modulate(bits, avg_bit_energy, gray=True):
    """Map pairs of bits to 4-ASK levels, normalised by √Eb."""
    bits2 = np.asarray(bits).reshape(-1, 2)
    n = len(bits2)
    syms = np.empty(n)
    for i in range(n):
        b0, b1 = bits2[i, 0], bits2[i, 1]
        if gray:
            if b0 == 0 and b1 == 0:   syms[i] = -3.0
            elif b0 == 0 and b1 == 1: syms[i] = -1.0
            elif b0 == 1 and b1 == 1: syms[i] =  1.0
            else:                      syms[i] =  3.0
        else:
            if b0 == 0 and b1 == 0:   syms[i] = -3.0
            elif b0 == 0 and b1 == 1: syms[i] = -1.0
            elif b0 == 1 and b1 == 0: syms[i] =  1.0
            else:                      syms[i] =  3.0
    return syms * np.sqrt(2 * avg_bit_energy / 5)


@njit(cache=True)
def demodulate(samples, noise_std, avg_bit_energy, a_priori_llr=None, gray=True):
    """4-ASK MAP soft demodulator.

    Returns the **extrinsic** LLR in standard convention ``L = ln P(b=1)/P(b=0)``,
    i.e. the channel-derived information *minus* the a-priori contribution
    that was supplied. With ``a_priori_llr=None`` this reduces to the
    plain channel LLR (extrinsic = posterior, since L_a = 0).

    Sign conventions
    ----------------
    The optional ``a_priori_llr`` argument follows the function-internal
    convention ``la = ln P(b=0)/P(b=1) = -L_a``. Callers in standard
    convention should pass ``-L_a``. The return is in standard convention
    (``L_ext > 0`` ⇒ ``b = 1`` more likely).

    Why extrinsic and not posterior
    -------------------------------
    Returning the extrinsic (and clipping it instead of the posterior) is
    essential for iterative turbo equalization. With confident a-priori,
    the posterior magnitude grows roughly like |channel-LLR| + |L_a|, so
    a ±50 clip on the posterior would saturate at high SNR / late
    iteration and the caller's ``L_ext = L_post − L_a`` would collapse to
    zero, destroying the channel information. The extrinsic itself is
    bounded by the channel-LLR scale, so clipping it at ±50 only kicks in
    at extreme SNR and never causes an iteration-to-iteration collapse.
    """
    y        = np.asarray(samples).ravel()
    n_sym    = len(y)
    n_bits   = 2 * n_sym
    var      = noise_std * noise_std
    inv2var  = 1.0 / (2.0 * var)

    # Normalised constellation levels (same order for both mappings)
    # index 0→-3 (b0=0,b1=0), 1→-1 (b0=0,b1=1),
    #       2→+1 (b0=1,b1=1 Gray | b0=1,b1=0 Natural),
    #       3→+3 (b0=1,b1=0 Gray | b0=1,b1=1 Natural)
    s = np.array([-3.0, -1.0, 1.0, 3.0]) * np.sqrt(2 * avg_bit_energy / 5)

    # A-priori LLRs (convention: ln P(b=0)/P(b=1))
    la = np.zeros(n_bits)
    if a_priori_llr is not None:
        t = np.asarray(a_priori_llr).ravel()
        m = min(t.size, n_bits)
        la[:m] = t[:m]
    # Clip a-priori at ±50 — this is OK because the a-priori is later
    # subtracted from the posterior to form the extrinsic, and we want
    # the bound on `la` to match the bound the caller saw.
    for i in range(n_bits):
        if   la[i] >  50.0: la[i] =  50.0
        elif la[i] < -50.0: la[i] = -50.0

    llr = np.empty(n_bits)

    for i in range(n_sym):
        yi       = y[i]
        la0, la1 = la[2 * i], la[2 * i + 1]

        # Channel branch metrics
        m00 = -(yi - s[0]) ** 2 * inv2var   # b0=0, b1=0  →  -3
        m01 = -(yi - s[1]) ** 2 * inv2var   # b0=0, b1=1  →  -1
        m10 = -(yi - s[2]) ** 2 * inv2var   # Gray:(b0=1,b1=1) / Nat:(b0=1,b1=0)  →  +1
        m11 = -(yi - s[3]) ** 2 * inv2var   # Gray:(b0=1,b1=0) / Nat:(b0=1,b1=1)  →  +3

        # Add a-priori contributions (la = ln P(b=0)/P(b=1)).
        # The two b0=0 levels carry the same labels under either mapping;
        # only the b0=1 pair swaps its b1 label.
        m00 += +0.5 * la0 + 0.5 * la1       # (b0=0, b1=0)
        m01 += +0.5 * la0 - 0.5 * la1       # (b0=0, b1=1)
        if gray:
            m10 += -0.5 * la0 - 0.5 * la1   # (b0=1, b1=1)
            m11 += -0.5 * la0 + 0.5 * la1   # (b0=1, b1=0)
        else:
            m10 += -0.5 * la0 + 0.5 * la1   # (b0=1, b1=0)
            m11 += -0.5 * la0 - 0.5 * la1   # (b0=1, b1=1)

        # Numerical stability: subtract max
        amax = max(m00, m01, m10, m11)
        m00 -= amax;  m01 -= amax;  m10 -= amax;  m11 -= amax

        # Posterior LLR (standard convention L = ln P(b=1)/P(b=0)).
        # b0 groups the two high levels against the two low ones either way,
        # so only b1's grouping depends on the mapping.
        v0 = np.logaddexp(m10, m11) - np.logaddexp(m00, m01)
        if gray:
            # b1=1: m01(b0=0,b1=1), m10(b0=1,b1=1)
            # b1=0: m00(b0=0,b1=0), m11(b0=1,b1=0)
            v1 = np.logaddexp(m01, m10) - np.logaddexp(m00, m11)
        else:
            # b1=1: m01(b0=0,b1=1), m11(b0=1,b1=1)
            # b1=0: m00(b0=0,b1=0), m10(b0=1,b1=0)
            v1 = np.logaddexp(m01, m11) - np.logaddexp(m00, m10)

        # Convert to extrinsic and clip there.  In standard convention
        # the a-priori is L_a = -la, so L_ext = L_post - L_a = v + la.
        # When `a_priori_llr=None`, la = 0 and L_ext = L_post (unchanged).
        e0 = v0 + la0
        e1 = v1 + la1
        llr[2 * i]     = max(-50.0, min(50.0, e0))
        llr[2 * i + 1] = max(-50.0, min(50.0, e1))

    return llr

# ── Parallel EXIT curve kernel for 4-ASK modem ───────────────────────────────

@njit(parallel=True, cache=True)
def exit_curve_ask4(IA, bits, rx, noise_std, avg_bit_energy, N_TRIALS, gray):
    """
    Compute the 4-ASK modem EXIT curve, parallelised across IA points.

    The 4-ASK MAP demodulator incorporates a-priori LLRs in a non-linear way
    (the a-priori resolves ambiguities between constellation points), so I_E
    genuinely increases with I_A. Gray and Natural labellings produce different
    EXIT shapes — select via the ``gray`` flag.

    Sign convention: ``demodulate`` accepts ``la = ln P(b=0)/P(b=1) = -L_a``
    and now returns the extrinsic LLR directly (= L_post - L_a) in
    standard ``ln P(b=1)/P(b=0)`` convention. We pass ``-l_a`` and use
    the return value as the extrinsic — no manual subtraction.

    Parameters
    ----------
    IA            : float64 array (n_pts,)
    bits          : int array (n_bits,) — true transmitted bits (0/1)
    rx            : float64 array — received AWGN samples
    noise_std     : float — σ of the AWGN channel
    avg_bit_energy: float — Eb
    N_TRIALS      : int
    gray          : bool — True for Gray labelling, False for Natural

    Returns
    -------
    ie_avg  : float64 array (n_pts,)
    ie_hist : float64 array (n_pts,)
    ia_meas : float64 array (n_pts,)
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
            l_a = _gen_llrs_jit(bits, sigma_a)

            if t == 0:
                ia_meas[k] = _mi_avg_jit(l_a, bits)

            # Pass -l_a (function expects "ln P(b=0)/P(b=1)").
            # demodulate now returns the extrinsic L_post - L_a directly.
            l_ext = demodulate(rx, noise_std, avg_bit_energy, -l_a, gray)

            sum_avg  += _mi_avg_jit(l_ext, bits)
            sum_hist += _mi_hist_jit(l_ext, bits)

        ie_avg[k]  = sum_avg  / N_TRIALS
        ie_hist[k] = sum_hist / N_TRIALS

    return ie_avg, ie_hist, ia_meas
