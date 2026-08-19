"""INVALID AS AN FTN MODEL. Retained only as an idealised ISI reference.

    MEASURED 2026-08-19 (scripts/ftn_msed.py): this module does NOT simulate
    faster-than-Nyquist signaling. `precompute` takes the ONE-SIDED pulse
    autocorrelation g[0..L_isi-1], normalises it to ||h||^2 = 1, and drives it
    with WHITE noise. Two errors compound:

      1. the one-sided autocorrelation is not the Forney channel response (that
         would be the minimum-phase spectral factor f with f*reverse(f) = g);
      2. the unit-energy normalisation pins the isolated-error squared distance
         to ||2h||^2 = 4, exactly the ISI-free 2-ASK value, INDEPENDENTLY OF
         TAU. The channel cannot lose to 2-ASK on an isolated error however
         hard it packs.

    The true tau=0.5 MSED is 2.03 at Es=1, a 2.95 dB deficit. The consequence
    was visible in the caches: coded FTN crossed 2e-5 at 5.61 dB against coded
    2-ASK at 5.59 dB while carrying twice the rate, which is impossible.

    scripts/ftn_whiten.py shows why no repair is available in this form: four
    of the six roots of G(z) lie on the unit circle and the folded spectrum is
    null over 21% of the band, so no stable causal whitener exists. A correct
    receiver uses the Ungerboeck observation model with noise covariance
    sigma^2 * G (Ungerboeck 1974; Li et al. TCOM 2018; Yang et al. JSAC 2026,
    which simulates exactly tau = 2/3 and 1/2).

    The paper no longer plots a simulated FTN curve; it cites Anderson's
    published results instead. Do not reinstate this module as an FTN
    benchmark without replacing the observation model. `results/ber/
    ftn_tau0p5_conv_K3_7iters` is honest data of the wrong channel.

Binary Faster-Than-Nyquist (FTN) signaling at packing factor τ.

A minimal reference implementation used as the §IV.B / Fig. 5 binary-FTN
benchmark. This is NOT the multicarrier MFTN of Table II, which is cited
from Anderson and not simulated here; do not conflate the two.
The construction follows the standard discrete-time FTN model
(see e.g. Anderson 2013, §3): bits are pulse-shaped at symbol rate
1/(τT), the matched filter sampled at the same rate produces a discrete
ISI channel whose taps are the pulse autocorrelation at lags k·τT:

    g[k] = ∫ p(t) p*(t − k τ T) dt,    k = -L_isi … +L_isi

For RRC roll-off α=0.3 and τ=0.5 the autocorrelation is a raised cosine
sampled every τT=T/2, so g[0]=1, |g[±1]| ≈ 0.62, and the *even* taps fall
on the raised-cosine Nyquist zeros (g[±2]=g[±4]=…=0); the next nonzero tap
is |g[±3]| ≈ 0.17. This gives a manageable trellis of 2^(L_isi−1) states
for L_isi=4–5.

Energy convention: ``modulate`` emits one unit-energy real symbol per bit
(the ISI taps are normalised to ‖g‖²=1), so the per-bit symbol energy is
1.0. Callers must pass that value as ``avg_bit_energy`` to the AWGN setup —
``precompute`` returns it as ``avg_bit_energy`` so the Eb/N0→σ² conversion
cannot silently disagree. (Using 0.5 here — the rate-2 MS-PRS per-bit
energy, where one symbol carries *two* bits — would halve σ² and hand FTN a
spurious 3 dB, letting it beat the ISI-free BPSK bound.)

Two simplifying assumptions, made deliberately to keep the BCJR
self-contained and comparable in complexity to the MS-PRS BCJR:

  1. The noise at the matched-filter output is treated as white with
     variance σ² (the receiver doesn't apply a whitening filter). This is
     mildly optimistic but standard in tutorial-level FTN comparisons; a
     Forney-style whitened matched filter would tighten the curve by a
     few tenths of a dB at high SNR. This is disclosed in §IV-B of the paper.
  2. The trellis ISI is truncated to L_isi taps (one-sided). For τ=0.5
     RRC α=0.3 this captures > 99 % of the tap energy at L_isi=5.

Both assumptions are disclosed in the paper's §IV-B FTN-benchmark description.
"""
from __future__ import annotations

import numpy as np
from numba import njit

from nsm._math import llr_to_log_prob, log_prob_to_llr, log_add


def _rrc_taps(rolloff: float, sps: int, span: int) -> np.ndarray:
    """Root-raised-cosine pulse, length 2*span*sps+1, peak-normalised so g[0]=1
    after symbol-rate sampling (we re-normalise after autocorrelation below)."""
    T = 1.0
    t = (np.arange(-span * sps, span * sps + 1)) / sps
    # Closed-form RRC, with the standard t=0 and t=±T/(4β) limits handled.
    a = rolloff
    h = np.zeros_like(t)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-12:
            h[i] = (1.0 + a * (4.0 / np.pi - 1.0)) / T
        elif abs(abs(ti) - T / (4 * a)) < 1e-12:
            h[i] = (a / (T * np.sqrt(2))) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * a))
                + (1 - 2 / np.pi) * np.cos(np.pi / (4 * a))
            )
        else:
            num = np.sin(np.pi * ti * (1 - a) / T) + 4 * a * ti / T * np.cos(np.pi * ti * (1 + a) / T)
            den = np.pi * ti * (1 - (4 * a * ti / T) ** 2) / T
            h[i] = num / den
    h /= np.sqrt(np.sum(h ** 2))
    return h


def precompute(input_length: int, *, tau: float = 0.5, rolloff: float = 0.3,
               L_isi: int = 5, sps: int = 32, span: int = 12) -> dict:
    """Build the FTN ISI channel and trellis used by ``modulate`` / ``demodulate``.

    Parameters
    ----------
    input_length : number of binary {0,1} input bits per packet.
    tau          : FTN time-packing factor (0 < τ ≤ 1); τ=0.5 ≡ 2× Nyquist.
    rolloff      : RRC roll-off α.
    L_isi        : one-sided ISI truncation; trellis has 2^(L_isi-1) states.
    sps, span    : oversampling + span used to *derive* g[k] (not Tx-rate).

    Returns
    -------
    Dict with: ``isi`` (length L_isi, normalised so ||isi||²=1),
    ``avg_bit_energy`` (= ||isi||² = 1.0, the per-bit symbol energy to pass to
    ``awgn.setup``), ``branch_labels`` (size 2·2^(L_isi-1) ≡ {ps,bit→symbol
    value}), ``next_states`` table, ``memory`` (= L_isi-1), ``total_states``,
    ``modulation_length``.
    """
    p = _rrc_taps(rolloff, sps, span)
    # Autocorrelation evaluated at symbol-rate lags k·τT.
    g = np.zeros(2 * L_isi - 1)
    step = int(round(sps * tau))
    for k in range(-(L_isi - 1), L_isi):
        lag = k * step
        n_overlap = len(p) - abs(lag)
        if n_overlap <= 0:
            g[k + L_isi - 1] = 0.0
        elif lag >= 0:
            g[k + L_isi - 1] = float(np.dot(p[:n_overlap], p[lag:lag + n_overlap]))
        else:
            g[k + L_isi - 1] = float(np.dot(p[-lag:-lag + n_overlap], p[:n_overlap]))
    # The one-sided ISI tap vector used as the discrete channel impulse
    # response. Normalise to unit energy so the BER curve compares like-for-like
    # against MS-PRS (which uses ||h0||²+h1²=1).
    h = g[L_isi - 1:]  # taps at lags 0..L_isi-1
    h = h / np.sqrt(np.sum(h ** 2))

    memory = L_isi - 1
    M = 2 ** memory

    # Branch labels: for each (ps, bit) → emitted real symbol s = Σ h[k]·x[t-k].
    # Encode ps as the L_isi-1 prior bits (MSB = oldest); bit is the newest.
    branch_labels = np.zeros(2 * M, dtype=np.float64)
    next_states = np.zeros((M, 2), dtype=np.uint16)
    for ps in range(M):
        # decode ps → (b_{t-1}, b_{t-2}, … b_{t-(L_isi-1)})  with MSB oldest
        bits_prev = np.zeros(L_isi - 1, dtype=np.int32)
        v = ps
        for j in range(L_isi - 1):
            bits_prev[L_isi - 2 - j] = (v & 1)
            v >>= 1
        for bit in (0, 1):
            # symbol value: h[0]·x_t + h[1]·x_{t-1} + … + h[L_isi-1]·x_{t-(L_isi-1)}
            x_t = 2 * bit - 1
            s = h[0] * x_t
            for k in range(1, L_isi):
                # x at lag k uses bits_prev[L_isi-1-k]
                xk = 2 * bits_prev[L_isi - 1 - k] - 1
                s += h[k] * xk
            branch_labels[ps + bit * M] = s
            # next state shifts in the new bit on the LSB side; oldest bit dropped
            ns = ((ps << 1) | bit) & (M - 1)
            next_states[ps, bit] = ns

    return {
        "isi": h.astype(np.float64),
        # Per-bit symbol energy (‖g‖²=1 → 1.0). Feed this to awgn.setup() as
        # avg_bit_energy so the Eb/N0→σ² conversion matches the emitted energy.
        "avg_bit_energy": float(np.sum(h ** 2)),
        "tau": float(tau),
        "rolloff": float(rolloff),
        "L_isi": int(L_isi),
        "memory": int(memory),
        "total_states": int(M),
        "branch_labels": branch_labels,
        "next_states": next_states,
        "modulation_length": int(input_length + L_isi - 1),
    }


@njit(cache=True)
def modulate(data, params_isi):
    """Convolve {0,1} bits → {-1,+1} symbols → ISI-filtered baseband samples.

    Returns a real-valued array of length ``len(data) + L_isi - 1``.
    """
    x = 2.0 * data.astype(np.float64) - 1.0
    return np.convolve(x, params_isi)


@njit(cache=True)
def _demodulate_jit(received, noise_variance, a_priori_llr,
                     branch_labels, next_states, memory, total_states,
                     llr_length):
    """Standard binary-input BCJR for the truncated FTN ISI channel.

    State = last ``memory`` bits, in MSB-oldest convention (matches
    ``next_states`` built by ``precompute``).
    """
    M = total_states
    L = llr_length + memory  # number of observed samples

    a_priori_llr = np.clip(a_priori_llr, -50.0, 50.0)

    # Reverse-edge table: for each ns find the two (ps, bit) pairs leading to it.
    prev_states = np.zeros((M, 2), dtype=np.uint16)
    prev_bits = np.zeros((M, 2), dtype=np.uint8)
    _cnt = np.zeros(M, dtype=np.uint16)
    for ps in range(M):
        for j in range(2):
            ns = next_states[ps, j]
            slot = _cnt[ns]
            prev_states[ns, slot] = ps
            prev_bits[ns, slot] = j
            _cnt[ns] = slot + 1

    Gamma = np.full((L, M, 2), -np.inf)
    Alpha = np.full((L + 1, M), -np.inf)
    Beta = np.full((L + 1, M), -np.inf)

    for t in range(L):
        Y = received[t]
        if t < llr_length:
            pa = a_priori_llr[t]
        else:
            pa = 0.0
        lp1 = llr_to_log_prob(pa)
        lp0 = llr_to_log_prob(-pa)
        for ps in range(M):
            for j in range(2):
                ns = next_states[ps, j]
                bi = ps + j * M
                # -(Y-s)^2/(2 sigma^2): the factor 2 matters, see nsm/modem/msprs.py demodulate
                lch = -(Y - branch_labels[bi]) ** 2 / (2.0 * noise_variance)
                Gamma[t, ps, j] = lch + (lp1 if j == 1 else lp0)

    # Force start state = 0 (the encoder begins with all-zero memory)
    Alpha[0, 0] = 0.0
    for t in range(L):
        norm = -np.inf
        for ns in range(M):
            for kk in range(2):
                ps = prev_states[ns, kk]
                j = prev_bits[ns, kk]
                Alpha[t + 1, ns] = log_add(Alpha[t + 1, ns],
                                            Alpha[t, ps] + Gamma[t, ps, j])
            norm = log_add(Alpha[t + 1, ns], norm)
        for ns in range(M):
            Alpha[t + 1, ns] -= norm

    # Soft termination: all final states equally likely (no zero-padding here)
    for ns in range(M):
        Beta[L, ns] = 0.0
    for t in range(L, 0, -1):
        norm = -np.inf
        for ps in range(M):
            for j in range(2):
                ns = next_states[ps, j]
                Beta[t - 1, ps] = log_add(Beta[t - 1, ps],
                                           Beta[t, ns] + Gamma[t - 1, ps, j])
            norm = log_add(Beta[t - 1, ps], norm)
        for ps in range(M):
            Beta[t - 1, ps] -= norm

    llr_out = np.zeros(llr_length)
    for t in range(llr_length):
        lp1_tot = -np.inf
        lp0_tot = -np.inf
        for ps in range(M):
            for j in range(2):
                ns = next_states[ps, j]
                s = Alpha[t, ps] + Gamma[t, ps, j] + Beta[t + 1, ns]
                if j == 1:
                    lp1_tot = log_add(lp1_tot, s)
                else:
                    lp0_tot = log_add(lp0_tot, s)
        llr_out[t] = log_prob_to_llr(lp1_tot, lp0_tot)

    return np.clip(llr_out - a_priori_llr, -50.0, 50.0)


def demodulate(received, noise_variance, a_priori_llr, params):
    """High-level BCJR entry point — strips ``params`` for the JIT kernel."""
    return _demodulate_jit(
        received.astype(np.float64),
        float(noise_variance),
        a_priori_llr.astype(np.float64),
        params["branch_labels"],
        params["next_states"],
        int(params["memory"]),
        int(params["total_states"]),
        len(a_priori_llr),
    )
