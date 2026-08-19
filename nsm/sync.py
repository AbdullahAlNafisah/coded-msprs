import numpy as np


# ── Sequence generation ───────────────────────────────────────────────────────

def generate_preamble(num_syms, seed=None):
    """Random QPSK preamble symbols."""
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, 2 * num_syms)
    bits = bits.reshape(-1, 2)
    syms = (2 * bits[:, 0] - 1 + 1j * (2 * bits[:, 1] - 1)) / np.sqrt(2)
    return syms.astype(np.complex64)


# ── Frame building ────────────────────────────────────────────────────────────

def build_frame(preamble_syms, payload_syms):
    return np.concatenate([preamble_syms, payload_syms])


def frame_to_samples(frame_syms, sps):
    """Zero-order-hold (rectangular) pulse shaping, normalised to DAC range."""
    tx = np.repeat(frame_syms, sps)
    tx /= np.max(np.abs(tx))
    return (tx * (2**14 - 1)).astype(np.complex64)


# ── Timing synchronisation ────────────────────────────────────────────────────

def sync_frame_rc(rx, preamble_syms, frame_len_syms, sps, h):
    """RC-pulse frame sync — correlate against RC-shaped preamble waveform.

    The correlation argmax is constrained to the *valid lock region*: the peak
    of the matched filter for a complete preamble lands at sample
    ``len(pre_shaped) - 1`` of the full-mode correlation, so any earlier peak
    cannot have a full preamble preceding it.  An unconstrained ``argmax`` that
    lands in the leading filter ramp gets ``start`` clamped to 0 by the
    ``max(..., 0)`` below, locking onto a spurious early peak: the channel-gain
    LS then returns |α|≈1 (vs ~26 for a real lock), preamble EVM ≈ 0 dB, a
    large spurious CFO, and the BCJR decodes the misaligned payload to ~0.5
    BER.  This is a sync edge-of-window failure, **not** an integer-symbol slip,
    and is independent of payload SNR.  Under cyclic TX the RX buffer holds
    ≥2 burst lengths, so a full, valid burst copy always exists at index
    ≥ ``len(pre_shaped) - 1``; restricting the search there recovers it.

    See the §V OTA characterisation: across 3672 clean bursts every valid lock
    had ``corr_peak_idx ≥ 613``; the only mis-decoding bursts had
    ``corr_peak_idx`` in 575..602 (< ``len(pre_shaped) - 1 = 611``) and decoded
    to ~0.5 BER.
    """
    pre_up = np.zeros(len(preamble_syms) * sps, dtype=np.complex64)
    pre_up[::sps] = preamble_syms.astype(np.complex64)
    pre_shaped = np.convolve(pre_up, h.astype(np.complex64), mode="full")

    corr = np.abs(np.convolve(rx, np.conj(pre_shaped[::-1]), mode="full"))

    # Valid region: a full preamble must precede the peak (peak ≥ Lp-1).
    lo = len(pre_shaped) - 1
    if lo < len(corr):
        peak = lo + int(np.argmax(corr[lo:]))
    else:                                  # capture shorter than the preamble
        peak = int(np.argmax(corr))
    start = max(peak - len(pre_shaped) + 1, 0)
    end = min(start + frame_len_syms * sps + 4 * len(h), len(rx))
    return rx[start:end], start, end, corr, peak


# ── Pilot-based phase tracking ────────────────────────────────────────────────

_PILOT_SYM = np.complex64((1 + 1j) / np.sqrt(2))   # QPSK +1+j, unit magnitude


def embed_pilots(data_syms, pilot_every: int = 100,
                 pilot_sym: complex = _PILOT_SYM):
    """Insert a pilot symbol before every ``pilot_every`` data symbols, and
    one trailing pilot at the end.

    Layout: ``[P  d0 d1 … d_{N-1}  P  d_N d_{N+1} …  P  …  P]``.
    The leading pilot lets the receiver bracket the first data block; the
    trailing pilot bounds the last block so linear interpolation works at
    both ends.

    Parameters
    ----------
    data_syms : complex array
    pilot_every : int
        Number of data symbols between successive pilots.  Must be ≥ 1.
    pilot_sym : complex
        Pilot value (unit modulus recommended).  Default: QPSK +1+j.

    Returns
    -------
    frame_syms : complex ndarray, length ``N + ceil(N / pilot_every) + 1``.
    pilot_positions : int ndarray, indices of pilots inside ``frame_syms``.
    pilot_value : complex, the pilot symbol used.
    """
    if pilot_every < 1:
        raise ValueError("pilot_every must be >= 1")
    data = np.asarray(data_syms, dtype=np.complex64)
    n = len(data)
    pilot_positions = []
    out = []
    for i in range(n):
        if i % pilot_every == 0:
            pilot_positions.append(len(out))
            out.append(pilot_sym)
        out.append(data[i])
    # Trailing pilot
    pilot_positions.append(len(out))
    out.append(pilot_sym)
    return (np.asarray(out, dtype=np.complex64),
            np.asarray(pilot_positions, dtype=np.int64),
            np.complex64(pilot_sym))


def phase_correct_with_pilots(frame_rx, pilot_positions, pilot_value):
    """Use pilots in ``frame_rx`` to linearly interpolate per-symbol phase
    correction across the burst, then strip the pilots and return the
    phase-corrected data symbols.

    Assumes the channel-gain magnitude has already been normalised (i.e.
    ``frame_rx`` ≈ ``tx_frame * e^{jφ(k)} + n`` after dividing by the
    preamble-derived complex α).
    """
    rx = np.asarray(frame_rx, dtype=np.complex64)
    pp = np.asarray(pilot_positions, dtype=np.int64)
    pv = np.complex64(pilot_value)

    # Measured phase at each pilot (relative to the known pilot value)
    pilot_phase = np.unwrap(np.angle(rx[pp] * np.conj(pv)))
    # Linearly interpolate to every sample.  np.interp returns the endpoint
    # value for indices outside [pp[0], pp[-1]] — fine because the leading
    # and trailing pilots already cover those endpoints.
    all_idx = np.arange(len(rx))
    phase_correction = np.interp(all_idx, pp, pilot_phase)
    rx_derot = rx * np.exp(-1j * phase_correction.astype(np.float64))

    # Drop pilot positions, return only data symbols
    mask = np.ones(len(rx), dtype=bool)
    mask[pp] = False
    return rx_derot[mask].astype(np.complex64)


def pilot_phase_diagnostics(frame_rx, pilot_positions, pilot_value):
    """Return raw per-pilot phase residuals (radians) for diagnostics.

    Pilot positions that fall beyond the recovered payload are dropped: frame
    sync can return fewer symbols than were transmitted, in which case the
    trailing pilot indices would otherwise overrun ``rx``.
    """
    rx = np.asarray(frame_rx, dtype=np.complex64)
    pp = np.asarray(pilot_positions, dtype=np.int64)
    pp = pp[pp < len(rx)]
    pv = np.complex64(pilot_value)
    if pp.size == 0:
        return np.zeros(0, dtype=np.float64)
    return np.unwrap(np.angle(rx[pp] * np.conj(pv)))
