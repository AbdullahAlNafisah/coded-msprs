"""16-QAM modulator with Gray-coded mapping (paper §V comparator).

Used as a standard-MQAM baseband reference for OTA PSD measurements
against MS-PRS.  Symbol energy is normalised to unit average power.
"""

import numpy as np

# Gray-coded 4-PAM amplitudes: bits 00,01,11,10  →  -3,-1,+3,+1
_PAM4_GRAY = np.array([-3, -1, +3, +1], dtype=np.float64)


def _pam4(bits_pair):
    """Map a 2-bit row to a Gray-coded 4-PAM amplitude."""
    idx = (bits_pair[:, 0] << 1) | bits_pair[:, 1]
    return _PAM4_GRAY[idx]


def modulate(bits):
    """16-QAM mapper.

    Parameters
    ----------
    bits : array-like of 0/1, length must be a multiple of 4.

    Returns
    -------
    syms : complex64 ndarray, unit average symbol power.
        4-PAM Gray on each axis; symbols ∈ {±1,±3}+j{±1,±3} / sqrt(10).
    """
    b = np.asarray(bits).astype(int).reshape(-1, 4)
    I = _pam4(b[:, 0:2])
    Q = _pam4(b[:, 2:4])
    syms = (I + 1j * Q) / np.sqrt(10.0)
    return syms.astype(np.complex64)


def demodulate(syms):
    """Hard-decision 16-QAM demapper (Gray)."""
    syms = np.asarray(syms) * np.sqrt(10.0)
    def _pam4_decide(x):
        # Decision boundaries at ±2; map back to bit pair via Gray inverse.
        # Amplitude → bits: -3→00, -1→01, +1→10, +3→11.
        bits = np.empty((len(x), 2), dtype=int)
        bits[:, 0] = (x >= 0).astype(int)
        bits[:, 1] = (np.abs(x) <= 2).astype(int)
        return bits
    bI = _pam4_decide(np.real(syms))
    bQ = _pam4_decide(np.imag(syms))
    return np.concatenate([bI, bQ], axis=1).reshape(-1)
