"""4-QAM modulator and hard-decision demodulator."""

import numpy as np


def modulate(bits):
    """
    4-QAM mapper.

    Parameters
    ----------
    bits : array-like of 0/1, length must be even.

    Returns
    -------
    syms : complex ndarray, unit average power.
        Mapping: (b0, b1) → I = 2*b0-1,  Q = 2*b1-1,  normalised by sqrt(2).
    """
    # Cast to signed int first — without this an unsigned input (uint8) wraps
    # around at "2*bits - 1" (0 → 255) and the modulator silently produces
    # garbage symbols.
    bits = np.asarray(bits).astype(int).reshape(-1, 2)
    I = 2 * bits[:, 0] - 1
    Q = 2 * bits[:, 1] - 1
    return (I + 1j * Q) / np.sqrt(2)


def demodulate(syms):
    """
    Hard-decision 4-QAM demapper.

    Parameters
    ----------
    syms : complex ndarray

    Returns
    -------
    bits : 1D ndarray of 0/1
    """
    syms = np.asarray(syms)
    bI = (np.real(syms) > 0).astype(int)
    bQ = (np.imag(syms) > 0).astype(int)
    return np.column_stack([bI, bQ]).reshape(-1)
