import numpy as np
from numba import njit


def setup(eb_no_range, avg_bit_energy, rate=1.0):
    """
    Convert an Eb/No sweep range to noise parameters.

    Parameters
    ----------
    eb_no_range : (min_dB, max_dB, step_dB)
    eb           : energy per bit of the signal

    Returns
    -------
    dict with keys: eb_no_db, noise_var, noise_std
    """
    lo, hi, step = eb_no_range
    eb_no_db = np.arange(lo, hi + step, step)
    eb_no_lin = 10 ** (eb_no_db / 10)
    noise_var = avg_bit_energy / (2*rate*eb_no_lin)
    return {
        "eb_no_db": eb_no_db,
        "noise_var": noise_var,
        "noise_std": np.sqrt(noise_var),
    }


@njit(cache=True)
def transmit(signal, noise_std):
    """Add real AWGN to signal (Numba-compiled)."""
    return signal + noise_std * np.random.randn(len(signal))
