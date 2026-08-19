import numpy as np
from numba import njit
from scipy.special import erfc

def ber_theory(eb_no_db):
    """Theoretical BER vs Eb/No for uncoded 2-ASK/BPSK over AWGN."""
    r = 10 ** (np.asarray(eb_no_db) / 10)
    return 0.5 * erfc(np.sqrt(r))

# Distance spectra (info-bit weight β_d) for common rate-1/2 codes
SPECTRUM_K3_57     = [(5,1),(6,2),(7,4),(8,8),(9,16),(10,32)]
SPECTRUM_K5_2335   = [(8,2),(9,3),(10,6),(11,12),(12,24)]
SPECTRUM_K7_133171 = [(10,36),(12,211),(14,1404)]

# Map octal generator tuples → (d_free, spectrum)
CODER_LOOKUP = {
    (0o5, 0o7):     (5,  SPECTRUM_K3_57),
    (0o23, 0o35):   (8,  SPECTRUM_K5_2335),
    (0o133, 0o171): (10, SPECTRUM_K7_133171),
}
def ber_coded_union_bound(eb_no_db, d_free, rate=0.5):
    """
    Dominant-term union bound for a rate-`rate` convolutional code with BPSK over AWGN.

    BER ≤ 0.5 · erfc(√(d_free · rate · Eb/N₀))

    Parameters
    ----------
    d_free : int   — free distance of the code
    rate   : float — code rate (default 0.5)
    """
    r = 10 ** (np.asarray(eb_no_db) / 10)
    return 0.5 * erfc(np.sqrt(d_free * rate * r))


def ber_coded_union_full(eb_no_db, spectrum, rate=0.5):
    """
    Multi-term union bound using the distance spectrum.

    BER ≤ Σ_d  β_d · Q(√(2·d·rate·Eb/N₀))

    Parameters
    ----------
    spectrum : list of (d, β_d) pairs — info-bit weight at each distance
    rate     : float — code rate (default 0.5)
    """
    r = 10 ** (np.asarray(eb_no_db) / 10)
    ber = np.zeros_like(r, dtype=float)
    for d, beta in spectrum:
        ber += beta * 0.5 * erfc(np.sqrt(d * rate * r))
    return ber


@njit(cache=True)
def modulate(bits, avg_bit_energy):
    """Map bits {0,1} → ±1/√Eb."""
    return (2 * bits - 1) * np.sqrt(avg_bit_energy)


@njit(cache=True)
def demodulate(samples, noise_variance, avg_bit_energy):
    """Soft LLR output for 2-ASK / BPSK."""
    return (2.0 * np.sqrt(avg_bit_energy) / noise_variance) * samples
