import numpy as np
from numba import njit


@njit(cache=True)
def llr_to_log_prob(llr):
    """log P(b=1 | LLR)  =  log( e^llr / (1 + e^llr) )"""
    if llr > 10.0:
        return -1e-6
    elif llr < -10.0:
        return llr
    return llr - np.log(1.0 + np.exp(llr))


@njit(cache=True)
def log_prob_to_llr(lp1, lp0):
    """LLR from log-probabilities, clipped to ±50."""
    if lp0 == -np.inf:
        return 50.0
    if lp1 == -np.inf:
        return -50.0
    return lp1 - lp0


@njit(cache=True)
def log_add(a, b):
    """Numerically stable log( e^a + e^b ) — the 'max-log with correction' step."""
    if a == -np.inf and b == -np.inf:
        return -np.inf
    if a == -np.inf:
        return b
    if b == -np.inf:
        return a
    diff = abs(a - b)
    if diff > 20.0:
        return max(a, b)
    return max(a, b) + np.log(1.0 + np.exp(-diff))
