import numpy as np
from numba import njit


@njit(cache=True)
def interleave(data, indices):
    """Permute data according to indices."""
    out = np.zeros(len(indices), dtype=data.dtype)
    for i in range(len(indices)):
        out[i] = data[indices[i]]
    return out


@njit(cache=True)
def deinterleave(data, indices):
    """Inverse permutation of interleave."""
    out = np.zeros(len(indices), dtype=data.dtype)
    for i in range(len(indices)):
        out[indices[i]] = data[i]
    return out


def random_indices(length, seed=None):
    """Generate a reproducible random interleaver index array."""
    rng = np.random.default_rng(seed)
    return rng.permutation(length).astype(np.int32)

