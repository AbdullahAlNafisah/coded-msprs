import numpy as np
import warnings
from scipy.sparse import csr_matrix
from numba import njit, int64, types, float64, prange
from nsm._exit_jit import _i_inv, _gen_llrs_jit, _mi_avg_jit, _mi_hist_jit

# Output type for Numba log-BP solvers
output_type_log2 = types.Tuple((float64[:, :, :], float64[:, :, :], float64[:, :]))


def decode(H, y, snr=0, maxiter=1000):
    """Decode a Gaussian-noise-corrupted codeword using the BP algorithm.

    Sign convention (pyldpc-native): ``y`` is interpreted as the channel LLR
    with the convention ``L = ln P(b=0)/P(b=1)`` (so ``L > 0 ⇒ x = 0``).
    This is the **opposite** of the convention used elsewhere in this
    package (``ln P(b=1)/P(b=0)``).  Use :func:`decode_ext` if you have an
    LLR vector in the package's standard convention.

    Decoding is performed in parallel if multiple codewords are passed in y.

    Parameters
    ----------
    H: array (n_equations, n_code). Decoding matrix H.
    y: array (n_code, n_messages) or (n_code,). Received message(s) in the
        codeword space (LLR with ``L > 0 ⇒ x = 0``).
    maxiter: int. Maximum number of iterations of the BP algorithm.

    Returns
    -------
    x: array (n_code,) or (n_code, n_messages) the solutions in the
        codeword space.

    """
    m, n = H.shape

    bits_hist, bits_values, nodes_hist, nodes_values = _bitsandnodes(H)

    _n_bits = np.unique(H.sum(0))
    _n_nodes = np.unique(H.sum(1))

    if _n_bits * _n_nodes == 1:
        solver = _logbp_numba_regular
        bits_values = bits_values.reshape(n, -1)
        nodes_values = nodes_values.reshape(m, -1)

    else:
        solver = _logbp_numba

    # var = 10 ** (-snr / 10)

    if y.ndim == 1:
        y = y[:, None]
    # step 0: initialization

    Lc = y  # 2 * y / var
    _, n_messages = y.shape

    Lq = np.zeros(shape=(m, n, n_messages))

    Lr = np.zeros(shape=(m, n, n_messages))
    for n_iter in range(maxiter):

        Lq, Lr, L_posteriori = solver(
            bits_hist, bits_values, nodes_hist, nodes_values, Lc, Lq, Lr, n_iter
        )

        x = np.array(L_posteriori <= 0).astype(int)
        product = incode(H, x)
        if product:
            break
    if n_iter == maxiter - 1:
        warnings.warn(
            """Decoding stopped before convergence. You may want
                       to increase maxiter"""
        )
    return x.squeeze()


@njit(
    output_type_log2(
        int64[:],  # bits_hist
        int64[:],  # bits_values
        int64[:],  # nodes_hist
        int64[:],  # nodes_values
        float64[:, :],  # Lc
        float64[:, :, :],  # Lq
        float64[:, :, :],  # Lr
        int64,  # n_iter
    ),
    cache=True,
)
def _logbp_numba(bits_hist, bits_values, nodes_hist, nodes_values, Lc, Lq, Lr, n_iter):
    """Log-domain Belief Propagation (irregular LDPC)"""

    m, n, n_messages = Lr.shape  # m: number of checks, n: number of variables

    # Step 1: Check node update (horizontal step)
    bits_counter = 0
    for i in range(m):  # for each check node
        num_neighbors = bits_hist[i]
        connected_vars = bits_values[bits_counter : bits_counter + num_neighbors]
        bits_counter += num_neighbors

        for j in connected_vars:
            X = np.ones(n_messages)
            # Multiply tanh(0.5 * incoming messages) from all neighbors except j
            for neighbor in connected_vars:
                if neighbor != j:
                    if n_iter == 0:
                        X *= np.tanh(0.5 * Lc[neighbor])
                    else:
                        X *= np.tanh(0.5 * Lq[i, neighbor])

            # Compute Lr[i, j] using log((1 + X) / (1 - X))
            num = 1 + X
            denom = 1 - X
            for ll in range(n_messages):
                if num[ll] == 0:
                    Lr[i, j, ll] = -1  # Prevent divide-by-zero
                elif denom[ll] == 0:
                    Lr[i, j, ll] = 1
                else:
                    Lr[i, j, ll] = np.log(num[ll] / denom[ll])  # sum-product rule

    # Step 2: Variable node update (vertical step)
    nodes_counter = 0
    for j in range(n):  # for each variable node
        num_neighbors = nodes_hist[j]
        connected_checks = nodes_values[nodes_counter : nodes_counter + num_neighbors]
        nodes_counter += num_neighbors

        for i in connected_checks:
            Lq[i, j] = Lc[j]
            for neighbor in connected_checks:
                if neighbor != i:
                    Lq[i, j] += Lr[neighbor, j]

    # Step 3: Compute a posteriori LLRs
    L_posteriori = np.zeros((n, n_messages))
    nodes_counter = 0
    for j in range(n):
        num_neighbors = nodes_hist[j]
        connected_checks = nodes_values[nodes_counter : nodes_counter + num_neighbors]
        nodes_counter += num_neighbors

        L_posteriori[j] = Lc[j] + Lr[connected_checks, j].sum(axis=0)

    return Lq, Lr, L_posteriori


@njit(
    output_type_log2(
        int64[:],  # bits_hist
        int64[:],  # bits_values
        int64[:],  # nodes_hist
        int64[:],  # nodes_values
        float64[:, :],  # Lc with shape (n_code, 1)
        float64[:, :, :],  # Lq with shape (m, n_code, 1)
        float64[:, :, :],  # Lr with shape (m, n_code, 1)
        int64,  # n_iter
    ),
    cache=True,
)
def _logbp_numba_n1(bits_hist, bits_values, nodes_hist, nodes_values, Lc, Lq, Lr, n_iter):
    """Scalar-specialised log-domain BP for the n_messages=1 path.

    Same algorithm as `_logbp_numba`, but every per-`(check, bit)` array
    operation on a length-1 vector is replaced with a scalar. This
    eliminates two per-edge ndarray allocations that baseline lprof
    showed at lines 129 (`X = np.ones(n_messages)`, 5.1 %) and 136
    (`X *= np.tanh(0.5 * Lq[i, neighbor])`, 20.6 %) — both unavoidable
    in the general-`n_messages` form, both pure overhead when the
    consumer only ever runs with `n_messages == 1` (which is the case
    for every caller in this repo: `coded_ber` and the LDPC turbo
    pipeline both call with `Lc.reshape(-1, 1)`).

    Output arrays keep the (…, 1) trailing dimension so the wrapper
    APIs (and pyldpc cross-check) are unchanged. Bit-exact agreement
    with `_logbp_numba` is expected at float64 precision because IEEE
    `*=` on a length-1 ndarray and scalar `*=` produce identical bits.
    Last cross-checked against pyldpc at rtol=1e-12, atol=1e-14.
    """

    m, n, _ = Lr.shape  # n_messages always 1; not used

    # Step 1: Check node update (horizontal). Scalar accumulator x replaces
    # X (a length-1 ndarray); per-(check, bit) np.ones() alloc gone, per-
    # (check, bit, neighbor) ndarray temp from X *= np.tanh(...) gone.
    bits_counter = 0
    for i in range(m):
        num_neighbors = bits_hist[i]
        connected_vars = bits_values[bits_counter : bits_counter + num_neighbors]
        bits_counter += num_neighbors

        for j in connected_vars:
            x = 1.0
            for neighbor in connected_vars:
                if neighbor != j:
                    if n_iter == 0:
                        x *= np.tanh(0.5 * Lc[neighbor, 0])
                    else:
                        x *= np.tanh(0.5 * Lq[i, neighbor, 0])

            num = 1.0 + x
            denom = 1.0 - x
            if num == 0.0:
                Lr[i, j, 0] = -1.0
            elif denom == 0.0:
                Lr[i, j, 0] = 1.0
            else:
                Lr[i, j, 0] = np.log(num / denom)

    # Step 2: Variable node update (vertical).
    nodes_counter = 0
    for j in range(n):
        num_neighbors = nodes_hist[j]
        connected_checks = nodes_values[nodes_counter : nodes_counter + num_neighbors]
        nodes_counter += num_neighbors

        for i in connected_checks:
            Lq[i, j, 0] = Lc[j, 0]
            for neighbor in connected_checks:
                if neighbor != i:
                    Lq[i, j, 0] += Lr[neighbor, j, 0]

    # Step 3: a-posteriori LLR. Scalar accumulator replaces the fancy-index
    # `Lr[connected_checks, j].sum(axis=0)` of the general form.
    L_posteriori = np.zeros((n, 1))
    nodes_counter = 0
    for j in range(n):
        num_neighbors = nodes_hist[j]
        connected_checks = nodes_values[nodes_counter : nodes_counter + num_neighbors]
        nodes_counter += num_neighbors

        total = Lc[j, 0]
        for idx in range(num_neighbors):
            total += Lr[connected_checks[idx], j, 0]
        L_posteriori[j, 0] = total

    return Lq, Lr, L_posteriori


@njit(
    output_type_log2(
        int64[:],  # bits_hist (unused here but preserved for signature consistency)
        int64[
            :, :
        ],  # bits_values (2D: each check node has same number of variable neighbors)
        int64[:],  # nodes_hist (unused here but preserved for signature consistency)
        int64[
            :, :
        ],  # nodes_values (2D: each variable node has same number of check neighbors)
        float64[:, :],  # Lc
        float64[:, :, :],  # Lq
        float64[:, :, :],  # Lr
        int64,  # n_iter
    ),
    cache=True,
)
def _logbp_numba_regular(
    bits_hist, bits_values, nodes_hist, nodes_values, Lc, Lq, Lr, n_iter
):
    """Log-domain Belief Propagation (regular LDPC)"""

    m, n, n_messages = Lr.shape

    # Step 1: Check node update (horizontal step)
    for i in range(m):
        connected_vars = bits_values[i]
        for j in connected_vars:
            X = np.ones(n_messages)
            for neighbor in connected_vars:
                if neighbor != j:
                    if n_iter == 0:
                        X *= np.tanh(0.5 * Lc[neighbor])
                    else:
                        X *= np.tanh(0.5 * Lq[i, neighbor])

            num = 1 + X
            denom = 1 - X
            for ll in range(n_messages):
                if num[ll] == 0:
                    Lr[i, j, ll] = -1
                elif denom[ll] == 0:
                    Lr[i, j, ll] = 1
                else:
                    Lr[i, j, ll] = np.log(num[ll] / denom[ll])  # sum-product rule

    # Step 2: Variable node update (vertical step)
    for j in range(n):
        connected_checks = nodes_values[j]
        for i in connected_checks:
            Lq[i, j] = Lc[j]
            for neighbor in connected_checks:
                if neighbor != i:
                    Lq[i, j] += Lr[neighbor, j]

    # Step 3: Compute a posteriori LLRs
    L_posteriori = np.zeros((n, n_messages))
    for j in range(n):
        connected_checks = nodes_values[j]
        L_posteriori[j] = Lc[j] + Lr[connected_checks, j].sum(axis=0)

    return Lq, Lr, L_posteriori


def get_message(tG, x):
    """Compute the original `n_bits` message from a `n_code` codeword `x`.

    Parameters
    ----------
    tG: array (n_code, n_bits) coding matrix tG.
    x: array (n_code,) decoded codeword of length `n_code`.

    Returns
    -------
    message: array (n_bits,). Original binary message.

    """
    n, k = tG.shape

    rtG, rx = gausselimination(tG, x)

    message = np.zeros(k).astype(int)

    message[k - 1] = rx[k - 1]
    for i in reversed(range(k - 1)):
        message[i] = rx[i]
        message[i] -= binaryproduct(
            rtG[i, list(range(i + 1, k))], message[list(range(i + 1, k))]
        )

    return abs(message)


def encode(tG, v, snr=0, seed=None):
    """Encode a binary message and adds Gaussian noise.

    Parameters
    ----------
    tG: array or scipy.sparse.csr_matrix (m, k). Transposed coding matrix
    obtained from `pyldpc.make_ldpc`.

    v: array (k, ) or (k, n_messages) binary messages to be encoded.

    snr, seed: accepted but IGNORED. Upstream pyldpc's encode also BPSK-maps
    the codeword and adds AWGN; this fork returns the bare codeword because
    the package does its own modulation (nsm.modem) and channel
    (nsm.channel.awgn). Callers still pass snr= positionally.

    Returns
    -------
    y: array (n,) or (n, n_messages) binary codeword. NOT noise-corrupted.

    """
    return binaryproduct(tG, v)


def parity_check_matrix(n_code, d_v, d_c, seed=None):
    """
    Build a regular Parity-Check Matrix H following Callager's algorithm.

    Parameters
    ----------
    n_code: int, Length of the codewords.
    d_v: int, Number of parity-check equations including a certain bit.
        Must be greater or equal to 2.
    d_c: int, Number of bits in the same parity-check equation. d_c Must be
        greater or equal to d_v and must divide n.
    seed: int, seed of the random generator.

    Returns
    -------
    H: array (n_equations, n_code). LDPC regular matrix H.
        Where n_equations = d_v * n / d_c, the total number of parity-check
        equations.

    """
    rng = check_random_state(seed)

    if d_v <= 1:
        raise ValueError("""d_v must be at least 2.""")

    if d_c <= d_v:
        raise ValueError("""d_c must be greater than d_v.""")

    if n_code % d_c:
        raise ValueError("""d_c must divide n for a regular LDPC matrix H.""")

    n_equations = (n_code * d_v) // d_c

    block = np.zeros((n_equations // d_v, n_code), dtype=int)
    H = np.empty((n_equations, n_code))
    block_size = n_equations // d_v

    # Filling the first block with consecutive ones in each row of the block

    for i in range(block_size):
        for j in range(i * d_c, (i + 1) * d_c):
            block[i, j] = 1
    H[:block_size] = block

    # reate remaining blocks by permutations of the first block's columns:
    for i in range(1, d_v):
        H[i * block_size : (i + 1) * block_size] = rng.permutation(block.T).T
    H = H.astype(int)
    return H


def coding_matrix(H, sparse=True):
    """Return the generating coding matrix G given the LDPC matrix H.

    Parameters
    ----------
    H: array (n_equations, n_code). Parity check matrix of an LDPC code with
        code length `n_code` and `n_equations` number of equations.
    sparse: (boolean, default True): if `True`, scipy.sparse format is used
        to speed up computation.

    Returns
    -------
    G.T: array (n_bits, n_code). Transposed coding matrix.

    """
    if type(H) == csr_matrix:
        H = H.toarray()
    n_equations, n_code = H.shape

    # DOUBLE GAUSS-JORDAN:

    Href_colonnes, tQ = gaussjordan(H.T, 1)

    Href_diag = gaussjordan(np.transpose(Href_colonnes))

    Q = tQ.T

    n_bits = n_code - Href_diag.sum()

    Y = np.zeros(shape=(n_code, n_bits)).astype(int)
    Y[n_code - n_bits :, :] = np.identity(n_bits)

    if sparse:
        Q = csr_matrix(Q)
        Y = csr_matrix(Y)

    tG = binaryproduct(Q, Y)

    return tG


def coding_matrix_systematic(H, sparse=True):
    """Compute a coding matrix G in systematic format with an identity block.

    Parameters
    ----------
    H: array (n_equations, n_code). Parity-check matrix.
    sparse: (boolean, default True): if `True`, scipy.sparse is used
    to speed up computation if n_code > 1000.

    Returns
    -------
    H_new: (n_equations, n_code) array. Modified parity-check matrix given by a
        permutation of the columns of the provided H.
    G_systematic.T: Transposed Systematic Coding matrix associated to H_new.

    """
    n_equations, n_code = H.shape

    if n_code > 1000 or sparse:
        sparse = True
    else:
        sparse = False

    P1 = np.identity(n_code, dtype=int)

    Hrowreduced = gaussjordan(H)

    n_bits = n_code - sum([a.any() for a in Hrowreduced])

    # After this loop, Hrowreduced will have the form H_ss : | I_(n-k)  A |

    while True:
        zeros = [i for i in range(min(n_equations, n_code)) if not Hrowreduced[i, i]]
        if len(zeros):
            indice_colonne_a = min(zeros)
        else:
            break
        list_ones = [
            j
            for j in range(indice_colonne_a + 1, n_code)
            if Hrowreduced[indice_colonne_a, j]
        ]
        if len(list_ones):
            indice_colonne_b = min(list_ones)
        else:
            break
        aux = Hrowreduced[:, indice_colonne_a].copy()
        Hrowreduced[:, indice_colonne_a] = Hrowreduced[:, indice_colonne_b]
        Hrowreduced[:, indice_colonne_b] = aux

        aux = P1[:, indice_colonne_a].copy()
        P1[:, indice_colonne_a] = P1[:, indice_colonne_b]
        P1[:, indice_colonne_b] = aux

    # Now, Hrowreduced has the form: | I_(n-k)  A | ,
    # the permutation above makes it look like :
    # |A  I_(n-k)|

    P1 = P1.T
    identity = list(range(n_code))
    sigma = identity[n_code - n_bits :] + identity[: n_code - n_bits]

    P2 = np.zeros(shape=(n_code, n_code), dtype=int)
    P2[identity, sigma] = np.ones(n_code)

    if sparse:
        P1 = csr_matrix(P1)
        P2 = csr_matrix(P2)
        H = csr_matrix(H)

    P = binaryproduct(P2, P1)

    if sparse:
        P = csr_matrix(P)

    H_new = binaryproduct(H, np.transpose(P))

    G_systematic = np.zeros((n_bits, n_code), dtype=int)
    G_systematic[:, :n_bits] = np.identity(n_bits)
    G_systematic[:, n_bits:] = (Hrowreduced[: n_code - n_bits, n_code - n_bits :]).T

    return H_new, G_systematic.T


def make_ldpc(n_code, d_v, d_c, systematic=False, sparse=True, seed=None):
    """Create an LDPC coding and decoding matrices H and G.

    Parameters
    ----------
    n_code: int, Length of the codewords.
    d_v: int, Number of parity-check equations including a certain bit.
    d_c: int, Number of bits in the same parity-check equation. d_c Must be
        greater or equal to d_v and must divide n.
    seed: int, seed of the random generator.
    systematic: boolean, default False. if True, constructs a systematic
    coding matrix G.

    Returns:
    --------
    H: array (n_equations, n_code). Parity check matrix of an LDPC code with
        code length `n_code` and `n_equations` number of equations.
    G: (n_code, n_bits) array coding matrix.

    """
    seed = check_random_state(seed)

    H = parity_check_matrix(n_code, d_v, d_c, seed=seed)
    if systematic:
        H, G = coding_matrix_systematic(H, sparse=sparse)
    else:
        G = coding_matrix(H, sparse=sparse)
    return H, G


"""Conversion tools."""

import math
import numbers
import numpy as np
import scipy
from scipy.stats import norm

pi = math.pi


def binaryproduct(X, Y):
    """Compute a matrix-matrix / vector product in Z/2Z."""
    A = X.dot(Y)
    try:
        A = A.toarray()
    except AttributeError:
        pass
    return A % 2


def gaussjordan(X, change=0):
    """Compute the binary row reduced echelon form of X.

    Parameters
    ----------
    X: array (m, n)
    change : boolean (default, False). If True returns the inverse transform

    Returns
    -------
    if `change` == 'True':
        A: array (m, n). row reduced form of X.
        P: tranformations applied to the identity
    else:
        A: array (m, n). row reduced form of X.

    """
    A = np.copy(X)
    m, n = A.shape

    if change:
        P = np.identity(m).astype(int)

    pivot_old = -1
    for j in range(n):
        filtre_down = A[pivot_old + 1 : m, j]
        pivot = np.argmax(filtre_down) + pivot_old + 1

        if A[pivot, j]:
            pivot_old += 1
            if pivot_old != pivot:
                aux = np.copy(A[pivot, :])
                A[pivot, :] = A[pivot_old, :]
                A[pivot_old, :] = aux
                if change:
                    aux = np.copy(P[pivot, :])
                    P[pivot, :] = P[pivot_old, :]
                    P[pivot_old, :] = aux

            for i in range(m):
                if i != pivot_old and A[i, j]:
                    if change:
                        P[i, :] = abs(P[i, :] - P[pivot_old, :])
                    A[i, :] = abs(A[i, :] - A[pivot_old, :])

        if pivot_old == m - 1:
            break

    if change:
        return A, P
    return A


def _bitsandnodes(H):
    """Return bits and nodes of a parity-check matrix H."""
    if type(H) != scipy.sparse.csr_matrix:
        bits_indices, bits = np.where(H)
        nodes_indices, nodes = np.where(H.T)
    else:
        bits_indices, bits = scipy.sparse.find(H)[:2]
        nodes_indices, nodes = scipy.sparse.find(H.T)[:2]
    bits_histogram = np.bincount(bits_indices)
    nodes_histogram = np.bincount(nodes_indices)

    return bits_histogram, bits, nodes_histogram, nodes


def incode(H, x):
    """Compute Binary Product of H and x."""
    return (binaryproduct(H, x) == 0).all()


def gausselimination(A, b):
    """Solve linear system in Z/2Z via Gauss Gauss elimination."""
    if type(A) == scipy.sparse.csr_matrix:
        A = A.toarray().copy()
    else:
        A = A.copy()
    b = b.copy()
    n, k = A.shape

    for j in range(min(k, n)):
        listedepivots = [i for i in range(j, n) if A[i, j]]
        if len(listedepivots):
            pivot = np.min(listedepivots)
        else:
            continue
        if pivot != j:
            aux = (A[j, :]).copy()
            A[j, :] = A[pivot, :]
            A[pivot, :] = aux

            aux = b[j].copy()
            b[j] = b[pivot]
            b[pivot] = aux

        for i in range(j + 1, n):
            if A[i, j]:
                A[i, :] = abs(A[i, :] - A[j, :])
                b[i] = abs(b[i] - b[j])

    return A, b


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance
    Parameters
    ----------
    seed : None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError(
        "%r cannot be used to seed a numpy.random.RandomState" " instance" % seed
    )


# ── Parallel EXIT curve kernel for LDPC decoder ──────────────────────────────

@njit(cache=True)
def _ldpc_extrinsic_jit(l_a, bits_hist, bits_values, nodes_hist, nodes_values,
                         n_equations, n_iter):
    """Run n_iter rounds of LDPC belief-propagation and return extrinsic LLRs.

    Input LLR Lc = l_a (a-priori only; no channel term).  The extrinsic is
    the information the check-node graph contributes beyond the a-priori:
        L_ext = L_posteriori − L_a
    """
    n_code = len(l_a)
    Lc  = l_a.reshape(-1, 1).copy()
    Lq  = np.zeros((n_equations, n_code, 1))
    Lr  = np.zeros((n_equations, n_code, 1))
    L_post = Lc.copy()
    for it in range(n_iter):
        Lq, Lr, L_post = _logbp_numba_n1(
            bits_hist, bits_values, nodes_hist, nodes_values, Lc, Lq, Lr, it
        )
    return L_post[:, 0] - l_a


@njit(parallel=True, cache=True)
def exit_curve_ldpc(IA, coded, bits_hist, bits_values, nodes_hist, nodes_values,
                    n_equations, n_iter, N_TRIALS):
    """
    Compute the LDPC decoder EXIT curve, parallelised across IA points.

    For each IA point, N_TRIALS independent a-priori LLR vectors are decoded
    by n_iter rounds of belief propagation; the IE estimates are averaged.
    Lq and Lr are allocated fresh per prange iteration (thread-local), so
    there are no data races.

    Parameters
    ----------
    IA           : float64 array (n_pts,)
    coded        : int array (n_code,) — true codeword bits (0/1)
    bits_hist    : int64 array  — check-node degree histogram from _bitsandnodes
    bits_values  : int64 array  — variable indices per check node (flattened)
    nodes_hist   : int64 array  — variable-node degree histogram
    nodes_values : int64 array  — check indices per variable node (flattened)
    n_equations  : int — number of parity-check equations (H.shape[0])
    n_iter       : int — number of BP iterations per trial
    N_TRIALS     : int

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
            l_a = _gen_llrs_jit(coded, sigma_a)

            if t == 0:
                ia_meas[k] = _mi_avg_jit(l_a, coded)

            l_ext = _ldpc_extrinsic_jit(
                l_a, bits_hist, bits_values, nodes_hist, nodes_values,
                n_equations, n_iter,
            )

            sum_avg  += _mi_avg_jit(l_ext, coded)
            sum_hist += _mi_hist_jit(l_ext, coded)

        ie_avg[k]  = sum_avg  / N_TRIALS
        ie_hist[k] = sum_hist / N_TRIALS

    return ie_avg, ie_hist, ia_meas
