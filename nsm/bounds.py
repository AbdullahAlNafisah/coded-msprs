"""
Analytical performance bounds for MS-PRS.

This module exposes three distance metrics over the rate-2 NSM trellis,
all computed by Dijkstra on (variants of) the error-state-pair graph
built from ``nsm.modem.msprs.precompute``:

SCALE.  Every distance below is returned on the internal ``E_s = 1``
normalisation applied by ``nsm.modem.msprs.load_coefficients``.  The
manuscript quotes distances on the ``E_s = 5`` scale (Table II), so
multiply by 5 to compare.  On that scale 4-ASK has d² = 4 and 2-ASK has
d² = 20.

* :func:`min_squared_distance` — full MSED over *all* error events,
  including "parallel transitions" that flip only the memoryless stream-1
  bit. Dominates the BER floor of the equalizer's joint detector.  **This
  is the d²_min reported in the technical report (Siala et al., 2025,
  App. E/F) and quoted throughout main.tex §III–IV.**  Its closed-form
  values reproduce here to ≤1e-6 on the corrected loader (post-69e3ad9);
  ×5 they are 7.388 / 7.760 / 8.373 / 8.966 unbalanced and 5.858 / 6.340 /
  6.897 / 7.673 balanced for L₀ = 3…6.  Under this metric,
  unbalanced > balanced for every ``L_0 ∈ {3,…,6}`` — the defining design
  property of the unbalanced family.

* :func:`min_squared_distance_trellis` — MSED over error events that
  disturb the stream-0 (memory-carrying) input at least once. Excludes
  parallel transitions.

* :func:`min_squared_distance_stream0` — free distance of the stream-0
  partial-response code in isolation, obtained by restricting both paths
  to identical stream-1 bits so ``h_1`` contributes nothing.

  The event sets nest, ``{stream-0 only} ⊂ {stream-0 disturbing} ⊂ {all}``,
  so the minima order the other way: ``full ≤ trellis ≤ stream0``.  For the
  unbalanced family all three coincide at every L₀ ∈ {3,…,6}, because the
  minimising event is the single stream-0 bit flip of squared distance
  4·η₀‖h₀‖² = 4·η₀.  The balanced family departs at L₀ ≥ 4, where the
  minimising event mixes both streams.

  Values here were wrong before 2026-07-31: both this function and
  ``min_squared_distance_trellis`` appended duplicate edges into
  ``csr_matrix``, which SUMS them instead of taking the minimum.  Pinned
  against brute force in ``test_stream0_free_distance_pinned``.

* :func:`parallel_min_squared_distance` — closed-form ``4 h_1^2``, the
  squared distance of the dominant parallel-transition event (single
  ``b_1`` flip).  Reported for completeness.

* :func:`union_bound_ber` — single-term Q-function union bound.

The full SED *spectrum* (multiplicity-vs-distance) is not enumerated;
only the minima are reported. For paper-matching analytical curves use
the full MSED metric (:func:`min_squared_distance`).
"""

from math import erfc, sqrt
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


__all__ = [
    "min_squared_distance",
    "min_squared_distance_trellis",
    "min_squared_distance_stream0",
    "parallel_min_squared_distance",
    "qfunc",
    "union_bound_ber",
    "clopper_pearson",
]


def clopper_pearson(errs, bits, alpha=0.05):
    """Exact two-sided Clopper-Pearson interval on BER = errs/bits (Beta form).

    Returns ``(lo, hi)``, scalars or arrays following the inputs. BER points
    are independent Monte-Carlo/measured Bernoulli trials, so a binomial-exact
    interval is the right uncertainty model. With the >=200-error stopping
    rule a typical waterfall point has a ~+/-14% relative interval; thinner
    tail points carry the visibly wider bars that bound the achievable Eb/N0
    resolution.

    ``errs == 0`` yields ``(0, ppf(1-alpha/2, 1, bits))``, the one-sided upper
    bound plotted for points where no error was observed.
    """
    from scipy.stats import beta as _beta

    errs = np.asarray(errs, dtype=float)
    bits = np.asarray(bits, dtype=float)
    lo = np.where(errs > 0,
                  _beta.ppf(alpha / 2, np.maximum(errs, 1), bits - errs + 1),
                  0.0)
    hi = np.where(errs < bits,
                  _beta.ppf(1 - alpha / 2, errs + 1, np.maximum(bits - errs, 1)),
                  1.0)
    return lo, hi


def qfunc(x):
    """Gaussian Q-function, vectorised."""
    x = np.asarray(x, dtype=np.float64)
    return 0.5 * np.vectorize(erfc)(x / sqrt(2.0))


def _pair_outputs(precomp):
    """Return ``out[s, j, b]``, the noiseless symbol emitted by state ``s``
    on stream-0 input ``j`` and stream-1 input ``b``."""
    M = precomp["total_states"]
    bl = precomp["branch_labels"]
    bi = precomp["branch_indices"]
    out = np.empty((M, 2, 2))
    for s in range(M):
        for j in (0, 1):
            for b in (0, 1):
                out[s, j, b] = bl[bi[s, j] + b * 2 * M]
    return out


def _build_pair_graph(precomp):
    """Build the directed pair-state graph used to find MSED.

    Nodes ``0..M^2-1`` encode the ordered pair ``(s1, s2) = s1*M + s2``.
    Each edge weight is the squared output difference produced by some
    combination of inputs ``(j1, b1, j2, b2)``; parallel edges are
    collapsed to their minimum weight.

    A virtual sink node ``M^2`` is added; every diagonal node ``(s, s)``
    is connected to the sink with weight zero so that Dijkstra on the
    transposed graph yields the shortest path from any non-diagonal
    node into the absorbing set ``{(s, s) : s}``.
    """
    M = precomp["total_states"]
    next_states = precomp["next_states"]
    out = _pair_outputs(precomp)

    N = M * M
    best = {}  # (src, dst) -> min weight

    for s1 in range(M):
        for s2 in range(M):
            src = s1 * M + s2
            for j1 in (0, 1):
                ns1 = next_states[s1, j1]
                for b1 in (0, 1):
                    o1 = out[s1, j1, b1]
                    for j2 in (0, 1):
                        ns2 = next_states[s2, j2]
                        for b2 in (0, 1):
                            o2 = out[s2, j2, b2]
                            w = (o1 - o2) ** 2
                            dst = ns1 * M + ns2
                            key = (src, dst)
                            if key not in best or best[key] > w:
                                best[key] = w

    rows, cols, data = [], [], []
    for (src, dst), w in best.items():
        rows.append(src)
        cols.append(dst)
        data.append(w)
    sink = N
    for s in range(M):
        rows.append(s * M + s)
        cols.append(sink)
        data.append(0.0)

    graph = csr_matrix((data, (rows, cols)), shape=(N + 1, N + 1))
    return graph, M, out, next_states


def min_squared_distance(precomp):
    """
    Minimum Squared Euclidean Distance of the rate-2 NSM trellis.

    SCALE: returns ``d^2_min`` on the internal ``E_s = 1`` code
    normalisation, i.e. with the energy normalisation already applied by
    :func:`nsm.modem.msprs.load_coefficients` (total symbol energy = 1 in
    the precomp tables).

    The manuscript quotes every squared distance on the ``E_s = 5`` scale
    instead (Table II), so multiply by ``5`` before comparing against the
    paper.  On that scale 4-ASK has ``d^2 = 4`` and 2-ASK has ``d^2 = 20``.
    :func:`union_bound_ber` expects the ``E_s = 1`` value directly.
    """
    graph, M, out, next_states = _build_pair_graph(precomp)
    sink = M * M

    # Dijkstra on the transposed graph from the sink gives shortest
    # path from every node TO the sink.
    dist_to_sink = dijkstra(graph.T, indices=sink, directed=True)

    best = np.inf
    for s in range(M):
        for j1 in (0, 1):
            ns1 = next_states[s, j1]
            for b1 in (0, 1):
                o1 = out[s, j1, b1]
                for j2 in (0, 1):
                    ns2 = next_states[s, j2]
                    for b2 in (0, 1):
                        if (j1, b1) == (j2, b2):
                            continue
                        o2 = out[s, j2, b2]
                        w = (o1 - o2) ** 2
                        dst = ns1 * M + ns2
                        total = w + dist_to_sink[dst]
                        if 0.0 < total < best:
                            best = float(total)
    return best


def parallel_min_squared_distance(precomp):
    """
    Squared distance of the smallest *parallel-transition* error event —
    one that flips only the memoryless stream-1 bit and keeps the trellis
    state trajectory unchanged.

    For ``L_1 = 1`` (this paper's setting) the smallest such event flips a
    single ``b_1`` and contributes ``4 h_1^2`` to the squared distance.
    """
    return 4.0 * float(precomp["h1"]) ** 2


def min_squared_distance_trellis(precomp):
    """
    MSED restricted to *trellis* error events — paths that disturb the
    stream-0 (memory-carrying) input ``j`` at least once before re-merging.

    This corresponds to the convolutional-code interpretation of MSED used
    by the MS-PRS paper: parallel transitions that only flip stream-1 bits
    are excluded because, with ``L_1 = 1``, they form an uncoded BPSK
    sub-channel whose dominant error event is the single-bit flip already
    reported by :func:`parallel_min_squared_distance`.

    For a meaningful comparison of the trellis design itself across
    ``balanced`` and ``unbalanced`` energy splits, use this function.
    """
    return _msed_over_pair_graph(precomp, tie_stream1=False)


def min_squared_distance_stream0(precomp):
    """
    Free distance of the stream-0 partial-response code in isolation.

    Computed as the minimum squared distance over error events in which
    *both* paths use identical stream-1 bits (so ``h_1`` contributes
    nothing) and the stream-0 input ``j`` differs at one or more steps.

    For ``L_1 = 1`` this is the natural extension of the convolutional-
    code free distance to the rate-2 NSM trellis and is the metric on
    which the unbalanced filter family is typically optimised.
    """
    return _msed_over_pair_graph(precomp, tie_stream1=True)


def _msed_over_pair_graph(precomp, *, tie_stream1):
    """Shared Dijkstra over the error-state-pair graph.

    ``tie_stream1=False`` lets the two paths choose stream-1 bits freely
    (:func:`min_squared_distance_trellis`); ``True`` forces b1 == b2 at every
    step so the h_1 contribution cancels and the distance is set entirely by
    h_0 (:func:`min_squared_distance_stream0`). That single constraint is the
    only difference between the two metrics.
    """
    M = precomp["total_states"]
    next_states = precomp["next_states"]
    out = _pair_outputs(precomp)

    # Doubled graph: node = s1*M + s2 + flag*M*M.
    # flag = 0 : trellis has not yet diverged
    # flag = 1 : trellis-divergence has occurred at some prior step
    # Source set: (s, s, 0) for all s (no error yet, on diagonal).
    # Sink:       virtual node connected from (s, s, 1) for all s.
    N = 2 * M * M
    sink = N
    best_edge = {}  # (src, dst) -> min weight

    def add(src, dst, w):
        # Parallel edges MUST collapse to their minimum. csr_matrix sums
        # duplicate (row, col) entries, so accumulating here is not optional.
        key = (src, dst)
        if key not in best_edge or best_edge[key] > w:
            best_edge[key] = w

    def stream1_pairs(b1):
        return (b1,) if tie_stream1 else (0, 1)

    for flag in (0, 1):
        for s1 in range(M):
            for s2 in range(M):
                src = flag * M * M + s1 * M + s2
                for j1 in (0, 1):
                    ns1 = next_states[s1, j1]
                    for b1 in (0, 1):
                        o1 = out[s1, j1, b1]
                        for j2 in (0, 1):
                            ns2 = next_states[s2, j2]
                            for b2 in stream1_pairs(b1):
                                o2 = out[s2, j2, b2]
                                w = (o1 - o2) ** 2
                                new_flag = 1 if (flag == 1 or j1 != j2) else 0
                                dst = new_flag * M * M + ns1 * M + ns2
                                add(src, dst, w)
    # Sinks: any merged-and-diverged state collapses to the virtual sink.
    for s in range(M):
        add(1 * M * M + s * M + s, sink, 0.0)

    rows = [k[0] for k in best_edge]
    cols = [k[1] for k in best_edge]
    data = list(best_edge.values())
    graph = csr_matrix((data, (rows, cols)), shape=(N + 1, N + 1))
    dist_to_sink = dijkstra(graph.T, indices=sink, directed=True)

    # Minimum over all "first edges" leaving a flag-0 diagonal node via a
    # stream-0 divergence (j1 != j2), plus the cost-to-sink from the
    # resulting flag-1 node.
    best = np.inf
    for s in range(M):
        for j1 in (0, 1):
            ns1 = next_states[s, j1]
            for b1 in (0, 1):
                o1 = out[s, j1, b1]
                for j2 in (0, 1):
                    if j1 == j2:
                        continue   # parallel transition — excluded here
                    ns2 = next_states[s, j2]
                    for b2 in stream1_pairs(b1):
                        o2 = out[s, j2, b2]
                        w = (o1 - o2) ** 2
                        dst = 1 * M * M + ns1 * M + ns2
                        total = w + dist_to_sink[dst]
                        if 0.0 < total < best:
                            best = float(total)
    return best


def union_bound_ber(d2_min, eb_no_db, k_e=1.0, rate=1.0, bits_per_symbol=2.0):
    """
    Single-term union upper bound on the bit error probability.

        P_b ≲ k_e · Q( sqrt( d²_min · rate · bits_per_symbol · (E_b/N_0) / 2 ) )

    ``d²_min`` is on the energy-normalised (``E_s = 1``) symbol scale, so the
    absolute squared distance is ``d²_min · E_s``.  Pairwise error probability
    for two signals at squared distance ``D²`` is ``Q(sqrt(D²/(2 N_0)))``, and
    Appendix A gives ``E_s = m R_c E_b`` with ``m = bits_per_symbol`` coded bits
    per channel symbol and ``R_c = rate``.  Hence the Q argument above.

    Sanity check (2-ASK/BPSK, ``d²_min = 4``, ``m = 1``, ``rate = 1``):
    the argument reduces to ``sqrt(2 E_b/N_0)``, the textbook result.

    Parameters
    ----------
    d2_min : float
        MSED on the energy-normalised symbol scale (i.e. the value returned
        by :func:`min_squared_distance`).
    eb_no_db : array-like
        Eb/N0 sweep in dB.
    k_e : float, optional
        Multiplicity of error events at ``d²_min`` per information bit.
        Defaults to 1; supply the true multiplicity if known.
    rate : float, optional
        Outer code rate (1 = uncoded).
    bits_per_symbol : float, optional
        Information bits per transmitted symbol; 2 for rate-2 NSM.
    """
    eb_no_db = np.asarray(eb_no_db, dtype=np.float64)
    eb_no_lin = 10.0 ** (eb_no_db / 10.0)
    arg = np.sqrt(d2_min * eb_no_lin * rate * bits_per_symbol / 2.0)
    return k_e * qfunc(arg)
