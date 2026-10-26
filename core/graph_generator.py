"""Problem-instance generation and MaxCut ground truth.

The optimisation problem visualised by Quantum Compass is **MaxCut**: given an
undirected graph ``G = (V, E)``, find a partition of the vertices into two sets
that maximises the number of edges crossing the cut.  Each graph becomes a QAOA
instance with one qubit per vertex.

This module provides

* :func:`generate_graph` – random instances from three standard families,
* :func:`optimal_maxcut` – exact brute-force ground truth (for ``n <= 16``),
* :func:`compute_graph_properties` – structural descriptors shown in the UI.

Bitstring convention
--------------------
A cut is encoded as an integer ``k`` (or an ``n``-character string) whose bit
``i`` is the side of vertex ``i``::

    bit_i(k) = (k >> i) & 1        # 0 or 1

Vertex ``i`` maps to qubit ``i``.  An edge ``(i, j)`` is *cut* when
``bit_i != bit_j``.  This little-endian convention is used consistently across
the whole package (including when reading Aer measurement outcomes).
"""

from __future__ import annotations

from typing import Dict, Tuple

import networkx as nx
import numpy as np

#: The graph families selectable in the UI, mapped to a short description.
GRAPH_FAMILIES: Dict[str, str] = {
    "3-regular": "Random 3-regular graph (every vertex has degree 3)",
    "erdos-renyi": "Erdős–Rényi G(n, p) random graph",
    "scale-free": "Barabási–Albert preferential-attachment (scale-free) graph",
}

_VALID_FAMILIES = ("3-regular", "erdos-renyi", "scale-free")


def generate_graph(family: str, n: int, seed: int = 42, **kwargs) -> nx.Graph:
    """Generate a random graph of the requested ``family`` on ``n`` vertices.

    Parameters
    ----------
    family:
        One of ``'3-regular'``, ``'erdos-renyi'`` or ``'scale-free'``.
    n:
        Number of vertices.
    seed:
        Random seed, for reproducible instances.
    **kwargs:
        Family-specific parameters:

        * ``d``   – degree for ``'3-regular'`` (default ``3``).
        * ``p``   – edge probability for ``'erdos-renyi'`` (default ``0.5``).
        * ``m``   – edges attached per new node for ``'scale-free'`` (default ``2``).

    Returns
    -------
    nx.Graph
        The generated graph.  Vertices are relabelled ``0 .. n-1`` so that
        vertex ``i`` corresponds to qubit ``i``.

    Raises
    ------
    ValueError
        If ``family`` is unknown or the parameters cannot produce a valid graph
        (e.g. a 3-regular graph on an odd number of vertices).
    """
    if family not in _VALID_FAMILIES:
        raise ValueError(
            f"Unknown graph family '{family}'. Choose one of {list(_VALID_FAMILIES)}."
        )
    if n < 2:
        raise ValueError(f"A MaxCut instance needs at least 2 vertices, got n={n}.")

    if family == "3-regular":
        d = int(kwargs.get("d", 3))
        if n <= d:
            raise ValueError(
                f"A {d}-regular graph needs n > d (got n={n}, d={d})."
            )
        if (n * d) % 2 != 0:
            raise ValueError(
                f"A {d}-regular graph on {n} vertices is impossible: "
                f"n*d = {n * d} must be even. Use an even n for d=3."
            )
        G = nx.random_regular_graph(d, n, seed=seed)

    elif family == "erdos-renyi":
        p = float(kwargs.get("p", 0.5))
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"Edge probability p must lie in [0, 1], got {p}.")
        G = nx.erdos_renyi_graph(n, p, seed=seed)

    else:  # 'scale-free'
        m = int(kwargs.get("m", 2))
        if m < 1:
            raise ValueError(f"Barabási–Albert parameter m must be >= 1, got {m}.")
        if n <= m:
            # Barabasi-Albert requires n > m; start from the smallest valid seed.
            raise ValueError(
                f"A scale-free graph with m={m} needs n > m (got n={n})."
            )
        G = nx.barabasi_albert_graph(n, m, seed=seed)

    # Guarantee a clean 0..n-1 labelling even if a generator reordered nodes.
    G = nx.convert_node_labels_to_integers(G, first_label=0, ordering="sorted")
    return G


def cut_value(bits, edges, n: int) -> int:
    """Number of edges cut by the assignment ``bits``.

    ``bits`` may be an ``int`` (bit ``i`` = side of vertex ``i``) or a sequence
    of 0/1 of length ``n`` (index ``i`` = vertex ``i``).
    """
    if isinstance(bits, (int, np.integer)):
        k = int(bits)
        side = [(k >> i) & 1 for i in range(n)]
    else:
        seq = list(bits)
        if len(seq) != n:
            raise ValueError(f"Expected {n} bits, got {len(seq)}.")
        side = [int(b) & 1 for b in seq]
    return sum(1 for (i, j) in edges if side[i] != side[j])


def optimal_maxcut(G: nx.Graph) -> Tuple[int, str]:
    """Exact maximum cut by brute force over all ``2^n`` assignments.

    Parameters
    ----------
    G:
        The graph.  Brute force is exact but exponential, so this is only
        supported for ``n <= 16`` (65 536 assignments — instantaneous).

    Returns
    -------
    (best_cut, best_bitstring):
        ``best_cut`` is the maximum number of cut edges and ``best_bitstring``
        is an ``n``-character ``'0'``/``'1'`` string in vertex order
        (character ``i`` is the side of vertex ``i``).

    Raises
    ------
    ValueError
        If ``n > 16`` — use a heuristic solver for larger instances.
    """
    n = G.number_of_nodes()
    if n == 0:
        return 0, ""
    if n > 16:
        raise ValueError(
            f"Brute-force MaxCut supports n <= 16 (got n={n}); "
            "the search space 2**n is too large."
        )

    edges = list(G.edges())
    best_cut = -1
    best_k = 0
    # Vectorised enumeration: build the (2**n, n) bit matrix once.
    ks = np.arange(2 ** n, dtype=np.int64)
    bits = ((ks[:, None] >> np.arange(n)[None, :]) & 1).astype(np.int8)
    if edges:
        i_idx = np.array([e[0] for e in edges])
        j_idx = np.array([e[1] for e in edges])
        cuts = (bits[:, i_idx] != bits[:, j_idx]).sum(axis=1)
    else:
        cuts = np.zeros(2 ** n, dtype=np.int64)
    best_k = int(np.argmax(cuts))
    best_cut = int(cuts[best_k])

    best_bitstring = "".join(str((best_k >> i) & 1) for i in range(n))
    return best_cut, best_bitstring


def compute_graph_properties(G: nx.Graph) -> Dict[str, float]:
    """Structural descriptors of ``G`` shown in the UI sidebar.

    Returns a dict with (at least) the four keys required by the spec:

    * ``treewidth``        – approximate treewidth (min-degree heuristic),
    * ``spectral_gap``     – algebraic connectivity, i.e. the second-smallest
      Laplacian eigenvalue (a standard measure of how well-connected / how
      "expander-like" the graph is),
    * ``clustering_coeff`` – average clustering coefficient,
    * ``density``          – edge density ``2|E| / (|V|(|V|-1))``.

    A few extra descriptive fields (sizes, mean degree) are included because
    they are handy in the UI; they do not affect the required contract.
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()

    props: Dict[str, float] = {
        "num_nodes": int(n),
        "num_edges": int(m),
    }

    # --- treewidth (approximate; exact treewidth is NP-hard) ---------------
    tw = None
    if n == 0:
        tw = 0
    else:
        for fn_name in ("treewidth_min_degree", "treewidth_min_fill_in"):
            fn = getattr(nx.algorithms.approximation, fn_name, None)
            if fn is not None:
                try:
                    _, tw_decomp = fn(G)
                    tw = int(tw_decomp)
                    break
                except Exception:
                    tw = None
        if tw is None:
            # Fall back to a cheap, always-valid upper bound: max degree + 1.
            tw = int(max(dict(G.degree()).values())) if n else 0
    props["treewidth"] = int(tw)

    # --- spectral gap = algebraic connectivity (2nd smallest Laplacian ev) --
    if n < 2:
        props["spectral_gap"] = 0.0
    else:
        try:
            lap_eigs = np.sort(nx.laplacian_spectrum(G))
            props["spectral_gap"] = float(lap_eigs[1])
        except Exception:
            props["spectral_gap"] = 0.0

    # --- average clustering coefficient -----------------------------------
    props["clustering_coeff"] = float(nx.average_clustering(G))

    # --- density ------------------------------------------------------------
    props["density"] = float(nx.density(G))

    # --- extra, non-required descriptors -----------------------------------
    degrees = [d for _, d in G.degree()]
    props["avg_degree"] = float(np.mean(degrees)) if degrees else 0.0
    props["max_degree"] = int(max(degrees)) if degrees else 0
    try:
        props["is_connected"] = bool(nx.is_connected(G))
    except Exception:
        props["is_connected"] = False

    return props
