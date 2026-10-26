"""Classical MaxCut baselines for the quantum-advantage comparison.

The Advantage Map asks a concrete question: *for which problem sizes and noise
levels does QAOA's approximation ratio exceed what a classical heuristic
achieves?*  This module supplies the classical side of that comparison.

Everything here is **pure numpy / networkx** — it constructs *no* Qiskit object —
so it is safe to call directly from the Streamlit parent process (unlike the
quantum evaluation, which is delegated to the worker subprocess; see
:mod:`core.aer_worker`).

All functions return an **approximation ratio** ``<C> / C_max`` in ``[0, 1]``,
where ``C_max`` is the exact brute-force MaxCut value, so they are directly
comparable to the QAOA ratio produced by :mod:`core.qaoa_evaluator`.

Baselines
---------
``goemans_willison_ratio``
    The Goemans–Williamson semidefinite-programming guarantee for MaxCut.  This
    is a *worst-case* approximation guarantee, used here as a fixed reference
    line (it does not depend on the instance).

``simulated_annealing_maxcut``
    Stochastic single-spin-flip local search with a geometric cooling schedule,
    repeated ``n_runs`` times.  Returns the mean, over runs, of each run's best
    cut ratio (a stable "typical best effort" estimate).

``random_sampling_ratio``
    Monte-Carlo estimate of the expected ratio of a uniformly random cut
    (``|E| / (2 C_max)`` in the limit), by averaging ``n_samples`` random
    assignments.

``best_classical_ratio``
    The maximum of the selected methods — the strongest classical baseline.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from core.graph_generator import optimal_maxcut

#: Goemans–Williamson worst-case approximation guarantee for MaxCut.
#: (The canonical constant is alpha_GW ~= 0.878567; this app uses 0.8746 as its
#: reference guarantee, per the Advantage-Map specification.)
GW_GUARANTEE: float = 0.8746

#: Recognised baseline selectors for :func:`best_classical_ratio`.
CLASSICAL_METHODS: Tuple[str, ...] = ("GW", "annealing", "random", "best")


def _edges_and_n(G: nx.Graph) -> Tuple[List[Tuple[int, int]], int]:
    """Normalise a graph to ``(edges_as_index_pairs, n)`` with 0..n-1 vertices."""
    nodes = list(G.nodes())
    index = {v: i for i, v in enumerate(nodes)}
    edges = [(index[u], index[v]) for (u, v) in G.edges()]
    return edges, len(nodes)


def _cmax(G: nx.Graph) -> int:
    """Exact MaxCut value (brute force; ``n <= 16``)."""
    best, _ = optimal_maxcut(G)
    return int(best)


def goemans_willison_ratio(G: nx.Graph) -> float:
    """Goemans–Williamson approximation guarantee for MaxCut (constant line).

    Returns the fixed worst-case guarantee :data:`GW_GUARANTEE` (0.8746).  GW is
    an SDP-based algorithm whose *guaranteed* approximation ratio does not depend
    on the instance, so this is a horizontal reference line on the advantage map.
    (On any specific graph GW typically achieves more than its guarantee; the
    constant is used as the standard classical reference.)
    """
    return float(GW_GUARANTEE)


def simulated_annealing_maxcut(G: nx.Graph, n_runs: int = 20, seed: int = 0,
                               steps_per_run: Optional[int] = None,
                               t_start: float = 2.0, t_end: float = 0.05) -> float:
    """Best-of-run simulated annealing for MaxCut, averaged over ``n_runs``.

    Each run starts from a random assignment and repeatedly proposes a
    single-vertex flip, accepting it with the Metropolis probability
    ``min(1, exp(dC / T))`` under a geometric cooling schedule ``T: t_start ->
    t_end``.  The best cut seen during a run is kept.  The returned value is the
    **mean over runs of each run's best-cut ratio** — a stable estimate of SA's
    typical best effort with this budget.

    The flip delta is computed incrementally (``dC = deg_i - 2 * c_i`` where
    ``c_i`` is the number of already-cut edges at vertex ``i``), so a sweep costs
    ``O(deg)`` rather than ``O(|E|)``.
    """
    edges, n = _edges_and_n(G)
    if n == 0 or not edges:
        return 0.0
    cmax = _cmax(G)
    if cmax <= 0:
        return 0.0

    # Adjacency lists for incremental delta evaluation.
    adj: List[List[int]] = [[] for _ in range(n)]
    for (i, j) in edges:
        adj[i].append(j)
        adj[j].append(i)

    n_runs = int(max(1, n_runs))
    if steps_per_run is None:
        steps_per_run = int(max(200, 60 * n))
    rng = np.random.default_rng(seed)

    ratios: List[float] = []
    for _ in range(n_runs):
        x = rng.integers(0, 2, size=n).astype(np.int8)
        # cut count at each vertex (number of neighbours on the other side)
        cur_cut = 0
        for (i, j) in edges:
            cur_cut += int(x[i] != x[j])
        best_cut = cur_cut

        for s in range(steps_per_run):
            frac = s / max(1, steps_per_run - 1)
            T = t_start * (t_end / t_start) ** frac
            i = int(rng.integers(0, n))
            # c_i = neighbours of i currently on the opposite side
            xi = x[i]
            c_i = 0
            for j in adj[i]:
                c_i += int(x[j] != xi)
            delta = len(adj[i]) - 2 * c_i   # change in cut if we flip i
            if delta >= 0 or rng.random() < np.exp(delta / max(T, 1e-9)):
                x[i] = 1 - xi
                cur_cut += delta
                if cur_cut > best_cut:
                    best_cut = cur_cut
        ratios.append(best_cut / cmax)

    return float(np.mean(ratios))


def random_sampling_ratio(G: nx.Graph, n_samples: int = 5000, seed: int = 0) -> float:
    """Expected approximation ratio of a uniformly random cut (Monte Carlo).

    Averages ``C(x) / C_max`` over ``n_samples`` random assignments ``x``.  In the
    limit this converges to ``|E| / (2 C_max)`` (each edge is cut with probability
    1/2), the natural "do-nothing" classical baseline.
    """
    edges, n = _edges_and_n(G)
    if n == 0 or not edges:
        return 0.0
    cmax = _cmax(G)
    if cmax <= 0:
        return 0.0
    n_samples = int(max(1, n_samples))
    rng = np.random.default_rng(seed)

    bits = rng.integers(0, 2, size=(n_samples, n)).astype(np.int8)
    i_idx = np.array([e[0] for e in edges], dtype=int)
    j_idx = np.array([e[1] for e in edges], dtype=int)
    cuts = (bits[:, i_idx] != bits[:, j_idx]).sum(axis=1)
    return float(cuts.mean() / cmax)


def best_classical_ratio(G: nx.Graph, method: str = "best",
                         sa_runs: int = 20, samples: int = 5000,
                         seed: int = 0) -> float:
    """Classical baseline ratio for a given ``method``.

    Parameters
    ----------
    method:
        ``'GW'`` (Goemans–Williamson guarantee), ``'annealing'`` / ``'SA'``
        (simulated annealing), ``'random'`` (random sampling), or ``'best'``
        (the maximum of all three — the strongest classical baseline).

    Returns
    -------
    float
        The approximation ratio ``<C> / C_max`` achieved by that baseline.
    """
    key = str(method).strip().lower()
    aliases = {
        "gw": "GW", "goemans": "GW", "goemans-williamson": "GW",
        "goemans_williamson": "GW", "sdp": "GW",
        "sa": "annealing", "annealing": "annealing", "simulated_annealing": "annealing",
        "simulated annealing": "annealing",
        "random": "random", "sampling": "random", "random_sampling": "random",
        "best": "best", "all": "best", "max": "best",
    }
    which = aliases.get(key)
    if which is None:
        raise ValueError(
            f"Unknown classical method '{method}'. Choose one of {list(CLASSICAL_METHODS)}."
        )
    if which == "GW":
        return goemans_willison_ratio(G)
    if which == "annealing":
        return simulated_annealing_maxcut(G, n_runs=sa_runs, seed=seed)
    if which == "random":
        return random_sampling_ratio(G, n_samples=samples, seed=seed)
    # 'best' → strongest of the three
    return float(max(
        goemans_willison_ratio(G),
        simulated_annealing_maxcut(G, n_runs=sa_runs, seed=seed),
        random_sampling_ratio(G, n_samples=samples, seed=seed),
    ))


def all_baselines(G: nx.Graph, sa_runs: int = 20, samples: int = 5000,
                  seed: int = 0) -> Dict[str, float]:
    """Convenience: every baseline ratio in one dict (for tables / tooltips)."""
    gw = goemans_willison_ratio(G)
    sa = simulated_annealing_maxcut(G, n_runs=sa_runs, seed=seed)
    rnd = random_sampling_ratio(G, n_samples=samples, seed=seed)
    return {"GW": gw, "annealing": sa, "random": rnd, "best": max(gw, sa, rnd)}
