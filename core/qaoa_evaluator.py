"""Single-point QAOA evaluation on the Qiskit Aer simulator.

Two evaluation modes are provided, both giving the *same physics*:

:func:`evaluate_qaoa`
    The spec-facing, **sampling** evaluator.  It runs the measured circuit for
    ``shots`` shots and derives the expected cut from the measurement counts.
    Aer applies gate *and* readout noise automatically when the noise model is
    attached to a ``density_matrix`` run with a ``measure`` instruction.

:func:`evaluate_qaoa_exact`
    A **deterministic** evaluator used for landscapes and optimisation, where
    shot noise would otherwise speckle the surface.  It obtains the exact output
    probability distribution (``save_probabilities``) and computes the
    expectation analytically.  Because ``save_probabilities`` has no ``measure``
    instruction, Aer does *not* apply readout error to it — so the symmetric
    readout channel is applied by hand (validated to match the sampling path
    within shot noise).

Both return the same dictionary schema::

    {
      'approximation_ratio': float,   # expected_cut / optimal_cut
      'expected_cut':        float,   # <C> on the QAOA state
      'optimal_cut':         int,     # brute-force MaxCut value
      'counts':              dict,    # {vertex-order bitstring: occurrences}
      'best_bitstring':      str,     # n chars, vertex order (char i = side of vertex i)
    }

All actual Aer execution is delegated to the persistent worker subprocess in
:mod:`core.aer_worker` (see that module for why).  This module only builds
picklable payloads and assembles the result dict.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import networkx as nx
import numpy as np

from core import _qaoa_sim, aer_worker
from core.graph_generator import optimal_maxcut

# Re-exported pure helpers (no Aer execution) for callers and tests.
cut_values_table = _qaoa_sim.cut_values_table
_apply_readout_channel = _qaoa_sim.apply_readout_channel
_readout_prob_from_model = _qaoa_sim.readout_prob_from_model


def cost_operator(G: nx.Graph):
    """The MaxCut cost Hamiltonian of ``G`` as a ``SparsePauliOp``.

    ``C = |E|/2 - (1/2) * sum_{(i,j) in E} Z_i Z_j``

    Its expectation on a state equals the expected number of cut edges.  Qiskit
    ``SparsePauliOp`` labels are big-endian (leftmost char = highest qubit), so
    the ``Z`` for qubit ``i`` sits at position ``n-1-i``.
    """
    return _qaoa_sim.cost_operator_from_edges(G.number_of_nodes(), list(G.edges()))


def _finalize(payload: Dict, G: nx.Graph) -> Dict:
    """Attach optimal_cut + approximation_ratio to a worker result payload."""
    optimal_cut, _ = optimal_maxcut(G)
    expected_cut = float(payload["expected_cut"])
    ratio = float(expected_cut / optimal_cut) if optimal_cut > 0 else 0.0
    return {
        "approximation_ratio": ratio,
        "expected_cut": expected_cut,
        "optimal_cut": int(optimal_cut),
        "counts": payload["counts"],
        "best_bitstring": payload["best_bitstring"],
    }


def evaluate_qaoa(
    G: nx.Graph,
    gammas: Sequence[float],
    betas: Sequence[float],
    noise_model=None,
    shots: int = 4096,
) -> Dict:
    """Sampling evaluator (spec-facing).

    Builds the measured QAOA circuit, runs it on ``AerSimulator`` for ``shots``
    shots (in the worker subprocess), and computes the expected cut from the
    measurement counts: ``<C> = sum_bitstring cut(bitstring) * count / shots``.

    When ``noise_model`` is not ``None`` the backend is forced to
    ``method='density_matrix'`` — Aer's ``automatic`` method silently discards
    noise for these circuits, which would make the noise slider a no-op.
    """
    n = G.number_of_nodes()
    edges = list(G.edges())
    payload = aer_worker.sampling(
        n, edges, [float(g) for g in gammas], [float(b) for b in betas],
        noise_model, int(shots),
    )
    return _finalize(payload, G)


def evaluate_qaoa_exact(
    G: nx.Graph,
    gammas: Sequence[float],
    betas: Sequence[float],
    noise_model=None,
    shots: int = 4096,
    readout_prob: Optional[float] = None,
) -> Dict:
    """Deterministic evaluator (exact output distribution, no shot noise).

    Uses ``save_probabilities`` to obtain the exact measurement distribution
    (``statevector`` when ideal, ``density_matrix`` when noisy), applies the
    readout channel by hand, and computes the expected cut analytically.  The
    returned ``counts`` are the exact probabilities scaled to ``shots`` (rounded
    to integers), so downstream code expecting a counts dict keeps working.
    """
    n = G.number_of_nodes()
    edges = list(G.edges())
    # readout_prob may be None: the worker resolves it from the (rebuilt) model.
    # Resolving it here would require a constructed NoiseModel in the parent,
    # which segfaults under Streamlit reruns.
    payload = aer_worker.exact(
        n, edges, [float(g) for g in gammas], [float(b) for b in betas],
        noise_model, int(shots), readout_prob,
    )
    return _finalize(payload, G)
