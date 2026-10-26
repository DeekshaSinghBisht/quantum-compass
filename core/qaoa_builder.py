"""QAOA ansatz construction for MaxCut.

For a graph ``G`` with ``n`` vertices the QAOA state at depth ``p`` is

.. math::

    |\\psi(\\gamma,\\beta)\\rangle
      = \\prod_{l=1}^{p} U_B(\\beta_l)\\, U_C(\\gamma_l)\\; |+\\rangle^{\\otimes n}

with the **cost unitary** (one ``RZZ`` per edge)

.. math::

    U_C(\\gamma_l) = \\exp\\!\\big(-i\\gamma_l \\sum_{(i,j)\\in E} Z_i Z_j\\big)

and the **mixer unitary** (one ``RX`` per qubit)

.. math::

    U_B(\\beta_l) = \\exp\\!\\big(-i\\beta_l \\sum_{i} X_i\\big).

Qiskit's ``rzz(theta)`` realises ``exp(-i theta/2 Z⊗Z)`` and ``rx(theta)``
realises ``exp(-i theta/2 X)``, so the physical angles are ``theta = 2*gamma``
and ``theta = 2*beta`` — exactly as required by the spec.

The MaxCut objective for a bitstring ``x`` is
``C(x) = sum_{(i,j) in E} (1 - z_i z_j)/2`` with ``z_i = (-1)^{x_i}``, whose
expectation on the QAOA state equals ``|E|/2 - (1/2) sum_{(i,j)} <Z_i Z_j>``.
"""

from __future__ import annotations

from typing import Optional, Sequence

import networkx as nx
from qiskit import QuantumCircuit


def build_qaoa_circuit(
    G: nx.Graph,
    gammas: Sequence[float],
    betas: Sequence[float],
    noise_model=None,
    measure: bool = True,
) -> QuantumCircuit:
    """Build the depth-``p`` QAOA circuit for MaxCut on ``G``.

    Parameters
    ----------
    G:
        The problem graph; vertex ``i`` becomes qubit ``i``.
    gammas:
        Cost-layer angles ``[gamma_1, ..., gamma_p]``.
    betas:
        Mixer-layer angles ``[beta_1, ..., beta_p]``.
    noise_model:
        Optional Aer ``NoiseModel``.  Noise is applied by the *backend* at
        simulation time, not baked into the circuit; the argument is accepted
        (and recorded in the circuit metadata) so that callers can build and
        simulate in one call and so the signature matches the spec.
    measure:
        If ``True`` (default) append a full computational-basis measurement.
        Set to ``False`` when you want to attach Aer ``save_*`` instructions to
        the pure state instead (used by the exact evaluators).

    Returns
    -------
    QuantumCircuit
        The QAOA ansatz on ``n`` qubits and ``n`` classical bits (when
        ``measure`` is ``True``).

    Structure
    ---------
    ::

        H on all qubits
        for l in range(p):
            for (i, j) in edges:  rzz(2*gamma[l], i, j)      # cost layer
            for q in range(n):    rx(2*beta[l], q)           # mixer layer
        measure_all()
    """
    gammas = list(gammas)
    betas = list(betas)
    if len(gammas) != len(betas):
        raise ValueError(
            f"gammas and betas must have equal length (got {len(gammas)} and "
            f"{len(betas)}); both must equal the QAOA depth p."
        )

    n = G.number_of_nodes()
    if n == 0:
        raise ValueError("Cannot build a QAOA circuit for an empty graph.")

    edges = list(G.edges())

    qc = QuantumCircuit(n, n if measure else 0, name=f"qaoa_p{len(gammas)}")

    # --- initial superposition -------------------------------------------
    qc.h(range(n))

    # --- p alternating cost / mixer layers --------------------------------
    for gamma, beta in zip(gammas, betas):
        # Cost layer: RZZ(2*gamma) on every edge.
        for (i, j) in edges:
            qc.rzz(2.0 * float(gamma), i, j)
        # Mixer layer: RX(2*beta) on every qubit.
        qc.rx(2.0 * float(beta), range(n))

    if measure:
        qc.measure(range(n), range(n))

    if noise_model is not None:
        # Metadata only — the simulator applies the actual noise.
        qc.metadata = dict(qc.metadata or {})
        qc.metadata["noise_model"] = repr(noise_model)

    return qc


def ansatz_circuit(
    G: nx.Graph,
    gammas: Sequence[float],
    betas: Sequence[float],
) -> QuantumCircuit:
    """Convenience wrapper: the QAOA ansatz *without* measurement.

    Used by the exact evaluators, which attach ``save_probabilities`` /
    ``save_expectation_value`` instructions to the final state.
    """
    return build_qaoa_circuit(G, gammas, betas, noise_model=None, measure=False)
