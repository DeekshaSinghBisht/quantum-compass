"""Low-level QAOA simulation routines (pure Qiskit/Aer, no Streamlit).

This module holds the actual quantum execution: building the ansatz, running it
on ``AerSimulator`` and post-processing the results.  It is deliberately free of
any dependency on :mod:`core.aer_worker`, :mod:`core.qaoa_evaluator` or
Streamlit so it can be imported cleanly *inside the simulation worker process*
(see :mod:`core.aer_worker`) without import cycles.

Rationale for the worker process
--------------------------------
Qiskit-Aer's C++/OpenMP backend is not safe to re-enter from the rotating
threads that Streamlit uses for each script rerun: running a simulation on a
second rerun can segfault the whole server.  Because dragging the noise slider
triggers a rerun (and therefore a fresh batch of simulations), *all* Aer
execution is delegated to a single long-lived worker subprocess.  The functions
here are what that subprocess calls.
"""

from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator

_MAX_THREADS = max(1, int(os.environ.get("QUANTUM_COMPASS_THREADS", os.cpu_count() or 1)))


# ---------------------------------------------------------------------------
# Pure helpers (no Aer execution) — safe to call from any process/thread
# ---------------------------------------------------------------------------
def cut_values_table(n: int, edges: Sequence) -> np.ndarray:
    """``C[k]`` = number of edges cut by assignment ``k`` for all ``k`` in ``2**n``."""
    ks = np.arange(2 ** n, dtype=np.int64)
    bits = ((ks[:, None] >> np.arange(n)[None, :]) & 1).astype(np.int8)
    if len(edges) == 0:
        return np.zeros(2 ** n, dtype=np.float64)
    i_idx = np.array([e[0] for e in edges])
    j_idx = np.array([e[1] for e in edges])
    return (bits[:, i_idx] != bits[:, j_idx]).sum(axis=1).astype(np.float64)


def brute_optimal_cut(n: int, edges: Sequence) -> int:
    """Exact MaxCut value by enumeration (``n <= 20``; UI caps at 12)."""
    if n == 0:
        return 0
    table = cut_values_table(n, edges)
    return int(table.max())


def cost_operator_from_edges(n: int, edges: Sequence) -> SparsePauliOp:
    """MaxCut cost Hamiltonian ``|E|/2 - (1/2) Σ Z_iZ_j`` as a ``SparsePauliOp``.

    Labels are big-endian (leftmost char = highest qubit), so the ``Z`` for
    qubit ``i`` is placed at position ``n-1-i``.
    """
    labels = ["I" * n]
    coeffs = [float(len(edges)) / 2.0]
    for (i, j) in edges:
        lab = ["I"] * n
        lab[n - 1 - i] = "Z"
        lab[n - 1 - j] = "Z"
        labels.append("".join(lab))
        coeffs.append(-0.5)
    return SparsePauliOp(labels, coeffs)


def apply_readout_channel(probs: np.ndarray, p_read: float, n: int) -> np.ndarray:
    """Apply an independent symmetric readout error to a distribution.

    ``p'(x) = sum_y prod_q T[x_q, y_q] p(y)`` with ``T = [[1-e, e], [e, 1-e]]``,
    implemented as ``n`` successive 2x2 mixes (``O(n 2^n)``).
    """
    if p_read <= 0.0:
        return np.asarray(probs, dtype=float)
    p = np.asarray(probs, dtype=float).reshape([2] * n)
    T = np.array([[1.0 - p_read, p_read], [p_read, 1.0 - p_read]])
    for ax in range(n):
        p = np.tensordot(T, p, axes=([1], [ax]))
        p = np.moveaxis(p, 0, ax)
    return p.reshape(2 ** n)


def readout_prob_from_model(noise_model) -> float:
    """Recover the symmetric readout probability ``P(1|0)`` from a noise model."""
    if noise_model is None:
        return 0.0
    tagged = getattr(noise_model, "_qc_readout_prob", None)
    if tagged is not None:
        return float(tagged)

    def _prob(err):
        try:
            probs = err.to_dict().get("probabilities")
            if probs and len(probs[0]) >= 2:
                return float(probs[0][1])
        except Exception:
            return None
        return None

    default_err = getattr(noise_model, "_default_readout_error", None)
    if default_err is not None:
        p = _prob(default_err)
        if p is not None:
            return p
    local_errs = getattr(noise_model, "_local_readout_errors", None)
    if local_errs:
        try:
            p = _prob(next(iter(local_errs.values())))
            if p is not None:
                return p
        except Exception:
            pass
    return 0.0


# ---------------------------------------------------------------------------
# Circuit construction + backend selection
# ---------------------------------------------------------------------------
def ansatz(n: int, edges: Sequence, gammas: Sequence[float], betas: Sequence[float]) -> QuantumCircuit:
    """QAOA ansatz (no measurement): H^{⊗n}, then per layer RZZ(2γ) + RX(2β)."""
    qc = QuantumCircuit(n, name="qaoa_ansatz")
    qc.h(range(n))
    for gamma, beta in zip(gammas, betas):
        for (i, j) in edges:
            qc.rzz(2.0 * float(gamma), i, j)
        qc.rx(2.0 * float(beta), range(n))
    return qc


def make_backend(noise_model) -> AerSimulator:
    """Statevector when ideal, density_matrix when noisy (Aer's 'automatic'
    method silently ignores noise for state-based save_* instructions)."""
    if noise_model is None:
        return AerSimulator(method="statevector", max_parallel_threads=_MAX_THREADS)
    return AerSimulator(
        method="density_matrix", noise_model=noise_model, max_parallel_threads=_MAX_THREADS
    )


# ---------------------------------------------------------------------------
# Single-point evaluation
# ---------------------------------------------------------------------------
def run_exact(
    n: int,
    edges: Sequence,
    gammas: Sequence[float],
    betas: Sequence[float],
    noise_model,
    shots: int,
    readout: float,
) -> Dict:
    """Deterministic evaluation via the exact output distribution."""
    cut_table = cut_values_table(n, edges)
    qc = ansatz(n, edges, gammas, betas)
    qc.save_probabilities(list(range(n)), label="probs")
    backend = make_backend(noise_model)
    probs = np.asarray(
        backend.run(qc, shots=1).result().data(0)["probs"], dtype=float
    )
    probs = apply_readout_channel(probs, readout, n)
    return _package(n, probs, cut_table, shots)


def run_sampling(
    n: int,
    edges: Sequence,
    gammas: Sequence[float],
    betas: Sequence[float],
    noise_model,
    shots: int,
) -> Dict:
    """Sampling evaluation via counts, normalized to vertex-order bitstrings.

    Aer applies readout error at physical measurement. Qiskit's raw classical
    key order is reversed once before return so all public evaluator results use
    character ``i`` = graph vertex/qubit ``i``.
    """
    qc = ansatz(n, edges, gammas, betas)
    qc.measure_all()
    backend = make_backend(noise_model)
    raw_counts = backend.run(qc, shots=int(shots)).result().get_counts()

    # Qiskit formats classical strings with c[n-1] on the left.  This project
    # measures qubit i into classical bit i and documents *vertex order* as
    # character i = graph vertex i, so normalize once at the worker boundary.
    # Exact and sampled evaluators then expose the same unambiguous schema.
    counts: Dict[str, int] = {}
    for key, val in raw_counts.items():
        raw_bits = bin(int(key, 16))[2:].zfill(n) if isinstance(key, str) and key.startswith("0x") \
            else str(key).replace(" ", "")
        if len(raw_bits) != n or any(bit not in "01" for bit in raw_bits):
            raise ValueError(f"Unexpected measurement key {key!r} for a {n}-qubit circuit.")
        vertex_bits = raw_bits[::-1]
        counts[vertex_bits] = counts.get(vertex_bits, 0) + int(val)

    total = sum(counts.values()) or 1
    expected_cut = 0.0
    best_cut, best_bits = -1, None
    for bits, c in counts.items():
        cv = sum(1 for (i, j) in edges if int(bits[i]) != int(bits[j]))
        expected_cut += cv * c
        if cv > best_cut:
            best_cut, best_bits = cv, bits
    expected_cut /= total
    best_vertex_order = best_bits or ""
    return {
        "expected_cut": float(expected_cut),
        "counts": counts,
        "best_bitstring": best_vertex_order,
    }


def _package(n: int, probs: np.ndarray, cut_table: np.ndarray, shots: int) -> Dict:
    """Turn an exact probability vector into the standard result payload."""
    expected_cut = float(probs @ cut_table)

    raw = probs * int(shots)
    int_counts = np.floor(raw).astype(np.int64)
    remainder = int(shots) - int(int_counts.sum())
    if remainder > 0:
        order = np.argsort(raw - int_counts)[::-1][:remainder]
        int_counts[order] += 1

    counts: Dict[str, int] = {}
    for k in np.nonzero(int_counts)[0]:
        bits = format(int(k), f"0{n}b")[::-1]  # vertex order
        counts[bits] = int(int_counts[k])

    contrib = probs * cut_table
    best_k = int(np.argmax(contrib)) if contrib.max() > 0 else int(np.argmax(probs))
    best_vertex_order = "".join(str((best_k >> i) & 1) for i in range(n))

    return {
        "expected_cut": expected_cut,
        "counts": counts,
        "best_bitstring": best_vertex_order,
    }


# ---------------------------------------------------------------------------
# Landscape sweep (batched; reuses one backend for the whole grid)
# ---------------------------------------------------------------------------
def run_landscape(
    n: int,
    edges: Sequence,
    gammas: np.ndarray,
    betas: np.ndarray,
    depth: int,
    noise_model,
    shots: int,
    readout: float,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> Dict:
    """Sweep the (γ, β) grid with one reused backend; report progress per row."""
    gammas = np.asarray(gammas, dtype=float)
    betas = np.asarray(betas, dtype=float)
    ri, rj = len(gammas), len(betas)
    cut_table = cut_values_table(n, edges)
    optimal_cut = brute_optimal_cut(n, edges)

    backend = make_backend(noise_model)
    ratios = np.zeros((ri, rj), dtype=float)
    expected_cuts = np.zeros((ri, rj), dtype=float)
    total = ri * rj
    denom = float(optimal_cut) if optimal_cut > 0 else 1.0

    done = 0
    for i, gamma in enumerate(gammas):
        for j, beta in enumerate(betas):
            qc = ansatz(n, edges, [gamma] * depth, [beta] * depth)
            qc.save_probabilities(list(range(n)), label="probs")
            probs = np.asarray(
                backend.run(qc, shots=1).result().data(0)["probs"], dtype=float
            )
            probs = apply_readout_channel(probs, readout, n)
            ec = float(probs @ cut_table)
            expected_cuts[i, j] = ec
            ratios[i, j] = ec / denom
            done += 1
        if progress_cb is not None:
            progress_cb(done, total)

    flat_best = int(np.argmax(ratios))
    bi, bj = np.unravel_index(flat_best, ratios.shape)
    return {
        "ratios": ratios,
        "expected_cuts": expected_cuts,
        "optimal_cut": int(optimal_cut),
        "best_ratio": float(ratios[bi, bj]),
        "best_gamma": float(gammas[bi]),
        "best_beta": float(betas[bj]),
    }
