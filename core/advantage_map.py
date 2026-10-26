"""Quantum-advantage boundary computation (pure orchestration, no Streamlit).

This module answers: *for which problem sizes ``n`` and noise levels ``p`` does
QAOA's approximation ratio exceed a classical baseline?*  It produces the data
behind the Advantage Map page — a 2-D grid of ``QAOA_ratio(n, p) −
classical_ratio(n)`` per environment, plus the crossing **boundary** and a
summary **advantage window**.

Cost model & the fixed-angle protocol
-------------------------------------
A noisy (density-matrix) evaluation costs ``O(4ⁿ)`` — measured at ~0.6 s per
point for ``n = 10`` and ~8–24 s for ``n = 12`` — so recomputing a *full*
landscape at every noise level is infeasible.  Instead we use the standard
**fixed-angle protocol**:

1. Tune the QAOA angles ``(γ*, β*)`` **once** on the *ideal* device (fast
   statevector optimisation) for each ``(n, depth, instance)``.
2. Evaluate the approximation ratio at those fixed angles under each
   environment's noise at every level — one cheap noisy evaluation per point.

This mirrors a realistic "tune on a simulator, run on hardware" workflow.  It is
also *accurate* here: for the isotropic (depolarising + burst) space environments
the optimal angle is noise-robust, so the fixed-angle ratio ≈ the noise-adapted
optimum.  (For the anisotropic satellite↔ground link the optimum shifts, so the
fixed-angle value is a conservative lower bound — that environment is not part of
the default map.)

All quantum execution is delegated to the worker subprocess via primitive noise
*specs* (never a constructed ``NoiseModel``), and the classical baselines are pure
numpy/networkx — so nothing here builds a Qiskit object in the caller's process.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from core.classical_baselines import best_classical_ratio
from core.graph_generator import generate_graph, optimal_maxcut
from core.noise_models import ENVIRONMENTS
from core.optimizer import run_optimization
from core.qaoa_evaluator import evaluate_qaoa_exact

#: Density-matrix memory is ``4ⁿ * 16 bytes``; n = 13 already exceeds ~1 GB.
MEMORY_MAX_N = 12

#: Environments offered on the Advantage Map (isotropic → fixed-angle accurate).
MAP_ENVIRONMENTS = ["earth-lab", "leo-satellite", "meo-van-allen", "deep-space"]

#: Friendly labels for the map environments.
ENV_LABELS = {
    "earth-lab": "Earth",
    "leo-satellite": "LEO",
    "meo-van-allen": "MEO",
    "deep-space": "Deep Space",
    "satellite-ground": "Satellite link",
    "ideal": "Ideal",
    "custom": "Custom",
}


def valid_sizes(family: str, n_min: int, n_max: int) -> List[int]:
    """Problem sizes usable for ``family``, capped by density-matrix memory.

    ``3-regular`` needs an even ``n``; other families accept any ``n >= 3``.
    Sizes are also capped at :data:`MEMORY_MAX_N` (noisy ``4ⁿ`` memory) and
    floored at 4.
    """
    n_min = int(max(4, n_min))
    n_max = int(min(MEMORY_MAX_N, n_max))
    if n_max < n_min:
        n_max = n_min
    sizes = list(range(n_min, n_max + 1))
    if family == "3-regular":
        sizes = [n for n in sizes if n % 2 == 0]
        if not sizes:  # e.g. range was a single odd number
            sizes = [n_min - 1 if n_min - 1 >= 4 else n_min + 1]
    return sizes


def _family_kwargs(family: str, n: int, p_er: float, m_ba: int) -> dict:
    if family == "3-regular":
        return {"d": 3}
    if family == "erdos-renyi":
        return {"p": float(p_er)}
    return {"m": int(min(m_ba, max(1, n - 1)))}


def noise_spec_for(environment: str, level: float, elevation: float = 45.0):
    """Primitive, picklable noise descriptor for the worker (no Qiskit built)."""
    if environment == "ideal" or level <= 0.0:
        return None
    params = {"noise_level": float(level)}
    if environment == "satellite-ground":
        params["elevation"] = float(elevation)
    return {"environment": environment, "params": params}


def reference_angles(G, depth: int, seed: int = 0, restarts: int = 3,
                     max_iter: int = 80) -> Tuple[List[float], List[float], float]:
    """Ideal-device optimal QAOA angles ``(gammas, betas, ideal_ratio)``.

    Optimised on the noiseless statevector simulator (fast), then reused across
    every environment/noise level for this instance (the fixed-angle protocol).
    """
    depth = int(max(1, depth))
    opt = run_optimization(
        G, depth=depth, noise_model=None, method="COBYLA",
        max_iter=int(max_iter), n_restarts=int(restarts),
        record_trajectory=False, readout_prob=None, shots=256, seed=int(seed),
    )
    params = np.asarray(opt["best_params"], dtype=float)
    gammas = list(params[:depth])
    betas = list(params[depth:2 * depth])
    return gammas, betas, float(opt["best_ratio"])


def estimate_map_cost(sizes: Sequence[int], depth: int, envs: Sequence[str],
                      noise_levels: Sequence[float], n_instances: int,
                      opt_restarts: int = 3, opt_max_iter: int = 80) -> float:
    """Rough wall-clock estimate (seconds) for :func:`compute_advantage_map`.

    Calibrated against measured single-point costs on the dev sandbox: a noisy
    density-matrix evaluation is ≈ ``0.12 ms * 16^((n-4)/2) * depth``, and the
    ideal reference optimisation is ≈ ``opt_restarts*opt_max_iter`` fast
    statevector evaluations (~3 ms each).
    """
    depth = int(max(1, depth))
    per_eval_ms = lambda n: 0.12 * (16.0 ** ((n - 4) / 2.0)) * depth  # noqa: E731
    ref_s = 0.0
    sweep_s = 0.0
    for n in sizes:
        ref_s += n_instances * (opt_restarts * opt_max_iter * 0.003)
        sweep_s += n_instances * len(envs) * len(noise_levels) * (per_eval_ms(n) / 1000.0)
    return float(ref_s + sweep_s)


def _column_boundary(noise_levels: np.ndarray, diff: np.ndarray) -> Optional[float]:
    """Noise level where ``diff`` first crosses from >0 to <=0 (interpolated).

    Returns ``None`` when there is no interior crossing:
      * ``float('inf')``  → advantage at every level in range (diff > 0 throughout),
      * ``None``          → no advantage at any level (diff <= 0 from the start).
    """
    diff = np.asarray(diff, dtype=float)
    if len(diff) == 0:
        return None
    if diff[0] <= 0.0:
        return None                       # never ahead, even at the lowest noise
    if np.all(diff > 0.0):
        return float("inf")               # ahead across the whole range
    # first index where it drops to <= 0, interpolate the zero crossing
    idx = int(np.where(diff <= 0.0)[0][0])
    if idx == 0:
        return None
    y0, y1 = diff[idx - 1], diff[idx]
    x0, x1 = noise_levels[idx - 1], noise_levels[idx]
    if y0 == y1:
        return float(x1)
    t = y0 / (y0 - y1)                     # fraction along [x0, x1]
    return float(x0 + t * (x1 - x0))


def compute_advantage_map(
    family: str,
    sizes: Sequence[int],
    depth: int = 1,
    envs: Sequence[str] = ("earth-lab",),
    baseline: str = "GW",
    noise_levels: Sequence[float] = (0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1),
    n_instances: int = 3,
    seed_base: int = 42,
    p_er: float = 0.5,
    m_ba: int = 2,
    opt_restarts: int = 3,
    opt_max_iter: int = 80,
    shots: int = 256,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Dict:
    """Compute the quantum-advantage map (see module docstring for the method).

    Returns a JSON-serialisable dict with the axes, per-environment QAOA ratio
    curves, the classical baseline per size, the signed ``difference`` grids, the
    crossing ``boundary`` per size, and an ``advantage`` summary per environment.
    """
    sizes = list(sizes)
    for e in envs:
        if e not in ENVIRONMENTS:
            raise ValueError(
                f"Unknown environment '{e}'. Choose from {list(ENVIRONMENTS)}.")
    envs = list(envs)
    noise_levels = [float(x) for x in noise_levels]
    n_instances = int(max(1, n_instances))
    depth = int(max(1, depth))
    nP = len(noise_levels)

    # total work units: one reference optimisation + (envs*levels) noisy evals per instance
    total = len(sizes) * n_instances * (1 + len(envs) * nP)
    done = 0

    def tick():
        nonlocal done
        done += 1
        if progress_callback is not None:
            progress_callback(done, total)

    # per-size accumulators
    qaoa = {e: {n: np.zeros((n_instances, nP), dtype=float) for n in sizes} for e in envs}
    classical = {n: np.zeros(n_instances, dtype=float) for n in sizes}
    ideal_ref = {n: np.zeros(n_instances, dtype=float) for n in sizes}
    cmax_acc = {n: np.zeros(n_instances, dtype=float) for n in sizes}

    for n in sizes:
        kw = _family_kwargs(family, n, p_er, m_ba)
        for inst in range(n_instances):
            seed = int(seed_base) + 1013 * inst + n
            G = generate_graph(family, n, seed=seed, **kw)
            cmax, _ = optimal_maxcut(G)
            cmax_acc[n][inst] = float(cmax)
            # 1) ideal-device reference angles (fast statevector optimisation)
            gammas, betas, ideal_ratio = reference_angles(
                G, depth=depth, seed=seed, restarts=opt_restarts, max_iter=opt_max_iter)
            ideal_ref[n][inst] = ideal_ratio
            # classical baseline for this instance (pure; env/noise independent)
            classical[n][inst] = best_classical_ratio(G, baseline, seed=seed)
            tick()
            # 2) fixed-angle ratio under each environment at each noise level
            for e in envs:
                for pi, p in enumerate(noise_levels):
                    spec = noise_spec_for(e, p)
                    r = evaluate_qaoa_exact(
                        G, gammas, betas, noise_model=spec, shots=shots, readout_prob=None,
                    )["approximation_ratio"]
                    qaoa[e][n][inst, pi] = float(r)
                    tick()

    # --- aggregate over instances -------------------------------------------
    classical_mean = {n: float(classical[n].mean()) for n in sizes}
    ideal_mean = {n: float(ideal_ref[n].mean()) for n in sizes}
    cmax_mean = {n: float(cmax_acc[n].mean()) for n in sizes}
    qaoa_mean = {e: {n: qaoa[e][n].mean(axis=0).tolist() for n in sizes} for e in envs}

    # signed difference grid per env: shape (len(sizes), nP) = QAOA - classical
    difference = {}
    boundary = {}
    advantage = {}
    nl = np.asarray(noise_levels, dtype=float)
    for e in envs:
        D = np.zeros((len(sizes), nP), dtype=float)
        bnd = {}
        for si, n in enumerate(sizes):
            q = np.asarray(qaoa_mean[e][n], dtype=float)
            D[si] = q - classical_mean[n]
            b = _column_boundary(nl, D[si])
            bnd[str(n)] = (None if b is None else (float("inf") if b == float("inf") else float(b)))
        difference[e] = D.tolist()
        boundary[e] = bnd
        advantage[e] = summarize_advantage(sizes, noise_levels, D, bnd, e, baseline)

    return {
        "sizes": sizes,
        "noise_levels": noise_levels,
        "envs": envs,
        "depth": depth,
        "family": family,
        "baseline": baseline,
        "n_instances": n_instances,
        "seed_base": int(seed_base),
        "classical": {str(n): classical_mean[n] for n in sizes},
        "ideal_qaoa": {str(n): ideal_mean[n] for n in sizes},
        "cmax": {str(n): cmax_mean[n] for n in sizes},
        "qaoa": {e: {str(n): qaoa_mean[e][n] for n in sizes} for e in envs},
        "difference": difference,
        "boundary": boundary,
        "advantage": advantage,
    }


def summarize_advantage(sizes: Sequence[int], noise_levels: Sequence[float],
                        diff: np.ndarray, boundary: Dict[str, Optional[float]],
                        env: str, baseline: str) -> Dict:
    """Turn a difference grid into a human-readable advantage-window summary."""
    sizes = list(sizes)
    diff = np.asarray(diff, dtype=float)
    # sizes that are ahead at the lowest computed noise level
    ahead_low = [sizes[i] for i in range(len(sizes)) if diff[i, 0] > 0.0]
    # sizes with any advantage somewhere in the range
    ahead_any = [sizes[i] for i in range(len(sizes)) if np.any(diff[i] > 0.0)]

    label = ENV_LABELS.get(env, env)
    if not ahead_any:
        return {
            "any_advantage": False,
            "text": (f"No quantum advantage over **{baseline}** anywhere in the computed "
                     f"range for {label} — classical stays ahead at every size and noise level."),
            "n_range": None, "p_threshold": None, "best_margin": float(diff.max()) if diff.size else 0.0,
        }

    n_lo, n_hi = min(ahead_any), max(ahead_any)
    # tightest finite boundary among advantaged sizes (most restrictive p)
    finite = [b for b in boundary.values() if b is not None and b != float("inf")]
    all_inf = all(boundary[str(n)] == float("inf") for n in ahead_any if str(n) in boundary)
    if all_inf:
        p_txt = f"p ≤ {max(noise_levels):g} (whole range)"
        p_thresh = float(max(noise_levels))
    elif finite:
        p_thresh = float(min(finite))
        p_txt = f"p < {p_thresh:.3f}"
    else:
        p_thresh = None
        p_txt = "only at the lowest noise"

    best_margin = float(diff.max()) if diff.size else 0.0
    n_txt = f"n = {n_lo}" if n_lo == n_hi else f"n ∈ [{n_lo}, {n_hi}]"
    text = f"Quantum wins for **{n_txt}** at **{p_txt}** in **{label}** (vs {baseline})."
    return {
        "any_advantage": True,
        "text": text,
        "n_range": [int(n_lo), int(n_hi)],
        "p_threshold": p_thresh,
        "best_margin": best_margin,
        "ahead_at_low_noise": [int(x) for x in ahead_low],
    }
