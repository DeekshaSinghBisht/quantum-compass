"""Classical optimisation of the QAOA angles.

The QAOA variational loop needs a classical optimiser to search the angle
space for parameters that maximise the approximation ratio.  This module wraps
:func:`scipy.optimize.minimize` with:

* several methods (``COBYLA`` by default — derivative-free, the standard choice
  for noisy quantum objectives — plus ``Nelder-Mead``, ``Powell`` and ``L-BFGS-B``),
* **multiple random restarts** to escape the local minima the landscape view
  makes visible,
* optional **trajectory recording** so the UI can draw the optimiser's path
  over the landscape.

The objective is the *negative* approximation ratio (scipy minimises), evaluated
with the deterministic exact evaluator so the optimiser sees a smooth function
rather than shot noise.  For depth ``p`` the parameter vector is
``[gamma_1, ..., gamma_p, beta_1, ..., beta_p]``.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from scipy.optimize import minimize

from core.qaoa_evaluator import evaluate_qaoa_exact

_SUPPORTED_METHODS = ("COBYLA", "Nelder-Mead", "Powell", "L-BFGS-B")

# Angle search domains (one full physical period).
_GAMMA_DOMAIN = (0.0, 2.0 * np.pi)
_BETA_DOMAIN = (0.0, np.pi)


def _wrap(params: np.ndarray, depth: int) -> Tuple[np.ndarray, np.ndarray]:
    """Split a flat parameter vector into (gammas, betas), each length ``depth``."""
    params = np.asarray(params, dtype=float)
    gammas = params[:depth]
    betas = params[depth : 2 * depth]
    return gammas, betas


def run_optimization(
    G: nx.Graph,
    depth: int = 1,
    noise_model=None,
    method: str = "COBYLA",
    max_iter: int = 200,
    n_restarts: int = 5,
    record_trajectory: bool = True,
    readout_prob: Optional[float] = None,
    shots: int = 2048,
    seed: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Dict:
    """Run multi-start optimisation of the QAOA angles on graph ``G``.

    Parameters
    ----------
    G:
        Problem graph.
    depth:
        QAOA depth ``p`` (parameter vector has length ``2p``).
    noise_model:
        Aer noise model or ``None`` for ideal.
    method:
        One of ``'COBYLA'``, ``'Nelder-Mead'``, ``'Powell'``, ``'L-BFGS-B'``.
    max_iter:
        Maximum iterations / function evaluations per restart.
    n_restarts:
        Number of independent random restarts; the best result is returned.
    record_trajectory:
        If ``True``, record ``(gamma, beta, ratio)`` at every objective call of
        the best restart (for ``depth == 1``) so the UI can overlay the path.
    readout_prob:
        Symmetric readout probability for the exact evaluator (noisy case).
    shots:
        Shot budget passed to the evaluator (only scales reported counts).
    seed:
        RNG seed for reproducible restarts.
    progress_callback:
        Optional ``callback(done_restarts, n_restarts)`` for a UI progress bar.

    Returns
    -------
    dict with keys
        * ``best_params``      – ``np.ndarray`` of length ``2p`` (gammas then betas),
        * ``best_ratio``       – the best approximation ratio found,
        * ``best_gamma`` / ``best_beta`` – first-layer angles (handy for ``p=1`` plots),
        * ``trajectory``       – list of ``(gamma, beta, ratio)`` for ``depth==1``
                                 (or ``(params_tuple, ratio)`` for ``depth>1``),
        * ``all_trajectories`` – list of per-restart trajectories,
        * ``method`` / ``n_restarts`` / ``nfev`` – metadata.
    """
    if method not in _SUPPORTED_METHODS:
        raise ValueError(
            f"Unsupported optimisation method '{method}'. "
            f"Choose one of {list(_SUPPORTED_METHODS)}."
        )
    depth = int(max(1, depth))
    n_restarts = int(max(1, n_restarts))
    rng = np.random.default_rng(seed)

    def objective(params: np.ndarray) -> float:
        gammas, betas = _wrap(params, depth)
        res = evaluate_qaoa_exact(
            G,
            list(gammas),
            list(betas),
            noise_model=noise_model,
            shots=shots,
            readout_prob=readout_prob,
        )
        return -float(res["approximation_ratio"])  # minimise the negative ratio

    options: Dict = {}
    if method == "COBYLA":
        options = {"maxiter": int(max_iter), "rhobeg": 0.5}
    elif method in ("Nelder-Mead", "Powell"):
        options = {"maxiter": int(max_iter), "maxfev": int(max_iter)}
    else:  # L-BFGS-B
        options = {"maxiter": int(max_iter), "maxfun": int(max_iter)}

    bounds = [(_GAMMA_DOMAIN[0], _GAMMA_DOMAIN[1])] * depth + [
        (_BETA_DOMAIN[0], _BETA_DOMAIN[1])
    ] * depth

    best_ratio = -np.inf
    best_params: Optional[np.ndarray] = None
    best_trajectory: List = []
    all_trajectories: List[List] = []
    total_nfev = 0

    for r in range(n_restarts):
        x0 = np.concatenate(
            [
                rng.uniform(_GAMMA_DOMAIN[0], _GAMMA_DOMAIN[1], depth),
                rng.uniform(_BETA_DOMAIN[0], _BETA_DOMAIN[1], depth),
            ]
        )

        traj: List = []

        def _objective_recording(params: np.ndarray, _traj=traj) -> float:
            val = objective(params)
            if record_trajectory:
                gammas, betas = _wrap(params, depth)
                ratio = -val
                if depth == 1:
                    _traj.append((float(gammas[0]), float(betas[0]), float(ratio)))
                else:
                    _traj.append((tuple(map(float, gammas)), tuple(map(float, betas)), float(ratio)))
            return val

        res = minimize(
            _objective_recording,
            x0,
            method=method,
            bounds=bounds if method == "L-BFGS-B" else None,
            options=options,
        )
        nfev = int(getattr(res, "nfev", 0) or 0)
        total_nfev += nfev

        ratio = -float(res.fun)
        all_trajectories.append(traj)
        if ratio > best_ratio:
            best_ratio = ratio
            best_params = np.asarray(res.x, dtype=float)
            best_trajectory = traj

        if progress_callback is not None:
            progress_callback(r + 1, n_restarts)

    if best_params is None:  # pragma: no cover - defensive
        best_params = np.zeros(2 * depth)

    gammas, betas = _wrap(best_params, depth)
    return {
        "best_params": best_params,
        "best_ratio": float(best_ratio),
        "best_gamma": float(gammas[0]),
        "best_beta": float(betas[0]),
        "trajectory": best_trajectory,
        "all_trajectories": all_trajectories,
        "method": method,
        "n_restarts": n_restarts,
        "nfev": total_nfev,
    }
