"""QAOA optimisation-landscape computation.

For QAOA at depth ``p = 1`` the variational parameters are a single pair
``(gamma, beta)`` and the landscape is the approximation ratio

    r(gamma, beta) = <C(gamma, beta)> / C_max

evaluated over a rectangular grid ``gamma in [0, 2*pi)``, ``beta in [0, pi)``.
This module sweeps that grid with the deterministic evaluator
(:func:`core.qaoa_evaluator.evaluate_qaoa_exact`) so the surface is smooth —
no shot noise speckle — and reports the grid, the ratio surface, and the
locations of local extrema.

For depth ``p > 1`` the full landscape is ``2p``-dimensional and cannot be
shown on a 2-D grid; as a standard visualisation slice we reuse the *same*
``(gamma, beta)`` for every layer (a symmetric "fixed-angle" cut through the
parameter space).  The optimiser in :mod:`core.optimizer`, by contrast, always
optimises the full ``2p`` parameters.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from core import aer_worker


def compute_landscape(
    G: nx.Graph,
    depth: int = 1,
    resolution: int = 30,
    noise_model=None,
    shots: int = 2048,
    readout_prob: Optional[float] = None,
    gamma_range: Tuple[float, float] = (0.0, 2.0 * np.pi),
    beta_range: Tuple[float, float] = (0.0, np.pi),
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, np.ndarray]:
    """Sweep the ``(gamma, beta)`` grid and evaluate the QAOA ratio everywhere.

    Parameters
    ----------
    G:
        Problem graph (vertex ``i`` = qubit ``i``).
    depth:
        QAOA depth ``p``.  For ``p > 1`` the same ``(gamma, beta)`` is applied
        to every layer (a 2-D symmetric slice of the true ``2p``-D landscape).
    resolution:
        Number of grid points per axis (``resolution x resolution`` total).
    noise_model:
        Aer ``NoiseModel`` or ``None`` for the ideal device.
    shots:
        Shot budget used only to scale the returned ``counts``; the ratio
        surface itself is exact (deterministic), independent of ``shots``.
    readout_prob:
        Symmetric readout probability to apply on the exact path (Aer does not
        apply readout error to ``save_probabilities``).  Ignored when ideal.
    gamma_range, beta_range:
        Sweep intervals.  Defaults are one full period of each angle.
    progress_callback:
        Optional ``callback(done, total)`` invoked after each grid row so the UI
        can drive a progress bar.

    Returns
    -------
    dict with keys
        * ``gammas``  – 1-D array of length ``resolution`` (gamma axis),
        * ``betas``   – 1-D array of length ``resolution`` (beta axis),
        * ``ratios``  – 2-D array ``ratios[i, j]`` at ``gammas[i], betas[j]``,
        * ``expected_cuts`` – 2-D array of ``<C>`` values,
        * ``optimal_cut``   – brute-force MaxCut value (scalar),
        * ``best_ratio``    – global max of ``ratios``,
        * ``best_gamma`` / ``best_beta`` – grid location of the global max,
        * ``depth`` / ``resolution`` – echoed for the UI/cache key.
    """
    if resolution < 2:
        raise ValueError("resolution must be >= 2 to form a grid.")
    depth = int(max(1, depth))

    gammas = np.linspace(gamma_range[0], gamma_range[1], resolution, endpoint=False)
    betas = np.linspace(beta_range[0], beta_range[1], resolution, endpoint=False)

    n = G.number_of_nodes()
    edges = list(G.edges())

    # One batched round-trip to the worker; progress streams back per grid row.
    # noise_model may be None, a primitive spec dict, or a real NoiseModel;
    # readout_prob may be None (the worker resolves it from the rebuilt model).
    # Nothing qiskit is constructed here, so this is safe under Streamlit reruns.
    out = aer_worker.landscape(
        n, edges, gammas, betas, depth, noise_model, int(shots), readout_prob,
        progress_cb=progress_callback,
    )

    ratios = np.asarray(out["ratios"], dtype=float)
    expected_cuts = np.asarray(out["expected_cuts"], dtype=float)
    optimal_cut = int(out["optimal_cut"])

    flat_best = int(np.argmax(ratios))
    bi, bj = np.unravel_index(flat_best, ratios.shape)

    return {
        "gammas": gammas,
        "betas": betas,
        "ratios": ratios,
        "expected_cuts": expected_cuts,
        "optimal_cut": optimal_cut,
        "best_ratio": float(ratios[bi, bj]),
        "best_gamma": float(gammas[bi]),
        "best_beta": float(betas[bj]),
        "depth": depth,
        "resolution": int(resolution),
    }


def _interior_extrema(
    values: np.ndarray,
    gammas: np.ndarray,
    betas: np.ndarray,
    kind: str,
) -> List[Dict[str, float]]:
    """Find strict interior grid extrema by comparing to the 8 neighbours.

    ``kind`` is ``'min'`` or ``'max'``.  Returns a list of dicts with the grid
    indices, the angle coordinates and the value at each extremum, sorted by
    value (best first).
    """
    ni, nj = values.shape
    found: List[Dict[str, float]] = []
    # Interior points only; the boundary is periodic-ish but not guaranteed.
    for i in range(1, ni - 1):
        for j in range(1, nj - 1):
            window = values[i - 1 : i + 2, j - 1 : j + 2]
            centre = values[i, j]
            others = np.delete(window.ravel(), 4)  # drop the centre
            if kind == "min":
                is_ext = centre <= others.min() and (centre < others).any()
            else:
                is_ext = centre >= others.max() and (centre > others).any()
            if is_ext:
                found.append(
                    {
                        "i": int(i),
                        "j": int(j),
                        "gamma": float(gammas[i]),
                        "beta": float(betas[j]),
                        "value": float(centre),
                    }
                )
    found.sort(key=lambda d: d["value"], reverse=(kind == "max"))
    return found


def detect_local_minima(
    ratios: np.ndarray,
    gammas: np.ndarray,
    betas: np.ndarray,
) -> List[Dict[str, float]]:
    """Local minima of the ratio surface (grid points lower than all neighbours).

    These are the "traps" a classical optimiser can fall into — the whole point
    of the Compass visualisation.
    """
    return _interior_extrema(np.asarray(ratios), gammas, betas, "min")


def detect_local_maxima(
    ratios: np.ndarray,
    gammas: np.ndarray,
    betas: np.ndarray,
) -> List[Dict[str, float]]:
    """Local maxima of the ratio surface (grid points higher than all neighbours).

    The global maximum is the best achievable ratio on the grid; the remaining
    local maxima are secondary peaks the optimiser might settle on.
    """
    return _interior_extrema(np.asarray(ratios), gammas, betas, "max")
