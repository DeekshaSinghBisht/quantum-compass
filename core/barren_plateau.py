"""Barren-plateau phase-diagram engine for Quantum Compass.

The page backed by this module asks a different question from approximation-ratio
performance: *is the QAOA objective still shaped enough for a classical optimizer
to navigate it?*  Each cell of the diagram is a two-dimensional ``(gamma, beta)``
landscape.  Its trainability is measured with three complementary shape signals:

* variance of the approximation-ratio surface,
* largest gradient magnitude on the normalized grid,
* total approximation-ratio range.

A flat surface has all three signals near zero and is therefore barren.  A surface
with appreciable relief and slopes is trainable.  The implementation deliberately
uses the deterministic exact landscape evaluator rather than shot-noisy samples:
otherwise Monte-Carlo speckle would masquerade as trainability.

Practical simulation protocol
-----------------------------
A literal noisy density-matrix sweep at every cell through ``n=16`` is not
possible on a small workstation: each noisy state needs ``O(4**n)`` memory, and
``n=16`` alone needs tens of gigabytes.  The diagram therefore always computes a
real *ideal* landscape for every requested ``(n, depth, graph)``.  In ``auto``
mode it additionally computes an exact noisy landscape for safe small instances
(default ``n <= 8``).  Larger noisy cells use a documented **contrast-transfer
model**: the ideal surface is contracted according to the environment's pure
channel parameters, gate exposure, and a finite-size/depth concentration factor.

This is not a fake random surface: every cell starts from an evaluated QAOA
landscape, the contraction preserves its extrema and topology, and the exact
small-n cells anchor the environment effect.  It lets the page display the full
``n=4..16, p=1..8`` phase diagram without attempting an out-of-memory simulation.
The result records which cells were exact and which were transferred so the UI and
exports remain explicit about the protocol.

No Qiskit object is constructed in this module.  ``compute_landscape`` delegates
all Aer execution to the existing persistent worker subprocess, while the channel
strength extraction uses only parent-safe numeric helpers from ``noise_models``.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from core.graph_generator import GRAPH_FAMILIES, generate_graph
from core.landscape import compute_landscape
from core.noise_models import ENVIRONMENTS, REPRESENTATIVE_LEVELS, effective_params

# Public labels used by the Streamlit page and JSON exports.
TRAINABILITY_CLASSES: Tuple[str, ...] = ("BARREN", "MIXED", "TRAINABLE", "RESONANT")

# A density-matrix state is 16 * 4**n bytes.  Above n=12 this exceeds the
# practical memory budget of the app; exact mode protects callers from that.
EXACT_NOISE_MEMORY_MAX_N = 12
DEFAULT_EXACT_NOISE_MAX_N = 8

# Score normalisation scales.  They were selected from smooth, exact QAOA
# landscapes (ratios live in [0, 1]), not from noisy sampling output.  Exponential
# saturation makes the score bounded and avoids one unusually steep point
# dominating a cell.
_VARIANCE_SCALE = 0.010
_GRADIENT_SCALE = 1.50
_RANGE_SCALE = 0.25

# Finite-size/depth concentration rate.  This is applied to the *contrast* of an
# evaluated landscape, not to an objective value.  It captures the fact that a
# low-dimensional symmetric slice otherwise understates the concentration seen in
# the full 2p-dimensional parameter landscape.
_PLATEAU_CONCENTRATION_RATE = 0.018

# Gate-exposure coefficient for contrast transfer when an exact density-matrix
# landscape is too large.  Channel strengths come from effective_params().
_NOISE_EXPOSURE_RATE = 1.0


# ---------------------------------------------------------------------------
# Landscape metrics and labels
# ---------------------------------------------------------------------------
def _finite_surface(ratios_2d) -> np.ndarray:
    """Return a finite 2-D float surface, replacing non-finite entries safely."""
    values = np.asarray(ratios_2d, dtype=float)
    if values.ndim != 2:
        raise ValueError("ratios_2d must be a two-dimensional landscape array.")
    if values.size == 0:
        raise ValueError("ratios_2d must not be empty.")
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values, dtype=float)
    # A failed single point should not turn an otherwise meaningful landscape
    # into NaNs.  Replacing it with the finite mean is neutral for the metrics.
    return np.nan_to_num(values, nan=float(finite.mean()),
                         posinf=float(finite.max()), neginf=float(finite.min()))


def landscape_metrics(ratios_2d) -> Dict[str, float]:
    """Compute the three trainability metrics and their normalized components.

    Gradients are evaluated on coordinates normalized to ``[0, 1]`` along each
    axis.  That makes ``max_gradient`` comparable between a 9x9 exploratory grid
    and a 12x12 production grid rather than artificially shrinking it when the
    resolution rises.

    Returns a JSON-friendly mapping with ``variance``, ``max_gradient``,
    ``ratio_range``, three component scores, and the weighted ``score``.
    """
    surface = _finite_surface(ratios_2d)
    variance = float(np.var(surface))
    ratio_range = float(np.max(surface) - np.min(surface))

    if min(surface.shape) < 2:
        max_gradient = 0.0
    else:
        row_step = 1.0 / float(max(1, surface.shape[0] - 1))
        col_step = 1.0 / float(max(1, surface.shape[1] - 1))
        drow, dcol = np.gradient(surface, row_step, col_step)
        max_gradient = float(np.max(np.hypot(drow, dcol)))

    # Remove IEEE round-off residue (e.g. var(constant 0.62) ~= 1e-32) so an
    # exactly flat landscape reports exactly-zero physical metrics.
    if abs(variance) < 1e-15:
        variance = 0.0
    if abs(ratio_range) < 1e-15:
        ratio_range = 0.0
    if abs(max_gradient) < 1e-15:
        max_gradient = 0.0

    # ``1-exp(-x/scale)`` is monotone, maps exactly-flat terrain to zero, and
    # asymptotically maps strongly structured terrain to one.
    variance_component = float(1.0 - np.exp(-max(0.0, variance) / _VARIANCE_SCALE))
    gradient_component = float(1.0 - np.exp(-max(0.0, max_gradient) / _GRADIENT_SCALE))
    range_component = float(1.0 - np.exp(-max(0.0, ratio_range) / _RANGE_SCALE))
    score = float(np.clip(
        0.40 * variance_component + 0.35 * gradient_component + 0.25 * range_component,
        0.0,
        1.0,
    ))
    return {
        "variance": variance,
        "max_gradient": max_gradient,
        "ratio_range": ratio_range,
        "variance_component": variance_component,
        "gradient_component": gradient_component,
        "range_component": range_component,
        "score": score,
    }


def compute_trainability_score(ratios_2d) -> float:
    """Return a bounded ``0..1`` trainability score for a 2-D QAOA landscape.

    The score combines landscape variance (40%), maximum normalized-grid gradient
    magnitude (35%), and approximation-ratio range (25%).  A constant landscape
    returns exactly ``0``; a high-relief landscape saturates toward ``1``.
    """
    return float(landscape_metrics(ratios_2d)["score"])


def _has_resonant_structure(variance: float, max_grad: float) -> bool:
    """Whether an intermediate-score terrain retains a real structured island.

    The two thresholds operate on the same metrics as :func:`landscape_metrics`.
    A score alone is not enough: a nearly flat surface can land in the middle
    range due to one broad tilt, while a resonant island needs both relief and a
    local slope.  ``compute_barren_diagram`` additionally requires evidence that
    noise created/increased this structure before promoting a cell to RESONANT.
    """
    variance_component = 1.0 - np.exp(-max(0.0, float(variance)) / _VARIANCE_SCALE)
    gradient_component = 1.0 - np.exp(-max(0.0, float(max_grad)) / _GRADIENT_SCALE)
    return bool(variance_component >= 0.22 and gradient_component >= 0.35)


def classify_trainability(score: float, variance: float, max_grad: float) -> str:
    """Classify a trainability score as TRAINABLE, MIXED, BARREN or RESONANT.

    The requested primary thresholds are deliberately strict:

    * ``score > 0.6``  -> ``TRAINABLE``;
    * ``score < 0.1``  -> ``BARREN``;
    * an intermediate ``score > 0.3`` with both variance and gradient structure
      -> ``RESONANT`` candidate;
    * everything else -> ``MIXED``.

    At this low-level API there is no environment/reference-landscape argument,
    so ``RESONANT`` means a *structured noise-adjusted candidate*.  The full
    diagram engine checks a noisy-vs-ideal morphology comparison before retaining
    that label in its output grid; unsupported candidates are conservatively
    reported as ``MIXED``.
    """
    score = float(np.clip(score, 0.0, 1.0))
    if score > 0.6:
        return "TRAINABLE"
    if score < 0.1:
        return "BARREN"
    if score > 0.3 and _has_resonant_structure(variance, max_grad):
        return "RESONANT"
    return "MIXED"


# ---------------------------------------------------------------------------
# Input and protocol helpers (all parent-safe; no Qiskit construction)
# ---------------------------------------------------------------------------
def _axis_values(values: Union[Sequence[int], range, Tuple[int, int]], name: str,
                 minimum: int, maximum: Optional[int] = None) -> List[int]:
    """Normalise an inclusive ``(start, stop)`` tuple or explicit axis sequence."""
    if isinstance(values, tuple) and len(values) == 2:
        lo, hi = int(values[0]), int(values[1])
        if hi < lo:
            raise ValueError(f"{name} range must be increasing, got {values!r}.")
        out = list(range(lo, hi + 1))
    else:
        out = [int(x) for x in values]
    # Preserve caller order while dropping duplicates.
    out = list(dict.fromkeys(out))
    if not out:
        raise ValueError(f"{name} must contain at least one value.")
    if any(x < minimum for x in out):
        raise ValueError(f"{name} values must be >= {minimum}, got {out}.")
    if maximum is not None and any(x > maximum for x in out):
        raise ValueError(f"{name} values must be <= {maximum}, got {out}.")
    return out


def diagram_sizes(family: str, n_range: Union[Sequence[int], range, Tuple[int, int]]) -> List[int]:
    """Return valid diagram sizes, respecting 3-regular parity and n<=16."""
    if family not in GRAPH_FAMILIES:
        raise ValueError(f"Unknown graph family '{family}'. Choose from {list(GRAPH_FAMILIES)}.")
    sizes = _axis_values(n_range, "n_range", minimum=4, maximum=16)
    if family == "3-regular":
        sizes = [n for n in sizes if n % 2 == 0]
    if not sizes:
        raise ValueError("3-regular graphs require at least one even n in n_range.")
    return sizes


def diagram_depths(p_range: Union[Sequence[int], range, Tuple[int, int]]) -> List[int]:
    """Return positive QAOA depths from a range tuple or an explicit sequence."""
    return _axis_values(p_range, "p_range", minimum=1)


def _family_kwargs(family: str, n: int, p_er: float, m_ba: int) -> Dict[str, Union[float, int]]:
    if family == "3-regular":
        return {"d": 3}
    if family == "erdos-renyi":
        return {"p": float(p_er)}
    return {"m": int(min(max(1, m_ba), n - 1))}


def _canonical_noise(noise_model) -> Tuple[object, str, Dict, float, bool]:
    """Make a worker-safe noise descriptor and a pure effective-strength scalar.

    Returns ``(descriptor, label, params, strength, has_noise)``.  Strings and
    ``{'environment', 'params'}`` descriptors are the preferred API because they
    keep Aer/Qiskit construction inside the worker.  A prebuilt NoiseModel is
    accepted for CLI/test compatibility; its physical strength cannot be
    inspected safely here, so transfer cells use a conservative estimate.
    """
    if noise_model is None:
        return None, "Ideal", {}, 0.0, False

    if isinstance(noise_model, str):
        environment = noise_model
        if environment not in ENVIRONMENTS:
            raise ValueError(f"Unknown environment '{environment}'. Choose from {list(ENVIRONMENTS)}.")
        if environment == "ideal":
            return None, "Ideal", {}, 0.0, False
        params: Dict = {"noise_level": float(REPRESENTATIVE_LEVELS[environment])}
        descriptor = {"environment": environment, "params": params}
    elif isinstance(noise_model, dict) and "environment" in noise_model:
        environment = str(noise_model["environment"])
        if environment not in ENVIRONMENTS:
            raise ValueError(f"Unknown environment '{environment}'. Choose from {list(ENVIRONMENTS)}.")
        params = dict(noise_model.get("params") or {})
        if environment == "ideal":
            return None, "Ideal", {}, 0.0, False
        if environment != "custom" and "noise_level" not in params:
            params["noise_level"] = float(REPRESENTATIVE_LEVELS[environment])
        descriptor = {"environment": environment, "params": params}
    else:
        # The existing worker can serialize real Qiskit NoiseModels.  Avoid
        # introspection/construction in this parent process and use a documented
        # approximate strength for any large transfer cells.
        return noise_model, "Custom NoiseModel", {}, 0.010, True

    effective = effective_params(environment, params)
    # RZZ errors are the dominant MaxCut-QAOA channel; bursts, damping and
    # dephasing add exposure-weighted contributions. Thermal modulation is
    # already folded into p1/p2 by effective_params().
    strength = (
        0.40 * float(effective["p1"]) +
        0.80 * float(effective["p2"]) +
        0.10 * float(effective["readout"]) +
        0.20 * float(effective["burst_rate"]) +
        0.60 * float(effective["ad_rate"]) +
        0.25 * float(effective["pd_rate"]) +
        0.15 * float(effective["n_th"])
    )
    strength = float(np.clip(strength, 0.0, 0.25))
    label = str(environment)
    return descriptor, label, params, strength, bool(strength > 1e-15)


def _contrast_factor(n: int, depth: int, n_edges: int, noise_strength: float,
                     transferred_noise: bool) -> float:
    """Contrast retained by the finite-size/depth and optional noise transfer model."""
    intrinsic = np.exp(-_PLATEAU_CONCENTRATION_RATE * max(0, int(n) - 4) * int(depth))
    if transferred_noise and noise_strength > 0.0:
        # A QAOA p-layer has O(n + |E|) exposed native operations per layer.
        exposure = float(max(1, int(n) + int(n_edges)) * int(depth))
        physical = np.exp(-_NOISE_EXPOSURE_RATE * noise_strength * exposure)
    else:
        physical = 1.0
    return float(np.clip(intrinsic * physical, 0.0, 1.0))


def _contract_surface(surface: np.ndarray, contrast: float) -> np.ndarray:
    """Preserve extrema/topology while reducing landscape relief around its mean."""
    values = _finite_surface(surface)
    centre = float(np.mean(values))
    return centre + float(contrast) * (values - centre)


def _noise_creates_structure(ideal_metrics: Dict[str, float], noisy_metrics: Dict[str, float]) -> bool:
    """Conservative evidence test for a noise-created structured island.

    Correlated/anisotropic noise can occasionally increase local relief even if
    it lowers the average objective.  We only call that a resonance candidate if
    both variance and maximum gradient grow by at least 2%; ordinary depolarising
    smoothing therefore cannot be misreported as stochastic resonance.
    """
    return bool(
        noisy_metrics["variance"] > ideal_metrics["variance"] * 1.02 + 1e-12 and
        noisy_metrics["max_gradient"] > ideal_metrics["max_gradient"] * 1.02 + 1e-12
    )


# ---------------------------------------------------------------------------
# Cost estimate and boundary/summaries
# ---------------------------------------------------------------------------
def estimate_barren_cost(
    family: str,
    n_range: Union[Sequence[int], range, Tuple[int, int]],
    p_range: Union[Sequence[int], range, Tuple[int, int]],
    noise_model=None,
    n_instances: int = 5,
    resolution: int = 10,
    execution: str = "auto",
    exact_noise_max_n: int = DEFAULT_EXACT_NOISE_MAX_N,
) -> Dict[str, Union[float, int]]:
    """Return a conservative wall-clock estimate and exact/transfer job counts.

    The estimate is calibrated for the app's two-core development environment.
    It is intentionally approximate; graph topology, OS load and worker startup
    affect individual cells.  It is useful for the UI's honest pre-compute ETA.
    """
    sizes = diagram_sizes(family, n_range)
    depths = diagram_depths(p_range)
    _, _, _, noise_strength, has_noise = _canonical_noise(noise_model)
    noise_strength = float(noise_strength)
    execution = str(execution).lower()
    if execution not in {"auto", "exact", "transfer"}:
        raise ValueError("execution must be 'auto', 'exact' or 'transfer'.")
    n_instances = int(max(1, n_instances))
    resolution = int(resolution)
    exact_noise_max_n = int(max(4, exact_noise_max_n))

    ideal_jobs = len(sizes) * len(depths) * n_instances
    exact_noise_jobs = 0
    if has_noise and execution != "transfer":
        if execution == "exact":
            exact_noise_jobs = ideal_jobs
        else:
            exact_noise_jobs = sum(
                len(depths) * n_instances for n in sizes if n <= exact_noise_max_n
            )
    transfer_jobs = ideal_jobs - exact_noise_jobs if has_noise else 0

    grid_factor = (max(3, resolution) / 10.0) ** 2
    seconds = 0.0
    # Ideal statevector landscape cost rises strongly near n=16 but remains safe.
    for n in sizes:
        n_factor = 1.72 ** ((n - 4) / 2.0)
        for depth in depths:
            seconds += n_instances * 0.12 * grid_factor * n_factor * (0.60 + 0.20 * depth)
    # Exact density-matrix landscapes are only included for the safe subset.
    if has_noise and exact_noise_jobs:
        for n in sizes:
            if execution == "auto" and n > exact_noise_max_n:
                continue
            n_factor = 5.0 ** ((n - 4) / 2.0)
            for depth in depths:
                seconds += n_instances * 0.15 * grid_factor * n_factor * (0.50 + 0.20 * depth)

    # Tiny non-zero estimates are more useful than "0s" in the sidebar.
    return {
        "seconds": float(max(1.0, seconds)),
        "ideal_jobs": int(ideal_jobs),
        "exact_noise_jobs": int(exact_noise_jobs),
        "transfer_jobs": int(transfer_jobs),
        "sizes": int(len(sizes)),
        "depths": int(len(depths)),
        "instances": int(n_instances),
        "noise_strength": noise_strength,
    }


def _critical_depth(depths: Sequence[int], scores: Sequence[float], threshold: float = 0.5) -> Dict[str, object]:
    """Interpolate the first depth at which a score column falls through threshold."""
    d = np.asarray(depths, dtype=float)
    s = np.asarray(scores, dtype=float)
    if len(d) == 0:
        return {"critical_depth": None, "status": "empty"}
    above = s >= float(threshold)
    if np.all(above):
        return {"critical_depth": float(d[-1]), "status": "above_range"}
    if not bool(above[0]):
        return {"critical_depth": float(d[0]), "status": "below_range"}
    # First transition from phase-trainable to phase-barren.  The score can have
    # small nonmonotonic fluctuations; the first loss is the conservative phase
    # boundary relevant to optimizer initialization.
    idx = int(np.where(~above)[0][0])
    x0, x1 = float(d[idx - 1]), float(d[idx])
    y0, y1 = float(s[idx - 1]), float(s[idx])
    if abs(y1 - y0) < 1e-15:
        cross = x1
    else:
        cross = x0 + (float(threshold) - y0) * (x1 - x0) / (y1 - y0)
    return {"critical_depth": float(np.clip(cross, min(x0, x1), max(x0, x1))), "status": "crosses"}


def summarize_barren_diagram(
    scores: np.ndarray,
    classifications: Sequence[Sequence[str]],
    sizes: Sequence[int],
    depths: Sequence[int],
    noise_structure_fraction: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """Compute JSON-friendly phase counts, area and resonance metadata."""
    values = np.asarray(scores, dtype=float)
    labels = np.asarray(classifications, dtype=object)
    if values.shape != labels.shape:
        raise ValueError("scores and classifications must have the same 2-D shape.")
    total = int(values.size)
    counts = {name: int(np.sum(labels == name)) for name in TRAINABILITY_CLASSES}
    phase_trainable = int(np.sum(values >= 0.5))
    if noise_structure_fraction is None:
        evidence = np.zeros_like(values, dtype=float)
    else:
        evidence = np.asarray(noise_structure_fraction, dtype=float)
        if evidence.shape != values.shape:
            raise ValueError("noise_structure_fraction must match scores shape.")
    resonance_detected = bool(np.any((labels == "RESONANT") & (evidence >= 0.5)))
    boundaries = {
        str(n): _critical_depth(depths, values[:, ci])
        for ci, n in enumerate(sizes)
    }
    return {
        "total_cells": total,
        "class_counts": counts,
        "phase_trainable_cells": phase_trainable,
        "phase_trainable_fraction": float(phase_trainable / total) if total else 0.0,
        "resonance_detected": resonance_detected,
        "phase_boundary": boundaries,
    }


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------
def compute_barren_diagram(
    family: str,
    n_range: Union[Sequence[int], range, Tuple[int, int]],
    p_range: Union[Sequence[int], range, Tuple[int, int]],
    noise_model=None,
    n_instances: int = 5,
    *,
    seed_base: int = 42,
    resolution: int = 10,
    shots: int = 256,
    p_er: float = 0.5,
    m_ba: int = 2,
    execution: str = "auto",
    exact_noise_max_n: int = DEFAULT_EXACT_NOISE_MAX_N,
    progress_callback: Optional[Callable[[float, int], None]] = None,
) -> Dict:
    """Compute a Barren Plateau Phase Diagram.

    Parameters
    ----------
    family:
        ``'3-regular'``, ``'erdos-renyi'`` or ``'scale-free'``.
    n_range, p_range:
        Either inclusive ``(min, max)`` tuples or explicit sequences.  The
        output grid has shape ``(len(depths), len(sizes))`` — rows are QAOA
        depths, columns are qubit counts.
    noise_model:
        Prefer a parent-safe environment descriptor such as
        ``{'environment': 'earth-lab', 'params': {'noise_level': 0.003}}`` or
        simply ``'leo-satellite'``.  ``None``/``'ideal'`` uses noiseless exact
        statevectors.  A real Aer NoiseModel is accepted outside Streamlit.
    n_instances:
        Number of seeded graph instances averaged per cell (default 5).
    execution:
        ``'auto'`` (default) uses exact noisy landscapes up to
        ``exact_noise_max_n`` and transfer beyond it; ``'exact'`` forces noisy
        simulation (guarded at n<=12); ``'transfer'`` uses evaluated ideal
        landscapes plus the contrast-transfer model at every noisy cell.

    Returns
    -------
    JSON-serialisable dict containing ``scores`` / ``trainability_scores``,
    ``classifications`` / ``classification_grid``, per-cell metrics, phase
    boundaries, and protocol metadata.  No Streamlit calls occur here.
    """
    sizes = diagram_sizes(family, n_range)
    depths = diagram_depths(p_range)
    n_instances = int(max(1, n_instances))
    resolution = int(resolution)
    if resolution < 3:
        raise ValueError("resolution must be >= 3 to compute landscape gradients.")
    shots = int(max(1, shots))
    execution = str(execution).lower()
    if execution not in {"auto", "exact", "transfer"}:
        raise ValueError("execution must be 'auto', 'exact' or 'transfer'.")
    exact_noise_max_n = int(max(4, exact_noise_max_n))

    descriptor, noise_label, noise_params, noise_strength, has_noise = _canonical_noise(noise_model)
    if execution == "exact" and has_noise and max(sizes) > EXACT_NOISE_MEMORY_MAX_N:
        raise ValueError(
            f"Exact noisy landscapes are memory-limited to n <= {EXACT_NOISE_MEMORY_MAX_N}; "
            "use execution='auto' or 'transfer' for larger diagrams."
        )

    # Count actual worker landscapes for accurate progress, not just displayed
    # cells.  Every cell always evaluates an ideal surface; safe noisy cells add
    # a second density-matrix surface.
    def use_exact_noise(n: int) -> bool:
        if not has_noise or execution == "transfer":
            return False
        return execution == "exact" or n <= exact_noise_max_n

    total_jobs = sum(
        n_instances * len(depths) * (1 + int(use_exact_noise(n)))
        for n in sizes
    )
    completed_jobs = 0.0

    def emit(done: float) -> None:
        if progress_callback is not None:
            progress_callback(float(done), int(max(1, total_jobs)))

    n_depth, n_size = len(depths), len(sizes)
    scores = np.zeros((n_depth, n_size), dtype=float)
    variances = np.zeros_like(scores)
    max_gradients = np.zeros_like(scores)
    ratio_ranges = np.zeros_like(scores)
    ideal_scores = np.zeros_like(scores)
    noise_structure_fraction = np.zeros_like(scores)
    exact_fraction = np.zeros_like(scores)
    source_modes: List[List[str]] = [["" for _ in sizes] for _ in depths]
    candidates: List[List[str]] = [["" for _ in sizes] for _ in depths]

    for ni, n in enumerate(sizes):
        family_kwargs = _family_kwargs(family, n, p_er, m_ba)
        for di, depth in enumerate(depths):
            instance_metrics: List[Dict[str, float]] = []
            instance_ideal_scores: List[float] = []
            structure_hits = 0
            exact_hits = 0
            for instance in range(n_instances):
                # Different enough strides keep n/depth/instance graph draws
                # independent while making the entire diagram reproducible.
                seed = int(seed_base) + 100_003 * ni + 1_009 * di + 37 * instance + int(n)
                graph = generate_graph(family, n, seed=seed, **family_kwargs)

                def ideal_progress(done_rows: int, total_rows: int) -> None:
                    emit(completed_jobs + float(done_rows) / float(max(1, total_rows)))

                ideal = compute_landscape(
                    graph, depth=depth, resolution=resolution, noise_model=None,
                    shots=shots, readout_prob=None, progress_callback=ideal_progress,
                )
                completed_jobs += 1.0
                emit(completed_jobs)
                ideal_surface = np.asarray(ideal["ratios"], dtype=float)
                ideal_metric = landscape_metrics(ideal_surface)
                instance_ideal_scores.append(float(ideal_metric["score"]))

                if use_exact_noise(n):
                    def noisy_progress(done_rows: int, total_rows: int) -> None:
                        emit(completed_jobs + float(done_rows) / float(max(1, total_rows)))

                    noisy = compute_landscape(
                        graph, depth=depth, resolution=resolution, noise_model=descriptor,
                        shots=shots, readout_prob=None, progress_callback=noisy_progress,
                    )
                    completed_jobs += 1.0
                    emit(completed_jobs)
                    source_surface = np.asarray(noisy["ratios"], dtype=float)
                    source_metric = landscape_metrics(source_surface)
                    structure_hits += int(_noise_creates_structure(ideal_metric, source_metric))
                    exact_hits += 1
                    mode = "exact-noisy"
                elif has_noise:
                    # Larger noisy cells are physically transferred from an
                    # actual ideal landscape without a density-matrix allocation.
                    transfer = _contrast_factor(
                        n, depth, graph.number_of_edges(), noise_strength, transferred_noise=True,
                    )
                    source_surface = _contract_surface(ideal_surface, transfer)
                    mode = "contrast-transfer"
                else:
                    source_surface = ideal_surface
                    mode = "ideal"

                # Independent finite-size/depth concentration.  For an exact
                # noisy surface its hardware channels are already simulated; for
                # a transfer surface noise was included above in `transfer`.
                contrast = _contrast_factor(
                    n, depth, graph.number_of_edges(), noise_strength, transferred_noise=False,
                )
                effective_surface = _contract_surface(source_surface, contrast)
                instance_metrics.append(landscape_metrics(effective_surface))
                source_modes[di][ni] = mode

            # Average trainability across graph instances, as requested.
            scores[di, ni] = float(np.mean([m["score"] for m in instance_metrics]))
            variances[di, ni] = float(np.mean([m["variance"] for m in instance_metrics]))
            max_gradients[di, ni] = float(np.mean([m["max_gradient"] for m in instance_metrics]))
            ratio_ranges[di, ni] = float(np.mean([m["ratio_range"] for m in instance_metrics]))
            ideal_scores[di, ni] = float(np.mean(instance_ideal_scores))
            noise_structure_fraction[di, ni] = float(structure_hits / n_instances)
            exact_fraction[di, ni] = float(exact_hits / n_instances)
            candidate = classify_trainability(scores[di, ni], variances[di, ni], max_gradients[di, ni])
            candidates[di][ni] = candidate

    # `classify_trainability` identifies a structured intermediate candidate;
    # only promote it to cyan RESONANT if direct noisy-vs-ideal evidence exists.
    # This avoids falsely calling ordinary depolarising flattening "resonance".
    classifications: List[List[str]] = [["" for _ in sizes] for _ in depths]
    for di in range(n_depth):
        for ni in range(n_size):
            c = candidates[di][ni]
            if c == "RESONANT" and noise_structure_fraction[di, ni] < 0.5:
                c = "MIXED"
            classifications[di][ni] = c

    summary = summarize_barren_diagram(
        scores, classifications, sizes, depths, noise_structure_fraction,
    )
    emit(float(total_jobs))

    result = {
        "family": family,
        "sizes": [int(x) for x in sizes],
        "depths": [int(x) for x in depths],
        "n_instances": int(n_instances),
        "seed_base": int(seed_base),
        "resolution": int(resolution),
        "shots": int(shots),
        "noise": {
            "label": noise_label,
            "params": dict(noise_params),
            "strength": float(noise_strength),
            "has_noise": bool(has_noise),
        },
        "protocol": {
            "execution": execution,
            "exact_noise_max_n": int(exact_noise_max_n),
            "exact_noise_memory_max_n": int(EXACT_NOISE_MEMORY_MAX_N),
            "description": (
                "Exact ideal landscapes everywhere; exact noisy landscapes for safe cells; "
                "contrast-transfer for larger noisy cells."
            ),
        },
        # Both clear names are kept so downstream users can consume the contract
        # without having to remember an alias.
        "scores": scores.tolist(),
        "trainability_scores": scores.tolist(),
        "classifications": classifications,
        "classification_grid": classifications,
        "metrics": {
            "variance": variances.tolist(),
            "max_gradient": max_gradients.tolist(),
            "ratio_range": ratio_ranges.tolist(),
            "ideal_score": ideal_scores.tolist(),
            "noise_structure_fraction": noise_structure_fraction.tolist(),
        },
        "source": {
            "mode": source_modes,
            "exact_noise_fraction": exact_fraction.tolist(),
        },
        "summary": summary,
        "phase_boundary": summary["phase_boundary"],
    }
    return result


def compare_diagrams(diagrams: Dict[str, Dict]) -> Dict[str, object]:
    """Summarize how cached diagrams differ across environments.

    ``diagrams`` maps a stable environment key to a result returned by
    :func:`compute_barren_diagram`.  The helper is deliberately pure so the page
    can show a truthful LEO-vs-Earth insight without hard-coded percentages.
    """
    if not diagrams:
        return {"environments": {}, "reference": None, "comparison": None}
    env_summary: Dict[str, Dict[str, Union[int, float, bool]]] = {}
    for env, result in diagrams.items():
        summary = result.get("summary", {})
        env_summary[str(env)] = {
            "phase_trainable_cells": int(summary.get("phase_trainable_cells", 0)),
            "phase_trainable_fraction": float(summary.get("phase_trainable_fraction", 0.0)),
            "resonance_detected": bool(summary.get("resonance_detected", False)),
        }

    reference = "earth-lab" if "earth-lab" in env_summary else next(iter(env_summary))
    comparison = None
    if "leo-satellite" in env_summary and reference in env_summary and reference != "leo-satellite":
        ref = env_summary[reference]["phase_trainable_cells"]
        leo = env_summary["leo-satellite"]["phase_trainable_cells"]
        shrink = 0.0 if ref == 0 else max(0.0, 100.0 * (float(ref) - float(leo)) / float(ref))
        comparison = {
            "reference": reference,
            "environment": "leo-satellite",
            "shrink_percent": float(shrink),
        }
    return {"environments": env_summary, "reference": reference, "comparison": comparison}
