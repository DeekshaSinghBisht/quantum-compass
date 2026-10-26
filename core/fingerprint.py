"""Optimization-landscape fingerprinting for Quantum Compass.

A QAOA landscape contains more diagnostic information than its best approximation
ratio alone.  Noise changes its variance, extrema, gradients, roughness and
symmetry; this module turns those shape signatures into a compact feature vector
and trains a credential-free environment classifier.

The classifier is deliberately trained on deterministic exact Aer landscapes,
not shot-noisy counts.  It is a *diagnostic aid* for the specific simulated
noise families in :mod:`core.noise_models`, not a forensic attribution tool for
arbitrary physical devices.  All Qiskit/Aer construction stays in the persistent
worker: this module passes only primitive environment descriptors to
:func:`core.landscape.compute_landscape`.
"""

from __future__ import annotations

import os
import pickle
import tempfile
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

from core.graph_generator import generate_graph
from core.landscape import compute_landscape
from core.noise_models import ENVIRONMENTS, REPRESENTATIVE_LEVELS, effective_params

# Stable ordering is part of the persisted-model contract.
FINGERPRINT_FEATURE_NAMES: Tuple[str, ...] = (
    "variance",
    "num_minima",
    "num_maxima",
    "peak_width",
    "roughness",
    "gradient_mean",
    "gradient_variance",
    "gradient_max",
    "symmetry_score",
    "ratio_range",
    "kurtosis",
    "skewness",
    "histogram_entropy",
    "neighbor_autocorrelation",
    "peak_prominence",
    "edge_center_contrast",
)

DEFAULT_FINGERPRINT_ENVIRONMENTS: Tuple[str, ...] = (
    "ideal",
    "earth-lab",
    "leo-satellite",
    "satellite-ground",
    "meo-van-allen",
    "deep-space",
)

FINGERPRINT_ENV_LABELS: Dict[str, str] = {
    "ideal": "Ideal simulator",
    "earth-lab": "Earth laboratory",
    "leo-satellite": "LEO satellite",
    "satellite-ground": "Satellite-to-ground link",
    "meo-van-allen": "MEO / Van Allen belts",
    "deep-space": "Deep space",
    "custom": "Custom profile",
}

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = _PROJECT_ROOT / "data" / "fingerprint_model.pkl"
_MODEL_VERSION = 1
_DEFAULT_RESOLUTION = 9  # avoids the p=1 aliasing of an 8x8 periodic grid
_DEFAULT_DEPTH = 1

# Hidden labels deliberately stay in core.  The page exposes only friendly
# mystery names until the model has made its prediction.
_MYSTERY_CONFIGS: Dict[str, Dict] = {
    "aurora": {
        "title": "Mystery A — Aurora", "hint": "A low-noise laboratory-style calibration snapshot.",
        "environment": "earth-lab", "params": {"noise_level": 0.0035},
    },
    "stratos": {
        "title": "Mystery B — Stratos", "hint": "An orbiting platform with radiation bursts and thermal cycling.",
        "environment": "leo-satellite", "params": {"noise_level": 0.008},
    },
    "horizon": {
        "title": "Mystery C — Horizon", "hint": "A free-space link with loss, turbulence and a thermal photon background.",
        "environment": "satellite-ground", "params": {"noise_level": 0.005, "elevation": 45.0},
    },
    "belt": {
        "title": "Mystery D — Belt", "hint": "A radiation-belt environment with elevated sustained error rates.",
        "environment": "meo-van-allen", "params": {"noise_level": 0.020},
    },
    "voyager": {
        "title": "Mystery E — Voyager", "hint": "An unshielded interplanetary setting with the strongest burst profile.",
        "environment": "deep-space", "params": {"noise_level": 0.050},
    },
}


def _finite_surface(ratios_2d) -> np.ndarray:
    """Coerce a finite nonempty two-dimensional landscape array."""
    values = np.asarray(ratios_2d, dtype=float)
    if values.ndim != 2:
        raise ValueError("ratios_2d must be a two-dimensional array.")
    if values.size == 0:
        raise ValueError("ratios_2d must not be empty.")
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values, dtype=float)
    return np.nan_to_num(values, nan=float(finite.mean()),
                         posinf=float(finite.max()), neginf=float(finite.min()))


def _local_extrema_counts(surface: np.ndarray) -> Tuple[int, int]:
    """Count strict-ish 8-neighbour minima/maxima away from grid boundaries."""
    rows, cols = surface.shape
    if rows < 3 or cols < 3:
        return 0, 0
    minima = maxima = 0
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            centre = surface[i, j]
            neighbours = np.delete(surface[i - 1:i + 2, j - 1:j + 2].ravel(), 4)
            if centre <= float(np.min(neighbours)) and np.any(centre < neighbours):
                minima += 1
            if centre >= float(np.max(neighbours)) and np.any(centre > neighbours):
                maxima += 1
    return int(minima), int(maxima)


def _neighbor_autocorrelation(surface: np.ndarray) -> float:
    """Mean Pearson correlation across horizontal/vertical neighbouring samples."""
    pairs: List[Tuple[np.ndarray, np.ndarray]] = []
    if surface.shape[1] >= 2:
        pairs.append((surface[:, :-1].ravel(), surface[:, 1:].ravel()))
    if surface.shape[0] >= 2:
        pairs.append((surface[:-1, :].ravel(), surface[1:, :].ravel()))
    correlations: List[float] = []
    for left, right in pairs:
        if left.size < 2 or np.std(left) < 1e-14 or np.std(right) < 1e-14:
            continue
        value = float(np.corrcoef(left, right)[0, 1])
        if np.isfinite(value):
            correlations.append(value)
    return float(np.mean(correlations)) if correlations else 1.0


def extract_landscape_fingerprint(ratios_2d) -> np.ndarray:
    """Extract a 16-feature, finite fingerprint from a QAOA landscape.

    Features include the requested variance, numbers of local minima/maxima,
    peak width, roughness, gradient mean/variance, symmetry score, ratio range
    and kurtosis, plus gradient maximum, skewness, entropy, neighbour
    autocorrelation, peak prominence and edge-centre contrast.  The order is
    fixed by :data:`FINGERPRINT_FEATURE_NAMES` and returns ``float64``.
    """
    surface = _finite_surface(ratios_2d)
    values = surface.ravel()
    mean = float(np.mean(values))
    variance = float(np.var(values))
    ratio_range = float(np.max(values) - np.min(values))
    std = float(np.sqrt(variance))
    minima, maxima = _local_extrema_counts(surface)

    # Gradient coordinates are normalized to [0, 1] so this feature does not
    # collapse merely because the uploader supplied a denser gamma/beta grid.
    if min(surface.shape) < 2:
        grad_mag = np.zeros_like(surface)
    else:
        dr = 1.0 / max(1, surface.shape[0] - 1)
        dc = 1.0 / max(1, surface.shape[1] - 1)
        gr, gc = np.gradient(surface, dr, dc)
        grad_mag = np.hypot(gr, gc)
    gradient_mean = float(np.mean(grad_mag))
    gradient_variance = float(np.var(grad_mag))
    gradient_max = float(np.max(grad_mag))

    if ratio_range <= 1e-14:
        peak_width = 1.0
        symmetry_score = 1.0
        kurtosis = 0.0
        skewness = 0.0
        peak_prominence = 0.0
        edge_center_contrast = 0.0
    else:
        high_threshold = float(np.min(surface) + 0.80 * ratio_range)
        peak_width = float(np.mean(surface >= high_threshold))
        horizontal = float(np.mean(np.abs(surface - surface[:, ::-1])) / ratio_range)
        vertical = float(np.mean(np.abs(surface - surface[::-1, :])) / ratio_range)
        rotational = float(np.mean(np.abs(surface - surface[::-1, ::-1])) / ratio_range)
        symmetry_score = float(np.clip(1.0 - (horizontal + vertical + rotational) / 3.0, 0.0, 1.0))
        normalized = (values - mean) / max(std, 1e-14)
        skewness = float(np.mean(normalized ** 3))
        kurtosis = float(np.mean(normalized ** 4) - 3.0)
        peak_prominence = float((np.max(values) - mean) / ratio_range)
        edge = np.concatenate((surface[0, :], surface[-1, :], surface[1:-1, 0], surface[1:-1, -1]))
        centre = surface[1:-1, 1:-1].ravel()
        edge_center_contrast = float((np.mean(centre) - np.mean(edge)) / ratio_range) if centre.size else 0.0

    # Mean absolute discrete Laplacian is a grid-scale roughness measure.
    laplacian = np.zeros_like(surface)
    if surface.shape[0] >= 3:
        laplacian[1:-1, :] += surface[:-2, :] - 2.0 * surface[1:-1, :] + surface[2:, :]
    if surface.shape[1] >= 3:
        laplacian[:, 1:-1] += surface[:, :-2] - 2.0 * surface[:, 1:-1] + surface[:, 2:]
    roughness = float(np.mean(np.abs(laplacian)))

    bins = int(max(4, min(16, round(np.sqrt(values.size)))))
    hist, _ = np.histogram(values, bins=bins, range=(float(np.min(values)), float(np.max(values))) if ratio_range > 1e-14 else None)
    prob = hist.astype(float) / max(1, int(hist.sum()))
    nz = prob[prob > 0.0]
    histogram_entropy = float(-np.sum(nz * np.log(nz)) / np.log(bins)) if bins > 1 else 0.0

    fingerprint = np.asarray([
        variance, float(minima), float(maxima), peak_width, roughness,
        gradient_mean, gradient_variance, gradient_max, symmetry_score,
        ratio_range, kurtosis, skewness, histogram_entropy,
        _neighbor_autocorrelation(surface), peak_prominence, edge_center_contrast,
    ], dtype=np.float64)
    return np.nan_to_num(fingerprint, nan=0.0, posinf=0.0, neginf=0.0)


def fingerprint_feature_dict(ratios_2d) -> Dict[str, float]:
    """Return named fingerprint features for display/debugging/export."""
    values = extract_landscape_fingerprint(ratios_2d)
    return {name: float(value) for name, value in zip(FINGERPRINT_FEATURE_NAMES, values)}


def _validate_environments(environments: Iterable[str]) -> List[str]:
    """Validate/de-duplicate environment keys while retaining caller ordering."""
    selected = list(dict.fromkeys(str(item) for item in environments))
    if len(selected) < 2:
        raise ValueError("Fingerprint training needs at least two environments.")
    unknown = [env for env in selected if env not in ENVIRONMENTS]
    if unknown:
        raise ValueError(f"Unknown environment(s): {unknown}. Choose from {list(ENVIRONMENTS)}.")
    return selected


def _noise_spec_for_training(environment: str, sample_index: int, n_per_env: int) -> Optional[Dict]:
    """Generate a deterministic, class-specific calibration variation descriptor."""
    if environment == "ideal":
        return None
    # A low-discrepancy sweep around each environment's representative value
    # gives a classifier tolerance to routine calibration drift without classes
    # overlapping their distinct physical profiles.
    u = (float(sample_index) + 0.5) / float(max(1, n_per_env))
    wave = 0.72 + 0.56 * u + 0.07 * np.sin(2.0 * np.pi * (sample_index + 1) / 7.0)
    params: Dict[str, float] = {
        "noise_level": float(max(0.0, REPRESENTATIVE_LEVELS[environment] * wave))
    }
    if environment == "satellite-ground":
        # Vary elevation in a physically meaningful 30–60° operating range.
        params["elevation"] = float(30.0 + 30.0 * ((sample_index * 7) % max(1, n_per_env)) / max(1, n_per_env - 1))
    return {"environment": environment, "params": params}


def _graph_variant(G: nx.Graph, sample_index: int) -> nx.Graph:
    """Make a small deterministic degree-preserving graph variation when possible.

    The same variant sequence is used for every environment, so the classifier
    cannot treat graph topology as an environment label.  Tiny graphs that
    cannot be rewired simply return a copy.
    """
    graph = nx.convert_node_labels_to_integers(G.copy(), first_label=0, ordering="sorted")
    if sample_index == 0 or graph.number_of_edges() < 4 or graph.number_of_nodes() < 5:
        return graph
    try:
        nx.double_edge_swap(
            graph, nswap=max(1, min(3, graph.number_of_edges() // 3)),
            max_tries=max(20, graph.number_of_edges() * 10), seed=10_007 + int(sample_index),
        )
    except Exception:
        pass
    return graph


def train_fingerprint_classifier(
    G: nx.Graph,
    environments: Sequence[str],
    n_per_env: int = 50,
    *,
    resolution: int = _DEFAULT_RESOLUTION,
    depth: int = _DEFAULT_DEPTH,
    seed: int = 42,
    progress_callback: Optional[Callable[[int, int], None]] = None,
):
    """Train a RandomForest environment classifier from simulated landscapes.

    ``G`` supplies the base logical MaxCut instance.  For each requested
    environment and sample, the function evaluates a deterministic landscape
    through the Aer worker, extracts its fingerprint, and fits a balanced
    :class:`sklearn.ensemble.RandomForestClassifier`.  ``n_per_env=50`` is the
    production default; tests/experiments may request fewer samples.

    The returned classifier carries serialisable ``_qc_*`` metadata including
    feature names, training environments, OOB score and representative parameter
    profiles.  No model is saved automatically; use
    :func:`load_default_classifier` for the persistent default cache.
    """
    try:
        from sklearn.ensemble import RandomForestClassifier
    except ImportError as exc:
        raise ImportError("scikit-learn is required for landscape fingerprinting.") from exc
    if not isinstance(G, nx.Graph):
        raise ValueError("G must be a networkx.Graph.")
    if G.number_of_nodes() < 2 or G.number_of_nodes() > 10:
        raise ValueError("Fingerprint training supports logical graphs with 2–10 nodes.")
    environments = _validate_environments(environments)
    n_per_env = int(n_per_env)
    if n_per_env < 2:
        raise ValueError("n_per_env must be at least 2.")
    resolution = int(resolution)
    if resolution < 3:
        raise ValueError("resolution must be at least 3.")
    depth = int(max(1, depth))

    fingerprints: List[np.ndarray] = []
    labels: List[str] = []
    total = len(environments) * n_per_env
    done = 0

    # Iterate variant first so every environment receives identical topology
    # variation and labels remain balanced.
    for sample_index in range(n_per_env):
        graph = _graph_variant(G, sample_index)
        for environment in environments:
            landscape = compute_landscape(
                graph, depth=depth, resolution=resolution,
                noise_model=_noise_spec_for_training(environment, sample_index, n_per_env),
                shots=256, readout_prob=None,
            )
            fingerprints.append(extract_landscape_fingerprint(landscape["ratios"]))
            labels.append(environment)
            done += 1
            if progress_callback is not None:
                progress_callback(done, total)

    X = np.vstack(fingerprints)
    y = np.asarray(labels, dtype=object)
    classifier = RandomForestClassifier(
        n_estimators=320, random_state=int(seed), n_jobs=1,
        class_weight="balanced", bootstrap=True, oob_score=True,
        min_samples_leaf=1,
    )
    classifier.fit(X, y)
    representative_params = {
        env: _representative_params(env) for env in environments
    }
    classifier._qc_fingerprint_model_version = _MODEL_VERSION
    classifier._qc_feature_names = tuple(FINGERPRINT_FEATURE_NAMES)
    classifier._qc_environments = tuple(environments)
    classifier._qc_environment_labels = {env: FINGERPRINT_ENV_LABELS.get(env, env) for env in environments}
    classifier._qc_representative_params = representative_params
    classifier._qc_training_resolution = resolution
    classifier._qc_training_depth = depth
    classifier._qc_n_per_env = n_per_env
    classifier._qc_training_accuracy = float(np.mean(classifier.predict(X) == y))
    classifier._qc_oob_score = float(getattr(classifier, "oob_score_", 0.0))
    return classifier


def _representative_params(environment: str) -> Dict[str, float]:
    """Return pure representative channel parameters for one class label."""
    if environment == "ideal":
        return {"p1": 0.0, "p2": 0.0, "readout": 0.0, "burst_rate": 0.0}
    params: Dict[str, float] = {"noise_level": float(REPRESENTATIVE_LEVELS[environment])}
    if environment == "satellite-ground":
        params["elevation"] = 45.0
    effective = effective_params(environment, params)
    return {
        "p1": float(effective["p1"]),
        "p2": float(effective["p2"]),
        "readout": float(effective["readout"]),
        "burst_rate": float(effective["burst_rate"]),
        "ad_rate": float(effective["ad_rate"]),
        "pd_rate": float(effective["pd_rate"]),
    }


def _classifier_metadata_valid(classifier) -> bool:
    """Ensure a cached pickle matches the current feature/model contract."""
    try:
        return (
            int(getattr(classifier, "_qc_fingerprint_model_version", -1)) == _MODEL_VERSION
            and tuple(getattr(classifier, "_qc_feature_names", ())) == FINGERPRINT_FEATURE_NAMES
            and len(getattr(classifier, "classes_", ())) >= 2
        )
    except Exception:
        return False


def load_default_classifier(
    force_retrain: bool = False,
    *,
    n_per_env: int = 50,
    progress_callback: Optional[Callable[[int, int], None]] = None,
):
    """Load or pre-train the default persistent fingerprint classifier.

    The model is cached as ``data/fingerprint_model.pkl``.  If the file is
    absent, corrupt, or from an incompatible feature contract, a new classifier
    is trained on a reproducible 4-qubit 3-regular graph and atomically written
    to disk.  The default uses 50 real landscapes per environment.
    """
    if not force_retrain and DEFAULT_MODEL_PATH.exists():
        try:
            with DEFAULT_MODEL_PATH.open("rb") as handle:
                cached = pickle.load(handle)
            if _classifier_metadata_valid(cached):
                return cached
        except Exception:
            pass

    training_graph = generate_graph("3-regular", 4, seed=42, d=3)
    classifier = train_fingerprint_classifier(
        training_graph, DEFAULT_FINGERPRINT_ENVIRONMENTS, n_per_env=n_per_env,
        resolution=_DEFAULT_RESOLUTION, depth=_DEFAULT_DEPTH, seed=42,
        progress_callback=progress_callback,
    )
    DEFAULT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(prefix="fingerprint_", suffix=".pkl", dir=str(DEFAULT_MODEL_PATH.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            pickle.dump(classifier, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary_path, DEFAULT_MODEL_PATH)
    finally:
        if os.path.exists(temporary_path):
            try:
                os.unlink(temporary_path)
            except OSError:
                pass
    return classifier


def predict_environment(ratios_2d, classifier) -> Dict:
    """Predict a simulated environment from a landscape fingerprint.

    Returns ``predicted_environment``, human-readable label, confidence in
    percent, ordered ``top_3`` probabilities, estimated weighted channel
    parameters (including requested ``p2`` and ``burst_rate``), and named raw
    features.  The classifier must be the RandomForest returned by
    :func:`train_fingerprint_classifier` or :func:`load_default_classifier`.
    """
    if not hasattr(classifier, "predict_proba") or not hasattr(classifier, "classes_"):
        raise ValueError("classifier must expose sklearn-style predict_proba() and classes_.")
    fingerprint = extract_landscape_fingerprint(ratios_2d)
    probabilities = np.asarray(classifier.predict_proba([fingerprint])[0], dtype=float)
    classes = [str(item) for item in classifier.classes_]
    if probabilities.size != len(classes):
        raise ValueError("classifier probabilities do not align with its classes.")
    order = np.argsort(probabilities)[::-1]
    labels = getattr(classifier, "_qc_environment_labels", {})
    top = [
        {
            "environment": classes[int(index)],
            "label": labels.get(classes[int(index)], FINGERPRINT_ENV_LABELS.get(classes[int(index)], classes[int(index)])),
            "probability": float(probabilities[int(index)]),
            "confidence_percent": float(100.0 * probabilities[int(index)]),
        }
        for index in order[:3]
    ]
    predicted = top[0]["environment"]

    representative = getattr(classifier, "_qc_representative_params", {})
    aggregate_keys = ("p1", "p2", "readout", "burst_rate", "ad_rate", "pd_rate")
    estimated = {}
    for key in aggregate_keys:
        estimated[key] = float(sum(
            float(probabilities[i]) * float(representative.get(classes[i], _representative_params(classes[i])).get(key, 0.0))
            for i in range(len(classes))
        ))
    feature_values = {name: float(value) for name, value in zip(FINGERPRINT_FEATURE_NAMES, fingerprint)}
    return {
        "predicted_environment": predicted,
        "predicted_label": labels.get(predicted, FINGERPRINT_ENV_LABELS.get(predicted, predicted)),
        "confidence": float(top[0]["confidence_percent"]),
        # Both spellings are retained to make the result pleasant for Python
        # callers while preserving the compact key used by the Streamlit page.
        "top_3": top,
        "top_three": top,
        "estimated_params": estimated,
        "estimated_parameters": estimated,
        "features": feature_values,
        "fingerprint": feature_values,
    }


def mystery_landscape_catalog() -> List[Dict[str, str]]:
    """Return public mystery choices without leaking their hidden environment labels."""
    return [
        {"id": key, "title": value["title"], "hint": value["hint"]}
        for key, value in _MYSTERY_CONFIGS.items()
    ]


def generate_mystery_landscape(mystery_id: str, resolution: int = _DEFAULT_RESOLUTION) -> Dict:
    """Generate one deterministic hidden-environment landscape for the page.

    The hidden key is included only as ``_hidden_environment`` so page code can
    assess an optional post-prediction reveal; it must not be rendered before
    analysis.  All simulation still occurs in the Aer worker via a primitive
    descriptor.
    """
    key = str(mystery_id).strip().lower()
    if key not in _MYSTERY_CONFIGS:
        raise ValueError(f"Unknown mystery landscape '{mystery_id}'.")
    config = _MYSTERY_CONFIGS[key]
    graph = generate_graph("3-regular", 4, seed=42, d=3)
    descriptor = {"environment": config["environment"], "params": dict(config["params"])}
    landscape = compute_landscape(
        graph, depth=1, resolution=int(resolution), noise_model=descriptor,
        shots=256, readout_prob=None,
    )
    return {
        "id": key,
        "title": config["title"],
        "hint": config["hint"],
        "landscape": landscape,
        "_hidden_environment": config["environment"],
    }
