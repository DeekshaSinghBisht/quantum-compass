"""Reproducible QAOA hardware-validation protocol and analysis utilities.

This module deliberately separates a *measured-hardware validation study* from
simulator-only demonstrations.  It creates a versioned experiment manifest with
an explicit graph, angle grid, randomized execution order and planned metrics;
validates normalized count data; and compares observations with independently
labelled simulation predictions using uncertainty intervals and distributional
distances.

The module is intentionally pure Python/NumPy/pandas/NetworkX.  It constructs no
Qiskit object and is therefore safe in Streamlit's parent process.  Actual IBM
Runtime submission is provided by :mod:`scripts.submit_ibm_validation`, which
runs as a separate one-shot CLI process and writes the normalized CSV schema
consumed here.

Scientific scope
----------------
A successful comparison measures agreement between a defined *small-QAOA
experiment* and selected simulator models.  It cannot identify a physical noise
source, establish a general quantum advantage, or validate a space environment.
Those boundaries are carried into generated reports so the UI cannot silently
overstate simulator agreement as hardware attribution.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
from copy import deepcopy
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd

from core.graph_generator import cut_value, optimal_maxcut

MANIFEST_SCHEMA_VERSION = 1
MANIFEST_KIND = "quantum-compass-qaoa-hardware-validation"
REQUIRED_COUNTS_COLUMNS: Tuple[str, ...] = ("condition_id", "bitstring", "count")
_VERTEX_ORDER = "vertex-order"
_QISKIT_CLASSICAL_ORDER = "qiskit-classical-order"


class ExperimentalValidationError(ValueError):
    """Raised when an experimental manifest or uploaded count table is invalid."""


def _json_ready(value):
    """Return a recursively JSON-native copy suitable for canonical hashing."""
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ExperimentalValidationError("Manifest values must be finite.")
        return float(value)
    return value


def _canonical_json(value) -> str:
    """Serialise a JSON-native object deterministically for IDs and reports."""
    return json.dumps(_json_ready(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _manifest_digest(manifest: Mapping) -> str:
    """SHA-256 digest of a manifest excluding its derived ID field."""
    payload = deepcopy(dict(manifest))
    payload.pop("manifest_id", None)
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _require_finite(value, name: str) -> float:
    """Coerce a finite numeric scalar or raise a contextual validation error."""
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ExperimentalValidationError(f"{name} must be numeric.") from exc
    if not math.isfinite(number):
        raise ExperimentalValidationError(f"{name} must be finite.")
    return number


def _validate_graph(G: nx.Graph, maximum_nodes: int = 10) -> nx.Graph:
    """Validate a small, contiguous-labelled simple graph for a hardware study."""
    if not isinstance(G, nx.Graph) or G.is_directed() or G.is_multigraph():
        raise ExperimentalValidationError("The experiment graph must be a simple undirected networkx.Graph.")
    n = int(G.number_of_nodes())
    if n < 2 or n > int(maximum_nodes):
        raise ExperimentalValidationError(
            f"Hardware-validation manifests support 2–{maximum_nodes} logical qubits; got {n}."
        )
    if set(G.nodes()) != set(range(n)):
        raise ExperimentalValidationError("Graph nodes must be contiguous integers 0 through n-1.")
    if any(int(left) == int(right) for left, right in G.edges()):
        raise ExperimentalValidationError("Self-loops are not supported by the MaxCut validation protocol.")
    return G


def standard_angle_grid(grid_size: int = 3) -> Tuple[List[float], List[float]]:
    """Return a compact, physically interpretable p=1 QAOA validation grid.

    The grid spans ``gamma ∈ [0, pi/2]`` and ``beta ∈ [0, pi/4]``.  It includes
    a zero-angle negative/control condition and multiple nontrivial cost/mixer
    settings while keeping a live-hardware study affordable.  A ``3×3`` grid is
    the recommended default: nine circuits at one fixed graph/depth.
    """
    grid_size = int(grid_size)
    if grid_size < 2 or grid_size > 7:
        raise ExperimentalValidationError("grid_size must be an integer from 2 through 7.")
    gammas = np.linspace(0.0, np.pi / 2.0, grid_size, dtype=float).tolist()
    betas = np.linspace(0.0, np.pi / 4.0, grid_size, dtype=float).tolist()
    return [float(value) for value in gammas], [float(value) for value in betas]


def graph_to_manifest_record(G: nx.Graph, metadata: Optional[Mapping] = None) -> Dict:
    """Encode a validated logical graph as an explicit JSON-safe manifest record."""
    graph = _validate_graph(G)
    edges = sorted((min(int(left), int(right)), max(int(left), int(right))) for left, right in graph.edges())
    record = {
        "n_qubits": int(graph.number_of_nodes()),
        "edges": [[int(left), int(right)] for left, right in edges],
    }
    if metadata:
        record["metadata"] = _json_ready(dict(metadata))
    return record


def graph_from_manifest_record(record: Mapping) -> nx.Graph:
    """Rebuild and validate the logical MaxCut graph encoded in a manifest."""
    if not isinstance(record, Mapping):
        raise ExperimentalValidationError("Manifest graph must be an object.")
    try:
        n = int(record["n_qubits"])
        raw_edges = record["edges"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ExperimentalValidationError("Manifest graph needs n_qubits and edges.") from exc
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    try:
        graph.add_edges_from((int(edge[0]), int(edge[1])) for edge in raw_edges)
    except (TypeError, IndexError, ValueError) as exc:
        raise ExperimentalValidationError("Manifest graph edges must be [u, v] pairs.") from exc
    return _validate_graph(graph)


def create_experiment_manifest(
    G: nx.Graph,
    *,
    graph_metadata: Optional[Mapping] = None,
    depth: int = 1,
    gamma_values: Optional[Sequence[float]] = None,
    beta_values: Optional[Sequence[float]] = None,
    shots: int = 4096,
    randomization_seed: int = 2027,
    bootstrap_resamples: int = 2000,
    created_utc: Optional[str] = None,
) -> Dict:
    """Create a versioned, randomized hardware-validation manifest.

    The manifest fixes the graph, shared-angle QAOA slice, all circuit settings,
    intended shot budget, bitstring convention and a randomized execution order
    *before* experimental data are uploaded.  This discourages post-hoc tuning
    of the angle grid or primary metrics after inspecting hardware results.

    ``depth > 1`` is supported as an explicit shared-angle slice: each
    condition repeats its scalar gamma/beta values for every QAOA layer.  The
    manifest labels that choice so it cannot be mistaken for a full 2p-parameter
    optimization experiment.
    """
    graph = _validate_graph(G)
    depth = int(depth)
    if depth < 1 or depth > 3:
        raise ExperimentalValidationError("Hardware-validation depth must be an integer from 1 through 3.")
    shots = int(shots)
    if shots < 128:
        raise ExperimentalValidationError("shots must be at least 128 for a meaningful count comparison.")
    bootstrap_resamples = int(bootstrap_resamples)
    if bootstrap_resamples < 200 or bootstrap_resamples > 20_000:
        raise ExperimentalValidationError("bootstrap_resamples must be between 200 and 20,000.")
    randomization_seed = int(randomization_seed)

    if gamma_values is None or beta_values is None:
        default_gammas, default_betas = standard_angle_grid(3)
        gamma_values = default_gammas if gamma_values is None else gamma_values
        beta_values = default_betas if beta_values is None else beta_values
    gammas = [_require_finite(value, "gamma") for value in gamma_values]
    betas = [_require_finite(value, "beta") for value in beta_values]
    if not gammas or not betas:
        raise ExperimentalValidationError("At least one gamma and one beta value are required.")
    if len(gammas) * len(betas) > 49:
        raise ExperimentalValidationError("Use at most 49 conditions in one hardware-validation manifest.")

    conditions: List[Dict] = []
    for gamma_index, gamma in enumerate(gammas):
        for beta_index, beta in enumerate(betas):
            conditions.append({
                "condition_id": f"g{gamma_index:02d}_b{beta_index:02d}",
                "grid_index": [int(gamma_index), int(beta_index)],
                "gamma": float(gamma),
                "beta": float(beta),
                "gammas": [float(gamma)] * depth,
                "betas": [float(beta)] * depth,
            })
    order = np.random.default_rng(randomization_seed).permutation(len(conditions))
    for run_order, condition_index in enumerate(order, start=1):
        conditions[int(condition_index)]["run_order"] = int(run_order)
    conditions.sort(key=lambda item: int(item["run_order"]))

    timestamp = created_utc or datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    manifest: Dict = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "kind": MANIFEST_KIND,
        "created_utc": str(timestamp),
        "graph": graph_to_manifest_record(graph, graph_metadata),
        "qaoa": {
            "depth": depth,
            "parameterization": "shared-angle-slice" if depth > 1 else "p=1",
            "cost_gate": "rzz(2*gamma)",
            "mixer_gate": "rx(2*beta)",
            "measurement": "computational-basis",
        },
        "execution_plan": {
            "shots_per_condition": shots,
            "randomization_seed": randomization_seed,
            "condition_count": len(conditions),
            "conditions": conditions,
            "measurement_bit_order": _VERTEX_ORDER,
            "qiskit_raw_counts_order": _QISKIT_CLASSICAL_ORDER,
        },
        "analysis_plan": {
            "primary_endpoint": "approximation_ratio",
            "secondary_endpoints": [
                "total_variation_distance",
                "hellinger_distance",
                "jensen_shannon_divergence",
                "absolute_ratio_error",
            ],
            "bootstrap_resamples": bootstrap_resamples,
            "confidence_level": 0.95,
            "interpretation": (
                "Agreement measures predictive fit for this manifest only; it does not identify "
                "a physical noise source or demonstrate general quantum advantage."
            ),
        },
        "provenance": {
            "software": "Quantum Compass",
            "protocol": "pre-specified hardware/simulator comparison",
            "normalization": "All stored bitstrings use vertex order: character i is vertex i.",
        },
    }
    manifest["manifest_id"] = _manifest_digest(manifest)
    validate_experiment_manifest(manifest)
    return manifest


def validate_experiment_manifest(manifest: Mapping) -> Dict:
    """Validate a manifest and return a detached JSON-native copy.

    A checksum mismatch catches accidental edits between circuit submission and
    count analysis.  Intentional protocol amendments should create a new
    manifest rather than overwrite an executed experiment record.
    """
    if not isinstance(manifest, Mapping):
        raise ExperimentalValidationError("Experiment manifest must be a JSON object.")
    normalized = json.loads(json.dumps(_json_ready(dict(manifest))))
    if int(normalized.get("schema_version", -1)) != MANIFEST_SCHEMA_VERSION:
        raise ExperimentalValidationError(
            f"Unsupported manifest schema version {normalized.get('schema_version')!r}."
        )
    if normalized.get("kind") != MANIFEST_KIND:
        raise ExperimentalValidationError("This file is not a Quantum Compass hardware-validation manifest.")
    graph_from_manifest_record(normalized.get("graph", {}))

    qaoa = normalized.get("qaoa")
    execution = normalized.get("execution_plan")
    analysis = normalized.get("analysis_plan")
    if not isinstance(qaoa, Mapping) or not isinstance(execution, Mapping) or not isinstance(analysis, Mapping):
        raise ExperimentalValidationError("Manifest needs qaoa, execution_plan and analysis_plan objects.")
    depth = int(qaoa.get("depth", 0))
    if depth < 1 or depth > 3:
        raise ExperimentalValidationError("Manifest QAOA depth must be in [1, 3].")
    shots = int(execution.get("shots_per_condition", 0))
    if shots < 128:
        raise ExperimentalValidationError("Manifest shots_per_condition must be at least 128.")
    if execution.get("measurement_bit_order") != _VERTEX_ORDER:
        raise ExperimentalValidationError("Manifest measurement convention must be vertex-order.")
    conditions = execution.get("conditions")
    if not isinstance(conditions, list) or not conditions:
        raise ExperimentalValidationError("Manifest needs at least one execution condition.")
    if int(execution.get("condition_count", -1)) != len(conditions):
        raise ExperimentalValidationError("Manifest condition_count does not match its conditions list.")

    condition_ids = set()
    run_orders = set()
    for condition in conditions:
        if not isinstance(condition, Mapping):
            raise ExperimentalValidationError("Every manifest condition must be an object.")
        identifier = str(condition.get("condition_id", "")).strip()
        if not identifier or identifier in condition_ids:
            raise ExperimentalValidationError("Condition IDs must be nonempty and unique.")
        condition_ids.add(identifier)
        run_order = int(condition.get("run_order", 0))
        if run_order < 1 or run_order in run_orders:
            raise ExperimentalValidationError("Condition run_order values must be unique positive integers.")
        run_orders.add(run_order)
        gamma = _require_finite(condition.get("gamma"), f"condition {identifier} gamma")
        beta = _require_finite(condition.get("beta"), f"condition {identifier} beta")
        layer_gammas = [_require_finite(value, f"condition {identifier} gammas") for value in condition.get("gammas", [])]
        layer_betas = [_require_finite(value, f"condition {identifier} betas") for value in condition.get("betas", [])]
        if len(layer_gammas) != depth or len(layer_betas) != depth:
            raise ExperimentalValidationError(f"Condition {identifier} does not provide {depth} gamma/beta values.")
        if not np.allclose(layer_gammas, [gamma] * depth) or not np.allclose(layer_betas, [beta] * depth):
            raise ExperimentalValidationError(
                f"Condition {identifier} is not the declared shared-angle QAOA slice."
            )
    if run_orders != set(range(1, len(conditions) + 1)):
        raise ExperimentalValidationError("Condition run_order must be a complete 1..N permutation.")

    resamples = int(analysis.get("bootstrap_resamples", 0))
    if resamples < 200 or resamples > 20_000:
        raise ExperimentalValidationError("Manifest bootstrap_resamples must be in [200, 20,000].")
    confidence = _require_finite(analysis.get("confidence_level"), "confidence_level")
    if not 0.0 < confidence < 1.0:
        raise ExperimentalValidationError("confidence_level must lie strictly between 0 and 1.")

    expected_id = _manifest_digest(normalized)
    if normalized.get("manifest_id") != expected_id:
        raise ExperimentalValidationError(
            "Manifest checksum mismatch. Use the original downloaded manifest or create a new protocol version."
        )
    return normalized


def manifest_to_json(manifest: Mapping) -> str:
    """Return a validated manifest as stable, human-readable JSON."""
    validated = validate_experiment_manifest(manifest)
    return json.dumps(validated, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def parse_experiment_manifest(raw: Union[bytes, str, Mapping]) -> Dict:
    """Parse and checksum-validate manifest bytes, JSON text, or a mapping."""
    if isinstance(raw, bytes):
        try:
            raw = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ExperimentalValidationError("Manifest must be UTF-8 JSON.") from exc
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ExperimentalValidationError(f"Could not parse manifest JSON: {exc.msg}.") from exc
    return validate_experiment_manifest(raw)


def normalize_bitstring(value, n_qubits: int, source_order: str = _VERTEX_ORDER) -> str:
    """Normalize a count key to the protocol's vertex-order bitstring.

    Qiskit formats a classical register with the highest classical bit at the
    left.  Since the submission script maps qubit ``i`` to classical bit ``i``,
    raw Qiskit keys must be reversed to obtain this project's vertex order.
    """
    n_qubits = int(n_qubits)
    if n_qubits < 1:
        raise ExperimentalValidationError("n_qubits must be positive when normalizing a bitstring.")
    raw = str(value).strip().replace(" ", "").replace("_", "")
    if raw.startswith("0x"):
        try:
            raw = format(int(raw, 16), f"0{n_qubits}b")
        except ValueError as exc:
            raise ExperimentalValidationError(f"Invalid hexadecimal count key {value!r}.") from exc
        source_order = _QISKIT_CLASSICAL_ORDER
    if len(raw) != n_qubits or any(character not in "01" for character in raw):
        raise ExperimentalValidationError(
            f"Bitstring {value!r} must contain exactly {n_qubits} binary characters."
        )
    if source_order == _VERTEX_ORDER:
        return raw
    if source_order == _QISKIT_CLASSICAL_ORDER:
        return raw[::-1]
    raise ExperimentalValidationError(
        f"Unknown source_order {source_order!r}; use {_VERTEX_ORDER!r} or {_QISKIT_CLASSICAL_ORDER!r}."
    )


def normalize_counts(
    counts: Mapping,
    n_qubits: int,
    source_order: str = _VERTEX_ORDER,
) -> Dict[str, int]:
    """Validate and aggregate one mapping of measured outcomes into vertex order."""
    if not isinstance(counts, Mapping) or not counts:
        raise ExperimentalValidationError("Counts must be a nonempty mapping of bitstrings to nonnegative integers.")
    out: Dict[str, int] = {}
    for bitstring, count in counts.items():
        try:
            numeric = float(count)
        except (TypeError, ValueError) as exc:
            raise ExperimentalValidationError(f"Count for {bitstring!r} is not numeric.") from exc
        if not math.isfinite(numeric) or numeric < 0 or not numeric.is_integer():
            raise ExperimentalValidationError(f"Count for {bitstring!r} must be a nonnegative integer.")
        normalized = normalize_bitstring(bitstring, n_qubits, source_order)
        out[normalized] = out.get(normalized, 0) + int(numeric)
    if sum(out.values()) <= 0:
        raise ExperimentalValidationError("Counts must contain at least one measured shot.")
    return dict(sorted(out.items()))


def parse_counts_csv(
    raw: Union[bytes, str],
    manifest: Mapping,
    *,
    source_order: str = _VERTEX_ORDER,
) -> pd.DataFrame:
    """Parse strict long-form measured counts against a validation manifest.

    Required CSV columns are ``condition_id,bitstring,count``.  Optional
    provenance columns such as ``backend``, ``job_id`` and ``manifest_id`` are
    retained.  The resulting frame is normalized to the documented vertex-order
    bitstring convention and duplicate rows are summed.
    """
    validated = validate_experiment_manifest(manifest)
    if isinstance(raw, str):
        raw = raw.encode("utf-8")
    try:
        # Preserve leading zeroes in bitstrings; default CSV inference would
        # turn `0001` into numeric 1 and destroy the measurement convention.
        frame = pd.read_csv(io.BytesIO(raw), dtype=str, keep_default_na=False)
    except Exception as exc:  # noqa: BLE001 - report a concise upload error
        raise ExperimentalValidationError("Could not read measurement data as UTF-8 CSV.") from exc
    columns = {str(column).strip().lower(): column for column in frame.columns}
    missing = [column for column in REQUIRED_COUNTS_COLUMNS if column not in columns]
    if missing:
        raise ExperimentalValidationError(
            "Counts CSV needs columns: condition_id, bitstring, count. Missing: " + ", ".join(missing)
        )
    canonical = frame[[columns["condition_id"], columns["bitstring"], columns["count"]]].copy()
    canonical.columns = list(REQUIRED_COUNTS_COLUMNS)
    canonical["condition_id"] = canonical["condition_id"].astype(str).str.strip()
    if canonical["condition_id"].eq("").any():
        raise ExperimentalValidationError("Counts CSV contains an empty condition_id.")
    canonical["count"] = pd.to_numeric(canonical["count"], errors="coerce")
    if canonical["count"].isna().any() or (~np.isfinite(canonical["count"])).any():
        raise ExperimentalValidationError("Counts CSV includes non-finite count values.")
    if (canonical["count"] < 0).any() or (~np.isclose(canonical["count"], np.round(canonical["count"]))).any():
        raise ExperimentalValidationError("Counts CSV values must be nonnegative integers.")
    canonical["count"] = canonical["count"].astype(np.int64)

    expected_id = validated["manifest_id"]
    if "manifest_id" in columns:
        identifiers = set(frame[columns["manifest_id"]].dropna().astype(str).str.strip()) - {""}
        if identifiers and identifiers != {expected_id}:
            raise ExperimentalValidationError("Counts CSV manifest_id does not match the selected protocol manifest.")
    known_conditions = {
        str(condition["condition_id"])
        for condition in validated["execution_plan"]["conditions"]
    }
    unknown = sorted(set(canonical["condition_id"]) - known_conditions)
    if unknown:
        raise ExperimentalValidationError(
            "Counts CSV references condition IDs not present in this manifest: " + ", ".join(unknown[:5])
        )
    n_qubits = int(validated["graph"]["n_qubits"])
    canonical["bitstring"] = [
        normalize_bitstring(value, n_qubits, source_order=source_order)
        for value in canonical["bitstring"]
    ]
    canonical = canonical.groupby(["condition_id", "bitstring"], as_index=False, sort=True)["count"].sum()
    totals = canonical.groupby("condition_id")["count"].sum()
    if (totals <= 0).any():
        raise ExperimentalValidationError("Every submitted condition must contain at least one measured shot.")
    return canonical.sort_values(["condition_id", "bitstring"], kind="stable").reset_index(drop=True)


def condition_counts_from_frame(frame: pd.DataFrame, manifest: Mapping) -> Dict[str, Dict[str, int]]:
    """Convert a validated long-form count DataFrame into per-condition mappings."""
    validated = validate_experiment_manifest(manifest)
    if not isinstance(frame, pd.DataFrame) or not set(REQUIRED_COUNTS_COLUMNS).issubset(frame.columns):
        raise ExperimentalValidationError("frame must contain condition_id, bitstring and count columns.")
    n_qubits = int(validated["graph"]["n_qubits"])
    out: Dict[str, Dict[str, int]] = {}
    for condition_id, group in frame.groupby("condition_id", sort=True):
        out[str(condition_id)] = normalize_counts(
            dict(zip(group["bitstring"], group["count"])), n_qubits, source_order=_VERTEX_ORDER
        )
    return out


def counts_summary(counts: Mapping, G: nx.Graph) -> Dict[str, float]:
    """Compute measured expected cut and approximation ratio from vertex-order counts."""
    graph = _validate_graph(G)
    normalized = normalize_counts(counts, graph.number_of_nodes(), source_order=_VERTEX_ORDER)
    shots = int(sum(normalized.values()))
    optimal_cut, _ = optimal_maxcut(graph)
    if optimal_cut <= 0:
        raise ExperimentalValidationError("A graph with no positive MaxCut cannot yield an approximation ratio.")
    expected_cut = sum(
        cut_value([int(bit) for bit in bitstring], list(graph.edges()), graph.number_of_nodes()) * count
        for bitstring, count in normalized.items()
    ) / float(shots)
    return {
        "shots": shots,
        "expected_cut": float(expected_cut),
        "optimal_cut": int(optimal_cut),
        "approximation_ratio": float(expected_cut / optimal_cut),
    }


def bootstrap_ratio_interval(
    counts: Mapping,
    G: nx.Graph,
    *,
    resamples: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 0,
) -> Dict[str, float]:
    """Estimate a nonparametric multinomial bootstrap interval for QAOA ratio.

    The interval represents finite-shot uncertainty conditional on the observed
    outcome distribution.  It does not include calibration drift, queue-time
    drift, compilation changes, or model misspecification; the protocol records
    those as experimental limitations rather than hiding them in one number.
    """
    graph = _validate_graph(G)
    resamples = int(resamples)
    if resamples < 200 or resamples > 20_000:
        raise ExperimentalValidationError("resamples must be between 200 and 20,000.")
    confidence_level = _require_finite(confidence_level, "confidence_level")
    if not 0.0 < confidence_level < 1.0:
        raise ExperimentalValidationError("confidence_level must lie strictly between 0 and 1.")
    normalized = normalize_counts(counts, graph.number_of_nodes(), source_order=_VERTEX_ORDER)
    summary = counts_summary(normalized, graph)
    bitstrings = list(normalized)
    observed = np.asarray([normalized[key] for key in bitstrings], dtype=np.int64)
    shots = int(observed.sum())
    probabilities = observed.astype(float) / float(shots)
    costs = np.asarray(
        [cut_value([int(bit) for bit in key], list(graph.edges()), graph.number_of_nodes()) for key in bitstrings],
        dtype=float,
    )
    samples = np.random.default_rng(int(seed)).multinomial(shots, probabilities, size=resamples)
    ratios = (samples @ costs) / float(shots * summary["optimal_cut"])
    alpha = (1.0 - confidence_level) / 2.0
    return {
        **summary,
        "confidence_level": float(confidence_level),
        "lower": float(np.quantile(ratios, alpha)),
        "upper": float(np.quantile(ratios, 1.0 - alpha)),
        "standard_error": float(np.std(ratios, ddof=1)),
        "resamples": int(resamples),
    }


def distribution_metrics(observed: Mapping, predicted: Mapping, n_qubits: int) -> Dict[str, float]:
    """Compare two vertex-order count distributions with three bounded metrics."""
    observed_normalized = normalize_counts(observed, n_qubits, source_order=_VERTEX_ORDER)
    predicted_normalized = normalize_counts(predicted, n_qubits, source_order=_VERTEX_ORDER)
    observed_total = float(sum(observed_normalized.values()))
    predicted_total = float(sum(predicted_normalized.values()))
    support = sorted(set(observed_normalized) | set(predicted_normalized))
    p = np.asarray([observed_normalized.get(key, 0) / observed_total for key in support], dtype=float)
    q = np.asarray([predicted_normalized.get(key, 0) / predicted_total for key in support], dtype=float)
    total_variation = 0.5 * float(np.abs(p - q).sum())
    hellinger = float(np.sqrt(0.5 * np.square(np.sqrt(p) - np.sqrt(q)).sum()))
    midpoint = 0.5 * (p + q)
    p_mask = p > 0.0
    q_mask = q > 0.0
    kl_pm = float(np.sum(p[p_mask] * np.log2(p[p_mask] / midpoint[p_mask])))
    kl_qm = float(np.sum(q[q_mask] * np.log2(q[q_mask] / midpoint[q_mask])))
    js_divergence = 0.5 * (kl_pm + kl_qm)
    return {
        "total_variation_distance": float(total_variation),
        "hellinger_distance": float(hellinger),
        "jensen_shannon_divergence": float(js_divergence),
    }


def analyze_validation_study(
    manifest: Mapping,
    measured_counts: pd.DataFrame,
    simulation_predictions: Optional[Mapping[str, Mapping[str, Mapping]]] = None,
    *,
    bootstrap_seed: int = 73,
) -> Dict:
    """Analyze uploaded hardware counts against one or more labelled simulators.

    ``simulation_predictions`` maps model name to condition ID to a counts
    mapping.  Typical model names are ``ideal`` and ``fake_backend``.  Missing
    predictions are recorded rather than imputed, preserving a transparent
    distinction between observed results and uncomputed comparisons.
    """
    validated = validate_experiment_manifest(manifest)
    graph = graph_from_manifest_record(validated["graph"])
    by_condition = condition_counts_from_frame(measured_counts, validated)
    execution = validated["execution_plan"]
    planned_conditions = list(execution["conditions"])
    condition_ids = [str(condition["condition_id"]) for condition in planned_conditions]
    prediction_models = dict(simulation_predictions or {})
    model_names = sorted(str(name) for name in prediction_models)
    per_condition: List[Dict] = []
    comparison_rows: Dict[str, List[Dict]] = {name: [] for name in model_names}
    resamples = int(validated["analysis_plan"]["bootstrap_resamples"])
    confidence = float(validated["analysis_plan"]["confidence_level"])

    for index, condition in enumerate(planned_conditions):
        condition_id = str(condition["condition_id"])
        if condition_id not in by_condition:
            continue
        observed = by_condition[condition_id]
        interval = bootstrap_ratio_interval(
            observed, graph, resamples=resamples, confidence_level=confidence,
            seed=int(bootstrap_seed) + index,
        )
        row: Dict = {
            "condition_id": condition_id,
            "run_order": int(condition["run_order"]),
            "gamma": float(condition["gamma"]),
            "beta": float(condition["beta"]),
            "observed": interval,
            "models": {},
        }
        for model_name in model_names:
            predicted = prediction_models[model_name].get(condition_id)
            if predicted is None:
                row["models"][model_name] = {"available": False}
                continue
            normalized_prediction = normalize_counts(
                predicted, graph.number_of_nodes(), source_order=_VERTEX_ORDER
            )
            predicted_summary = counts_summary(normalized_prediction, graph)
            metrics = distribution_metrics(observed, normalized_prediction, graph.number_of_nodes())
            signed_error = float(predicted_summary["approximation_ratio"] - interval["approximation_ratio"])
            model_result = {
                "available": True,
                "predicted": predicted_summary,
                "signed_ratio_error": signed_error,
                "absolute_ratio_error": abs(signed_error),
                **metrics,
            }
            row["models"][model_name] = model_result
            comparison_rows[model_name].append({"shots": interval["shots"], **model_result})
        per_condition.append(row)

    summaries: Dict[str, Dict] = {}
    for model_name, rows in comparison_rows.items():
        if not rows:
            summaries[model_name] = {"available_conditions": 0}
            continue
        weights = np.asarray([row["shots"] for row in rows], dtype=float)
        absolute = np.asarray([row["absolute_ratio_error"] for row in rows], dtype=float)
        signed = np.asarray([row["signed_ratio_error"] for row in rows], dtype=float)
        tvd = np.asarray([row["total_variation_distance"] for row in rows], dtype=float)
        hellinger = np.asarray([row["hellinger_distance"] for row in rows], dtype=float)
        jsd = np.asarray([row["jensen_shannon_divergence"] for row in rows], dtype=float)
        summaries[model_name] = {
            "available_conditions": int(len(rows)),
            "weighted_mean_absolute_ratio_error": float(np.average(absolute, weights=weights)),
            "weighted_mean_signed_ratio_error": float(np.average(signed, weights=weights)),
            "weighted_root_mean_squared_ratio_error": float(np.sqrt(np.average(np.square(signed), weights=weights))),
            "weighted_mean_total_variation_distance": float(np.average(tvd, weights=weights)),
            "weighted_mean_hellinger_distance": float(np.average(hellinger, weights=weights)),
            "weighted_mean_jensen_shannon_divergence": float(np.average(jsd, weights=weights)),
        }

    observed_ids = set(by_condition)
    return {
        "manifest_id": validated["manifest_id"],
        "graph": validated["graph"],
        "coverage": {
            "planned_conditions": len(condition_ids),
            "observed_conditions": len(per_condition),
            "missing_condition_ids": [identifier for identifier in condition_ids if identifier not in observed_ids],
            "observed_shots": int(sum(row["observed"]["shots"] for row in per_condition)),
        },
        "conditions": per_condition,
        "model_summary": summaries,
        "interpretation_boundary": validated["analysis_plan"]["interpretation"],
    }


def validation_report_markdown(manifest: Mapping, analysis: Mapping) -> str:
    """Create a transparent Markdown report from a manifest and computed study."""
    validated = validate_experiment_manifest(manifest)
    if str(analysis.get("manifest_id", "")) != validated["manifest_id"]:
        raise ExperimentalValidationError("Analysis manifest_id does not match the requested manifest.")
    coverage = analysis.get("coverage", {})
    lines = [
        "# Quantum Compass — Experimental Validation Report",
        "",
        f"- **Manifest ID:** `{validated['manifest_id']}`",
        f"- **Protocol created:** {validated['created_utc']}",
        f"- **Logical graph:** n={validated['graph']['n_qubits']}, |E|={len(validated['graph']['edges'])}",
        f"- **QAOA depth:** {validated['qaoa']['depth']} ({validated['qaoa']['parameterization']})",
        f"- **Planned / observed conditions:** {coverage.get('planned_conditions', 0)} / {coverage.get('observed_conditions', 0)}",
        f"- **Observed shots:** {coverage.get('observed_shots', 0)}",
        "",
        "## Protocol boundary",
        "",
        str(analysis.get("interpretation_boundary", "No interpretation boundary supplied.")),
        "",
        "## Model agreement summary",
        "",
    ]
    summaries = analysis.get("model_summary", {})
    if not summaries:
        lines.append("No simulator predictions were supplied for this analysis.")
    else:
        lines.extend([
            "| Model | Conditions | Weighted MAE ratio | Weighted TVD | Weighted Hellinger |",
            "|---|---:|---:|---:|---:|",
        ])
        for model_name, summary in summaries.items():
            if not summary.get("available_conditions"):
                lines.append(f"| {model_name} | 0 | — | — | — |")
                continue
            lines.append(
                f"| {model_name} | {summary['available_conditions']} | "
                f"{summary['weighted_mean_absolute_ratio_error']:.5f} | "
                f"{summary['weighted_mean_total_variation_distance']:.5f} | "
                f"{summary['weighted_mean_hellinger_distance']:.5f} |"
            )
    missing = coverage.get("missing_condition_ids", [])
    if missing:
        lines.extend(["", "## Incomplete execution", "", "Missing planned conditions: " + ", ".join(missing)])
    lines.extend([
        "",
        "## Limits",
        "",
        "This report compares uploaded computational-basis counts with explicitly named simulations. "
        "Bootstrap intervals quantify finite-shot uncertainty conditional on observed counts only. They do not "
        "capture device drift, compilation variation, calibration changes, queue-time effects, or model misspecification.",
        "",
    ])
    return "\n".join(lines)
