#!/usr/bin/env python3
"""Submit a pre-specified Quantum Compass QAOA validation manifest to IBM Quantum.

This is intentionally a one-shot command-line program rather than a Streamlit
feature.  A remote quantum job needs a user's own IBM Quantum credentials and
should be submitted deliberately, with a saved manifest and an auditable output
directory.  It never stores a token in the project or output files.

Typical use::

    # Set IBM_QUANTUM_TOKEN securely in your shell first.
    python scripts/submit_ibm_validation.py \
      --manifest qaoa_validation_manifest.json \
      --backend ibm_brisbane \
      --output-dir results/ibm_brisbane_run

Use ``--dry-run`` first to build and inspect the circuits without connecting to
IBM Quantum.  A successful remote run writes ``counts_long.csv`` in the strict
vertex-order schema accepted by page 10, plus raw counts and provenance JSON.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.experimental_validation import (  # noqa: E402
    ExperimentalValidationError,
    graph_from_manifest_record,
    manifest_to_json,
    normalize_counts,
    parse_experiment_manifest,
)


def _utc_now() -> str:
    """Return a compact UTC timestamp for output provenance."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_manifest(path: Path) -> Dict:
    """Load and checksum-validate one UTF-8 experiment-manifest file."""
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ExperimentalValidationError(f"Could not read manifest {path}: {exc}") from exc
    return parse_experiment_manifest(raw)


def build_qaoa_circuits(manifest: Mapping):
    """Build measured circuits in manifest run order for a one-shot CLI process."""
    try:
        from qiskit import __version__ as qiskit_version
    except ImportError as exc:
        raise RuntimeError("Qiskit is required; install requirements.txt before submitting hardware jobs.") from exc
    from core.qaoa_builder import build_qaoa_circuit

    graph = graph_from_manifest_record(manifest["graph"])
    circuits = []
    conditions = sorted(manifest["execution_plan"]["conditions"], key=lambda item: int(item["run_order"]))
    for condition in conditions:
        circuit = build_qaoa_circuit(graph, condition["gammas"], condition["betas"], measure=True)
        circuit.name = f"qc_{condition['condition_id']}"
        circuit.metadata = {
            "quantum_compass_manifest_id": manifest["manifest_id"],
            "condition_id": condition["condition_id"],
            "run_order": int(condition["run_order"]),
            "gamma": float(condition["gamma"]),
            "beta": float(condition["beta"]),
            "measurement_bit_order": "qiskit-classical-order",
            "qiskit_version": qiskit_version,
        }
        circuits.append(circuit)
    return circuits, conditions, qiskit_version


def transpile_for_backend(circuits: Sequence, backend, optimization_level: int):
    """Transpile to a selected backend target across current Qiskit API variants."""
    level = int(optimization_level)
    if level < 0 or level > 3:
        raise ValueError("optimization_level must be between 0 and 3.")
    try:
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

        manager = generate_preset_pass_manager(backend=backend, optimization_level=level)
        compiled = manager.run(list(circuits))
        return list(compiled) if isinstance(compiled, (list, tuple)) else [compiled]
    except Exception:
        # The legacy transpile path remains useful for older Runtime/Qiskit
        # installations where a preset pass manager cannot consume a backend.
        from qiskit import transpile

        compiled = transpile(list(circuits), backend=backend, optimization_level=level)
        return list(compiled) if isinstance(compiled, (list, tuple)) else [compiled]


def _result_counts(pub_result) -> Dict[str, int]:
    """Extract a counts mapping from one Runtime Sampler V2 publication result."""
    data = getattr(pub_result, "data", None)
    candidates = []
    if data is not None:
        for name in ("meas", "c", "c0", "measure"):
            value = getattr(data, name, None)
            if value is not None:
                candidates.append(value)
        if hasattr(data, "join_data"):
            try:
                candidates.append(data.join_data())
            except Exception:
                pass
        if hasattr(data, "items"):
            try:
                candidates.extend(value for _, value in data.items())
            except Exception:
                pass
    for candidate in candidates:
        if hasattr(candidate, "get_counts"):
            raw = candidate.get_counts()
            if isinstance(raw, Mapping):
                return {str(key): int(value) for key, value in raw.items()}
    raise RuntimeError(
        "Could not find a BitArray count register in the Runtime result. "
        "This script expects measured circuits and Sampler V2 publication data."
    )


def submit_with_runtime(
    circuits: Sequence,
    *,
    backend_name: str,
    token: str,
    channel: str,
    instance: str | None,
    shots: int,
    optimization_level: int,
):
    """Submit transpiled circuits with IBM Runtime Sampler V2 and return raw counts."""
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    except ImportError as exc:
        raise RuntimeError(
            "qiskit-ibm-runtime with SamplerV2 is required; install requirements.txt."
        ) from exc

    service_kwargs = {"channel": str(channel), "token": str(token)}
    if instance:
        service_kwargs["instance"] = str(instance)
    service = QiskitRuntimeService(**service_kwargs)
    backend = service.backend(str(backend_name))
    compiled = transpile_for_backend(circuits, backend, optimization_level)
    sampler = SamplerV2(mode=backend)
    job = sampler.run(compiled, shots=int(shots))
    job_id = job.job_id() if hasattr(job, "job_id") else str(job)
    print(f"Submitted IBM Runtime job {job_id}. Keep this ID and wait for results…", flush=True)
    result = job.result()
    if len(result) != len(compiled):
        raise RuntimeError(
            f"Runtime returned {len(result)} publication results for {len(compiled)} submitted circuits."
        )
    raw_counts = [_result_counts(publication) for publication in result]
    return raw_counts, str(job_id), backend, compiled


def _safe_write_json(path: Path, payload: Mapping, overwrite: bool) -> None:
    """Write provenance JSON without silently overwriting an existing experiment."""
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {path}; pass --overwrite to replace it.")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_run_outputs(
    output_dir: Path,
    manifest: Mapping,
    conditions: Sequence[Mapping],
    raw_counts: Sequence[Mapping],
    *,
    backend_name: str,
    job_id: str,
    qiskit_version: str,
    optimization_level: int,
    overwrite: bool,
) -> Tuple[Path, Path, Path]:
    """Write normalized long-form counts, raw counts, and reproducibility metadata."""
    if len(conditions) != len(raw_counts):
        raise ValueError("conditions and raw_counts must have identical lengths.")
    output_dir.mkdir(parents=True, exist_ok=True)
    counts_path = output_dir / "counts_long.csv"
    raw_path = output_dir / "raw_qiskit_counts.json"
    metadata_path = output_dir / "run_metadata.json"
    for path in (counts_path, raw_path, metadata_path):
        if path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite {path}; pass --overwrite to replace it.")

    n_qubits = int(manifest["graph"]["n_qubits"])
    shots = int(manifest["execution_plan"]["shots_per_condition"])
    timestamp = _utc_now()
    rows: List[Dict[str, object]] = []
    raw_by_condition: Dict[str, Dict[str, int]] = {}
    for condition, raw in zip(conditions, raw_counts):
        condition_id = str(condition["condition_id"])
        raw_by_condition[condition_id] = {str(key): int(value) for key, value in raw.items()}
        normalized = normalize_counts(raw, n_qubits, source_order="qiskit-classical-order")
        measured_shots = int(sum(normalized.values()))
        for bitstring, count in normalized.items():
            rows.append({
                "manifest_id": manifest["manifest_id"],
                "condition_id": condition_id,
                "run_order": int(condition["run_order"]),
                "gamma": float(condition["gamma"]),
                "beta": float(condition["beta"]),
                "bitstring": bitstring,
                "count": int(count),
                "shots": measured_shots,
                "planned_shots": shots,
                "backend": str(backend_name),
                "job_id": str(job_id),
                "timestamp_utc": timestamp,
                "measurement_bit_order": "vertex-order",
            })
    fieldnames = [
        "manifest_id", "condition_id", "run_order", "gamma", "beta", "bitstring", "count",
        "shots", "planned_shots", "backend", "job_id", "timestamp_utc", "measurement_bit_order",
    ]
    with counts_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    _safe_write_json(raw_path, raw_by_condition, overwrite=True)
    _safe_write_json(metadata_path, {
        "manifest_id": manifest["manifest_id"],
        "backend": str(backend_name),
        "job_id": str(job_id),
        "submitted_result_retrieved_utc": timestamp,
        "qiskit_version": str(qiskit_version),
        "optimization_level": int(optimization_level),
        "planned_shots_per_condition": shots,
        "condition_count": len(conditions),
        "raw_count_order": "qiskit-classical-order",
        "normalized_count_order": "vertex-order",
        "token_stored": False,
    }, overwrite=True)
    return counts_path, raw_path, metadata_path


def write_dry_run_plan(
    output_dir: Path,
    manifest: Mapping,
    circuits: Sequence,
    conditions: Sequence[Mapping],
    qiskit_version: str,
    overwrite: bool,
) -> Path:
    """Write an inspectable no-network circuit plan for protocol review."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "dry_run_submission_plan.json"
    operation_counts = [dict(circuit.count_ops()) for circuit in circuits]
    payload = {
        "manifest": json.loads(manifest_to_json(manifest)),
        "dry_run_created_utc": _utc_now(),
        "network_submission": False,
        "qiskit_version": qiskit_version,
        "circuits": [
            {
                "condition_id": str(condition["condition_id"]),
                "run_order": int(condition["run_order"]),
                "circuit_name": str(circuit.name),
                "num_qubits": int(circuit.num_qubits),
                "num_clbits": int(circuit.num_clbits),
                "operation_counts": operation_count,
            }
            for condition, circuit, operation_count in zip(conditions, circuits, operation_counts)
        ],
    }
    _safe_write_json(path, payload, overwrite=overwrite)
    return path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the explicit, credential-safe command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Path to a downloaded validation manifest JSON.")
    parser.add_argument("--backend", help="IBM Quantum backend name; required unless --dry-run is used.")
    parser.add_argument("--output-dir", type=Path, default=Path("hardware_validation_output"))
    parser.add_argument("--token-env", default="IBM_QUANTUM_TOKEN", help="Environment variable containing the IBM token.")
    parser.add_argument("--channel", default="ibm_quantum_platform", help="QiskitRuntimeService channel.")
    parser.add_argument("--instance", default=None, help="Optional IBM Runtime instance / hub-group-project.")
    parser.add_argument("--optimization-level", type=int, choices=range(4), default=1)
    parser.add_argument("--dry-run", action="store_true", help="Build and record circuits without authenticating or submitting.")
    parser.add_argument("--overwrite", action="store_true", help="Permit replacement of existing files in output-dir.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Execute a dry-run plan or a credential-gated IBM Runtime submission."""
    args = parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        circuits, conditions, qiskit_version = build_qaoa_circuits(manifest)
        if args.dry_run:
            plan = write_dry_run_plan(
                args.output_dir, manifest, circuits, conditions, qiskit_version, args.overwrite
            )
            print(f"Dry run completed. Inspect {plan}")
            return 0
        if not args.backend:
            raise ExperimentalValidationError("--backend is required for a real IBM Runtime submission.")
        token = os.environ.get(args.token_env, "").strip()
        if not token:
            raise ExperimentalValidationError(
                f"No token found in ${args.token_env}. Export it in your shell; never commit it to this repository."
            )
        raw_counts, job_id, _backend, _compiled = submit_with_runtime(
            circuits,
            backend_name=args.backend,
            token=token,
            channel=args.channel,
            instance=args.instance,
            shots=int(manifest["execution_plan"]["shots_per_condition"]),
            optimization_level=args.optimization_level,
        )
        paths = write_run_outputs(
            args.output_dir, manifest, conditions, raw_counts,
            backend_name=args.backend, job_id=job_id, qiskit_version=qiskit_version,
            optimization_level=args.optimization_level, overwrite=args.overwrite,
        )
        print(f"IBM Runtime job {job_id} completed.")
        print("Wrote:\n  " + "\n  ".join(str(path) for path in paths))
        return 0
    except (ExperimentalValidationError, FileExistsError, RuntimeError, ValueError) as exc:
        print(f"Validation submission failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
