#!/usr/bin/env python3
"""Submit or recover a pre-specified Quantum Compass QAOA validation study.

A remote quantum job needs the user's own IBM Quantum credentials and should be
submitted deliberately, with a saved manifest and an auditable output directory.
This program never stores a token, API key, CRN, or other credential in the
project or its evidence files.

Typical new submission::

    # Set IBM_QUANTUM_TOKEN securely in the shell first.
    python scripts/submit_ibm_validation.py \
      --manifest qaoa_validation_manifest.json \
      --backend ibm_brisbane \
      --output-dir results/ibm_brisbane_run

Use ``--dry-run`` first to build and inspect circuits without contacting IBM
Quantum. A successful submission writes a ``submission_receipt.json`` as soon
as IBM returns a job ID, then writes normalized counts, raw counts, and runtime
provenance when the job completes.

If a computer closes after submission, do not resubmit the experiment. Recover
the original job by supplying its recorded ID::

    python scripts/submit_ibm_validation.py \
      --manifest qaoa_validation_manifest.json \
      --backend ibm_brisbane \
      --job-id EXISTING_IBM_RUNTIME_JOB_ID \
      --output-dir results/ibm_brisbane_run

The public ``counts_long.csv`` contract uses vertex-order bitstrings: character
``i`` is logical graph vertex ``i``. Raw IBM/Qiskit counts are retained
unchanged in a separate audit file.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

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


SubmissionCallback = Callable[[str, Mapping[str, object]], None]


def _utc_now() -> str:
    """Return a compact, timezone-aware UTC timestamp."""
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
        # This path keeps the CLI usable with older Qiskit installations whose
        # preset pass manager cannot consume the selected backend object.
        from qiskit import transpile

        compiled = transpile(list(circuits), backend=backend, optimization_level=level)
        return list(compiled) if isinstance(compiled, (list, tuple)) else [compiled]


def _result_counts(pub_result) -> Dict[str, int]:
    """Extract a count mapping from one Runtime sampler publication result."""
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
        "This script expects measured circuits and sampler publication data."
    )


def _runtime_client_version() -> str:
    """Return the installed Runtime client version without persisting credentials."""
    try:
        import qiskit_ibm_runtime

        return str(getattr(qiskit_ibm_runtime, "__version__", "unknown"))
    except ImportError:
        return "unknown"


def _make_runtime_service(*, token: str, channel: str, instance: str | None):
    """Create a credential-scoped Runtime service without saving an account locally."""
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
    except ImportError as exc:
        raise RuntimeError("qiskit-ibm-runtime is required; install requirements.txt.") from exc

    service_kwargs: Dict[str, str] = {"channel": str(channel), "token": str(token)}
    if instance:
        service_kwargs["instance"] = str(instance)
    return QiskitRuntimeService(**service_kwargs)


def _make_runtime_sampler(backend):
    """Prefer Runtime's current client-side sampler, with a legacy fallback.

    qiskit-ibm-runtime 0.50 introduced ``executor_sampler.Sampler`` as the
    client-side replacement for deprecated top-level ``SamplerV2``. Both expose
    the V2 ``run(pubs, shots=...)`` contract required by this CLI. Older
    compatible installations fall back to the legacy implementation.
    """
    try:
        from qiskit_ibm_runtime.executor_sampler import Sampler

        return Sampler(mode=backend), "client-side-executor-sampler"
    except ImportError:
        try:
            from qiskit_ibm_runtime import SamplerV2
        except ImportError as exc:
            raise RuntimeError(
                "No compatible IBM Runtime sampler was found; install qiskit-ibm-runtime."
            ) from exc
        return SamplerV2(mode=backend), "legacy-server-side-sampler-v2"


def _safe_write_json(path: Path, payload: Mapping, overwrite: bool) -> None:
    """Write sorted provenance JSON without silently overwriting evidence."""
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {path}; pass --overwrite to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _require_output_paths_available(paths: Sequence[Path], overwrite: bool) -> None:
    """Fail before remote submission if an evidence directory would be overwritten."""
    if overwrite:
        return
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing evidence files: " + ", ".join(existing) + ". "
            "Use a new output directory, recover an existing job with --job-id, or pass --overwrite deliberately."
        )


def _new_submission_paths(output_dir: Path) -> Tuple[Path, Path, Path, Path]:
    """Return the files a new remote submission reserves before it contacts IBM."""
    return (
        output_dir / "counts_long.csv",
        output_dir / "raw_qiskit_counts.json",
        output_dir / "run_metadata.json",
        output_dir / "submission_receipt.json",
    )


def _recovery_paths(output_dir: Path) -> Tuple[Path, Path, Path, Path]:
    """Return output files that a recovered existing job is allowed to create."""
    return (
        output_dir / "counts_long.csv",
        output_dir / "raw_qiskit_counts.json",
        output_dir / "run_metadata.json",
        output_dir / "job_recovery_receipt.json",
    )


def write_submission_receipt(
    output_dir: Path,
    manifest: Mapping,
    *,
    backend_name: str,
    job_id: str,
    optimization_level: int,
    runtime_client_version: str,
    sampler_implementation: str,
    channel: str,
    instance_provided: bool,
    overwrite: bool,
) -> Path:
    """Persist a credential-free receipt immediately after job acceptance."""
    path = output_dir / "submission_receipt.json"
    payload = {
        "schema_version": 1,
        "event": "submitted_waiting_for_result",
        "recorded_utc": _utc_now(),
        "manifest_id": str(manifest["manifest_id"]),
        "backend": str(backend_name),
        "job_id": str(job_id),
        "planned_shots_per_condition": int(manifest["execution_plan"]["shots_per_condition"]),
        "condition_count": int(manifest["execution_plan"]["condition_count"]),
        "optimization_level": int(optimization_level),
        "runtime_channel": str(channel),
        "instance_provided": bool(instance_provided),
        "qiskit_ibm_runtime_version": str(runtime_client_version),
        "sampler_implementation": str(sampler_implementation),
        "token_stored": False,
    }
    _safe_write_json(path, payload, overwrite=overwrite)
    return path


def write_job_recovery_receipt(
    output_dir: Path,
    manifest: Mapping,
    *,
    backend_name: str,
    job_id: str,
    remote_status: str,
    optimization_level: int,
    runtime_client_version: str,
    channel: str,
    instance_provided: bool,
    overwrite: bool,
) -> Path:
    """Record retrieval of an already-submitted Runtime job without credentials."""
    path = output_dir / "job_recovery_receipt.json"
    payload = {
        "schema_version": 1,
        "event": "recovered_existing_job",
        "recorded_utc": _utc_now(),
        "manifest_id": str(manifest["manifest_id"]),
        "backend": str(backend_name),
        "job_id": str(job_id),
        "remote_status_before_result_retrieval": str(remote_status),
        "planned_shots_per_condition": int(manifest["execution_plan"]["shots_per_condition"]),
        "condition_count": int(manifest["execution_plan"]["condition_count"]),
        "optimization_level": int(optimization_level),
        "runtime_channel": str(channel),
        "instance_provided": bool(instance_provided),
        "qiskit_ibm_runtime_version": str(runtime_client_version),
        "token_stored": False,
    }
    _safe_write_json(path, payload, overwrite=overwrite)
    return path


def submit_with_runtime(
    circuits: Sequence,
    *,
    backend_name: str,
    token: str,
    channel: str,
    instance: str | None,
    shots: int,
    optimization_level: int,
    on_submitted: SubmissionCallback | None = None,
):
    """Submit circuits, record the returned job ID, then retrieve raw counts."""
    service = _make_runtime_service(token=token, channel=channel, instance=instance)
    backend = service.backend(str(backend_name))
    compiled = transpile_for_backend(circuits, backend, optimization_level)
    sampler, sampler_implementation = _make_runtime_sampler(backend)
    runtime_client_version = _runtime_client_version()

    job = sampler.run(compiled, shots=int(shots))
    job_id = str(job.job_id() if hasattr(job, "job_id") else job)
    submission_details: Dict[str, object] = {
        "runtime_client_version": runtime_client_version,
        "sampler_implementation": sampler_implementation,
        "runtime_channel": str(channel),
        "instance_provided": bool(instance),
    }
    print(f"Submitted IBM Runtime job {job_id}. Keep this ID and wait for results…", flush=True)
    if on_submitted is not None:
        try:
            on_submitted(job_id, submission_details)
        except Exception as exc:  # noqa: BLE001
            # The remote job already exists. The visible job ID lets users
            # recover it later even if a local filesystem receipt cannot be written.
            print(f"WARNING: could not write submission receipt: {exc}", file=sys.stderr, flush=True)

    result = job.result()
    if len(result) != len(compiled):
        raise RuntimeError(
            f"Runtime returned {len(result)} publication results for {len(compiled)} submitted circuits."
        )
    raw_counts = [_result_counts(publication) for publication in result]
    return raw_counts, job_id, backend, compiled, submission_details


def _runtime_job_backend_name(job) -> str | None:
    """Best-effort backend name extraction for a retrieved Runtime job."""
    try:
        accessor = getattr(job, "backend", None)
        backend = accessor() if callable(accessor) else accessor
        if backend is None:
            return None
        name = getattr(backend, "name", backend)
        name = name() if callable(name) else name
        return str(name) if name else None
    except Exception:  # noqa: BLE001
        # Result recovery must still be possible when a client version cannot
        # expose historical backend metadata after successfully returning counts.
        return None


def recover_existing_job_with_runtime(
    *,
    job_id: str,
    backend_name: str,
    token: str,
    channel: str,
    instance: str | None,
    expected_condition_count: int,
):
    """Retrieve raw counts for an existing IBM Runtime job without resubmitting it."""
    service = _make_runtime_service(token=token, channel=channel, instance=instance)
    job = service.job(str(job_id))
    remote_status = str(job.status()) if hasattr(job, "status") else "unknown"
    print(f"Retrieved existing IBM Runtime job {job_id} (status: {remote_status}).", flush=True)
    result = job.result()
    if len(result) != int(expected_condition_count):
        raise RuntimeError(
            f"Runtime returned {len(result)} publication results for "
            f"{expected_condition_count} manifest conditions."
        )
    raw_counts = [_result_counts(publication) for publication in result]
    remote_backend_name = _runtime_job_backend_name(job)
    if remote_backend_name is not None and remote_backend_name != str(backend_name):
        raise RuntimeError(
            f"Recovered job {job_id} reports backend {remote_backend_name!r}, "
            f"but --backend was {backend_name!r}. Re-run recovery with the recorded backend name."
        )
    details = {
        "runtime_client_version": _runtime_client_version(),
        "runtime_channel": str(channel),
        "instance_provided": bool(instance),
        "remote_status_before_result_retrieval": remote_status,
        "result_source": "recovered_existing_job",
    }
    if remote_backend_name is not None:
        details["remote_backend_name"] = remote_backend_name
    return raw_counts, str(job_id), details


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
    runtime_client_version: str | None = None,
    sampler_implementation: str | None = None,
    result_source: str = "direct_submission",
    submission_receipt_path: str | None = None,
) -> Tuple[Path, Path, Path]:
    """Write normalized counts, raw counts, and non-secret reproducibility metadata."""
    if len(conditions) != len(raw_counts):
        raise ValueError("conditions and raw_counts must have identical lengths.")
    output_dir.mkdir(parents=True, exist_ok=True)
    counts_path = output_dir / "counts_long.csv"
    raw_path = output_dir / "raw_qiskit_counts.json"
    metadata_path = output_dir / "run_metadata.json"
    _require_output_paths_available((counts_path, raw_path, metadata_path), overwrite)

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
    _safe_write_json(raw_path, raw_by_condition, overwrite=overwrite)
    metadata: Dict[str, object] = {
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
        "result_source": str(result_source),
        "token_stored": False,
    }
    if runtime_client_version is not None:
        metadata["qiskit_ibm_runtime_version"] = str(runtime_client_version)
    if sampler_implementation is not None:
        metadata["sampler_implementation"] = str(sampler_implementation)
    if submission_receipt_path is not None:
        # Keep output metadata portable and public-safe: the receipt's basename
        # is sufficient to locate it beside the run files and reveals no local path.
        metadata["result_receipt_file"] = Path(submission_receipt_path).name
    _safe_write_json(metadata_path, metadata, overwrite=overwrite)
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
    parser.add_argument("--backend", help="IBM Quantum backend name; required for a real submission or --job-id recovery.")
    parser.add_argument("--output-dir", type=Path, default=Path("hardware_validation_output"))
    parser.add_argument("--token-env", default="IBM_QUANTUM_TOKEN", help="Environment variable containing the IBM token.")
    parser.add_argument("--channel", default="ibm_quantum_platform", help="QiskitRuntimeService channel.")
    parser.add_argument("--instance", default=None, help="Optional IBM Runtime instance / CRN / service name.")
    parser.add_argument("--optimization-level", type=int, choices=range(4), default=1)
    parser.add_argument("--dry-run", action="store_true", help="Build and record circuits without authenticating or submitting.")
    parser.add_argument(
        "--job-id",
        default=None,
        help="Recover an existing IBM Runtime job by ID instead of submitting a new job.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Permit replacement of existing files in output-dir.")
    args = parser.parse_args(argv)
    if args.dry_run and args.job_id:
        parser.error("--dry-run and --job-id cannot be used together.")
    if args.job_id and not args.backend:
        parser.error("--backend is required with --job-id so recovered evidence records its intended backend.")
    return args


def _token_from_environment(args: argparse.Namespace) -> str:
    """Read a token only from the configured process environment variable."""
    token = os.environ.get(args.token_env, "").strip()
    if not token:
        raise ExperimentalValidationError(
            f"No token found in ${args.token_env}. Export it in your shell; never commit it to this repository."
        )
    return token


def main(argv: Sequence[str] | None = None) -> int:
    """Execute a dry run, a new credential-gated submission, or job recovery."""
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
        token = _token_from_environment(args)
        planned_shots = int(manifest["execution_plan"]["shots_per_condition"])

        if args.job_id:
            _require_output_paths_available(_recovery_paths(args.output_dir), args.overwrite)
            raw_counts, job_id, recovery_details = recover_existing_job_with_runtime(
                job_id=str(args.job_id),
                backend_name=str(args.backend),
                token=token,
                channel=str(args.channel),
                instance=args.instance,
                expected_condition_count=len(conditions),
            )
            recovery_receipt = write_job_recovery_receipt(
                args.output_dir,
                manifest,
                backend_name=str(args.backend),
                job_id=job_id,
                remote_status=str(recovery_details["remote_status_before_result_retrieval"]),
                optimization_level=int(args.optimization_level),
                runtime_client_version=str(recovery_details["runtime_client_version"]),
                channel=str(args.channel),
                instance_provided=bool(args.instance),
                overwrite=args.overwrite,
            )
            paths = write_run_outputs(
                args.output_dir,
                manifest,
                conditions,
                raw_counts,
                backend_name=str(args.backend),
                job_id=job_id,
                qiskit_version=qiskit_version,
                optimization_level=int(args.optimization_level),
                overwrite=args.overwrite,
                runtime_client_version=str(recovery_details["runtime_client_version"]),
                result_source="recovered_existing_job",
                submission_receipt_path=str(recovery_receipt),
            )
            print(f"Recovered IBM Runtime job {job_id}.")
            print("Wrote:\n  " + "\n  ".join(str(path) for path in (recovery_receipt, *paths)))
            return 0

        _require_output_paths_available(_new_submission_paths(args.output_dir), args.overwrite)
        receipt_path: Path | None = None

        def record_submission(job_id: str, details: Mapping[str, object]) -> None:
            nonlocal receipt_path
            receipt_path = write_submission_receipt(
                args.output_dir,
                manifest,
                backend_name=str(args.backend),
                job_id=job_id,
                optimization_level=int(args.optimization_level),
                runtime_client_version=str(details["runtime_client_version"]),
                sampler_implementation=str(details["sampler_implementation"]),
                channel=str(args.channel),
                instance_provided=bool(args.instance),
                overwrite=args.overwrite,
            )
            print(f"Saved submission receipt: {receipt_path}", flush=True)

        raw_counts, job_id, _backend, _compiled, submission_details = submit_with_runtime(
            circuits,
            backend_name=str(args.backend),
            token=token,
            channel=str(args.channel),
            instance=args.instance,
            shots=planned_shots,
            optimization_level=int(args.optimization_level),
            on_submitted=record_submission,
        )
        paths = write_run_outputs(
            args.output_dir,
            manifest,
            conditions,
            raw_counts,
            backend_name=str(args.backend),
            job_id=job_id,
            qiskit_version=qiskit_version,
            optimization_level=int(args.optimization_level),
            overwrite=args.overwrite,
            runtime_client_version=str(submission_details["runtime_client_version"]),
            sampler_implementation=str(submission_details["sampler_implementation"]),
            result_source="direct_submission",
            submission_receipt_path=str(receipt_path) if receipt_path is not None else None,
        )
        print(f"IBM Runtime job {job_id} completed.")
        print("Wrote:\n  " + "\n  ".join(str(path) for path in paths))
        return 0
    except (ExperimentalValidationError, FileExistsError, RuntimeError, ValueError) as exc:
        print(f"Validation submission failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
