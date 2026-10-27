from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path("/app")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qiskit_ibm_runtime import QiskitRuntimeService
from scripts.submit_ibm_validation import (
    _result_counts,
    build_qaoa_circuits,
    load_manifest,
    write_run_outputs,
)


def require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def main() -> None:
    token = require_env("IBM_QUANTUM_TOKEN")
    instance = require_env("IBM_QUANTUM_INSTANCE")
    job_id = require_env("IBM_RUNTIME_JOB_ID")

    manifest_path = Path("studies/final_v1/qaoa_validation_manifest.json")
    output_dir = Path("studies/final_v1/ibm_kingston_run")

    manifest = load_manifest(manifest_path)
    _, conditions, qiskit_version = build_qaoa_circuits(manifest)

    service = QiskitRuntimeService(
        channel="ibm_quantum_platform",
        token=token,
        instance=instance,
    )
    job = service.job(job_id)
    print(f"Retrieved existing IBM Runtime job {job_id}.")
    print(f"Remote job status: {job.status()}")

    result = job.result()
    raw_counts = [_result_counts(publication) for publication in result]

    if len(raw_counts) != len(conditions):
        raise RuntimeError(
            f"Retrieved {len(raw_counts)} result publications for "
            f"{len(conditions)} manifest conditions."
        )

    paths = write_run_outputs(
        output_dir,
        manifest,
        conditions,
        raw_counts,
        backend_name="ibm_kingston",
        job_id=job_id,
        qiskit_version=qiskit_version,
        optimization_level=1,
        overwrite=False,
    )

    print("Recovered completed-job evidence files:")
    for path in paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
