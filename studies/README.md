# Study evidence directory

This directory holds evidence from real and simulated experimental-validation studies. It is intentionally not a source of pre-generated example hardware data: every raw count, manifest, receipt, analysis, checksum, and environment record must come from the actual study checkout and execution environment.

## Preserve a study as an immutable evidence bundle

Keep each study in its own directory. A practical layout is:

```text
studies/
└── descriptive-study-name/
    ├── qaoa_validation_manifest.json
    ├── protocol_check/
    │   └── dry_run_submission_plan.json
    ├── backend_run/
    │   ├── submission_receipt.json       # new submission only
    │   ├── job_recovery_receipt.json     # recovery only, if applicable
    │   ├── counts_long.csv
    │   ├── raw_qiskit_counts.json
    │   └── run_metadata.json
    ├── analysis/
    │   ├── qaoa_hardware_validation_analysis.json
    │   └── qaoa_hardware_validation_report.md
    ├── environment.json
    └── SHA256SUMS.txt
```

Directory names may reflect an approved internal study name. Do not rename, rewrite, or delete the historic pilot/final evidence merely to match this example.

## Evidence rules

1. Generate and commit the manifest before the hardware submission. Preserve the preregistration commit and annotated tag.
2. Run the submission CLI once for a given manifest/backend execution. If the local process ends after a job ID is issued, use `--job-id` to recover that remote job instead of submitting again.
3. Retain both `raw_qiskit_counts.json` and the normalized `counts_long.csv`; they document the count-order transformation used by the app.
4. Store analysis exports beside the manifest and raw results, not as replacements for them.
5. Capture the environment and checksums after the files are complete, from the actual checkout that produced or analysed them:

   ```bash
   python scripts/capture_study_provenance.py --study-dir studies/descriptive-study-name
   ```

   In Docker, run the same command against the bind-mounted study directory:

   ```bash
   docker compose run --rm --no-deps quantum-compass python scripts/capture_study_provenance.py --study-dir studies/descriptive-study-name
   ```

6. Do not place credentials, account screenshots, full CRNs, dashboard exports, or unrelated personal information in this directory.

The detailed protocol and reproducibility guidance are in [`docs/research/experimental_validation_protocol.md`](../docs/research/experimental_validation_protocol.md) and [`REPRODUCIBILITY.md`](../REPRODUCIBILITY.md).
