# Quantum Compass Experimental Validation Protocol

## Purpose and scope

This protocol turns Quantum Compass from a simulator-only demonstration into a **reproducible small-QAOA hardware/simulation comparison study**. It is deliberately narrow. It asks whether named simulation models reproduce measurements from a fixed set of QAOA circuits on a specified backend.

It does **not** claim to identify a physical radiation environment, certify a mission profile, establish quantum advantage, or validate all QAOA instances. Any result is limited to the graph, depth, compiled circuits, backend, calibration period, shot budget, and metrics recorded in its manifest.

## Research question

> For a pre-specified small MaxCut instance and a pre-specified shared-angle QAOA slice, how closely do the ideal and calibration-derived simulator predictions match measured computational-basis counts from a named IBM Quantum backend?

The primary endpoint is the measured **approximation ratio** `E[C] / C_max`. Secondary endpoints compare the full count distributions through total variation distance, Hellinger distance, and Jensen–Shannon divergence.

## Hypotheses

Before hardware submission, the experimenter commits to the following qualitative hypotheses:

1. The ideal statevector model will generally be less predictive than a calibration-derived noise model on a noisy backend.
2. Agreement varies over the QAOA angle grid; a single best-ratio point is insufficient to assess a model.
3. Finite-shot uncertainty is only one source of disagreement. Compilation choices, calibration drift, readout error, crosstalk, routing, and non-Markovian effects can dominate it.

No universal numerical threshold defines model “truth.” Results are reported as effect sizes and uncertainty intervals, not pass/fail evidence of a physical mechanism.

## Pre-specified design

The app's **Experimental Validation** page creates a checksum-identified JSON manifest. The manifest fixes all of the following before submission:

- a contiguous-labelled MaxCut graph and its explicit edge list;
- QAOA depth, using `RZZ(2γ)` cost gates and `RX(2β)` mixer gates;
- a compact gamma/beta grid, with the default `3×3` grid spanning `γ ∈ [0, π/2]` and `β ∈ [0, π/4]`;
- a shared-angle convention for depths above one;
- shots per circuit, defaulting to 4,096;
- a randomized circuit execution order generated from a recorded seed;
- vertex-order bitstring convention;
- planned primary and secondary metrics; and
- bootstrap resamples and confidence level.

The zero-angle condition is retained as a control. It helps reveal count-order, readout, or data-ingestion mistakes before interpreting nontrivial QAOA settings.

Do not alter a downloaded manifest after looking at hardware data. Amend the graph, grid, backend strategy, shots, or metrics by creating a new manifest with a new checksum and reporting it as a new experiment.

A checksum detects accidental changes but is **not** a cryptographic timestamp or an external preregistration service. Before submitting a real job, commit the manifest and this protocol to a dated Git tag, repository release, or suitable research archive. Record that immutable public/private reference alongside the resulting IBM job ID.

## Running a real backend study

### 1. Create and archive a manifest

Use page **🧪 Experimental Validation** to generate and download `qaoa_validation_manifest.json`. Save it with the project commit/hash, backend choice, and date in a research folder. For a Git-backed preregistration record, commit the protocol and manifest before submission:

```bash
git add docs/research/experimental_validation_protocol.md qaoa_validation_manifest.json
git commit -m "Preregister QAOA hardware validation manifest"
git tag qaoa-validation-v1
```

### 2. Inspect circuits without contacting a backend

```bash
python scripts/submit_ibm_validation.py \
  --manifest qaoa_validation_manifest.json \
  --output-dir results/protocol_check \
  --dry-run
```

Review `dry_run_submission_plan.json`. Confirm the number of qubits, classical bits, `h`, `rzz`, `rx`, and measurement operations before spending hardware credits.

### 3. Submit to IBM Quantum deliberately

```bash
# Set IBM_QUANTUM_TOKEN securely in your shell before submission.
python scripts/submit_ibm_validation.py \
  --manifest qaoa_validation_manifest.json \
  --backend ibm_brisbane \
  --output-dir results/ibm_brisbane_run
```

Use a backend available to your own IBM Quantum account; `ibm_brisbane` above is a concrete example, not a guaranteed allocation. The CLI transpiles circuits to the selected backend target and prefers the current client-side IBM Runtime sampler, with a compatibility fallback for older supported Runtime clients. As soon as IBM returns a job ID, it writes:

- `submission_receipt.json` — immediate credential-free proof of the accepted job ID and submission configuration;
- `counts_long.csv` — normalized vertex-order counts for analysis after result retrieval;
- `raw_qiskit_counts.json` — unmodified Runtime count keys for auditability; and
- `run_metadata.json` — job ID, backend, timestamp, Qiskit version, Runtime sampler/compiler details, count-order provenance, and direct-versus-recovered result source.

The IBM token is never stored in any output file. Do not commit it, a `.env` file, or Streamlit secrets. If the local client stops after a job ID is issued, do **not** submit again. Recover the original remote job with:

```bash
python scripts/submit_ibm_validation.py \
  --manifest qaoa_validation_manifest.json \
  --backend ibm_brisbane \
  --job-id RECORDED_RUNTIME_JOB_ID \
  --output-dir results/ibm_brisbane_run
```

Recovery writes `job_recovery_receipt.json` and retrieves `service.job(job_id)` without sending a replacement circuit batch.

### 4. Analyze without changing the protocol

Upload the original manifest and `counts_long.csv` into page **🧪 Experimental Validation**. Always compare with the ideal simulator. Enable a named FakeProvider reference only when its backend snapshot is a meaningful reference for the measured backend; otherwise leave it disabled rather than implying a calibration match that does not exist. Download the generated analysis JSON and Markdown report together with the raw data.

## Data conventions and integrity

- The Qiskit submission API normally returns classical strings in descending classical-bit order.
- The project stores normalized **vertex-order** strings: character `i` is the bit assigned to logical graph vertex `i`.
- The submission script reverses Qiskit's raw keys because it maps logical qubit `i` to classical bit `i`.
- The page rejects unknown condition IDs, manifest checksum mismatches, nonbinary keys, noninteger counts, empty conditions, and malformed grids.
- The total counts recorded per condition are preserved rather than assumed to equal the planned shots. This exposes provider-side shot differences or incomplete data.

## Analysis plan

For each observed condition, the app computes:

1. expected cut and approximation ratio;
2. a multinomial bootstrap confidence interval for the ratio, conditional on observed counts;
3. ideal-model predicted ratio and distribution, when selected;
4. calibration-derived fake-backend predicted ratio and distribution, when selected;
5. signed and absolute ratio error; and
6. total variation, Hellinger, and Jensen–Shannon divergence for the full observed versus predicted distribution.

Model summaries are weighted by the number of shots per condition. This avoids treating a partially executed condition as equally informative as a complete one.

The bootstrap interval is **not** a complete hardware uncertainty model. It does not include drift, compiler variation, measurement mitigation uncertainty, backend calibration updates, queue time, crosstalk, leakage, or model misspecification.

## Inclusion and exclusion rules

- Include every condition in the original manifest for which count data are available.
- Report missing conditions explicitly; do not replace them with simulator values.
- Do not delete unfavorable angles, post-select bitstrings, or choose a preferred model after seeing results without describing this as exploratory analysis.
- If a backend fails, a job is cancelled, or a condition returns no counts, preserve its metadata and report the event.
- A repeat run is a new dataset. Keep its manifest ID, job ID, calibration period, and compilation settings distinct from prior runs.

## Minimum publishable evidence package

A credible supplement, repository release, report, or application portfolio entry should include:

1. this protocol version;
2. the immutable manifest JSON;
3. the submission command, dry-run plan, and immediate submission or recovery receipt;
4. `counts_long.csv`, raw count JSON, and run metadata;
5. the exact project commit or release ZIP;
6. analysis JSON/Markdown and plotted results;
7. backend and calibration date/time;
8. environment/checksum records captured from the actual study checkout; and
9. a limitations paragraph distinguishing simulator agreement from physical-noise attribution.

## Interpretation boundaries

A calibration-derived simulation beating the ideal model on one manifest supports a limited statement:

> “For these circuits, this backend, and this execution period, the specified calibration-derived model had lower discrepancy from the observed count distribution than the ideal model.”

It does **not** support stronger statements such as:

- “The model identified the device’s true noise process.”
- “The classifier diagnosed a space environment.”
- “QAOA achieved a quantum advantage.”
- “The result transfers to another backend, date, graph family, depth, or application.”

These boundaries are scientific strengths, not weaknesses: they demonstrate that the study understands what its measurements can and cannot establish.
