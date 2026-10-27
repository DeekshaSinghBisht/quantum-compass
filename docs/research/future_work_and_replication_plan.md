# Future Work and Replication Plan

## Scope of the completed evidence

Quantum Compass has a real IBM Quantum workflow consisting of a four-qubit pipeline pilot and a preregistered six-qubit `p=1` 3-regular MaxCut final study on `ibm_kingston`. The final study completed all nine planned angle conditions with 4,096 shots each. Its documented result is deliberately narrow: ideal Aer approximated the observed approximation ratios closely for that fixed manifest and execution period, while the full output distributions retained measurable disagreement.

This is useful feasibility and analysis evidence. It is not a hardware-noise attribution study, a calibration benchmark across devices, a claim of physical radiation effects, or a demonstration of quantum advantage.

## Next high-value replication

The next hardware experiment should be an explicitly labelled replication rather than a modification of the completed study. It should retain the original manifest unchanged when direct comparability is desired and capture a new job ID, backend timestamp, runtime/compiler versions, raw counts, normalized counts, and study-local checksums.

A careful replication sequence is:

1. Select an account-visible operational backend and document the selection criteria before submission.
2. Decide whether the goal is same-backend temporal replication, cross-backend comparison, or a new graph/depth study. These are different questions and should not be pooled without a plan.
3. Commit a new replication note and output directory name before data collection. Do not move the historic preregistration tag.
4. Run a credential-free `--dry-run` circuit inspection.
5. Submit once, retain `submission_receipt.json`, and use `--job-id` recovery if the local client stops.
6. Preserve all conditions, including failed/missing conditions, and report deviations rather than silently replacing them.
7. Run the analysis with the original manifest and store analysis exports beside the raw evidence.
8. Capture `environment.json` and `SHA256SUMS.txt` from the actual final study folder.

## Designs that would strengthen inference

### Temporal and backend replication

Repeat the same fixed circuit family at multiple documented times and, where accessible, on more than one backend. Analyse condition-level results and report variability rather than treating one queue window as a general device property.

### Compiler and calibration sensitivity

Pre-specify optimization levels, transpilation choices, and any calibration-reference model. Where backend metadata are permissible to archive, record their retrieval time and source. Avoid presenting a FakeProvider snapshot as a live calibration match without evidence.

### Larger but still interpretable protocols

Increase graph sizes or QAOA depth only after resource estimates, simulator feasibility, and a revised analysis plan are documented. Keep the zero-angle control, retain all planned angle conditions, and separate exploratory scans from confirmatory endpoints.

### Better uncertainty treatment

The current bootstrap quantifies finite-shot variability conditional on observed counts. A future study could model repeated execution windows, calibration variation, compiler alternatives, and readout-mitigation uncertainty. Such extensions require pre-specified assumptions and should not be retrofitted to force a preferred conclusion.

### Physical-noise studies

Claims about radiation, space environments, or physical noise mechanisms require independent hardware instrumentation, calibration records, environmental measurements, and domain-specific experimental design. The app's phenomenological space profiles are planning and sensitivity tools, not substitutes for that evidence.

## Success criteria for future work

A stronger follow-up should improve auditability and inference at the same time: immutable protocol before data, complete raw evidence, documented recovery behavior, exact analysis inputs, honest environment/checksum records, and conclusions scoped to the data actually collected.
