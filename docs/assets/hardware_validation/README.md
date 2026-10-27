# Hardware-validation visual assets

This directory holds reviewed static exports from Quantum Compass’s **Experimental Validation** page for the completed IBM Quantum `ibm_kingston` studies.

## Included evidence triplets

Each completed study has the same three-result set: a pre-specified approximation-ratio comparison, the full per-condition output table, and the aggregate distributional-agreement table.

### Four-qubit pipeline pilot

- `pilot_ibm_kingston_ratio_comparison.png` — four-condition QAOA approximation-ratio comparison in randomized execution order. Orange is measured hardware and blue dotted is the ideal-Aer reference.
- `pilot_ibm_kingston_condition_results.png` — all four retained conditions with angle values, 1,024 shots each, observed ratio, 95% interval, ideal-Aer ratio, total-variation distance, and absolute ratio error.
- `pilot_ibm_kingston_aggregate_metrics.png` — aggregate ideal-Aer-versus-hardware agreement: weighted MAE ratio `0.0079`, RMSE ratio `0.0082`, TVD `0.0642`, Hellinger `0.0559`, and JS divergence `0.0047`.

### Six-qubit preregistered final study

- `final_ibm_kingston_ratio_comparison.png` — nine-condition QAOA approximation-ratio comparison in preregistered randomized execution order. Orange is measured hardware and blue dotted is the ideal-Aer reference.
- `final_ibm_kingston_condition_results.png` — all nine retained conditions with angle values, 4,096 shots each, observed ratio, 95% interval, ideal-Aer ratio, total-variation distance, and absolute ratio error.
- `final_ibm_kingston_aggregate_metrics.png` — aggregate ideal-Aer-versus-hardware agreement: weighted MAE ratio `0.0050`, RMSE ratio `0.0078`, TVD `0.0798`, Hellinger `0.0923`, and JS divergence `0.0152`.

The source material was supplied from completed app output. The final ratio export has only been cropped to remove Plotly controls and excess canvas margin; no plotted value, table cell, or scientific data was changed. PNG metadata was stripped on copy.

## Interpretation boundary

These visuals document two small, fixed hardware/simulator comparisons. They must be read with the provenance and limitations in [`../../research/hardware_validation_results.md`](../../research/hardware_validation_results.md). They do not support claims of quantum advantage, a universal hardware-noise model, device superiority, or physical-noise attribution.

The pilot and final metrics are not a direct device-performance ranking: their graph, angle grid, and shot budget differ. The progression demonstrates an end-to-end pilot followed by a larger pre-specified study on the same named backend.

## Publication rule

Use only app-generated exports or reviewed app screenshots. Do not publish IBM account dashboards, API-key pages, CRNs, private workload pages, terminal screenshots, browser profile data, personal email addresses, account/trial details, or full unredacted local filesystem paths.
