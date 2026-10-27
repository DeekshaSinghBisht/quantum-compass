# Hardware-validation visual assets

This directory holds static, public-safe assets for the documented IBM Quantum hardware studies.

## Included assets

- `final_ibm_kingston_aggregate_metrics.png` — a cropped export from Quantum Compass’s **Experimental Validation** page showing aggregate ideal-Aer-versus-measured-hardware metrics for the completed six-qubit study (manifest ID `3af2ae6317092dbcbf6a0dde42ee32d4305570f15160e173995ffa97ab408f6c`).
- `final_ibm_kingston_ratio_comparison.png` — the app-exported nine-condition **Pre-specified QAOA ratio comparison** for the same final study. Orange markers/line represent measured hardware; the blue dotted line represents the ideal-Aer reference, in preregistered randomized execution order.

Both figures were exported from the real Quantum Compass application using the completed final-study inputs. They contain no API key, CRN, account usage, dashboard identity information, local filesystem path, browser chrome, or personal profile information.

## Interpretation boundary

These visuals document a small, fixed hardware/simulator comparison. They must be accompanied by the aggregate metrics and limitations in [`../../research/hardware_validation_results.md`](../../research/hardware_validation_results.md). They do not support claims of quantum advantage, a universal hardware-noise model, or physical-noise attribution.

## Publication rule

Use only application-chart exports or reviewed app screenshots. Do not publish IBM account dashboards, API-key pages, CRNs, private workload pages, terminal screenshots, browser profile data, personal email addresses, or account/trial details.
