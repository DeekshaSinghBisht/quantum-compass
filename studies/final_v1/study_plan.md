# QAOA Hardware Validation v1 — Pre-submission Study Plan

## Status
Planned. No final hardware data have been collected.

## Research question
For a pre-specified six-qubit, depth-one QAOA MaxCut circuit family, how closely do ideal simulator predictions match measured computational-basis counts from IBM Quantum backend ibm_kingston?

## Hardware plan
- Backend: ibm_kingston
- Access plan: IBM Quantum Open Plan
- Backend selection rationale: the completed four-qubit pilot established a functioning end-to-end data-collection and analysis workflow on this backend.
- No API key, token, or credential is stored in this study folder.

## Circuit design
- Graph family: 3-regular MaxCut
- Logical qubits: 6
- Graph seed: 42
- QAOA depth: p = 1
- Gamma-beta grid: 3 by 3
- Shots per condition: 4096
- Planned conditions: 9
- Planned total shots: 36864
- Execution-order seed: 2027
- Bitstring convention: vertex order

## Analysis plan
- Primary endpoint: approximation ratio.
- Secondary endpoints: total variation distance, Hellinger distance, Jensen-Shannon divergence, and absolute approximation-ratio error.
- Use all returned conditions. Do not delete unfavourable angles or replace missing hardware data with simulation.
- Compare with the ideal simulator. Do not label an unmatched FakeProvider snapshot as the live ibm_kingston calibration.

## Interpretation boundary
Any result applies only to this graph, p = 1 circuit family, backend, compiler configuration, execution period, shot budget, and analysis protocol. It does not establish a general hardware-noise mechanism, environmental diagnosis, or quantum advantage.
