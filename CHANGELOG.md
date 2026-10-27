# Changelog

All notable changes to Quantum Compass are documented in this file. The project follows the spirit of [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html) when a public release is cut.

## [Unreleased]

### Added

- A credential-safe IBM Runtime submission receipt written immediately after a new job receives an ID, plus native `--job-id` recovery for retrieving an existing job without duplicate QPU submission.
- A current client-side Qiskit Runtime sampler path with a compatibility fallback for older supported Runtime clients.
- Token-free, local-environment provenance tooling that writes `environment.json` and SHA-256 checksums only when run from the real study checkout.
- Public-release research guidance: reproducibility, evidence-layout, future-work, contribution, security, and release-process documents.
- GitHub Actions regression checks for the supported Python runtime.
- Citation metadata in `CITATION.cff`.
- A reviewed Docker-hosted application gallery covering Landscape View, Noise Analysis, Space Mode, Advantage Map, Barren Plateau, Hardware Profiles, Mission Planner, Fingerprinting, and Experimental Validation, with captions that distinguish simulator/planning views from real hardware evidence.
- A real app-exported final `ibm_kingston` ratio-comparison figure, with evidence-faithful README/report captions.

### Changed

- The hardware-validation workflow now distinguishes direct submission from recovery in its metadata and receipts.
- Research documentation now records the completed `ibm_kingston` pilot-to-final workflow with a narrow, evidence-faithful interpretation.

### Evidence boundary

- The archived pilot and preregistered final study are not regenerated, amended, or reinterpreted by this maintenance work. The six-qubit final hardware result remains limited to its fixed manifest, backend, compiler configuration, and execution period.
