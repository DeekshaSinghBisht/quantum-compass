# Application visual assets

This directory contains public-safe screenshots captured from the running Docker-hosted Quantum Compass Streamlit application. The screenshots are real interface captures, not generated mock-ups or reconstructed figures.

## Included captures

| Asset | Page / workflow | Evidence class |
|---|---|---|
| `quantum_compass_landscape_view.png` | Four-qubit 3-regular `earth-lab` QAOA contour and 3-D surface at noise level `0.003` | Simulator interface |
| `quantum_compass_noise_analysis.png` | Completed LEO-satellite noise sweep and resonance-model comparison | Simulator analysis |
| `quantum_compass_space_mode.png` | Earth/LEO/deep-space controlled landscape comparison | Simulator sensitivity study |
| `quantum_compass_advantage_map.png` | Fixed-angle QAOA/classical-baseline map configuration workflow | Protocol/configuration interface |
| `quantum_compass_barren_plateau.png` | Trainability phase-diagram configuration workflow | Protocol/configuration interface |
| `quantum_compass_hardware_profiles.png` | IBM Cairo FakeProvider calibration-derived profile | Credential-free simulator profile |
| `quantum_compass_mission_planner.png` | SPOQC-like mission-plan configuration | Planning interface |
| `quantum_compass_fingerprinting.png` | Mystery-landscape classifier workflow | Controlled simulator diagnostic |
| `quantum_compass_experimental_validation.png` | Manifest/pre-registration and scientific-boundary interface | Research protocol interface |

The completed real-hardware figures are intentionally stored separately in [`../hardware_validation/`](../hardware_validation/README.md).

## Privacy and claim boundary

All included files were reviewed for the absence of browser address bars, operating-system chrome, local filesystem paths, account/profile identifiers, API keys, IBM dashboard content, CRNs, and terminal output. These assets demonstrate the user interface and documented workflows only. They must not be presented as direct hardware measurements, calibration evidence, quantum advantage, forensic noise attribution, or mission qualification.

## Publication rule

Future screenshots must be captured from the real app and reviewed before commit. Crop browser chrome, operating-system chrome, account/profile controls, local paths, and unrelated windows. Do not create synthetic UI screenshots or present a simulator visual as hardware evidence. The complete, captioned gallery is [`../../application_gallery.md`](../../application_gallery.md).
