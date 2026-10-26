# Quantum Compass — Experimental Validation Report

- **Manifest ID:** `eb79bbd427e5f77d949dcc46ebb6eff16e64f48469558ccbd43cb75d4211ec30`
- **Protocol created:** 2026-10-26T12:39:10+00:00
- **Logical graph:** n=4, |E|=6
- **QAOA depth:** 1 (p=1)
- **Planned / observed conditions:** 4 / 4
- **Observed shots:** 4096

## Protocol boundary

Agreement measures predictive fit for this manifest only; it does not identify a physical noise source or demonstrate general quantum advantage.

## Model agreement summary

| Model | Conditions | Weighted MAE ratio | Weighted TVD | Weighted Hellinger |
|---|---:|---:|---:|---:|
| ideal_aer | 4 | 0.00787 | 0.06421 | 0.05586 |

## Limits

This report compares uploaded computational-basis counts with explicitly named simulations. Bootstrap intervals quantify finite-shot uncertainty conditional on observed counts only. They do not capture device drift, compilation variation, calibration changes, queue-time effects, or model misspecification.
