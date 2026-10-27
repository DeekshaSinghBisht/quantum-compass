# Evidence Layout and Publication Guidance

This document describes a public-safe way to organize the completed Quantum Compass hardware-validation evidence without changing the frozen preregistration or inventing supplemental records.

## What the published evidence should contain

For each study, retain the original artifacts produced by the actual workflow:

- pre-data manifest JSON and its manifest ID;
- dry-run circuit plan;
- immediate submission receipt or recovery receipt, when applicable;
- raw Qiskit count mapping;
- normalized long-form count CSV;
- run metadata;
- app-generated analysis JSON and Markdown report;
- approved visual exports; and
- environment/checksum records captured from the actual study checkout.

The completed final study's public interpretation belongs in [`hardware_validation_results.md`](hardware_validation_results.md), not in a marketing-only caption divorced from limitations.

## Safe visual assets

The repository uses two safe, app-generated hardware-result exports:

```text
docs/assets/hardware_validation/final_ibm_kingston_aggregate_metrics.png
docs/assets/hardware_validation/final_ibm_kingston_ratio_comparison.png
```

The first displays real final aggregate analysis metrics. The second is the real nine-condition, hardware-versus-ideal approximation-ratio comparison in preregistered randomized execution order. Neither reveals a token, dashboard, local path, browser profile, or personal account context. They are suitable for README-level context only when accompanied by the claim boundary in the hardware-results report.

The repository also includes a reviewed real-application gallery at:

```text
docs/application_gallery.md
docs/assets/application/
```

It demonstrates the simulator, planning, and protocol interfaces and is explicitly captioned by evidence class; none of these application captures may be presented as hardware evidence.

Do not publish terminal screenshots, IBM Quantum dashboards, browser tabs, credential prompts, service-instance pages, or full unredacted local filesystem paths. A visual being authentic does not automatically make it public-safe. Do not redraw a plot from prose values or create a stand-in chart: retain the exact data and app-export route used for every public figure.

## Public claim language

Appropriate language is:

> For this pre-specified six-qubit circuit family, backend, compiler configuration, and execution period, ideal Aer predicted observed approximation ratios closely, while full computational-basis distributions showed non-zero mismatch.

Avoid language that asserts a universal hardware model, physical-noise diagnosis, device superiority, QAOA quantum advantage, or generalization to other backends/times/instances.

## Before release

Use [`public_release_checklist.md`](public_release_checklist.md), [`RELEASE.md`](../../RELEASE.md), and [`REPRODUCIBILITY.md`](../../REPRODUCIBILITY.md). The final decision to make the repository public should follow a history-wide secret/privacy review, including commit-author email addresses and attachments.
