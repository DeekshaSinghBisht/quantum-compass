# Contributing to Quantum Compass

Thank you for improving Quantum Compass. The project welcomes code, documentation, accessibility, testing, and research-methodology contributions that preserve its transparent scientific scope.

## Ground rules

- Keep claims proportionate to the evidence. Simulator studies, fake-backend studies, and real-hardware studies are different evidence classes and must be labelled accordingly.
- Never add API keys, IBM Quantum tokens, CRNs, `.env` files, Streamlit secrets, browser exports containing account details, or other credentials to source control.
- Never alter a manifest, raw-count file, preregistration tag, or reported analysis to improve a result after hardware data exist. A repeat experiment is a new study with its own provenance.
- Do not present synthetic counts, screenshots, calibration values, or model outputs as real hardware evidence.
- Keep public-facing visuals free of personal names, local paths, account identifiers, dashboard content, and credentials unless the owner has explicitly approved disclosure.

## Local development

Use the supported Python version declared in `runtime.txt` and install the pinned project constraints:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest tests/test_core.py -q
python -m compileall -q app.py core utils pages scripts tests
```

For an interactive check, run `streamlit run app.py`. The app's Qiskit/Aer work is intentionally isolated in a worker process; preserve that boundary when changing UI or simulation code.

## Making a change

1. Create a focused branch from the current default branch.
2. Read the surrounding tests and documentation before editing behavior.
3. Add or update tests for every behavior change, especially manifest validation, count-order conversion, evidence output, and credential handling.
4. Run the regression and compilation checks shown above.
5. Explain the scientific or user-facing impact in the pull request, including any changed assumptions or limitations.

## Hardware-validation contributions

The submission CLI must remain deliberate and recoverable. A new network submission must only happen when the user explicitly requests it; recovery must use a supplied existing job ID. Output records must be credential-free and must retain raw Qiskit counts alongside normalized vertex-order counts.

See [`docs/research/experimental_validation_protocol.md`](docs/research/experimental_validation_protocol.md), [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md), and [`docs/research/public_release_checklist.md`](docs/research/public_release_checklist.md) before changing study-related code or documentation.

## Reporting issues

Use a clear, reproducible issue description: Python version, package versions, command, expected behavior, observed behavior, and a minimal non-sensitive example. Do not paste tokens, private manifests, raw account pages, or personal filesystem paths into a public issue. For a security-sensitive report, follow [`SECURITY.md`](SECURITY.md).
