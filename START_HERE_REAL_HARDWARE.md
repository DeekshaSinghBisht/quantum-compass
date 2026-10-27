# Start here: from the ZIP file to a real IBM Quantum validation study

This guide is for someone who has **only the Quantum Compass ZIP file** and wants to turn it into an honest, reproducible hardware-validation result.

> **Prefer Docker rather than installing Python locally?** Skip to [`DOCKER_HARDWARE_GUIDE.md`](DOCKER_HARDWARE_GUIDE.md). Docker still runs the same browser interface, but packages Streamlit, Qiskit, and the submission CLI inside a container.

You will complete two runs:

1. a small **pilot** to prove your computer, account, and data pipeline work; and
2. a pre-specified **final study** whose real hardware data can be included in a portfolio or technical report.

Do not describe anything as a hardware experiment until the IBM Runtime job has returned real counts and you have preserved its outputs.

---

## What you need before beginning

- A laptop or desktop with internet access.
- Python **3.12**.
- A modern browser.
- Git, strongly recommended for preserving the protocol before data collection.
- Later, an IBM Quantum Platform account with access to a physical backend and an API key. Access, queue time, and any usage limits depend on your account; check them before submitting a job.

You do **not** need an IBM token for installation, app exploration, manifest creation, tests, or a CLI dry run.

Never paste an IBM API key into source code, a chat, GitHub, `.env`, Streamlit Secrets, screenshots, or the app itself.

---

## Part A — extract and install the project

### A1. Extract the ZIP

Put `quantum-compass-docker.zip` in a location you can find, such as Downloads, then extract it. It creates a `quantum-compass` folder.

### A2. Open a terminal in the extracted project folder

Use **one** of the following command blocks.

#### Windows PowerShell

Open PowerShell, then run:

```powershell
cd "$HOME\Downloads"
Expand-Archive -Path .\quantum-compass-docker.zip -DestinationPath .\quantum-compass-release
cd .\quantum-compass-release\quantum-compass
```

If you extracted with File Explorer instead, use `cd` to enter the resulting `quantum-compass` folder.

#### macOS or Linux terminal

```bash
cd ~/Downloads
unzip quantum-compass-docker.zip -d quantum-compass-release
cd quantum-compass-release/quantum-compass
```

### A3. Check Python

The project is designed for Python 3.12.

**Windows:**

```powershell
py -3.12 --version
```

**macOS/Linux:**

```bash
python3.12 --version
```

You should see a version beginning with `Python 3.12`.

If that command fails, install Python 3.12 from <https://www.python.org/downloads/>. On Windows, select **Add Python to PATH** during installation, close PowerShell, reopen it, and retry.

### A4. Create an isolated Python environment

This prevents Quantum Compass packages from interfering with other projects.

**Windows PowerShell:**

```powershell
py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**macOS/Linux:**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The install can take several minutes because Qiskit and its simulation packages are substantial. Leave the terminal open until it finishes without an error.

### A5. Verify the installation

Run this in the activated environment:

```bash
python -m pytest tests/test_core.py -q
python -m compileall -q app.py core utils pages scripts tests
```

On Windows, the same commands work after environment activation. A successful test run should report passing tests and the compilation command should return quietly.

### A6. Start the app

```bash
streamlit run app.py
```

Open the local URL printed in the terminal, normally:

```text
http://localhost:8501
```

Keep this terminal open while using the app. To stop the app later, click in the terminal and press `Ctrl+C`.

**Checkpoint:** you should see the Quantum Compass home page and ten sidebar pages, including **🧪 Experimental Validation**.

---

## Part B — create a Git evidence trail before collecting data

A ZIP file has no built-in history. Git lets you preserve what you planned before seeing the data.

### B1. Install/check Git

Run:

```bash
git --version
```

If unavailable, install Git from <https://git-scm.com/downloads> and reopen the terminal.

### B2. Make a baseline commit

From the `quantum-compass` folder:

```bash
git init
git add .
git commit -m "Initial Quantum Compass release"
git branch -M main
```

If Git asks who you are, configure a name and email once, then rerun the commit:

```bash
git config --global user.name "Your Name"
git config --global user.email "your-email@example.com"
```

For stronger evidence, create a private or public GitHub repository and push this baseline. Use the exact remote URL GitHub gives your new repository; do not invent one.

```bash
git remote add origin YOUR-REPOSITORY-URL-FROM-GITHUB
git push -u origin main
```

A local Git commit is useful. A pushed commit/tag or an archival release is stronger because it is harder to rewrite silently.

---

## Part C — run a no-cost pilot first

The pilot verifies the complete workflow. It is **not** your final study.

### C1. Create the pilot manifest in the app

Open **🧪 Experimental Validation**. In the left sidebar choose:

| Control | Pilot value |
|---|---:|
| Graph family | `3-regular` |
| Logical qubits n | `4` |
| QAOA depth | `1` |
| Angle-grid size | `2` |
| Shots per circuit | `1024` |
| Graph seed | leave the default (`42`) |
| Execution-order seed | leave the default (`2027`) |

Click **Create checksum-identified validation manifest**.

Under **1. Pre-specify the experiment**, click **Download preregistered manifest**. Save the file as downloaded:

```text
qaoa_validation_manifest.json
```

### C2. Put the pilot manifest in the project

Create a folder named `studies/pilot` inside the project, then move the downloaded manifest into it.

**Windows PowerShell:**

```powershell
New-Item -ItemType Directory -Force -Path studies\pilot | Out-Null
Move-Item "$HOME\Downloads\qaoa_validation_manifest.json" studies\pilot\
```

**macOS/Linux:**

```bash
mkdir -p studies/pilot
mv ~/Downloads/qaoa_validation_manifest.json studies/pilot/
```

If your browser downloaded it elsewhere, move it using File Explorer/Finder or adjust the source path.

### C3. Run the dry run

In the project terminal, with the virtual environment active:

```bash
python scripts/submit_ibm_validation.py --manifest studies/pilot/qaoa_validation_manifest.json --output-dir studies/pilot/protocol_check --dry-run
```

This does **not** contact IBM Quantum and does **not** need a token. It should create:

```text
studies/pilot/protocol_check/dry_run_submission_plan.json
```

Open that JSON file in a text editor. For this four-qubit 3-regular pilot, each circuit should have:

```text
num_qubits: 4
num_clbits: 4
h: 4
rzz: 6
rx: 4
measure: 4
```

**Checkpoint:** if the dry run works, your local installation and circuit-generation workflow are ready.

---

## Part D — set up IBM Quantum access safely

### D1. Create/login to IBM Quantum Platform

Create or sign in to an IBM Quantum Platform account. From its home/dashboard area, create an API key and check which **physical** backends your account can access.

Current official references:

- IBM's first-hardware-job setup: <https://docs.quantum.ibm.com/start/hello-world>
- `QiskitRuntimeService` authentication/API reference: <https://docs.quantum.ibm.com/api/qiskit-ibm-runtime/qiskit_ibm_runtime.QiskitRuntimeService>

The name `ibm_brisbane` is only an example. Use it only if it appears as an accessible hardware backend in **your** account.

If the dashboard shows an instance/CRN requirement, copy it somewhere private. You may add it to the submission command later with `--instance`; it is not a secret like an API key, but do not guess it.

### D2. Put the API key into the current terminal only

Do this only after activating the project environment and only in the terminal you will use for submission.

**Windows PowerShell:**

```powershell
$secureToken = Read-Host -AsSecureString "IBM Quantum API key"
$env:IBM_QUANTUM_TOKEN = [System.Net.NetworkCredential]::new("", $secureToken).Password
Remove-Variable secureToken
```

**macOS/Linux:**

```bash
read -rs -p "IBM Quantum API key: " IBM_QUANTUM_TOKEN; echo
export IBM_QUANTUM_TOKEN
```

Confirm that a value exists without printing it:

**Windows PowerShell:**

```powershell
if ($env:IBM_QUANTUM_TOKEN) { "Token is set for this terminal." }
```

**macOS/Linux:**

```bash
test -n "$IBM_QUANTUM_TOKEN" && echo "Token is set for this terminal."
```

Closing the terminal removes this session-only value. That is desirable.

---

## Part E — submit the pilot to a real QPU

Choose an accessible physical backend in your IBM dashboard. In the command below, use the exact backend name you saw there. Here `ibm_brisbane` is an example only.

```bash
python scripts/submit_ibm_validation.py --manifest studies/pilot/qaoa_validation_manifest.json --backend ibm_brisbane --output-dir studies/pilot/ibm_brisbane_run
```

If your account requires an instance/CRN, append:

```text
--instance YOUR-INSTANCE-CRN
```

The command does four things:

1. authenticates using the `IBM_QUANTUM_TOKEN` in memory;
2. transpiles the manifest circuits for the selected backend;
3. submits one IBM Runtime sampler job; and
4. waits for the result.

As soon as IBM accepts the job, the terminal prints an IBM Runtime job ID and writes `submission_receipt.json`. Copy the ID into a notebook or plain-text study log. Queue time can vary, so keep the terminal open when practical.

A completed run writes:

```text
studies/pilot/ibm_brisbane_run/
├── submission_receipt.json
├── counts_long.csv
├── raw_qiskit_counts.json
└── run_metadata.json
```

Keep all files. Do not edit them manually.

If the terminal or container closes after a job ID is issued, do **not** submit the manifest again. Restore the token in a new terminal and recover the original remote job:

```bash
python scripts/submit_ibm_validation.py --manifest studies/pilot/qaoa_validation_manifest.json --backend ibm_brisbane --job-id RECORDED_RUNTIME_JOB_ID --output-dir studies/pilot/ibm_brisbane_run
```

This writes `job_recovery_receipt.json` and retrieves the existing job without spending credits on a duplicate circuit batch. If submission fails before any job ID appears, preserve the error message; do not keep retrying blindly or modify the downloaded manifest after a failed/partial run.

---

## Part F — create the final pre-specified study

Only begin this after the pilot has worked.

### F1. Make a final manifest

In **🧪 Experimental Validation**, choose these suggested final settings:

| Control | Final-study value |
|---|---:|
| Graph family | `3-regular` |
| Logical qubits n | `6` |
| QAOA depth | `1` |
| Angle-grid size | `3` |
| Shots per circuit | `4096` |
| Graph seed | `42` |
| Execution-order seed | `2027` |

This creates nine angle conditions and plans:

```text
9 conditions × 4096 shots = 36,864 planned shots
```

Check your IBM account's usage/credit policy before submission. If this is too large for your access level, choose a smaller **new** manifest before any final data are collected; do not reduce the shot count in an already downloaded manifest.

Download the new manifest and place it here:

```text
studies/final_v1/qaoa_validation_manifest.json
```

Create the directory with either:

```bash
mkdir -p studies/final_v1
```

or, in PowerShell:

```powershell
New-Item -ItemType Directory -Force -Path studies\final_v1 | Out-Null
```

### F2. Commit and tag the final manifest before data collection

Run:

```bash
git add docs/research/experimental_validation_protocol.md studies/final_v1/qaoa_validation_manifest.json
git commit -m "Preregister QAOA hardware validation v1"
git tag -a qaoa-validation-v1 -m "QAOA hardware validation v1 preregistration"
```

If you have a GitHub remote, publish the evidence before submitting:

```bash
git push origin main --tags
```

The checksum in the manifest detects edits, but a commit/tag or archive is the evidence that the manifest existed before you saw the hardware data.

### F3. Dry-run the final manifest

```bash
python scripts/submit_ibm_validation.py --manifest studies/final_v1/qaoa_validation_manifest.json --output-dir studies/final_v1/protocol_check --dry-run
```

Open the resulting plan. A six-qubit 3-regular, depth-1 circuit should show:

```text
num_qubits: 6
num_clbits: 6
h: 6
rzz: 9
rx: 6
measure: 6
```

### F4. Submit the final study

Use the same safe token setup from Part D in a fresh terminal, then run a command of this form with an accessible backend:

```bash
python scripts/submit_ibm_validation.py --manifest studies/final_v1/qaoa_validation_manifest.json --backend ibm_brisbane --output-dir studies/final_v1/ibm_brisbane_run
```

Do not change the manifest after submission. Preserve the job ID, immediate `submission_receipt.json`, all completed-run outputs, and any error/queue information. If the local client stops after job acceptance, use the `--job-id` recovery command shown in Part E rather than resubmitting the final study.

---

## Part G — analyse the measured counts in the app

1. Start the app again if it is not open:

   ```bash
   streamlit run app.py
   ```

2. Open **🧪 Experimental Validation**.
3. Under **1. Pre-specify the experiment**, upload the exact original final `qaoa_validation_manifest.json`.
4. Under **3. Upload measured data**, upload:

   ```text
   studies/final_v1/ibm_brisbane_run/counts_long.csv
   ```

5. Leave **Uploaded bitstring convention** as `vertex-order`. The supplied CLI already converts IBM/Qiskit output into this project convention.
6. Always retain the ideal simulator comparison.
7. Enable **Include a FakeProvider calibration-reference model** only if the listed fake snapshot is a meaningful comparison for your measured backend. It is not a live calibration. If no appropriate snapshot exists, uncheck it and use the scientifically safer ideal-only comparison.
8. Click **Run preregistered simulator comparison**.
9. Download both files the page generates:

   ```text
   qaoa_hardware_validation_analysis.json
   qaoa_hardware_validation_report.md
   ```

Put them in the same study folder, for example:

```text
studies/final_v1/analysis/
```

**Do not select only favourable angles or delete poor-fitting conditions.** Disagreement between simulation and hardware is a valid scientific outcome.

---

## Part H — make the result stronger

If your account access permits it, repeat the unchanged final manifest:

- on a different day; or
- on a different physical backend.

Use a new folder for every actual run:

```text
studies/final_v1/
├── qaoa_validation_manifest.json
├── protocol_check/
├── ibm_backend_A_run_1/
├── ibm_backend_A_run_2/
├── ibm_backend_B_run_1/
└── analysis/
```

Do not merge or overwrite raw count files. A repeated run is a new dataset, even when the manifest is identical.

Then write a short 2–4 page technical report with:

1. the research question;
2. the fixed circuit/graph/shot design;
3. backend and job IDs, date, and compiler settings;
4. ratio, bootstrap, and distributional results;
5. what the models predicted well or poorly; and
6. limitations: small graph, p=1, finite shots, calibration drift, backend-specific findings, and no physical-noise attribution.

A scientifically honest conclusion is narrow, for example:

> For this pre-specified QAOA circuit set, backend, and execution period, the reported simulator model had lower or higher discrepancy from the measured count distributions than the ideal reference.

It is **not** evidence that the tool diagnosed a real space environment, discovered the device's true noise mechanism, or demonstrated quantum advantage.

---

## Troubleshooting

| Problem | What to do |
|---|---|
| `py` or `python3.12` is not found | Install Python 3.12, reopen the terminal, and retry Part A3. |
| PowerShell says scripts are disabled | Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, then activate `.venv` again. |
| `No module named ...` | Confirm `.venv` is activated, then rerun `python -m pip install -r requirements.txt`. |
| Browser cannot open the app | Copy the `Local URL` printed by Streamlit into a browser. If port 8501 is occupied, use `streamlit run app.py --server.port 8502`. |
| `No token found in $IBM_QUANTUM_TOKEN` | Set the token again in the **same terminal** used to run the CLI. It is intentionally not persistent. |
| IBM reports unauthorized/no backend access | Confirm the API key, account/instance, and exact hardware backend name in the IBM dashboard. Account access and usage policy are controlled by IBM, not Quantum Compass. |
| A job is queued | Keep the terminal open and retain the printed job ID. Queue time is normal for shared hardware. |
| A real run has partial/missing counts | Keep the outputs and report the missing conditions. Do not fill them with simulation data. |
| You need different final parameters after seeing a pilot | Create a brand-new manifest, commit/tag it, and call it a new protocol version. Do not edit the previous manifest. |

---

## Final checklist

Before describing the work as a real hardware study, make sure you can tick every box:

- [ ] Python environment installed and tests passed.
- [ ] Pilot manifest dry run succeeded.
- [ ] Pilot real-hardware run produced raw counts and metadata.
- [ ] Final manifest was downloaded before final data collection.
- [ ] Final manifest and protocol were committed/tagged before submission.
- [ ] Final dry-run plan was reviewed.
- [ ] Final run produced `counts_long.csv`, raw counts, and metadata.
- [ ] App analysis was generated from the original manifest and actual measured counts.
- [ ] Analysis JSON and Markdown report were saved.
- [ ] Limitations and any failed/missing data were reported honestly.

Once those boxes are checked, you have a defensible small-scale QAOA hardware-validation study rather than only a simulation project.
