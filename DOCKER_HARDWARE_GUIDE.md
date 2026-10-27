# Docker guide: run Quantum Compass and its hardware workflow without a local Python install

## Important distinction

**Docker is not a replacement for the Quantum Compass interface.** It is the package that runs the app and its Python tools in a reproducible container.

With this route:

- you do **not** install Python, Qiskit, or Streamlit directly on your computer;
- Docker runs the Streamlit interface at `http://localhost:8501` for manifest creation and analysis; and
- Docker runs the IBM submission CLI in a one-off container for dry runs and real hardware jobs.

The supplied Compose configuration deliberately keeps research evidence in the host-side `studies/` folder. It does not place an IBM API key in the image, Compose file, `.env`, or browser app.

---

## 1. Install Docker

Install and start one of the following:

- **Windows or macOS:** Docker Desktop, <https://www.docker.com/products/docker-desktop/>.
- **Linux:** Docker Engine plus the Docker Compose plugin, <https://docs.docker.com/engine/install/>.

Open a terminal in the extracted `quantum-compass` folder and confirm both commands work:

```bash
docker --version
docker compose version
```

On Windows, use PowerShell. On macOS/Linux, use Terminal.

---

## 2. Prepare the local folders

The ZIP already includes an empty `studies/` folder. It is where you will preserve manifests, dry-run plans, real counts, job metadata, and analysis outputs.

If it is missing, create it:

**Windows PowerShell**

```powershell
New-Item -ItemType Directory -Force -Path studies | Out-Null
Copy-Item .env.example .env
```

**macOS/Linux**

```bash
mkdir -p studies
cp .env.example .env
```

Do not put an IBM API key in `.env`. The `.env` file only configures the local app port, thread count, repository link, and host studies directory.

### Linux bind-mount note

The container runs as a non-root user. If a native Linux Docker installation reports `Permission denied` when writing under `studies/`, grant the local study folder read/write access, then retry:

```bash
chmod ugo+rwx studies
```

This applies only to the local study folder on your own computer; it contains no API key.

---

## 3. Build and start the app

Run:

```bash
docker compose up --build -d
```

The first build can take several minutes because it installs Qiskit and prepares the bundled fingerprint model. Later starts are much faster.

Check the container:

```bash
docker compose ps
docker compose logs -f quantum-compass
```

When the logs show Streamlit is ready, open:

```text
http://localhost:8501
```

Stop viewing logs with `Ctrl+C`; this does **not** stop the app because it is running in the background.

To stop the app later:

```bash
docker compose down
```

Do not add `--volumes` unless you intentionally want to delete the named cache volume. Your host `studies/` folder remains separate regardless.

---

## 4. Create a manifest in the browser

In the browser app, open **🧪 Experimental Validation**.

For a first local pilot, use:

| Setting | Value |
|---|---:|
| Graph family | `3-regular` |
| Logical qubits | `4` |
| QAOA depth | `1` |
| Angle-grid size | `2` |
| Shots per circuit | `1024` |

Click **Create checksum-identified validation manifest**, then **Download preregistered manifest**.

Move the browser-downloaded file into this host folder:

```text
studies/pilot/qaoa_validation_manifest.json
```

The Compose bind mount makes the same file available inside the container at:

```text
/app/studies/pilot/qaoa_validation_manifest.json
```

---

## 5. Run a Docker-based dry run

From the project folder, run:

```bash
docker compose run --rm quantum-compass python scripts/submit_ibm_validation.py --manifest studies/pilot/qaoa_validation_manifest.json --output-dir studies/pilot/protocol_check --dry-run
```

This command starts a temporary container, builds the QAOA circuits, writes a plan into your host `studies/` folder, and removes the temporary container when done. It does not contact IBM Quantum and needs no token.

You should find:

```text
studies/pilot/protocol_check/dry_run_submission_plan.json
```

For the four-qubit 3-regular pilot, inspect the JSON and confirm each circuit has:

```text
h: 4
rzz: 6
rx: 4
measure: 4
```

---

## 6. Obtain IBM Quantum access

Create/sign in to IBM Quantum Platform, create an API key, and check which real backends your own account can access. The current official setup guide is:

<https://docs.quantum.ibm.com/start/hello-world>

Do not assume that `ibm_brisbane` is available to you; it is only an example backend name.

Never put the API key in:

- `.env`;
- `docker-compose.yml`;
- Dockerfile;
- source code;
- GitHub;
- the Streamlit app; or
- a screenshot/chat message.

---

## 7. Submit a real job from a one-off Docker container

First set the token in the **current host terminal only**.

### Windows PowerShell

```powershell
$secureToken = Read-Host -AsSecureString "IBM Quantum API key"
$env:IBM_QUANTUM_TOKEN = [System.Net.NetworkCredential]::new("", $secureToken).Password
Remove-Variable secureToken
```

### macOS/Linux

```bash
read -rs -p "IBM Quantum API key: " IBM_QUANTUM_TOKEN; echo
export IBM_QUANTUM_TOKEN
```

Then run the job. Replace `ibm_brisbane` only with the exact hardware backend visible in your IBM account:

```bash
docker compose run --rm -e IBM_QUANTUM_TOKEN quantum-compass python scripts/submit_ibm_validation.py --manifest studies/pilot/qaoa_validation_manifest.json --backend ibm_brisbane --output-dir studies/pilot/ibm_brisbane_run
```

The `-e IBM_QUANTUM_TOKEN` option copies the value from the current host shell into **this temporary one-off container only**. It is not placed in the running browser-app container, the image, the Compose file, or your research output. Docker documents this named-variable pass-through behavior for `docker compose run -e NAME`.

The CLI prints the IBM job ID and writes `submission_receipt.json` immediately after submission. Record the ID and keep the terminal open while the job waits in the queue and retrieves results when practical.

On completion, these host files should exist:

```text
studies/pilot/ibm_brisbane_run/
├── submission_receipt.json
├── counts_long.csv
├── raw_qiskit_counts.json
└── run_metadata.json
```

If the one-off container or terminal exits after the receipt is written, do **not** submit again. Re-export the token only in the current host shell and recover the already accepted remote job:

```bash
docker compose run --rm -e IBM_QUANTUM_TOKEN quantum-compass python scripts/submit_ibm_validation.py --manifest studies/pilot/qaoa_validation_manifest.json --backend ibm_brisbane --job-id RECORDED_RUNTIME_JOB_ID --output-dir studies/pilot/ibm_brisbane_run
```

Recovery writes `job_recovery_receipt.json` and retrieves the original result without a duplicate QPU submission.

Remove the token from your current shell afterward:

**Windows PowerShell**

```powershell
Remove-Item Env:IBM_QUANTUM_TOKEN
```

**macOS/Linux**

```bash
unset IBM_QUANTUM_TOKEN
```

---

## 8. Create and pre-register the final study

After the pilot works, make a new final manifest in the browser app using:

| Setting | Value |
|---|---:|
| Graph family | `3-regular` |
| Logical qubits | `6` |
| QAOA depth | `1` |
| Angle-grid size | `3` |
| Shots per circuit | `4096` |

This plans nine circuits and 36,864 total shots. Check your IBM plan/usage policy before submitting.

Save the downloaded manifest as:

```text
studies/final_v1/qaoa_validation_manifest.json
```

Before collecting final data, commit/tag it in Git:

```bash
git add docs/research/experimental_validation_protocol.md studies/final_v1/qaoa_validation_manifest.json
git commit -m "Preregister QAOA hardware validation v1"
git tag -a qaoa-validation-v1 -m "QAOA hardware validation v1 preregistration"
```

Then use the same Docker commands from sections 5 and 7, substituting the final manifest/output directory.

---

## 9. Analyse the real counts in the browser app

1. Visit `http://localhost:8501`.
2. Open **🧪 Experimental Validation**.
3. Upload the exact original final manifest from `studies/final_v1/`.
4. Upload the real output:

   ```text
   studies/final_v1/ibm_brisbane_run/counts_long.csv
   ```

5. Leave the bitstring convention as `vertex-order`.
6. Always compare with the ideal model.
7. Enable a FakeProvider reference only if its stored snapshot is actually meaningful for the measured backend. Otherwise disable it and perform the ideal-only analysis.
8. Download the generated JSON and Markdown report into your `studies/final_v1/analysis/` folder.

---

## Short command reference

```bash
# Start the browser app.
docker compose up --build -d

# Watch the logs.
docker compose logs -f quantum-compass

# Dry run: no IBM account/token required.
docker compose run --rm quantum-compass python scripts/submit_ibm_validation.py --manifest studies/pilot/qaoa_validation_manifest.json --output-dir studies/pilot/protocol_check --dry-run

# Real run: token must already exist only in the current host terminal.
docker compose run --rm -e IBM_QUANTUM_TOKEN quantum-compass python scripts/submit_ibm_validation.py --manifest studies/pilot/qaoa_validation_manifest.json --backend ibm_brisbane --output-dir studies/pilot/ibm_brisbane_run

# Stop the browser app.
docker compose down
```
