# Release Procedure

This procedure preserves the scientific record while preparing a private working repository for a public code release. Perform it from the actual repository checkout; do not invent environment records, checksums, calibration snapshots, hardware results, or release metadata in a separate workspace.

## 1. Protect historic evidence

1. Confirm the preregistration commit and annotated tag still point to the original pre-data protocol.
2. Confirm raw pilot/final manifests, counts, receipts, run metadata, and analysis exports are tracked as intended.
3. Do not amend the preregistration commit, move the tag, or rerun the completed final job merely to make a cleaner release.
4. Treat any newly submitted QPU run as a separately named replication with a new job ID and evidence folder.

## 2. Run the release checks

```bash
python -m pytest tests/test_core.py -q
python -m compileall -q app.py core utils pages scripts tests
python scripts/submit_ibm_validation.py --help
python scripts/capture_study_provenance.py --help
```

If Docker is the intended deployment route, build and smoke-test it from the real repository root as well:

```bash
docker compose build
docker compose up -d
docker compose ps
docker compose down
```

Record only successful checks that were actually performed. A Docker result should not be claimed from a machine that lacks the Docker daemon.

## 3. Capture final study provenance locally

After the evidence folders are complete, run the provenance command for each study that will be published:

```bash
python scripts/capture_study_provenance.py --study-dir studies/pilot
python scripts/capture_study_provenance.py --study-dir studies/final_v1
```

Use the actual folder names in the checkout. Review the resulting files before committing. The command intentionally excludes credentials and host identity, but it cannot decide whether the underlying study data themselves are suitable for public release.

## 4. Audit privacy and secrets before changing visibility

Review tracked files and full commit history, not only the current working tree:

```bash
git status --short
git ls-files
git log --format='%h %an <%ae>' | sort -u
git grep -nEI 'IBM_QUANTUM_TOKEN|api[_ -]?key|token|crn|secret' -- ':!*.md'
```

Also inspect images, PDFs, CSVs, notebooks, GitHub Actions logs, release attachments, and untracked files that may be added. Remove or rotate exposed credentials before making the repository public. Decide whether the commit-author email addresses are appropriate for public disclosure; rewriting a published history is consequential, so make that decision before changing visibility.

Do not publish IBM dashboard screenshots, terminal screenshots with local usernames/paths, browser autofill, personal account data, or a private service instance/CRN unless disclosure is explicitly intended.

## 5. Prepare public metadata

- Review `README.md`, `CITATION.cff`, `CHANGELOG.md`, `LICENSE`, `SECURITY.md`, and `CONTRIBUTING.md` for accuracy.
- Replace only facts you can verify from the real release: canonical repository URL, tagged version, release date, DOI if one is minted, and approved author/organization information.
- Keep the narrow hardware claim and its limitations adjacent to the real metrics.
- Verify all documentation links and image paths on the hosted repository preview.

## 6. Commit, tag, and publish

Create a reviewed release commit, then create an annotated release tag from that exact commit. Push the branch and tag while the repository is still private, inspect the hosted rendering, and change visibility only after the privacy audit is complete.

When creating a hosted release, include the source archive and a concise note that distinguishes software changes from completed research evidence. If a DOI archive service is used, archive the final tag and add its verified DOI to the public release metadata afterward.

## 7. Post-release verification

From a fresh clone or downloaded release archive, verify installation, core tests, documentation rendering, and at least one app startup. Confirm that the public repository contains no secret files and that citations resolve correctly. Record any known limitations in the release notes rather than silently editing historical evidence.
