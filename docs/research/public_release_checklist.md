# Public-release checklist for the completed hardware study

Use this checklist before changing the Quantum Compass repository from private to public or creating a public release.

## Evidence and reproducibility

- [ ] Keep the pre-data tag `qaoa-final-v1-preregistered` unchanged.
- [ ] Retain the final manifest, study plan, dry-run plan, submission/recovery receipt when applicable, normalized counts, raw counts, run metadata, analysis JSON, generated Markdown report, and locally captured environment/checksum records under `studies/final_v1/`.
- [ ] Retain the equivalent pilot package under `studies/pilot/`.
- [ ] Generate `environment.json` and `SHA256SUMS.txt` only from the actual final local study checkout with `scripts/capture_study_provenance.py`; do not copy or fabricate them.
- [ ] Verify that the final analysis includes all 9 planned conditions and 36,864 observed shots.
- [ ] Link or include `docs/research/hardware_validation_results.md`.
- [ ] Keep the claim boundary: this is a small hardware/simulation comparison, not quantum advantage or physical-noise attribution.

## Visuals

- [x] Keep the public-safe aggregate metrics figure at `docs/assets/hardware_validation/final_ibm_kingston_aggregate_metrics.png`.
- [x] Include the reviewed, app-exported final ratio-comparison chart at `docs/assets/hardware_validation/final_ibm_kingston_ratio_comparison.png`; do not redraw it from prose metrics.
- [x] Include the reviewed real-application gallery at `docs/application_gallery.md` and `docs/assets/application/`, with captions that distinguish simulator/planning interfaces from hardware evidence.
- [ ] Do not publish IBM dashboard screenshots containing account name, trial/usage details, CRN, API key, or private workload information.

## Secrets and privacy

- [ ] Run `git status --ignored --short` and confirm `.env` and `.streamlit/secrets.toml` are ignored.
- [ ] Search the repository for `IBM_QUANTUM_TOKEN`, API keys, and CRNs before publishing.
- [ ] Review `git log --format='%h %an <%ae> %s'` before going public: commit author email addresses can be visible in a public repository.
- [ ] If a personal email should not be public, keep this repository private or create a clean public release/repository using a GitHub no-reply email.

## Final Git workflow

```powershell
# Inspect what will be committed.
git status

# Add reviewed code, release metadata, results documentation, public-safe assets, and real study evidence.
git add README.md CHANGELOG.md CITATION.cff CONTRIBUTING.md SECURITY.md REPRODUCIBILITY.md RELEASE.md .github scripts docs studies tests

git commit -m "Prepare reproducible Quantum Compass public release"
git push origin main
```

After the public documentation is complete, optionally create a post-data release tag. Do not reuse or move the pre-data tag.

```powershell
git tag -a v1.0.0-hardware-validation -m "Completed IBM Kingston QAOA hardware validation release"
git push origin v1.0.0-hardware-validation
```
