# Security Policy

## Supported code

Security fixes are made on the current default branch. Users of a released archive should upgrade to the latest maintained revision before reporting a suspected code issue.

## Credential handling

Quantum Compass is designed so IBM Quantum credentials are supplied only at execution time through an environment variable such as `IBM_QUANTUM_TOKEN`. Tokens, API keys, CRNs, `.env` files, Streamlit secret files, browser dashboards, and terminal screenshots with personal context must not be committed, attached to issues, or included in public evidence packages.

If a credential is exposed, revoke or rotate it immediately through the relevant provider before opening an issue. Treat repository history, build logs, chat transcripts, and release attachments as potentially public once pushed or shared.

## Reporting a vulnerability

Do not disclose a suspected credential leak, unsafe command path, arbitrary file access, or other security vulnerability in a public issue. Use the private reporting mechanism offered by the repository host when available. If that mechanism is unavailable, contact the repository owner privately through the repository-host profile and include only the minimum information needed to reproduce the issue safely.

A useful private report includes:

- affected revision or release;
- a concise description of impact;
- safe reproduction steps that use fake credentials and synthetic data; and
- a suggested mitigation, if known.

The maintainer should acknowledge the report, assess impact, prepare a fix without exposing secrets, and publish a coordinated advisory or release note after remediation.
