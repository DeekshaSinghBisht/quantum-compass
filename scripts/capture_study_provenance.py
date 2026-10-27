#!/usr/bin/env python3
"""Capture checksums and a token-free Python environment record for one study.

Run this only from the checkout and environment that produced (or analysed) the
study evidence. The command deliberately derives its records from local files
and installed packages instead of inventing values in documentation.

Example::

    python scripts/capture_study_provenance.py --study-dir studies/final_v1

The command writes ``environment.json`` and ``SHA256SUMS.txt`` inside the study
directory by default. It never reads shell credentials and intentionally omits
hostnames, usernames, executable paths, and environment variables from the
captured record.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


CHUNK_SIZE = 1024 * 1024


def utc_now() -> str:
    """Return a compact UTC timestamp suitable for JSON provenance."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    """Calculate the SHA-256 digest of one regular file without loading it at once."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def installed_packages() -> Dict[str, str]:
    """Return a deterministic mapping of installed Python distributions and versions."""
    packages: Dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            packages[str(name).lower()] = str(distribution.version)
    return dict(sorted(packages.items(), key=lambda item: item[0]))


def environment_record() -> Dict[str, object]:
    """Build a portable package/runtime record without machine-identity fields."""
    return {
        "schema_version": 1,
        "captured_utc": utc_now(),
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "version_info": {
                "major": int(sys.version_info.major),
                "minor": int(sys.version_info.minor),
                "micro": int(sys.version_info.micro),
            },
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "packages": installed_packages(),
        "credentials_stored": False,
        "host_identity_stored": False,
    }


def _relative_posix(path: Path, root: Path) -> str:
    """Return a portable relative filename and reject files outside study root."""
    try:
        return path.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(f"{path} is not inside study directory {root}.") from exc


def study_files(study_dir: Path, excluded_paths: Iterable[Path]) -> List[Path]:
    """List regular evidence files deterministically, excluding generated checksum output."""
    root = study_dir.resolve()
    exclusions = {path.resolve() for path in excluded_paths}
    files: List[Path] = []
    for path in root.rglob("*"):
        if path.is_file() and not path.is_symlink() and path.resolve() not in exclusions:
            files.append(path)
    return sorted(files, key=lambda item: _relative_posix(item, root))


def checksum_lines(study_dir: Path, excluded_paths: Iterable[Path]) -> List[str]:
    """Return standard SHA256SUMS-formatted rows for all included study files."""
    root = study_dir.resolve()
    rows = []
    for path in study_files(root, excluded_paths):
        rows.append(f"{sha256_file(path)}  {_relative_posix(path, root)}")
    return rows


def _write_json(path: Path, payload: Dict[str, object], overwrite: bool) -> None:
    """Write one provenance JSON file with explicit overwrite protection."""
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {path}; pass --overwrite to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_checksums(path: Path, lines: Sequence[str], overwrite: bool) -> None:
    """Write a newline-terminated standard SHA-256 digest listing."""
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {path}; pass --overwrite to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def capture_study_provenance(
    study_dir: Path,
    environment_path: Path,
    checksums_path: Path,
    *,
    overwrite: bool,
) -> Tuple[Path, Path, int]:
    """Write environment then checksums, including environment JSON but not self-checksum."""
    root = study_dir.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Study directory does not exist or is not a directory: {study_dir}")
    environment = environment_path.resolve()
    checksums = checksums_path.resolve()
    if environment == checksums:
        raise ValueError("--environment-file and --checksums-file must name different files.")
    _relative_posix(environment, root)
    _relative_posix(checksums, root)
    if not overwrite:
        existing = [str(path) for path in (environment, checksums) if path.exists()]
        if existing:
            raise FileExistsError(
                "Refusing to overwrite existing provenance file(s): " + ", ".join(existing) + ". "
                "Pass --overwrite only when recapturing from the intended local environment."
            )

    _write_json(environment, environment_record(), overwrite=True)
    lines = checksum_lines(root, excluded_paths=(checksums,))
    _write_checksums(checksums, lines, overwrite=True)
    return environment, checksums, len(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse paths relative to the current checkout without requiring Qiskit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True, help="Existing study directory to document.")
    parser.add_argument(
        "--environment-file",
        type=Path,
        default=None,
        help="Environment JSON path; defaults to STUDY_DIR/environment.json.",
    )
    parser.add_argument(
        "--checksums-file",
        type=Path,
        default=None,
        help="SHA-256 listing path; defaults to STUDY_DIR/SHA256SUMS.txt.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace prior generated provenance records.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Capture the records and return a conventional command-line exit status."""
    args = parse_args(argv)
    environment_path = args.environment_file or args.study_dir / "environment.json"
    checksums_path = args.checksums_file or args.study_dir / "SHA256SUMS.txt"
    try:
        environment, checksums, file_count = capture_study_provenance(
            args.study_dir,
            environment_path,
            checksums_path,
            overwrite=bool(args.overwrite),
        )
    except (FileNotFoundError, FileExistsError, OSError, ValueError) as exc:
        print(f"Study provenance capture failed: {exc}", file=sys.stderr)
        return 2
    print(f"Captured token-free environment record: {environment}")
    print(f"Wrote SHA-256 checksums for {file_count} file(s): {checksums}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
