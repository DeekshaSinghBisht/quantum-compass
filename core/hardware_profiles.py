"""IBM fake-backend hardware profiles for Quantum Compass.

This module makes the app's simulations use calibration data shipped with
``qiskit-ibm-runtime`` rather than an invented generic error rate.  The public
API deliberately works without IBM credentials: Qiskit's local ``Fake*``
backends snapshot real backend configuration/properties and are deterministic.

Two noise-model modes are useful:

``get_hardware_noise_model(name)``
    Returns the literal :class:`qiskit_aer.noise.NoiseModel` produced by
    ``NoiseModel.from_backend(fake_backend)``.  This is the requested real
    hardware model and is useful in scripts/tests that transpile circuits to the
    backend's native basis.

``get_hardware_noise_model(name, subset_size=n)``
    Returns a reduced, QAOA-native model for an ``n``-qubit landscape.  A
    full 27/127-qubit density matrix is infeasible for a small MaxCut instance.
    The reduced model maps the fake backend's calibrated aggregate 1Q / 2Q /
    readout errors onto this project's native ``h``, ``rx`` and ``rzz`` gates.
    It is still calibration-derived, but avoids claiming that a 127-qubit
    density-matrix simulation was performed.

Streamlit pages must not instantiate these Qiskit objects in the rotating parent
thread.  They pass :func:`hardware_noise_spec` to the persistent Aer worker,
which constructs this model safely in the child.  The direct public functions
remain available for CLI use and unit tests.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Keep this mapping entirely primitive so importing it does not instantiate a
# fake backend or a Qiskit object.  The class is resolved lazily in the helper.
AVAILABLE_HARDWARE: Dict[str, Dict[str, str]] = {
    "ibm_cairo": {
        "label": "IBM Cairo",
        "fake_class": "FakeCairoV2",
        "description": "27-qubit IBM heavy-hex fake backend calibration snapshot.",
    },
    "ibm_kyoto": {
        "label": "IBM Kyoto",
        "fake_class": "FakeKyoto",
        "description": "127-qubit IBM heavy-hex fake backend calibration snapshot.",
    },
    "ibm_sherbrooke": {
        "label": "IBM Sherbrooke",
        "fake_class": "FakeSherbrooke",
        "description": "127-qubit IBM heavy-hex fake backend calibration snapshot.",
    },
    "ibm_hanoi": {
        "label": "IBM Hanoi",
        "fake_class": "FakeHanoiV2",
        "description": "27-qubit IBM heavy-hex fake backend calibration snapshot.",
    },
}

# The fake backends expose ECR on newer heavy-hex hardware and CX on older ones.
# We report either as the requested CX-equivalent two-qubit error.
_ONE_QUBIT_GATES = {"id", "x", "sx", "rx", "h", "u1", "u2", "u3"}
_TWO_QUBIT_GATES = {"cx", "ecr", "cz", "rzz"}


def _require_backend_name(backend_name: str) -> str:
    key = str(backend_name).strip().lower().replace("-", "_").replace(" ", "_")
    if key not in AVAILABLE_HARDWARE:
        raise ValueError(
            f"Unknown hardware '{backend_name}'. Choose from {list(AVAILABLE_HARDWARE)}."
        )
    return key


def _fake_backend(backend_name: str):
    """Instantiate one local fake backend lazily (Qiskit object construction)."""
    key = _require_backend_name(backend_name)
    try:
        from qiskit_ibm_runtime import fake_provider
    except ImportError as exc:  # clear install hint outside the app as well
        raise ImportError(
            "qiskit-ibm-runtime is required for hardware profiles; install requirements.txt."
        ) from exc
    class_name = AVAILABLE_HARDWARE[key]["fake_class"]
    backend_cls = getattr(fake_provider, class_name, None)
    if backend_cls is None:
        raise RuntimeError(
            f"Installed qiskit-ibm-runtime does not provide {class_name}. "
            "Upgrade qiskit-ibm-runtime to a version with its fake provider data."
        )
    return backend_cls()


def _property_value(properties, qubit: int, name: str) -> Optional[float]:
    """Read one backend-property value across Runtime/Qiskit API variants."""
    try:
        item = properties.qubit_property(int(qubit), name)
        value = item[0] if isinstance(item, tuple) else item
        value = float(value)
        return value if np.isfinite(value) else None
    except Exception:
        return None


def _gate_error(gate) -> Optional[float]:
    """Extract ``gate_error`` from a FakeProvider GateProperties object."""
    try:
        for parameter in gate.parameters:
            if str(parameter.name) == "gate_error":
                value = float(parameter.value)
                return value if np.isfinite(value) else None
    except Exception:
        pass
    return None


def _robust_mean(values: Sequence[float], default: float) -> float:
    """Mean with a small trim to prevent a few failed fake calibration entries.

    Some fake snapshots contain a literal error probability of 1.0 for a
    disabled/placeholder coupling.  Those are retained in diagnostics but are
    not representative of a working calibrated gate subset.
    """
    clean = np.asarray([float(v) for v in values if np.isfinite(v)], dtype=float)
    if clean.size == 0:
        return float(default)
    clean.sort()
    if clean.size >= 10:
        trim = max(1, int(clean.size * 0.10))
        clean = clean[trim:-trim]
    return float(np.mean(clean))


def _calibration_values(backend, subset_size: Optional[int] = None) -> Dict:
    """Collect raw calibration values from a fake backend into primitive lists."""
    properties = backend.properties()
    n_qubits = int(getattr(backend, "num_qubits", len(properties.qubits)))
    if subset_size is None:
        selected = list(range(n_qubits))
    else:
        subset_size = int(subset_size)
        if subset_size < 1 or subset_size > n_qubits:
            raise ValueError(
                f"subset_size must be in [1, {n_qubits}] for {backend.name}, got {subset_size}."
            )
        # Stable logical-to-physical convention for a reduced density-matrix
        # study.  The result remains calibrated aggregate noise, not a claim of
        # connectivity-aware placement optimization.
        selected = list(range(subset_size))
    selected_set = set(selected)

    t1: List[float] = []
    t2: List[float] = []
    readout: List[float] = []
    for q in selected:
        value = _property_value(properties, q, "T1")
        if value is not None and value > 0.0:
            t1.append(value)
        value = _property_value(properties, q, "T2")
        if value is not None and value > 0.0:
            t2.append(value)
        value = _property_value(properties, q, "readout_error")
        if value is not None and 0.0 <= value <= 0.5:
            readout.append(value)

    one_errors: List[float] = []
    two_errors: List[float] = []
    raw_two_errors: List[float] = []
    for gate in properties.gates:
        try:
            gate_name = str(gate.gate).lower()
            qubits = [int(q) for q in gate.qubits]
        except Exception:
            continue
        if not set(qubits).issubset(selected_set):
            continue
        error = _gate_error(gate)
        if error is None:
            continue
        if gate_name in _ONE_QUBIT_GATES and 0.0 <= error < 0.5:
            one_errors.append(error)
        elif gate_name in _TWO_QUBIT_GATES:
            raw_two_errors.append(error)
            if 0.0 <= error < 0.5:
                two_errors.append(error)

    # A tiny first-N subset can contain no complete calibrated coupling.  Fall
    # back to backend-wide gate averages in that case, while recording it.
    used_backend_twoq_fallback = False
    if not two_errors:
        for gate in properties.gates:
            try:
                gate_name = str(gate.gate).lower()
                qubits = list(gate.qubits)
            except Exception:
                continue
            if len(qubits) != 2 or gate_name not in _TWO_QUBIT_GATES:
                continue
            error = _gate_error(gate)
            if error is not None:
                raw_two_errors.append(error)
                if 0.0 <= error < 0.5:
                    two_errors.append(error)
        used_backend_twoq_fallback = bool(two_errors)

    return {
        "n_qubits": n_qubits,
        "selected_qubits": selected,
        "t1_seconds": t1,
        "t2_seconds": t2,
        "readout_errors": readout,
        "single_qubit_errors": one_errors,
        "two_qubit_errors": two_errors,
        "raw_two_qubit_errors": raw_two_errors,
        "used_backend_twoq_fallback": used_backend_twoq_fallback,
    }


def _stats_from_backend(backend, backend_name: str, subset_size: Optional[int] = None) -> Dict:
    """Produce the public stats mapping from one instantiated fake backend."""
    key = _require_backend_name(backend_name)
    values = _calibration_values(backend, subset_size)
    one_error = _robust_mean(values["single_qubit_errors"], default=0.001)
    valid_two = list(values["two_qubit_errors"])
    raw_two = list(values["raw_two_qubit_errors"])
    warnings: List[str] = []
    if valid_two:
        two_error = _robust_mean(valid_two, default=max(0.005, one_error * 30.0))
        two_source = "fake-backend calibrated 2Q gates"
    else:
        # Runtime's current FakeKyoto snapshot reports 1.0 for every ECR gate,
        # a disabled/placeholder calibration rather than usable device physics.
        # Derive a conservative, explicit proxy from actual 1Q calibration so a
        # small logical landscape remains meaningful; expose the raw 1.0 values
        # and warning to callers rather than hiding the source limitation.
        two_error = float(np.clip(max(0.005, one_error * 30.0), 0.005, 0.05))
        two_source = "conservative 2Q proxy from 1Q calibration"
        if raw_two and all(v >= 0.5 for v in raw_two):
            warnings.append(
                "Fake-provider 2Q calibration is placeholder p=1.0; reduced QAOA model uses a tagged conservative proxy."
            )
        else:
            warnings.append("No usable fake-provider 2Q calibration found; reduced model uses a tagged conservative proxy.")

    readout_error = _robust_mean(values["readout_errors"], default=0.02)
    avg_t1_seconds = _robust_mean(values["t1_seconds"], default=100e-6)
    avg_t2_seconds = _robust_mean(values["t2_seconds"], default=80e-6)
    raw_avg_two = float(np.mean(raw_two)) if raw_two else None
    invalid_two = int(sum(1 for v in raw_two if v >= 0.5))

    if values["used_backend_twoq_fallback"]:
        warnings.append("Selected qubit subset had no complete 2Q calibration; used backend-wide calibrated 2Q aggregate.")
    if invalid_two:
        warnings.append(f"Excluded {invalid_two} disabled/placeholder 2Q calibration entries (error ≥ 0.5).")

    return {
        "backend_name": key,
        "backend_display_name": AVAILABLE_HARDWARE[key]["label"],
        "description": AVAILABLE_HARDWARE[key]["description"],
        "n_qubits": int(values["n_qubits"]),
        "subset_size": int(len(values["selected_qubits"])),
        "selected_qubits": [int(q) for q in values["selected_qubits"]],
        # Required keys: use calibrated CX/ECR-equivalent error rates.
        "avg_cx_error": float(two_error),
        "avg_single_qubit_error": float(one_error),
        # Raw SI seconds plus display-ready microseconds remove any unit ambiguity.
        "avg_T1": float(avg_t1_seconds),
        "avg_T2": float(avg_t2_seconds),
        "avg_T1_us": float(avg_t1_seconds * 1e6),
        "avg_T2_us": float(avg_t2_seconds * 1e6),
        "readout_error": float(readout_error),
        "two_qubit_calibration_source": two_source,
        "raw_avg_two_qubit_error": raw_avg_two,
        "invalid_two_qubit_entries": invalid_two,
        "calibration_warnings": warnings,
    }


def extract_hardware_stats(backend_name: str, subset_size: Optional[int] = None) -> Dict:
    """Extract calibration statistics from an IBM Runtime fake backend.

    Parameters
    ----------
    backend_name:
        One of ``ibm_cairo``, ``ibm_kyoto``, ``ibm_sherbrooke`` or
        ``ibm_hanoi``.
    subset_size:
        Optional logical simulation subset.  It controls which physical-qubit
        calibration data is summarized; omitted means all physical qubits.

    Returns
    -------
    dict
        Includes the required ``n_qubits``, ``avg_cx_error`` (CX/ECR-equivalent),
        ``avg_single_qubit_error``, ``avg_T1``/``avg_T2`` (seconds) and
        ``readout_error`` fields, alongside display-ready microseconds, selected
        qubits and provenance/warnings.
    """
    backend = _fake_backend(backend_name)
    return _stats_from_backend(backend, backend_name, subset_size)


def get_hardware_noise_model(backend_name: str, subset_size: Optional[int] = None):
    """Return an Aer noise model derived from a local IBM fake backend.

    With no ``subset_size``, this is the literal
    ``NoiseModel.from_backend(FakeXxx())`` result.  With a subset, this is a
    calibration-derived native-QAOA reduction carrying real aggregate 1Q/2Q and
    readout error rates on ``h``, ``rx`` and ``rzz``; it is suitable for the
    small density-matrix landscape simulator.
    """
    key = _require_backend_name(backend_name)
    backend = _fake_backend(key)
    try:
        from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error
    except ImportError as exc:
        raise ImportError("qiskit-aer is required for hardware noise models.") from exc

    # Always build the literal real fake-backend model first.  That is the
    # provenance source for both modes and validates the provider calibration.
    full_model = NoiseModel.from_backend(backend)
    stats = _stats_from_backend(backend, key, subset_size)
    if subset_size is None:
        model = full_model
    else:
        # Full backend models target native ECR/CX/SX gates and cannot affect the
        # app's logical H/RX/RZZ ansatz without transpiling a 27/127-qubit
        # circuit.  Map the extracted calibrated aggregate onto its native gate
        # vocabulary in a fresh, small logical NoiseModel.
        model = NoiseModel()
        p1 = float(np.clip(stats["avg_single_qubit_error"], 0.0, 0.749999))
        p2 = float(np.clip(stats["avg_cx_error"], 0.0, 0.937499))
        readout = float(np.clip(stats["readout_error"], 0.0, 0.499999))
        if p1 > 0.0:
            model.add_all_qubit_quantum_error(depolarizing_error(p1, 1), ["h", "rx"])
        if p2 > 0.0:
            model.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ["rzz"])
        if readout > 0.0:
            model.add_all_qubit_readout_error(
                ReadoutError([[1.0 - readout, readout], [readout, 1.0 - readout]])
            )

    # Tags are read by the existing exact evaluator without needing a circuit
    # measurement.  They also make provenance inspectable in tests/CLI output.
    model._qc_readout_prob = float(stats["readout_error"])
    model._qc_hardware_backend = key
    model._qc_hardware_stats = dict(stats)
    model._qc_hardware_mode = "full-fake-backend" if subset_size is None else "reduced-qaoa-native"
    return model


def hardware_noise_spec(backend_name: str, subset_size: int) -> Dict[str, object]:
    """Return a parent-safe worker descriptor for a reduced hardware model.

    The Streamlit parent passes this primitive dict to ``aer_worker``; the child
    then calls :func:`get_hardware_noise_model`.  No Qiskit object is built in
    the parent process.
    """
    key = _require_backend_name(backend_name)
    return {"hardware_backend": key, "subset_size": int(subset_size)}
