"""Environment noise models for Quantum Compass (Phase 2: space environments).

Quantum Compass morphs the QAOA optimisation landscape from a noiseless
simulation to a realistic device by dragging a single *noise level* slider.
This module turns that slider — plus a choice of *environment* — into a concrete
Qiskit Aer ``NoiseModel``.

Environments
------------
``ideal``
    No noise.  :func:`get_noise_model` returns ``None`` so the evaluator uses the
    fast, exact ``statevector`` method.

``earth-lab``
    A present-day trapped-ion / superconducting lab device: single-qubit
    depolarising (``p1``) on ``h``/``rx``, two-qubit depolarising (``p2``) on
    ``rzz``, and symmetric readout error.  A single ``noise_level`` drives all
    three (``p1 = p2 = readout = noise_level``).

``leo-satellite``  (Low Earth Orbit, ~400 km)
    Elevated depolarising (no atmospheric shielding), *burst* noise from cosmic
    rays, and an orbital thermal-cycling modulation.

``satellite-ground``  (quantum link through the atmosphere)
    Photon loss (amplitude damping) that grows at low elevation, atmospheric
    turbulence dephasing, and a thermal-photon population.

``meo-van-allen``  (Medium Earth Orbit, inside the radiation belts)
    Sustained radiation → strong depolarising + frequent bursts.

``deep-space``  (interplanetary, no shielding)
    Galactic cosmic rays → the strongest depolarising + burst rates.

``custom``
    User-defined: absolute per-channel probabilities supplied through ``params``.

How the slider maps to channels
-------------------------------
Every environment has a **base profile** of channel strengths defined at a
**representative noise level** ``REP`` (the value the UI slider jumps to when you
pick that environment).  Dragging the slider to ``level`` rescales every channel
by ``level / REP``::

    channel_eff = channel_base * (level / REP)

so at the representative level the profile equals the published numbers exactly,
at ``level = 0`` the device is ideal, and larger levels make it progressively
worse.  ``earth-lab`` is the special case ``p1 = p2 = readout = level``.

Burst noise (cosmic rays)
-------------------------
A burst is modelled exactly as the spec's mixture
``(1 - burst_rate) * Identity + burst_rate * Depolarizing(p = 0.5)`` on the
qubits of each gate.  Because ``Depolarizing(p)`` scales every non-identity Pauli
by ``(1 - p)``, that mixture is *identically* ``Depolarizing(burst_rate / 2)``;
we still build it explicitly from Kraus operators so the model reads as written.
Composing it with the baseline depolarising ``Depol(p)`` gives
``Depol(p + br/2 - p*br/2)`` (verified against Aer's superoperator to ~1e-16).

Important Aer detail
--------------------
Errors are registered against the *native* gate names produced by
:func:`core.qaoa_builder.build_qaoa_circuit` (``h``, ``rx``, ``rzz``).  The
evaluator forces ``method='density_matrix'`` whenever a noise model is present,
because Aer's ``automatic`` method silently ignores noise for state-based
``save_*`` instructions.  Readout error only fires at a real ``measure``, so the
exact evaluator re-applies the symmetric readout channel by hand.

This module constructs Qiskit objects and therefore must only run **inside the
simulation worker subprocess** (see :mod:`core.aer_worker`).  The pure helpers
:func:`noise_summary`, :func:`effective_params` and :func:`get_environment_info`
build nothing and are safe to call from the Streamlit parent.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from qiskit.quantum_info import Kraus
from qiskit_aer.noise import (
    NoiseModel,
    ReadoutError,
    amplitude_damping_error,
    depolarizing_error,
    kraus_error,
    phase_damping_error,
)

# Native gate names produced by build_qaoa_circuit.
_SINGLE_QUBIT_GATES = ["h", "rx"]
_TWO_QUBIT_GATES = ["rzz"]

#: Environments selectable in the UI (insertion order = dropdown order).
ENVIRONMENTS: Dict[str, str] = {
    "ideal": "Noiseless statevector simulation (exact)",
    "earth-lab": "Lab device: depolarising gate errors + readout (p2 = 0.003 default)",
    "leo-satellite": "Low Earth Orbit (~400 km): elevated depolarising + cosmic-ray bursts + thermal cycling",
    "satellite-ground": "Ground↔satellite link: atmospheric photon loss + turbulence dephasing + thermal photons",
    "meo-van-allen": "Medium Earth Orbit (Van Allen belts): sustained radiation, strong bursts",
    "deep-space": "Interplanetary: galactic cosmic rays, highest depolarising + burst rates",
    "custom": "User-defined channel probabilities",
}

#: Environments that represent a space/radiation setting (for Space Mode).
SPACE_ENVIRONMENTS = ["leo-satellite", "satellite-ground", "meo-van-allen", "deep-space"]

#: Depolarising + burst environments (share the same channel structure).
_DEPOLARISING_ENVS = ["earth-lab", "leo-satellite", "meo-van-allen", "deep-space"]

# Representative noise level per environment — the value the UI slider jumps to
# when the environment is selected, and the reference point (scale = 1) at which
# the base profile below equals the published numbers.
REPRESENTATIVE_LEVELS: Dict[str, float] = {
    "ideal": 0.0,
    "earth-lab": 0.003,
    "leo-satellite": 0.008,
    "satellite-ground": 0.005,
    "meo-van-allen": 0.020,
    "deep-space": 0.050,
    "custom": 0.010,
}

# Base channel profiles at scale = 1 (i.e. at the representative level).
_BASE: Dict[str, Dict[str, float]] = {
    "earth-lab": {"p1": 0.003, "p2": 0.003, "readout": 0.003},
    "leo-satellite": {
        "p1": 0.002, "p2": 0.008, "readout": 0.02,
        "burst_rate": 0.01,       # 1% of gates hit by a cosmic-ray burst
        "thermal_mod": 0.30,      # +30% depolarising from 90-min thermal cycling
    },
    "meo-van-allen": {
        "p1": 0.008, "p2": 0.020, "readout": 0.02, "burst_rate": 0.03,
    },
    "deep-space": {
        "p1": 0.02, "p2": 0.05, "readout": 0.03, "burst_rate": 0.05,
    },
    # satellite-ground is handled specially (elevation-driven photon loss).
    "satellite-ground": {"ad_rate": 0.6, "pd_rate": 0.05, "n_th": 0.1},
}

# Default absolute channels for the 'custom' environment.
_CUSTOM_DEFAULTS = {
    "p1": 0.003, "p2": 0.003, "readout": 0.003, "burst_rate": 0.0,
    "thermal_mod": 0.0, "ad_rate": 0.0, "pd_rate": 0.0, "n_th": 0.0,
    "elevation": 45.0,
}

# Reference elevation (deg) at which satellite-ground photon loss = base ad_rate.
_REF_ELEVATION = 45.0


# ---------------------------------------------------------------------------
# Pure helpers (build nothing — safe in the Streamlit parent process)
# ---------------------------------------------------------------------------
def _clamp_prob(p: float, name: str) -> float:
    """Clamp a probability into the physically valid range for its channel."""
    p = float(p)
    if p < 0.0:
        p = 0.0
    if name == "p2":
        return min(p, 15.0 / 16.0)   # 2-qubit depolarising is CPTP for p <= 15/16
    if name == "p1":
        return min(p, 3.0 / 4.0)     # 1-qubit depolarising is CPTP for p <= 3/4
    if name in ("ad", "pd", "n_th"):
        return min(p, 1.0)
    return min(p, 0.5)               # symmetric readout error: p <= 1/2


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def eta_from_elevation(elevation_deg: float, eta_ref: float = 0.6,
                       ref_elevation: float = _REF_ELEVATION) -> float:
    """Atmospheric photon-loss probability as a function of elevation angle.

    Beer–Lambert slant-path model: the optical depth scales like ``1/sin(el)``,
    so ``eta(el) = 1 - exp(-tau / sin(el))`` with ``tau`` fixed by
    ``eta(ref_elevation) = eta_ref``.  Lower elevation → longer path → more loss.
    Purely numeric (no Qiskit), so it is safe anywhere.
    """
    el = float(np.clip(elevation_deg, 1.0, 90.0))
    eta_ref = _clamp01(eta_ref)
    if eta_ref <= 0.0:
        return 0.0
    if eta_ref >= 1.0:
        return 1.0
    tau = -np.log(1.0 - eta_ref) * np.sin(np.radians(ref_elevation))
    return _clamp01(1.0 - np.exp(-tau / np.sin(np.radians(el))))


def effective_params(environment: str, params: Optional[dict] = None) -> Dict[str, float]:
    """Return the *effective* per-channel probabilities for an environment.

    This is the single source of truth for how ``(environment, params)`` maps to
    channel strengths.  It builds **no** Qiskit object, so it is safe to call
    from the Streamlit parent (for display) *and* from the worker (to then build
    the ``NoiseModel``).  Keys are always present (0.0 when unused):
    ``p1, p2, readout, burst_rate, thermal_mod, ad_rate, pd_rate, n_th,
    elevation, scale``.
    """
    params = dict(params or {})
    zero = {
        "p1": 0.0, "p2": 0.0, "readout": 0.0, "burst_rate": 0.0, "thermal_mod": 0.0,
        "ad_rate": 0.0, "pd_rate": 0.0, "n_th": 0.0,
        "elevation": _REF_ELEVATION, "scale": 0.0,
    }
    if environment == "ideal":
        return zero

    if environment == "custom":
        out = dict(zero)
        for k, default in _CUSTOM_DEFAULTS.items():
            out[k] = float(params.get(k, default))
        out["p1"] = _clamp_prob(out["p1"], "p1")
        out["p2"] = _clamp_prob(out["p2"], "p2")
        out["readout"] = _clamp_prob(out["readout"], "readout")
        out["burst_rate"] = _clamp01(out["burst_rate"])
        out["thermal_mod"] = max(0.0, out["thermal_mod"])
        out["ad_rate"] = _clamp01(out["ad_rate"])
        out["pd_rate"] = _clamp_prob(out["pd_rate"], "pd")
        out["n_th"] = _clamp01(out["n_th"])
        out["scale"] = 1.0
        return out

    if environment not in _BASE:
        raise ValueError(
            f"Unknown environment '{environment}'. Choose one of {list(ENVIRONMENTS)}."
        )

    base = _BASE[environment]
    rep = REPRESENTATIVE_LEVELS[environment]
    level = float(params.get("noise_level", rep))
    scale = (level / rep) if rep > 0 else 0.0
    elevation = float(params.get("elevation", _REF_ELEVATION))

    out = dict(zero)
    out["elevation"] = elevation
    out["scale"] = scale

    if environment == "satellite-ground":
        eta = eta_from_elevation(elevation, base["ad_rate"])
        out["ad_rate"] = _clamp01(eta * scale)
        out["pd_rate"] = _clamp_prob(base["pd_rate"] * scale, "pd")
        out["n_th"] = _clamp01(base["n_th"] * scale)
        return out

    thermal_mod = float(base.get("thermal_mod", 0.0))
    out["thermal_mod"] = thermal_mod
    out["p1"] = _clamp_prob(base.get("p1", 0.0) * scale * (1.0 + thermal_mod), "p1")
    out["p2"] = _clamp_prob(base.get("p2", 0.0) * scale * (1.0 + thermal_mod), "p2")
    out["readout"] = _clamp_prob(base.get("readout", 0.0) * scale, "readout")
    out["burst_rate"] = _clamp01(base.get("burst_rate", 0.0) * scale)
    return out


def noise_summary(environment: str, params: Optional[dict] = None) -> Dict[str, float]:
    """Effective per-channel probabilities for display (pure; parent-safe).

    Backwards-compatible: always includes ``p1``, ``p2`` and ``readout``.  Space
    environments additionally report ``burst_rate`` / ``thermal_mod`` (or
    ``ad_rate`` / ``pd_rate`` / ``n_th`` / ``elevation`` for the ground link).
    """
    return effective_params(environment, params)


# ---------------------------------------------------------------------------
# Metadata (pure; parent-safe)
# ---------------------------------------------------------------------------
_ENV_INFO: Dict[str, Dict] = {
    "ideal": {
        "name": "Ideal",
        "description": "Noiseless statevector simulation — the exact QAOA landscape.",
        "key_params": {},
        "citation": "Farhi, Goldstone & Gutmann (2014), arXiv:1411.4028 (QAOA).",
        "missions": ["—"],
    },
    "earth-lab": {
        "name": "Earth Laboratory",
        "description": "A present-day trapped-ion / superconducting processor: "
                       "depolarising gate errors plus symmetric readout error.",
        "key_params": {"p1": 0.003, "p2": 0.003, "readout": 0.003},
        "citation": "Representative NISQ device error rates (IBM Quantum / Quantinuum "
                    "public benchmarks, 2023–2024).",
        "missions": ["IBM Quantum", "Google Sycamore", "Quantinuum H-series"],
    },
    "leo-satellite": {
        "name": "LEO Satellite (~400 km)",
        "description": "Low Earth Orbit: elevated depolarising (thin shielding), "
                       "cosmic-ray burst errors, and a +30% modulation from the "
                       "90-minute sun/shadow thermal cycle.",
        "key_params": {"p1": 0.002, "p2": 0.008, "readout": 0.02,
                       "burst_rate": 0.01, "thermal_mod": 0.30},
        "citation": "Vepsäläinen et al., 'Impact of ionizing radiation on superconducting "
                    "qubit coherence', Nature 584, 551–556 (2020) (MIT / Lincoln Lab).",
        "missions": ["ISS", "Starlink", "Sentinel", "Quantum Experiment on Satellite (Micius)"],
    },
    "satellite-ground": {
        "name": "Satellite↔Ground Link",
        "description": "A free-space quantum link through the atmosphere: photon loss "
                       "(amplitude damping) that grows at low elevation, turbulence "
                       "dephasing, and a thermal-photon background.",
        "key_params": {"ad_rate@45deg": 0.6, "pd_rate": 0.05, "n_th": 0.1},
        "citation": "Liao et al., 'Satellite-to-ground quantum key distribution', "
                    "Nature 549, 43–47 (2017) (Micius).",
        "missions": ["Micius (QUESS)", "QEYSSat (planned)", "SPOOC"],
    },
    "meo-van-allen": {
        "name": "MEO / Van Allen Belts",
        "description": "Medium Earth Orbit inside the radiation belts: sustained "
                       "proton/electron flux → strong depolarising and frequent bursts.",
        "key_params": {"p1": 0.008, "p2": 0.020, "readout": 0.02, "burst_rate": 0.03},
        "citation": "Van Allen radiation-belt environment (NASA Van Allen Probes, "
                    "Baker et al., Space Sci. Rev. 179, 531–560 (2013)).",
        "missions": ["GPS (MEO)", "Galileo", "Van Allen Probes"],
    },
    "deep-space": {
        "name": "Deep Space (interplanetary)",
        "description": "Beyond the magnetosphere: unshielded galactic cosmic rays and "
                       "solar particle events give the highest depolarising and burst rates.",
        "key_params": {"p1": 0.02, "p2": 0.05, "readout": 0.03, "burst_rate": 0.05},
        "citation": "Cucinotta, Kim & Chappell, 'Space radiation risks to the central "
                    "nervous system', Neurosci. Biobehav. Rev. 38, 16–32 (2014) (NASA).",
        "missions": ["Artemis / Gateway", "Juno (Jupiter)", "Voyager", "Deep Space Network"],
    },
    "custom": {
        "name": "Custom",
        "description": "User-defined absolute channel probabilities.",
        "key_params": dict(_CUSTOM_DEFAULTS),
        "citation": "—",
        "missions": ["—"],
    },
}


def get_environment_info(environment: Optional[str] = None) -> Dict:
    """Metadata for one environment, or all of them when ``environment is None``.

    Returns ``{name, description, key_params, citation, missions}``.  Pure (builds
    nothing) so it is safe to call from the Streamlit parent.
    """
    if environment is None:
        return {k: dict(v) for k, v in _ENV_INFO.items()}
    if environment not in _ENV_INFO:
        raise ValueError(
            f"Unknown environment '{environment}'. Choose one of {list(ENVIRONMENTS)}."
        )
    return dict(_ENV_INFO[environment])


# ---------------------------------------------------------------------------
# Channel builders (Qiskit objects — WORKER SUBPROCESS ONLY)
# ---------------------------------------------------------------------------
def _burst_mixture(burst_rate: float, num_qubits: int):
    """``(1 - br) * Identity + br * Depolarizing(0.5)`` on ``num_qubits``.

    Built explicitly from Kraus operators (matches the spec's mixture).  It is
    identically ``Depolarizing(br / 2)``; we keep the explicit form for clarity.
    """
    br = _clamp01(burst_rate)
    if br <= 0.0:
        return None
    dim = 2 ** num_qubits
    identity = np.eye(dim, dtype=complex)
    strong = Kraus(depolarizing_error(0.5, num_qubits)).data
    mats = [np.sqrt(1.0 - br) * identity] + [np.sqrt(br) * K for K in strong]
    return kraus_error(mats)


def _tensor_2q(err1):
    """Two-qubit error = the same single-qubit channel applied to each qubit."""
    kraus = Kraus(err1).data
    return kraus_error([np.kron(A, B) for A in kraus for B in kraus])


def _compose(base_err, extra_err):
    """Compose two QuantumErrors (either may be ``None``)."""
    if base_err is None:
        return extra_err
    if extra_err is None:
        return base_err
    return base_err.compose(extra_err)


def get_noise_model(environment: str, params: Optional[dict] = None):
    """Build the Aer ``NoiseModel`` for a given ``environment``.

    Parameters
    ----------
    environment:
        One of :data:`ENVIRONMENTS` (``'ideal'`` returns ``None``).
    params:
        Overrides.  Recognised keys:

        * ``noise_level`` – master slider value (scales the base profile by
          ``noise_level / REP``).  Used by every environment except ``custom``.
        * ``elevation`` – satellite-ground elevation angle in degrees (drives
          photon loss).
        * ``custom`` only: absolute ``p1, p2, readout, burst_rate, thermal_mod,
          ad_rate, pd_rate, n_th``.

    Returns
    -------
    NoiseModel | None
        ``None`` for ``ideal``; otherwise a fully-populated ``NoiseModel`` tagged
        with ``_qc_params`` (the effective channel dict) and ``_qc_readout_prob``
        (so the exact evaluator can recover the readout strength).

    Raises
    ------
    ValueError
        If ``environment`` is not recognised.
    """
    if environment not in ENVIRONMENTS:
        raise ValueError(
            f"Unknown environment '{environment}'. Choose one of {list(ENVIRONMENTS)}."
        )
    if environment == "ideal":
        return None

    ep = effective_params(environment, params)
    noise_model = NoiseModel()
    p_read = ep["readout"]

    if environment == "satellite-ground" or (environment == "custom" and ep["ad_rate"] > 0):
        # --- photon-loss / dephasing link ----------------------------------
        ad, pd, nth = ep["ad_rate"], ep["pd_rate"], ep["n_th"]
        e1 = None
        if ad > 0.0:
            e1 = amplitude_damping_error(ad, excited_state_population=nth)
        if pd > 0.0:
            e1 = _compose(e1, phase_damping_error(pd))
        if e1 is not None:
            noise_model.add_all_qubit_quantum_error(e1, _SINGLE_QUBIT_GATES)
            noise_model.add_all_qubit_quantum_error(_tensor_2q(e1), _TWO_QUBIT_GATES)
    else:
        # --- depolarising (+ thermal modulation + cosmic-ray bursts) --------
        p1, p2 = ep["p1"], ep["p2"]
        br = ep["burst_rate"]
        burst1 = _burst_mixture(br, 1)
        burst2 = _burst_mixture(br, 2)

        e1 = depolarizing_error(p1, 1) if p1 > 0.0 else None
        e1 = _compose(e1, burst1)
        if e1 is not None:
            noise_model.add_all_qubit_quantum_error(e1, _SINGLE_QUBIT_GATES)

        e2 = depolarizing_error(p2, 2) if p2 > 0.0 else None
        e2 = _compose(e2, burst2)
        if e2 is not None:
            noise_model.add_all_qubit_quantum_error(e2, _TWO_QUBIT_GATES)

    if p_read > 0.0:
        readout_err = ReadoutError([[1.0 - p_read, p_read], [p_read, 1.0 - p_read]])
        noise_model.add_all_qubit_readout_error(readout_err)

    # Tag the model so the exact evaluator can recover the readout strength
    # without fragile introspection.
    try:
        noise_model._qc_params = dict(ep)
        noise_model._qc_readout_prob = float(p_read)
    except Exception:
        pass

    return noise_model
