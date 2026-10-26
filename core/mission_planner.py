"""Space-mission planning heuristics for Quantum Compass.

The Mission Planner is intentionally a transparent engineering estimator, not a
flight-certification tool.  It turns a mission's orbit, qubit technology,
coherence, calibrated gate fidelity, duration and shielding into a coherent
noise/risk assessment and a conservative QAOA operating recommendation.

All calculations are pure Python/numpy-free arithmetic and JSON-serialisable,
so the planner is safe in Streamlit's parent process and easy to export/test.
The model is explicit about its assumptions:

* ``T1`` / ``T2`` are supplied in microseconds;
* ``gate_fidelity`` is a representative two-qubit entangling-gate fidelity;
* radiation events are planning-level relative rates, not a radiation transport
  calculation; and
* recommendations target small MaxCut/QAOA demonstrations, not fault-tolerant
  workloads.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, List, Mapping


# Orbit-level planning priors.  Values are deliberately exposed in the output so
# users can see where the recommendation came from and replace inputs in Custom.
_ORBIT_PROFILES: Dict[str, Dict] = {
    "leo": {
        "label": "Low Earth Orbit (LEO)",
        "radiation_events_per_day": 0.8,
        "thermal_cycles_per_day": 15.5,
        "solar_flare_risk": "moderate",
        "radiation_multiplier": 1.0,
        "operating_window": "Schedule high-value circuits in thermally stable orbital segments; recalibrate around eclipse transitions.",
    },
    "iss": {
        "label": "ISS / shielded LEO",
        "radiation_events_per_day": 0.55,
        "thermal_cycles_per_day": 15.5,
        "solar_flare_risk": "moderate",
        "radiation_multiplier": 0.80,
        "operating_window": "Use station shielding and avoid high-latitude / South-Atlantic-Anomaly passes for long variational sweeps.",
    },
    "cislunar": {
        "label": "Cislunar transfer / lunar orbit",
        "radiation_events_per_day": 3.0,
        "thermal_cycles_per_day": 1.0,
        "solar_flare_risk": "high",
        "radiation_multiplier": 2.5,
        "operating_window": "Run short batches behind available shielding; gate long jobs on space-weather alerts.",
    },
    "lunar_gateway": {
        "label": "Lunar Gateway / NRHO",
        "radiation_events_per_day": 4.2,
        "thermal_cycles_per_day": 1.0,
        "solar_flare_risk": "high",
        "radiation_multiplier": 3.2,
        "operating_window": "Favor protected module orientations and execute only after radiation-monitor green status.",
    },
    "mars_orbit": {
        "label": "Mars Orbit",
        "radiation_events_per_day": 6.5,
        "thermal_cycles_per_day": 0.55,
        "solar_flare_risk": "very high",
        "radiation_multiplier": 4.8,
        "operating_window": "Use radiation-monitor gating, short calibration-verified bursts, and defer variational scans during solar-particle events.",
    },
    "deep_space": {
        "label": "Deep Space / interplanetary cruise",
        "radiation_events_per_day": 7.5,
        "thermal_cycles_per_day": 0.35,
        "solar_flare_risk": "very high",
        "radiation_multiplier": 5.4,
        "operating_window": "Operate only in shielded/calibration-verified windows; preserve a classical fallback for all science products.",
    },
}

_TECH_PROFILES: Dict[str, Dict] = {
    "superconducting": {
        "label": "Superconducting",
        "two_qubit_gate_us": 0.30,
        "radiation_sensitivity": 1.15,
        "default_mitigation": "Active quasiparticle/radiation monitoring plus dynamical decoupling",
    },
    "trapped_ion": {
        "label": "Trapped-ion",
        "two_qubit_gate_us": 20.0,
        "radiation_sensitivity": 0.65,
        "default_mitigation": "Laser/clock recalibration and sympathetic cooling checks",
    },
    "spin": {
        "label": "Spin / semiconductor",
        "two_qubit_gate_us": 0.20,
        "radiation_sensitivity": 0.85,
        "default_mitigation": "Charge-noise monitoring, recalibration and dynamical decoupling",
    },
    "photonic": {
        "label": "Photonic",
        "two_qubit_gate_us": 0.05,
        "radiation_sensitivity": 0.30,
        "default_mitigation": "Source-brightness monitoring and loss-aware post-selection",
    },
}

_SHIELDING_FACTORS = {
    "none": 1.25,
    "light": 1.00,
    "moderate": 0.65,
    "heavy": 0.40,
}

# Templates are illustrative mission concepts, not claims about actual flight
# hardware. Keep all values primitive and in the same units as Custom inputs.
_MISSION_TEMPLATES: Dict[str, Dict] = {
    "spoqc_like": {
        "name": "SPOQC-like LEO demonstrator",
        "orbit": "LEO",
        "qubit_tech": "superconducting",
        "T1": 200.0,
        "T2": 160.0,
        "gate_fidelity": 0.9970,
        "available_qubits": 10,
        "duration": 180.0,
        "shielding": "moderate",
        "notes": "Illustrative compact quantum-computing payload in LEO; not a specification for a named mission.",
    },
    "iss_quantum": {
        "name": "ISS quantum laboratory",
        "orbit": "ISS",
        "qubit_tech": "trapped-ion",
        "T1": 600.0,
        "T2": 450.0,
        "gate_fidelity": 0.9960,
        "available_qubits": 12,
        "duration": 365.0,
        "shielding": "heavy",
        "notes": "Illustrative shielded-LEO laboratory profile with crewed-station operations constraints.",
    },
    "lunar_gateway": {
        "name": "Lunar Gateway payload",
        "orbit": "Lunar Gateway",
        "qubit_tech": "superconducting",
        "T1": 150.0,
        "T2": 100.0,
        "gate_fidelity": 0.9940,
        "available_qubits": 12,
        "duration": 365.0,
        "shielding": "moderate",
        "notes": "Illustrative cislunar payload profile with elevated radiation and one-cycle-per-day thermal operations.",
    },
    "mars_orbiter": {
        "name": "Mars Orbiter payload",
        "orbit": "Mars Orbit",
        "qubit_tech": "superconducting",
        "T1": 100.0,
        "T2": 60.0,
        "gate_fidelity": 0.9900,
        "available_qubits": 12,
        "duration": 730.0,
        "shielding": "light",
        "notes": "Illustrative Mars-orbit technology demonstration profile; solar-particle operations dominate planning.",
    },
}

_ORBIT_ALIASES = {
    "leo": "leo", "low earth orbit": "leo",
    "iss": "iss", "iss / shielded leo": "iss",
    "cislunar": "cislunar", "lunar orbit": "cislunar", "cislunar transfer": "cislunar",
    "lunar gateway": "lunar_gateway", "gateway": "lunar_gateway", "nrho": "lunar_gateway",
    "mars orbit": "mars_orbit", "mars_orbiter": "mars_orbit", "mars": "mars_orbit",
    "deep space": "deep_space", "interplanetary": "deep_space",
}
_TECH_ALIASES = {
    "superconducting": "superconducting", "transmon": "superconducting",
    "trapped-ion": "trapped_ion", "trapped ion": "trapped_ion", "ion": "trapped_ion",
    "spin": "spin", "semiconductor": "spin", "spin / semiconductor": "spin",
    "photonic": "photonic", "photonics": "photonic",
}


def _canonical_template_name(name: str) -> str:
    key = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    if key not in _MISSION_TEMPLATES:
        raise ValueError(f"Unknown mission template '{name}'. Choose from {list(_MISSION_TEMPLATES)}.")
    return key


def get_mission_template(name: str) -> Dict:
    """Return an independent mission-parameter template.

    Valid names are ``spoqc_like``, ``iss_quantum``, ``lunar_gateway`` and
    ``mars_orbiter``.  A deep copy prevents a UI caller from mutating the
    canonical template for later users.
    """
    return deepcopy(_MISSION_TEMPLATES[_canonical_template_name(name)])


def _canonical_orbit(value: object) -> str:
    key = str(value).strip().lower()
    if key not in _ORBIT_ALIASES:
        raise ValueError(f"Unknown orbit '{value}'. Choose one of {sorted(set(_ORBIT_ALIASES))}.")
    return _ORBIT_ALIASES[key]


def _canonical_tech(value: object) -> str:
    key = str(value).strip().lower()
    if key not in _TECH_ALIASES:
        raise ValueError(f"Unknown qubit technology '{value}'. Choose one of {sorted(set(_TECH_ALIASES))}.")
    return _TECH_ALIASES[key]


def _normalise_params(mission_params: Mapping) -> Dict:
    """Validate and normalize mission input into stable primitive values."""
    if not isinstance(mission_params, Mapping):
        raise ValueError("mission_params must be a mapping/dict.")
    raw = dict(mission_params)
    # Permit lowercase entries in programmatic callers while retaining the
    # documented T1/T2 spelling.
    t1 = raw.get("T1", raw.get("t1"))
    t2 = raw.get("T2", raw.get("t2"))
    required = {
        "orbit": raw.get("orbit"), "qubit_tech": raw.get("qubit_tech"),
        "T1": t1, "T2": t2, "gate_fidelity": raw.get("gate_fidelity"),
        "available_qubits": raw.get("available_qubits"), "duration": raw.get("duration"),
        "shielding": raw.get("shielding"),
    }
    missing = [key for key, value in required.items() if value is None]
    if missing:
        raise ValueError(f"mission_params is missing required field(s): {', '.join(missing)}.")

    orbit = _canonical_orbit(required["orbit"])
    tech = _canonical_tech(required["qubit_tech"])
    shielding = str(required["shielding"]).strip().lower()
    if shielding not in _SHIELDING_FACTORS:
        raise ValueError(f"shielding must be one of {list(_SHIELDING_FACTORS)}, got '{required['shielding']}'.")

    try:
        t1_us = float(required["T1"])
        t2_us = float(required["T2"])
        fidelity = float(required["gate_fidelity"])
        qubits = int(required["available_qubits"])
        duration_days = float(required["duration"])
    except (TypeError, ValueError) as exc:
        raise ValueError("T1, T2, gate_fidelity, available_qubits and duration must be numeric.") from exc
    if t1_us <= 0.0 or t2_us <= 0.0:
        raise ValueError("T1 and T2 must be positive microseconds.")
    if not 0.0 < fidelity <= 1.0:
        raise ValueError("gate_fidelity must lie in (0, 1].")
    if qubits < 1:
        raise ValueError("available_qubits must be at least 1.")
    if duration_days <= 0.0:
        raise ValueError("duration must be positive days.")

    return {
        "orbit_key": orbit,
        "orbit": _ORBIT_PROFILES[orbit]["label"],
        "qubit_tech_key": tech,
        "qubit_tech": _TECH_PROFILES[tech]["label"],
        "T1_us": t1_us,
        "T2_us": t2_us,
        "gate_fidelity": fidelity,
        "available_qubits": qubits,
        "duration_days": duration_days,
        "shielding": shielding,
        "notes": str(raw.get("notes", "")),
    }


def _noise_band(noise: float) -> str:
    if noise < 0.005:
        return "low"
    if noise < 0.015:
        return "moderate"
    if noise < 0.035:
        return "high"
    return "severe"


def _strategy_for_noise(noise: float, available_qubits: int) -> Dict:
    """Map effective noise to a conservative small-QAOA strategy."""
    if noise < 0.005:
        depth = 3
        max_n = min(available_qubits, 10)
        return {
            "algorithm": "QAOA with noise-aware warm start",
            "optimal_depth": depth,
            "depth_range": [1, 3],
            "max_problem_size": max(4, max_n),
            "expected_advantage": "Plausible for small structured MaxCut instances after a matched classical baseline check.",
            "mitigation": ["Readout mitigation", "Pauli twirling", "Zero-noise extrapolation on final candidates"],
        }
    if noise < 0.015:
        depth = 2
        max_n = min(available_qubits, 8)
        return {
            "algorithm": "Shallow QAOA with warm start",
            "optimal_depth": depth,
            "depth_range": [1, 2],
            "max_problem_size": max(4, max_n),
            "expected_advantage": "Conditional: benchmark p=1 and p=2 against a classical heuristic before allocating mission time.",
            "mitigation": ["Readout mitigation", "Dynamical decoupling", "Randomized compiling / Pauli twirling", "Frequent calibration checks"],
        }
    if noise < 0.035:
        depth = 1
        max_n = min(available_qubits, 6)
        return {
            "algorithm": "p=1 QAOA demonstration with classical warm start",
            "optimal_depth": depth,
            "depth_range": [1, 1],
            "max_problem_size": max(4, max_n),
            "expected_advantage": "Exploratory only: useful for calibration-aware demonstrations, not a robust algorithmic advantage claim.",
            "mitigation": ["Aggressive readout mitigation", "Dynamical decoupling", "Error-detection sentinels", "Classical fallback"],
        }
    depth = 1
    max_n = min(available_qubits, 4)
    return {
        "algorithm": "Calibration experiment / p=1 QAOA micro-benchmark",
        "optimal_depth": depth,
        "depth_range": [1, 1],
        "max_problem_size": max(4, max_n),
        "expected_advantage": "No robust quantum advantage expected under the present noise budget; prioritize telemetry and hardware characterization.",
        "mitigation": ["Radiation-event veto", "Readout mitigation", "Short circuits only", "Classical fallback as primary path"],
    }


def _risk_level(score: float) -> str:
    if score < 0.25:
        return "low"
    if score < 0.50:
        return "moderate"
    if score < 0.75:
        return "high"
    return "very high"


def generate_mission_plan(mission_params: Mapping) -> Dict:
    """Generate a coherent, JSON-serialisable space-QAOA mission plan.

    Parameters are ``orbit``, ``qubit_tech``, ``T1``/``T2`` in microseconds,
    representative two-qubit ``gate_fidelity``, ``available_qubits``, mission
    ``duration`` in days, and categorical ``shielding`` (`none`/`light`/
    `moderate`/`heavy`).

    The returned dict always contains the requested sections:
    ``environment_assessment``, ``recommended_strategy``, ``risk_assessment``
    and ``advantage_window``.
    """
    params = _normalise_params(mission_params)
    orbit = _ORBIT_PROFILES[params["orbit_key"]]
    tech = _TECH_PROFILES[params["qubit_tech_key"]]
    shielding_factor = float(_SHIELDING_FACTORS[params["shielding"]])

    # A representative two-qubit gate spends this fraction of a coherence time
    # exposed.  It is intentionally a simple planning approximation, not a
    # replacement for gate-level T1/T2 simulation.
    gate_time_us = float(tech["two_qubit_gate_us"])
    coherence_loss = 0.5 * gate_time_us * (1.0 / params["T1_us"] + 1.0 / params["T2_us"])
    base_gate_error = max(0.0, 1.0 - params["gate_fidelity"])
    radiation_events = float(orbit["radiation_events_per_day"] * shielding_factor * tech["radiation_sensitivity"])
    thermal_cycles = float(orbit["thermal_cycles_per_day"])
    radiation_penalty = radiation_events * 8.0e-4
    thermal_penalty = thermal_cycles * 3.0e-5
    # Slow radiation-induced calibration drift grows with mission duration but
    # is capped so it remains a planning term rather than an accumulating error
    # probability over every gate.
    duration_drift = min(0.015, (params["duration_days"] / 365.0) * radiation_events * 2.0e-4)
    effective_noise = min(0.50, base_gate_error + coherence_loss + radiation_penalty + thermal_penalty + duration_drift)
    band = _noise_band(effective_noise)

    strategy = _strategy_for_noise(effective_noise, params["available_qubits"])
    strategy["mitigation"] = [tech["default_mitigation"], *strategy["mitigation"]]
    strategy["operating_point"] = orbit["operating_window"]
    strategy["recommended_problem_size"] = f"n = 4…{strategy['max_problem_size']} logical qubits"

    # The same inputs drive operational risks.  Mars/deep-space becomes more
    # conservative than LEO through orbit multiplier, event rate and duration.
    solar_score = min(1.0, 0.10 * float(orbit["radiation_multiplier"]) * shielding_factor)
    barren_score = min(1.0, 0.45 * (effective_noise / 0.035) + 0.08 * max(0, strategy["optimal_depth"] - 1))
    drift_score = min(1.0, duration_drift / 0.010)
    review_days = 7 if band in {"high", "severe"} else (14 if band == "moderate" else 30)
    if params["duration_days"] >= 365.0:
        timeline = f"Track calibration drift every {review_days} days; radiation-driven retuning is likely within months without telemetry-gated operation."
    else:
        timeline = f"Track calibration drift every {review_days} days; retain before/after-pass reference circuits throughout the mission." 

    max_n = int(strategy["max_problem_size"])
    advantage_window = {
        "n_range": [4, max_n],
        "depth_range": list(strategy["depth_range"]),
        "problem_types": [
            "Small sparse MaxCut / scheduling graphs",
            "Calibration-aware variational benchmark instances",
            "Hybrid warm-start combinatorial optimization",
        ],
        "conditions": (
            "Use only after calibration/reference circuits pass; compare against a matched classical heuristic and pause during radiation alerts."
        ),
        "expected_advantage": strategy["expected_advantage"],
    }

    return {
        "mission_parameters": {
            "orbit": params["orbit"],
            "qubit_tech": params["qubit_tech"],
            "T1_us": float(params["T1_us"]),
            "T2_us": float(params["T2_us"]),
            "gate_fidelity": float(params["gate_fidelity"]),
            "available_qubits": int(params["available_qubits"]),
            "duration_days": float(params["duration_days"]),
            "shielding": params["shielding"],
            "notes": params["notes"],
        },
        "environment_assessment": {
            "orbit": params["orbit"],
            "noise_band": band,
            "estimated_effective_noise": float(effective_noise),
            "estimated_effective_noise_percent": float(100.0 * effective_noise),
            "radiation_events_per_day": float(radiation_events),
            "thermal_cycles_per_day": float(thermal_cycles),
            "base_gate_error": float(base_gate_error),
            "coherence_loss_per_2q_gate": float(coherence_loss),
            "radiation_penalty": float(radiation_penalty),
            "thermal_penalty": float(thermal_penalty),
            "duration_drift_penalty": float(duration_drift),
            "assumptions": "Planning-level aggregate estimate; T1/T2 are in microseconds and gate fidelity is representative of a two-qubit entangler.",
        },
        "recommended_strategy": strategy,
        "risk_assessment": {
            "solar_flare_risk": orbit["solar_flare_risk"],
            "solar_flare_risk_level": _risk_level(solar_score),
            "barren_plateau_risk": _risk_level(barren_score),
            "barren_plateau_risk_score": float(barren_score),
            "calibration_drift_risk": _risk_level(drift_score),
            "degradation_timeline": timeline,
            "abort_conditions": [
                "Radiation monitor exceeds mission threshold",
                "Reference-circuit fidelity drops by >20% from last calibration",
                "Readout mitigation residual exceeds the science error budget",
            ],
        },
        "advantage_window": advantage_window,
    }
