"""Space Mission Planner for Quantum Compass.

The planner converts an orbit and hardware budget into a transparent,
engineering-level QAOA recommendation.  It is a planning aid, not a flight
qualification analysis: all radiation rates and mitigation choices are exposed
in the generated report so mission teams can replace assumptions with their own
models.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
import streamlit as st

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from core.mission_planner import generate_mission_plan, get_mission_template  # noqa: E402
from utils.export import export_analysis_pdf  # noqa: E402

st.set_page_config(page_title="Space Mission Planner · Quantum Compass", page_icon="🛰️", layout="wide")

_TEMPLATE_OPTIONS = {
    "SPOQC-like": "spoqc_like",
    "ISS quantum": "iss_quantum",
    "Lunar Gateway": "lunar_gateway",
    "Mars Orbiter": "mars_orbiter",
    "Custom": None,
}


def _custom_params() -> Dict:
    """Render the custom mission parameter form and return primitive inputs."""
    st.sidebar.subheader("Custom mission parameters")
    orbit = st.sidebar.selectbox(
        "Orbit", ["LEO", "ISS", "Cislunar", "Lunar Gateway", "Mars Orbit", "Deep Space"],
    )
    qubit_tech = st.sidebar.selectbox(
        "Qubit technology", ["superconducting", "trapped-ion", "spin", "photonic"],
    )
    c1, c2 = st.sidebar.columns(2)
    t1 = c1.number_input("T1 (µs)", min_value=1.0, max_value=10000.0, value=150.0, step=10.0)
    t2 = c2.number_input("T2 (µs)", min_value=1.0, max_value=10000.0, value=100.0, step=10.0)
    fidelity = st.sidebar.slider("Representative 2Q gate fidelity", 0.9000, 0.9999, 0.9950, 0.0001,
                                 format="%.4f")
    c3, c4 = st.sidebar.columns(2)
    qubits = c3.number_input("Available logical qubits", min_value=1, max_value=128, value=8, step=1)
    duration = c4.number_input("Mission duration (days)", min_value=1.0, max_value=5000.0, value=365.0, step=30.0)
    shielding = st.sidebar.selectbox("Shielding", ["none", "light", "moderate", "heavy"], index=2)
    notes = st.sidebar.text_area("Mission notes", value="", placeholder="Payload, thermal-control or operations constraints…")
    return {
        "name": "Custom mission", "orbit": orbit, "qubit_tech": qubit_tech,
        "T1": float(t1), "T2": float(t2), "gate_fidelity": float(fidelity),
        "available_qubits": int(qubits), "duration": float(duration),
        "shielding": shielding, "notes": notes,
    }


def _section_dataframe(mapping: Dict, labels: Dict[str, str] | None = None) -> pd.DataFrame:
    """Turn a simple dict into an attractive label/value table."""
    labels = labels or {}
    rows = []
    for key, value in mapping.items():
        label = labels.get(key, key.replace("_", " ").title())
        if isinstance(value, float):
            rendered = f"{value:.5f}" if abs(value) < 0.01 else f"{value:.3f}"
        elif isinstance(value, list):
            rendered = " · ".join(str(x) for x in value)
        else:
            rendered = str(value)
        rows.append({"Item": label, "Assessment": rendered})
    return pd.DataFrame(rows)


def _mission_pdf(plan: Dict, template_label: str) -> bytes:
    """Create a real multi-page PDF mission report from pure plan data."""
    env = plan["environment_assessment"]
    strategy = plan["recommended_strategy"]
    risk = plan["risk_assessment"]
    window = plan["advantage_window"]
    params = plan["mission_parameters"]
    blocks: List[Dict] = [
        {
            "type": "text", "heading": "Mission inputs",
            "lines": [
                f"Template: {template_label}",
                f"Orbit: {params['orbit']} | technology: {params['qubit_tech']}",
                f"T1/T2: {params['T1_us']:.1f} / {params['T2_us']:.1f} us",
                f"2Q gate fidelity: {params['gate_fidelity']:.5f} | available qubits: {params['available_qubits']}",
                f"Duration: {params['duration_days']:.0f} days | shielding: {params['shielding']}",
                params.get("notes", "") or "No extra mission notes provided.",
            ],
        },
        {
            "type": "text", "heading": "Environment assessment",
            "lines": [
                f"Noise band: {env['noise_band']}",
                f"Estimated effective noise: {env['estimated_effective_noise_percent']:.2f}%",
                f"Radiation events/day: {env['radiation_events_per_day']:.2f}",
                f"Thermal cycles/day: {env['thermal_cycles_per_day']:.2f}",
                f"Coherence loss per 2Q gate: {100 * env['coherence_loss_per_2q_gate']:.3f}%",
                env['assumptions'],
            ],
        },
        {
            "type": "text", "heading": "Recommended strategy",
            "lines": [
                f"Algorithm: {strategy['algorithm']}",
                f"Optimal depth: p={strategy['optimal_depth']} (range {strategy['depth_range']})",
                f"Recommended problem size: {strategy['recommended_problem_size']}",
                f"Expected advantage: {strategy['expected_advantage']}",
                "Mitigation: " + "; ".join(strategy['mitigation']),
                "Operating point: " + strategy['operating_point'],
            ],
        },
        {
            "type": "text", "heading": "Risk assessment and advantage window",
            "lines": [
                f"Solar-flare risk: {risk['solar_flare_risk']} ({risk['solar_flare_risk_level']})",
                f"Barren-plateau risk: {risk['barren_plateau_risk']} (score {risk['barren_plateau_risk_score']:.2f})",
                f"Calibration drift: {risk['calibration_drift_risk']}",
                risk['degradation_timeline'],
                f"Advantage n range: {window['n_range']} | depth range: {window['depth_range']}",
                "Problem types: " + "; ".join(window['problem_types']),
                "Conditions: " + window['conditions'],
            ],
        },
    ]
    return export_analysis_pdf(
        title="Space Mission Plan", subtitle=f"{template_label} — Quantum Compass", blocks=blocks,
        footer="Generated by Quantum Compass Mission Planner · Planning aid, not flight qualification",
    )


def main() -> None:
    st.title("🛰️ Space Mission Planner")
    st.caption(
        "Turn orbital environment and hardware assumptions into a conservative QAOA strategy, risk register, "
        "and quantum-advantage operating window."
    )
    st.markdown(
        "> **Planning model, not flight certification.** Radiation-event rates and coherence penalties are transparent "
        "heuristics for comparing mission concepts. Replace them with payload-specific radiation transport, thermal, and "
        "hardware reliability models before making operational decisions."
    )

    st.sidebar.header("🛰️ Mission configuration")
    template_label = st.sidebar.selectbox("Mission template", list(_TEMPLATE_OPTIONS), index=0)
    template_key = _TEMPLATE_OPTIONS[template_label]
    if template_key is None:
        mission_params = _custom_params()
    else:
        mission_params = get_mission_template(template_key)
        st.sidebar.success(f"Loaded {mission_params['name']}")
        with st.sidebar.expander("Template inputs", expanded=False):
            st.json({k: v for k, v in mission_params.items() if k not in {"name", "notes"}})
            st.caption(mission_params.get("notes", ""))

    config_key = json.dumps(mission_params, sort_keys=True, default=str)
    generate = st.sidebar.button("🚀 Generate Mission Plan", type="primary", width="stretch")
    if generate:
        try:
            plan = generate_mission_plan(mission_params)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Mission-plan generation failed: {exc}")
            return
        st.session_state["mission_plan_view"] = {
            "key": config_key, "plan": plan, "template_label": template_label,
        }

    view = st.session_state.get("mission_plan_view")
    if view is None or view.get("key") != config_key:
        st.info("Select a mission concept and press **🚀 Generate Mission Plan**.")
        st.markdown(
            "Start with **SPOQC-like** for a compact LEO demonstration, then compare the much more conservative "
            "**Mars Orbiter** recommendation under higher radiation exposure."
        )
        return

    plan = view["plan"]
    env = plan["environment_assessment"]
    strategy = plan["recommended_strategy"]
    risk = plan["risk_assessment"]
    window = plan["advantage_window"]

    # ---------------- Environment assessment ----------------
    st.subheader("🌍 Environment Assessment")
    c1, c2, c3 = st.columns(3)
    c1.metric("Estimated effective noise", f"{env['estimated_effective_noise_percent']:.2f}%", env["noise_band"])
    c2.metric("Radiation events / day", f"{env['radiation_events_per_day']:.2f}")
    c3.metric("Thermal cycles / day", f"{env['thermal_cycles_per_day']:.1f}")
    st.dataframe(
        _section_dataframe({
            "base_gate_error": env["base_gate_error"],
            "coherence_loss_per_2q_gate": env["coherence_loss_per_2q_gate"],
            "radiation_penalty": env["radiation_penalty"],
            "thermal_penalty": env["thermal_penalty"],
            "duration_drift_penalty": env["duration_drift_penalty"],
            "assumptions": env["assumptions"],
        }, {
            "base_gate_error": "Base 2Q gate error",
            "coherence_loss_per_2q_gate": "Coherence loss / 2Q gate",
            "radiation_penalty": "Radiation contribution",
            "thermal_penalty": "Thermal contribution",
            "duration_drift_penalty": "Duration/drift contribution",
        }), width="stretch", hide_index=True,
    )

    # ---------------- Strategy ----------------
    st.subheader("🎯 Recommended Strategy")
    st.success(
        f"**{strategy['algorithm']}** · target **p={strategy['optimal_depth']}** · "
        f"{strategy['recommended_problem_size']}"
    )
    s1, s2 = st.columns([1, 1])
    with s1:
        st.markdown("**Mitigation stack**")
        for mitigation in strategy["mitigation"]:
            st.markdown(f"- {mitigation}")
    with s2:
        st.markdown("**Operating point**")
        st.write(strategy["operating_point"])
        st.markdown("**Expected advantage**")
        st.write(strategy["expected_advantage"])

    # ---------------- Risks ----------------
    st.subheader("⚠️ Risk Assessment")
    r1, r2, r3 = st.columns(3)
    r1.metric("Solar-flare risk", risk["solar_flare_risk_level"].title(), risk["solar_flare_risk"])
    r2.metric("Barren-plateau risk", risk["barren_plateau_risk"].title(), f"score {risk['barren_plateau_risk_score']:.2f}")
    r3.metric("Calibration drift", risk["calibration_drift_risk"].title())
    st.warning(risk["degradation_timeline"])
    with st.expander("Abort / pause conditions", expanded=False):
        for condition in risk["abort_conditions"]:
            st.markdown(f"- {condition}")

    # ---------------- Advantage window ----------------
    st.subheader("📊 Quantum Advantage Window")
    w1, w2, w3 = st.columns(3)
    w1.metric("Logical-qubit window", f"n={window['n_range'][0]}–{window['n_range'][1]}")
    w2.metric("Depth window", " / ".join(f"p={x}" for x in window["depth_range"]))
    w3.metric("Problem families", f"{len(window['problem_types'])} recommended")
    st.info(window["expected_advantage"])
    st.markdown("**Suitable problem types**")
    for problem_type in window["problem_types"]:
        st.markdown(f"- {problem_type}")
    st.caption("**Operating conditions:** " + window["conditions"])

    # ---------------- Export ----------------
    st.subheader("📥 Export Mission Plan")
    json_payload = json.dumps(plan, indent=2)
    pdf_payload = _mission_pdf(plan, view["template_label"])
    e1, e2 = st.columns(2)
    with e1:
        st.download_button("⬇️ Download JSON data", json_payload,
                           file_name="quantum_compass_mission_plan.json", mime="application/json",
                           width="stretch")
    with e2:
        st.download_button("⬇️ Download PDF report", pdf_payload,
                           file_name="quantum_compass_mission_plan.pdf", mime="application/pdf",
                           width="stretch")


main()
