"""Experimental Validation — preregistered QAOA hardware/simulator comparison."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, Mapping, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from core.experimental_validation import (  # noqa: E402
    ExperimentalValidationError,
    analyze_validation_study,
    condition_counts_from_frame,
    create_experiment_manifest,
    graph_from_manifest_record,
    manifest_to_json,
    parse_counts_csv,
    parse_experiment_manifest,
    standard_angle_grid,
    validation_report_markdown,
)
from core.graph_generator import GRAPH_FAMILIES, generate_graph  # noqa: E402
from core.hardware_profiles import AVAILABLE_HARDWARE, hardware_noise_spec  # noqa: E402
from core.qaoa_evaluator import evaluate_qaoa_exact  # noqa: E402

st.set_page_config(page_title="Experimental Validation · Quantum Compass", page_icon="🧪", layout="wide")


def _build_graph(family: str, n: int, seed: int, er_probability: float, ba_m: int) -> nx.Graph:
    """Build a small hardware-study graph from primitive Streamlit controls."""
    if family == "3-regular":
        return generate_graph(family, n, seed=seed, d=3)
    if family == "erdos-renyi":
        return generate_graph(family, n, seed=seed, p=er_probability)
    return generate_graph(family, n, seed=seed, m=min(int(ba_m), int(n) - 1))


def _counts_signature(manifest: Mapping, frame: pd.DataFrame, hardware_key: str | None) -> str:
    """Create a session-state key that prevents stale validation reports."""
    csv_text = frame.sort_values(["condition_id", "bitstring"]).to_csv(index=False)
    payload = f"{manifest['manifest_id']}|{hardware_key or 'ideal-only'}|{csv_text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _simulation_predictions(
    manifest: Mapping,
    measured_frame: pd.DataFrame,
    fake_backend_key: str | None,
    progress_callback=None,
) -> Dict[str, Dict[str, Dict[str, int]]]:
    """Evaluate ideal and selected fake-calibration models for observed conditions.

    Qiskit/Aer execution remains in the existing persistent worker subprocess;
    this Streamlit page only passes graph/angle/noise primitives to the core API.
    """
    graph = graph_from_manifest_record(manifest["graph"])
    observed = condition_counts_from_frame(measured_frame, manifest)
    conditions = {
        str(item["condition_id"]): item
        for item in manifest["execution_plan"]["conditions"]
    }
    model_specs = {"ideal_aer": None}
    if fake_backend_key:
        model_specs[f"fake_{fake_backend_key}_reduced"] = hardware_noise_spec(
            fake_backend_key, graph.number_of_nodes()
        )
    output: Dict[str, Dict[str, Dict[str, int]]] = {name: {} for name in model_specs}
    total = len(observed) * len(model_specs)
    done = 0
    for condition_id in sorted(observed, key=lambda identifier: int(conditions[identifier]["run_order"])):
        condition = conditions[condition_id]
        observed_shots = int(sum(observed[condition_id].values()))
        for model_name, spec in model_specs.items():
            result = evaluate_qaoa_exact(
                graph,
                condition["gammas"],
                condition["betas"],
                noise_model=spec,
                shots=observed_shots,
                readout_prob=None,
            )
            output[model_name][condition_id] = {
                str(bitstring): int(count)
                for bitstring, count in result["counts"].items()
            }
            done += 1
            if progress_callback is not None:
                progress_callback(done, total)
    return output


def _condition_dataframe(analysis: Mapping) -> pd.DataFrame:
    """Flatten condition-level ratio estimates and model errors for display."""
    rows = []
    for condition in analysis.get("conditions", []):
        observed = condition["observed"]
        row = {
            "Run": int(condition["run_order"]),
            "Condition": condition["condition_id"],
            "γ": float(condition["gamma"]),
            "β": float(condition["beta"]),
            "Shots": int(observed["shots"]),
            "Observed ratio": float(observed["approximation_ratio"]),
            "95% lower": float(observed["lower"]),
            "95% upper": float(observed["upper"]),
        }
        for model_name, result in condition.get("models", {}).items():
            if result.get("available"):
                row[f"{model_name} ratio"] = float(result["predicted"]["approximation_ratio"])
                row[f"{model_name} TVD"] = float(result["total_variation_distance"])
                row[f"{model_name} |ratio error|"] = float(result["absolute_ratio_error"])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("Run") if rows else pd.DataFrame()


def _ratio_figure(table: pd.DataFrame) -> go.Figure:
    """Plot observed QAOA ratios with bootstrap intervals and model predictions."""
    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=table["Run"], y=table["Observed ratio"], mode="markers+lines", name="Measured hardware",
        marker=dict(color="#FFD166", size=10), line=dict(color="#FFD166", width=2),
        error_y=dict(
            type="data", symmetric=False,
            array=(table["95% upper"] - table["Observed ratio"]).tolist(),
            arrayminus=(table["Observed ratio"] - table["95% lower"]).tolist(),
            color="#FFD166",
        ),
        hovertemplate=(
            "run %{x}<br>observed ratio=%{y:.5f}<extra>bootstrap 95% interval shown</extra>"
        ),
    ))
    for column in table.columns:
        if column.endswith(" ratio") and column != "Observed ratio":
            name = column[:-6]
            figure.add_trace(go.Scatter(
                x=table["Run"], y=table[column], mode="markers+lines", name=name,
                marker=dict(size=7), line=dict(width=1.6, dash="dot"),
            ))
    figure.update_layout(
        title=dict(text="Pre-specified QAOA ratio comparison", x=0.5),
        xaxis=dict(title="Randomized execution order", dtick=1, gridcolor="#2A2A3E"),
        yaxis=dict(title="Approximation ratio", range=[0, 1.02], gridcolor="#2A2A3E"),
        height=430, paper_bgcolor="#1A1A2E", plot_bgcolor="#0A0A0A",
        font=dict(color="#E0E0E0"), legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return figure


def _model_summary_dataframe(analysis: Mapping) -> pd.DataFrame:
    """Format aggregate model-comparison metrics for transparent interpretation."""
    rows = []
    for name, summary in analysis.get("model_summary", {}).items():
        if not summary.get("available_conditions"):
            rows.append({"Model": name, "Conditions": 0})
            continue
        rows.append({
            "Model": name,
            "Conditions": int(summary["available_conditions"]),
            "Weighted MAE ratio": float(summary["weighted_mean_absolute_ratio_error"]),
            "Weighted RMSE ratio": float(summary["weighted_root_mean_squared_ratio_error"]),
            "Weighted TVD": float(summary["weighted_mean_total_variation_distance"]),
            "Weighted Hellinger": float(summary["weighted_mean_hellinger_distance"]),
            "Weighted JS divergence": float(summary["weighted_mean_jensen_shannon_divergence"]),
        })
    return pd.DataFrame(rows)


def main() -> None:
    st.title("🧪 Experimental Validation")
    st.caption(
        "Pre-specify a small QAOA study, submit it to real hardware with your own IBM Quantum credentials, "
        "then compare measured counts against explicitly named simulator models."
    )
    st.warning(
        "**Scientific boundary:** this page never fabricates a hardware result. Uploaded counts are required for "
        "an experimental claim; agreement with a simulator validates only this manifest/back-end/time window, not "
        "a general noise mechanism, environment diagnosis, or quantum advantage."
    )

    st.sidebar.header("🧪 Protocol design")
    family = st.sidebar.selectbox("Graph family", list(GRAPH_FAMILIES), index=0)
    if family == "3-regular":
        n = st.sidebar.select_slider("Logical qubits n", options=[4, 6, 8], value=6)
        er_probability, ba_m = 0.5, 2
    elif family == "erdos-renyi":
        n = st.sidebar.slider("Logical qubits n", 4, 8, 6, 1)
        er_probability = st.sidebar.slider("Edge probability", 0.20, 0.80, 0.50, 0.05)
        ba_m = 2
    else:
        n = st.sidebar.slider("Logical qubits n", 4, 8, 6, 1)
        ba_m = st.sidebar.slider("Scale-free attachment m", 1, min(3, int(n) - 1), min(2, int(n) - 1), 1)
        er_probability = 0.5
    graph_seed = int(st.sidebar.number_input("Graph seed", min_value=0, max_value=999_999, value=42, step=1))
    depth = st.sidebar.selectbox("QAOA depth", [1, 2], index=0,
                                 help="Depth 2 uses an explicitly labelled shared-angle slice, not a full four-parameter optimization.")
    grid_size = st.sidebar.selectbox("Angle-grid size", [2, 3, 4], index=1)
    shots = st.sidebar.select_slider("Shots per circuit", options=[1024, 2048, 4096, 8192], value=4096)
    protocol_seed = int(st.sidebar.number_input("Execution-order seed", min_value=0, max_value=999_999, value=2027, step=1))

    if st.sidebar.button("Create checksum-identified validation manifest", type="primary", width="stretch"):
        try:
            graph = _build_graph(family, int(n), graph_seed, er_probability, ba_m)
            gamma_values, beta_values = standard_angle_grid(int(grid_size))
            metadata = {
                "family": family,
                "seed": graph_seed,
                "erdos_renyi_probability": er_probability if family == "erdos-renyi" else None,
                "scale_free_m": ba_m if family == "scale-free" else None,
            }
            manifest = create_experiment_manifest(
                graph,
                graph_metadata=metadata,
                depth=int(depth),
                gamma_values=gamma_values,
                beta_values=beta_values,
                shots=int(shots),
                randomization_seed=protocol_seed,
                bootstrap_resamples=2000,
            )
            st.session_state["validation_manifest"] = manifest
            st.session_state.pop("validation_result", None)
            st.success("Protocol created. Download it before submitting any hardware job.")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not create protocol: {exc}")

    st.subheader("1. Pre-specify the experiment")
    st.markdown(
        "The manifest fixes the graph, angle grid, shot budget, execution order, bit convention, and analysis metrics "
        "before data are collected. This prevents choosing only favourable angle settings after inspecting a device run."
    )
    uploaded_manifest = st.file_uploader("Or upload an existing validation manifest", type=["json"])
    manifest = st.session_state.get("validation_manifest")
    if uploaded_manifest is not None:
        try:
            manifest = parse_experiment_manifest(uploaded_manifest.getvalue())
            st.session_state["validation_manifest"] = manifest
            st.session_state.pop("validation_result", None)
            st.success("Validated uploaded manifest checksum.")
        except ExperimentalValidationError as exc:
            st.error(f"Invalid manifest: {exc}")
            manifest = None

    if manifest is None:
        st.info("Create a manifest in the sidebar or upload an existing one to begin.")
        return

    manifest = parse_experiment_manifest(manifest)
    execution = manifest["execution_plan"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Manifest conditions", execution["condition_count"])
    c2.metric("Logical qubits", manifest["graph"]["n_qubits"])
    c3.metric("Shots / condition", execution["shots_per_condition"])
    c4.metric("Total planned shots", execution["shots_per_condition"] * execution["condition_count"])
    st.code(f"Manifest ID: {manifest['manifest_id']}", language="text")
    st.download_button(
        "⬇️ Download preregistered manifest",
        data=manifest_to_json(manifest),
        file_name="qaoa_validation_manifest.json",
        mime="application/json",
    )
    st.caption(
        "The checksum detects accidental protocol edits; it is not an external timestamp. For a genuine preregistration, "
        "commit the downloaded JSON and protocol to a dated Git tag, repository release, or research archive before job submission."
    )
    with st.expander("Review randomized execution plan", expanded=False):
        condition_table = pd.DataFrame(execution["conditions"])[["run_order", "condition_id", "gamma", "beta"]]
        st.dataframe(condition_table, width="stretch", hide_index=True)
        st.caption("Bitstrings stored by the analysis use vertex order: character i represents logical graph vertex i.")

    st.subheader("2. Collect actual hardware counts")
    st.markdown(
        "Run the repository CLI outside Streamlit. It first supports a credential-free dry run, then submits measured "
        "circuits through IBM Runtime only when you provide your own token in the shell environment."
    )
    st.code(
        "python scripts/submit_ibm_validation.py \\\n"
        "  --manifest qaoa_validation_manifest.json \\\n"
        "  --output-dir results/protocol_check \\\n"
        "  --dry-run\n\n"
        "# Set IBM_QUANTUM_TOKEN securely in your shell before submission.\n"
        "python scripts/submit_ibm_validation.py \\\n"
        "  --manifest qaoa_validation_manifest.json \\\n"
        "  --backend ibm_brisbane \\\n"
        "  --output-dir results/ibm_brisbane_run",
        language="bash",
    )
    st.caption(
        "The CLI writes `counts_long.csv`, `raw_qiskit_counts.json`, and `run_metadata.json`. "
        "Replace `ibm_brisbane` with a backend available to your account, and never upload or commit your IBM token."
    )

    st.subheader("3. Upload measured data and evaluate the registered comparison")
    counts_upload = st.file_uploader("Upload counts_long.csv from a hardware run", type=["csv"])
    source_order = st.selectbox(
        "Uploaded bitstring convention",
        ["vertex-order", "qiskit-classical-order"],
        index=0,
        help="Use vertex-order for the CLI output. Select qiskit-classical-order only for raw counts exported directly by Qiskit.",
    )
    if counts_upload is None:
        st.info("Hardware analysis remains unavailable until measured counts are uploaded. This is intentional.")
        return
    try:
        measured_frame = parse_counts_csv(counts_upload.getvalue(), manifest, source_order=source_order)
    except ExperimentalValidationError as exc:
        st.error(f"Invalid measured-counts CSV: {exc}")
        return
    except Exception as exc:  # noqa: BLE001
        st.error(f"Unexpected count-upload error: {exc}")
        return

    measured_conditions = sorted(measured_frame["condition_id"].unique())
    total_measured = int(measured_frame["count"].sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Observed conditions", len(measured_conditions))
    c2.metric("Measured shots", total_measured)
    c3.metric("Missing planned conditions", execution["condition_count"] - len(measured_conditions))
    with st.expander("Normalized measured counts", expanded=False):
        st.dataframe(measured_frame, width="stretch", hide_index=True)

    include_fake_reference = st.checkbox(
        "Include a FakeProvider calibration-reference model",
        value=True,
        help=(
            "Enable this only when the selected local fake snapshot is a meaningful reference for the measured backend. "
            "It is never the live backend's exact current calibration."
        ),
    )
    hardware_key = None
    if include_fake_reference:
        hardware_key = st.selectbox(
            "Calibration-reference model",
            list(AVAILABLE_HARDWARE),
            format_func=lambda key: AVAILABLE_HARDWARE[key]["label"],
            help="A local reduced-noise model derived from a shipped FakeProvider calibration snapshot.",
        )
    else:
        st.caption(
            "Only the ideal simulator will be compared. This is the scientifically safer choice when no matching "
            "FakeProvider snapshot exists for the measured backend."
        )
    signature = _counts_signature(manifest, measured_frame, hardware_key)
    if st.button("Run preregistered simulator comparison", type="primary", width="stretch"):
        progress = st.progress(0.0, text="Preparing model comparisons…")
        try:
            def update(done: int, total: int) -> None:
                progress.progress(done / max(1, total), text=f"Evaluating simulator condition {done}/{total}…")

            predictions = _simulation_predictions(manifest, measured_frame, hardware_key, update)
            analysis = analyze_validation_study(manifest, measured_frame, predictions, bootstrap_seed=73)
            st.session_state["validation_result"] = {
                "signature": signature,
                "analysis": analysis,
                "predictions": predictions,
            }
            progress.empty()
            st.success("Comparison completed. Interpret model agreement only within the protocol boundary below.")
        except Exception as exc:  # noqa: BLE001
            progress.empty()
            st.error(f"Could not complete simulator comparison: {exc}")
            return

    saved = st.session_state.get("validation_result")
    if not saved or saved.get("signature") != signature:
        st.info("Run the registered comparison to calculate bootstrap intervals and full-distribution metrics.")
        return

    analysis = saved["analysis"]
    st.subheader("4. Results with interpretation boundary")
    table = _condition_dataframe(analysis)
    if table.empty:
        st.warning("No observed conditions were available after validation.")
        return
    st.plotly_chart(_ratio_figure(table), width="stretch", key=f"validation_ratio_{signature}")
    st.dataframe(table, width="stretch", hide_index=True)
    st.markdown("**Aggregate distributional agreement**")
    st.dataframe(_model_summary_dataframe(analysis), width="stretch", hide_index=True)
    st.info(str(analysis["interpretation_boundary"]))

    report = validation_report_markdown(manifest, analysis)
    left, right = st.columns(2)
    with left:
        st.download_button(
            "⬇️ Download analysis JSON",
            data=json.dumps(analysis, indent=2),
            file_name="qaoa_hardware_validation_analysis.json",
            mime="application/json",
            width="stretch",
        )
    with right:
        st.download_button(
            "⬇️ Download transparent Markdown report",
            data=report,
            file_name="qaoa_hardware_validation_report.md",
            mime="text/markdown",
            width="stretch",
        )

    with st.expander("Research protocol, limitations, and next experiment", expanded=False):
        st.markdown(
            "Read `docs/research/experimental_validation_protocol.md` before reporting results. A strong follow-up "
            "repeats the same immutable manifest on a second backend or date, preserves failed/missing conditions, "
            "and reports model disagreement rather than selecting only visually favourable curves."
        )


main()
