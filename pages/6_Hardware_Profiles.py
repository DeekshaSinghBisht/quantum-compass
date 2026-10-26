"""IBM hardware-profile landscapes for Quantum Compass.

This page uses the calibration snapshots distributed by
``qiskit_ibm_runtime.fake_provider``.  It never needs cloud credentials and does
not submit a job to IBM hardware.  The Streamlit parent passes a primitive
hardware descriptor to the persistent Aer worker; the worker constructs the
Fake backend and its calibration-derived QAOA noise model safely in the child.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from core import aer_worker  # noqa: E402
from core.graph_generator import GRAPH_FAMILIES, generate_graph  # noqa: E402
from core.hardware_profiles import AVAILABLE_HARDWARE, hardware_noise_spec  # noqa: E402
from core.landscape import compute_landscape  # noqa: E402
from utils.plotting import plot_landscape_contour_plotly  # noqa: E402

st.set_page_config(page_title="Hardware Profiles · Quantum Compass", page_icon="🖥️", layout="wide")

_HARDWARE_LABELS = {meta["label"]: key for key, meta in AVAILABLE_HARDWARE.items()}
_HARDWARE_OPTIONS = list(_HARDWARE_LABELS)


@st.cache_data(show_spinner=False)
def cached_hardware_stats(backend_name: str, subset_size: int) -> Dict:
    """Read calibration stats inside the Aer worker, never in the Streamlit parent."""
    return aer_worker.hardware_stats(backend_name, int(subset_size))


@st.cache_data(show_spinner=False)
def cached_landscape_pair(
    backend_name: str,
    family: str,
    n_requested: int,
    seed: int,
    depth: int,
    resolution: int,
    shots: int,
    p_er: float,
    m_ba: int,
) -> Tuple[Dict, Dict, Dict]:
    """Compute ideal + calibration-derived hardware landscapes for one graph."""
    n = int(n_requested)
    graph_kwargs: Dict = {}
    note = ""
    if family == "3-regular":
        if n % 2:
            n -= 1
            note = f"3-regular requires even n; using n={n}."
        graph_kwargs["d"] = 3
    elif family == "erdos-renyi":
        graph_kwargs["p"] = float(p_er)
    else:
        graph_kwargs["m"] = min(int(m_ba), n - 1)
    graph = generate_graph(family, n, seed=int(seed), **graph_kwargs)
    ideal = compute_landscape(graph, depth=int(depth), resolution=int(resolution),
                              noise_model=None, shots=int(shots), readout_prob=None)
    # This dict contains no Qiskit object; the worker builds the fake backend
    # reduction and resulting Aer NoiseModel.
    hardware = compute_landscape(
        graph, depth=int(depth), resolution=int(resolution),
        noise_model=hardware_noise_spec(backend_name, graph.number_of_nodes()),
        shots=int(shots), readout_prob=None,
    )
    meta = {
        "n": graph.number_of_nodes(), "edges": graph.number_of_edges(),
        "family": family, "seed": int(seed), "depth": int(depth), "note": note,
    }
    return ideal, hardware, meta


def stats_table(stats: Dict) -> pd.DataFrame:
    """Human-friendly display of real FakeProvider calibration statistics."""
    rows = [
        ("Physical qubits", f"{int(stats['n_qubits'])}"),
        ("Logical subset used", f"{int(stats['subset_size'])}  ({stats['selected_qubits']})"),
        ("Average CX/ECR-equivalent error", f"{100 * float(stats['avg_cx_error']):.3f}%"),
        ("Average 1-qubit gate error", f"{100 * float(stats['avg_single_qubit_error']):.4f}%"),
        ("Average T1", f"{float(stats['avg_T1_us']):.1f} µs"),
        ("Average T2", f"{float(stats['avg_T2_us']):.1f} µs"),
        ("Average readout error", f"{100 * float(stats['readout_error']):.3f}%"),
        ("2-qubit calibration source", str(stats['two_qubit_calibration_source'])),
    ]
    return pd.DataFrame(rows, columns=["Hardware statistic", "Calibration value"])


def loss_figure(ideal: Dict, hardware: Dict, backend_label: str) -> go.Figure:
    """Visualize pointwise approximation-ratio loss versus ideal."""
    loss = np.asarray(ideal["ratios"], dtype=float) - np.asarray(hardware["ratios"], dtype=float)
    fig = go.Figure(go.Heatmap(
        x=hardware["betas"], y=hardware["gammas"], z=loss.T,
        colorscale="Reds", zmin=0.0, zmax=max(1e-4, float(np.max(loss))),
        colorbar=dict(title="ideal −<br>hardware"),
        hovertemplate="β=%{x:.3f}<br>γ=%{y:.3f}<br>ratio loss=%{z:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=f"Noise cost across the landscape — {backend_label}", x=0.5),
        height=420, margin=dict(l=50, r=25, t=60, b=45),
        xaxis_title="β (mixer angle, rad)", yaxis_title="γ (cost angle, rad)",
        paper_bgcolor="#1A1A2E", plot_bgcolor="#0A0A0A", font=dict(color="#E0E0E0"),
    )
    return fig


def main() -> None:
    st.title("🖥️ Real IBM Hardware Profiles")
    st.caption(
        "Calibration snapshots from IBM Runtime fake backends — no account, credentials, or cloud job needed. "
        "The hardware landscape uses calibration-derived 1Q, 2Q and readout noise."
    )
    st.markdown(
        "> **Scope.** This is a local Aer simulation calibrated from a FakeProvider snapshot, not a live IBM job. "
        "For a small logical QAOA graph, the worker reduces the real backend calibration onto native `h` / `rx` / `rzz` gates; "
        "it does not pretend to density-matrix simulate all 27 or 127 physical qubits."
    )

    st.sidebar.header("🖥️ Hardware run")
    hardware_label = st.sidebar.selectbox("IBM hardware selector", _HARDWARE_OPTIONS, index=0)
    backend_name = _HARDWARE_LABELS[hardware_label]
    st.sidebar.caption(AVAILABLE_HARDWARE[backend_name]["description"])

    family = st.sidebar.selectbox("Graph family", list(GRAPH_FAMILIES), index=0)
    n_requested = st.sidebar.slider("Logical qubits / graph vertices", 4, 8, 4, 1,
                                    help="Small logical subsets keep noisy density-matrix landscapes practical.")
    depth = st.sidebar.selectbox("QAOA depth p", [1, 2], index=0)
    resolution = st.sidebar.selectbox("Landscape resolution", [9, 10, 11, 12], index=1)
    seed = st.sidebar.number_input("Graph seed", 0, 9999, 42, 1)
    shots = st.sidebar.select_slider("Reported-count shots", [256, 512, 1024, 2048], value=512)
    p_er = 0.5
    m_ba = 2
    if family == "erdos-renyi":
        p_er = st.sidebar.slider("Erdős–Rényi edge probability", 0.3, 0.8, 0.5, 0.05)
    elif family == "scale-free":
        m_ba = st.sidebar.slider("Scale-free attachment m", 1, 3, 2, 1)

    effective_subset = int(n_requested if family != "3-regular" or n_requested % 2 == 0 else n_requested - 1)
    try:
        with st.spinner(f"Loading {hardware_label} calibration snapshot…"):
            stats = cached_hardware_stats(backend_name, effective_subset)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not load fake-backend calibration: {exc}")
        return

    st.subheader(f"Calibration snapshot — {hardware_label}")
    st.dataframe(stats_table(stats), width="stretch", hide_index=True)
    for warning in stats.get("calibration_warnings", []):
        st.warning(warning)

    compute = st.sidebar.button("🚀 Compute Landscape on This Hardware", type="primary", width="stretch")
    config = (backend_name, family, int(n_requested), int(seed), int(depth), int(resolution),
              int(shots), float(p_er), int(m_ba))
    if compute:
        with st.spinner("Computing ideal and calibration-derived noisy landscapes in the Aer worker…"):
            try:
                ideal, hardware, meta = cached_landscape_pair(*config)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Hardware landscape computation failed: {exc}")
                return
        st.session_state["hardware_profile_view"] = {
            "config": config, "ideal": ideal, "hardware": hardware, "meta": meta, "stats": stats,
        }

    view = st.session_state.get("hardware_profile_view")
    if view is None or tuple(view.get("config", ())) != config:
        st.info("Select an IBM profile, inspect its calibration statistics, then press **Compute Landscape on This Hardware**.")
        return

    ideal = view["ideal"]
    hardware = view["hardware"]
    meta = view["meta"]
    if meta.get("note"):
        st.info(meta["note"])
    st.subheader(f"QAOA-MaxCut landscape — n={meta['n']}, {meta['edges']} edges, p={meta['depth']}")

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            plot_landscape_contour_plotly(ideal["gammas"], ideal["betas"], ideal["ratios"],
                                          title="Ideal simulator"),
            width="stretch", key=f"hardware_ideal_{backend_name}_{meta['n']}_{meta['depth']}",
        )
    with c2:
        st.plotly_chart(
            plot_landscape_contour_plotly(hardware["gammas"], hardware["betas"], hardware["ratios"],
                                          title=f"{hardware_label} calibration profile"),
            width="stretch", key=f"hardware_noisy_{backend_name}_{meta['n']}_{meta['depth']}",
        )

    ideal_best = float(ideal["best_ratio"])
    hardware_best = float(hardware["best_ratio"])
    performance_cost = 0.0 if ideal_best <= 0.0 else 100.0 * max(0.0, ideal_best - hardware_best) / ideal_best
    m1, m2, m3 = st.columns(3)
    m1.metric("Ideal best ratio", f"{ideal_best:.4f}")
    m2.metric(f"{hardware_label} best ratio", f"{hardware_best:.4f}", delta=f"-{performance_cost:.1f}% vs ideal")
    m3.metric("Landscape range retained", f"{100 * np.ptp(hardware['ratios']) / max(1e-9, np.ptp(ideal['ratios'])):.0f}%")
    st.info(
        f"**Hardware insight.** On **{hardware_label}**, calibration-derived noise costs you "
        f"**{performance_cost:.1f}%** of the best approximation ratio compared with the ideal simulator "
        f"for this n={meta['n']}, p={meta['depth']} instance. The dominant aggregate two-qubit error is "
        f"{100 * float(stats['avg_cx_error']):.3f}% and readout error is {100 * float(stats['readout_error']):.3f}%."
    )
    st.plotly_chart(loss_figure(ideal, hardware, hardware_label), width="stretch",
                    key=f"hardware_loss_{backend_name}_{meta['n']}_{meta['depth']}")

    with st.expander("📥 Export hardware comparison"):
        payload = {
            "hardware_stats": stats, "graph": meta,
            "ideal": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in ideal.items()},
            "hardware": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in hardware.items()},
            "performance_cost_percent": performance_cost,
        }
        st.download_button("⬇️ Download comparison JSON", json.dumps(payload, indent=2),
                           file_name=f"{backend_name}_qaoa_landscape.json", mime="application/json",
                           width="stretch")


main()
