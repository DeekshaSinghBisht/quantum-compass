"""Barren Plateau Phase Diagram — trainability across size, depth and environment.

The page visualises whether an optimizer still sees usable structure in QAOA's
``(gamma, beta)`` landscape.  Green cells are trainable, yellow cells are mixed,
red cells are barren, and cyan is reserved for a genuinely noise-created
structured (resonant) island.  A black contour marks trainability score 0.5.

Large noisy density-matrix landscapes cannot fit at n=16.  The core engine is
explicit about its practical protocol: exact ideal landscapes everywhere, exact
noisy landscapes for safe small n, and a channel-strength/gate-exposure contrast
transfer for larger noisy cells.  Every result is persisted as JSON under
``data/barren_diagrams/`` and reloads instantly after its first computation.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from core.barren_plateau import (  # noqa: E402
    DEFAULT_EXACT_NOISE_MAX_N,
    TRAINABILITY_CLASSES,
    compare_diagrams,
    compute_barren_diagram,
    diagram_depths,
    diagram_sizes,
    estimate_barren_cost,
)
from core.noise_models import REPRESENTATIVE_LEVELS  # noqa: E402

DIAGRAM_DIR = _project_root / "data" / "barren_diagrams"
DIAGRAM_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="Barren Plateau Phase Diagram · Quantum Compass",
                   page_icon="🧊", layout="wide")

_ENV_LABELS = {
    "ideal": "Ideal",
    "earth-lab": "Earth",
    "leo-satellite": "LEO",
    "satellite-ground": "Satellite link",
    "meo-van-allen": "MEO / Van Allen",
    "deep-space": "Deep Space",
}
_LABEL_TO_ENV = {label: key for key, label in _ENV_LABELS.items()}
_ENV_OPTIONS = list(_LABEL_TO_ENV)

_CLASS_CODE = {"BARREN": 0, "MIXED": 1, "TRAINABLE": 2, "RESONANT": 3}
_CLASS_COLOUR = {
    "BARREN": "#c62828",      # red
    "MIXED": "#f9a825",       # yellow
    "TRAINABLE": "#2e7d32",   # green
    "RESONANT": "#00acc1",    # cyan
}
_ENV_COLOUR = {
    "ideal": "#455a64", "earth-lab": "#2e7d32", "leo-satellite": "#1565c0",
    "satellite-ground": "#c62828", "meo-van-allen": "#ef6c00",
    "deep-space": "#6a1b9a",
}


# --------------------------------------------------------------------------- #
# Durable cache: session state -> @st.cache_data disk read -> computation
# --------------------------------------------------------------------------- #
def cache_key(cfg: Dict) -> str:
    """Stable, filename-safe key for one environment's phase diagram."""
    canon = json.dumps(cfg, sort_keys=True, default=str, separators=(",", ":"))
    digest = hashlib.sha1(canon.encode("utf-8")).hexdigest()[:12]
    family = re.sub(r"[^A-Za-z0-9]+", "", cfg["family"])[:12]
    environment = re.sub(r"[^A-Za-z0-9]+", "", cfg["environment"])[:12]
    return (f"{family}_{environment}_n{cfg['n_min']}-{cfg['n_max']}_"
            f"p{cfg['p_min']}-{cfg['p_max']}_i{cfg['n_instances']}_{digest}")[:140]


def view_key(configs: Sequence[Dict]) -> str:
    """One key for a one-to-three-environment comparison currently on screen."""
    canon = json.dumps([cache_key(c) for c in configs], sort_keys=True)
    return hashlib.sha1(canon.encode("utf-8")).hexdigest()[:16]


@st.cache_data(show_spinner=False)
def read_diagram_json(path_str: str) -> Optional[Dict]:
    """Cached immutable JSON read; computation itself stays outside Streamlit cache."""
    try:
        return json.loads(Path(path_str).read_text(encoding="utf-8"))
    except Exception:
        return None


def _session_store() -> Dict[str, Dict]:
    return st.session_state.setdefault("barren_diagrams", {})


def try_load_cached(cfg: Dict) -> Tuple[Optional[Dict], Optional[str]]:
    """Load a configuration from session/disk without triggering a computation."""
    key = cache_key(cfg)
    store = _session_store()
    if key in store:
        return store[key], "session"
    path = DIAGRAM_DIR / f"{key}.json"
    if path.exists():
        data = read_diagram_json(str(path))
        if data is not None:
            store[key] = data
            return data, "disk"
    return None, None


def _persist(cfg: Dict, result: Dict) -> None:
    """Store a JSON-serialisable result in both the current session and disk cache."""
    key = cache_key(cfg)
    _session_store()[key] = result
    try:
        (DIAGRAM_DIR / f"{key}.json").write_text(json.dumps(result), encoding="utf-8")
    except Exception:
        # The in-memory result remains usable even on a read-only filesystem.
        pass


def _compute_pending_with_progress(pending: Sequence[Dict], estimates: Dict[str, Dict], bar) -> Dict[str, Dict]:
    """Compute uncached diagrams in a background thread while the main thread paints progress."""
    holder: Dict[str, object] = {"results": {}, "done": 0.0, "total": 1.0, "current": ""}
    done_event = threading.Event()
    errors: List[Optional[BaseException]] = [None]

    # Weight multi-environment progress by actual worker landscape jobs rather
    # than treating a small cached/ideal configuration as equal to a noisy one.
    weights = {
        cache_key(cfg): max(1, int(estimates[cache_key(cfg)]["ideal_jobs"]) +
                               int(estimates[cache_key(cfg)]["exact_noise_jobs"]))
        for cfg in pending
    }
    total_weight = float(sum(weights.values()))
    holder["total"] = total_weight

    def work() -> None:
        offset = 0.0
        try:
            for cfg in pending:
                key = cache_key(cfg)
                weight = float(weights[key])
                holder["current"] = _ENV_LABELS.get(cfg["environment"], cfg["environment"])

                def callback(done: float, total: int, *, _offset=offset, _weight=weight) -> None:
                    frac = float(done) / float(max(1, total))
                    holder["done"] = min(total_weight, _offset + _weight * frac)

                result = compute_barren_diagram(
                    family=cfg["family"], n_range=cfg["n_range"], p_range=cfg["p_range"],
                    noise_model=cfg["noise_model"], n_instances=cfg["n_instances"],
                    seed_base=cfg["seed_base"], resolution=cfg["resolution"], shots=cfg["shots"],
                    p_er=cfg["p_er"], m_ba=cfg["m_ba"], execution=cfg["execution"],
                    exact_noise_max_n=cfg["exact_noise_max_n"], progress_callback=callback,
                )
                holder["results"][key] = result  # type: ignore[index]
                offset += weight
                holder["done"] = offset
        except BaseException as exc:  # noqa: BLE001 - re-raised on the UI thread
            errors[0] = exc
        finally:
            done_event.set()

    thread = threading.Thread(target=work, daemon=True)
    thread.start()
    while not done_event.is_set():
        done = float(holder.get("done", 0.0))
        total = max(1.0, float(holder.get("total", 1.0)))
        fraction = min(1.0, done / total)
        current = str(holder.get("current", ""))
        suffix = f" · {current}" if current else ""
        bar.progress(fraction, text=f"Computing phase diagram{suffix} — {100 * fraction:.0f}%")
        time.sleep(0.20)
    if errors[0] is not None:
        raise errors[0]
    bar.progress(1.0, text="Phase diagram complete")
    return dict(holder["results"])  # type: ignore[arg-type]


def get_diagrams(configs: Sequence[Dict], estimates: Dict[str, Dict], bar) -> Tuple[Dict[str, Dict], Dict[str, str]]:
    """Return all requested environment diagrams, computing only cache misses."""
    results: Dict[str, Dict] = {}
    sources: Dict[str, str] = {}
    pending: List[Dict] = []
    for cfg in configs:
        environment = cfg["environment"]
        cached, source = try_load_cached(cfg)
        if cached is None:
            pending.append(cfg)
        else:
            results[environment] = cached
            sources[environment] = str(source)
    if pending:
        computed = _compute_pending_with_progress(pending, estimates, bar)
        for cfg in pending:
            environment = cfg["environment"]
            result = computed[cache_key(cfg)]
            _persist(cfg, result)
            results[environment] = result
            sources[environment] = "computed"
    else:
        bar.progress(1.0, text="Loaded from cache")
    return results, sources


# --------------------------------------------------------------------------- #
# Plotting and tables
# --------------------------------------------------------------------------- #
def plot_phase_heatmap(result: Dict, environment: str) -> go.Figure:
    """Categorical phase heatmap with a continuous score=0.5 boundary contour."""
    sizes = [int(x) for x in result["sizes"]]
    depths = [int(x) for x in result["depths"]]
    labels = np.asarray(result["classifications"], dtype=object)
    scores = np.asarray(result["scores"], dtype=float)
    variance = np.asarray(result["metrics"]["variance"], dtype=float)
    gradients = np.asarray(result["metrics"]["max_gradient"], dtype=float)
    ranges = np.asarray(result["metrics"]["ratio_range"], dtype=float)
    modes = np.asarray(result["source"]["mode"], dtype=object)
    codes = np.vectorize(lambda x: _CLASS_CODE.get(str(x), 1))(labels)

    custom = np.empty(scores.shape + (4,), dtype=object)
    custom[:, :, 0] = variance
    custom[:, :, 1] = gradients
    custom[:, :, 2] = ranges
    custom[:, :, 3] = modes

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=codes, x=sizes, y=depths, text=labels, customdata=custom,
        zmin=-0.5, zmax=3.5,
        # Discrete: red barren -> yellow mixed -> green trainable -> cyan resonant.
        colorscale=[
            [0.00, _CLASS_COLOUR["BARREN"]], [0.249, _CLASS_COLOUR["BARREN"]],
            [0.250, _CLASS_COLOUR["MIXED"]], [0.499, _CLASS_COLOUR["MIXED"]],
            [0.500, _CLASS_COLOUR["TRAINABLE"]], [0.749, _CLASS_COLOUR["TRAINABLE"]],
            [0.750, _CLASS_COLOUR["RESONANT"]], [1.000, _CLASS_COLOUR["RESONANT"]],
        ],
        colorbar=dict(
            title="phase", thickness=15, len=0.78,
            tickvals=[0, 1, 2, 3], ticktext=["Barren", "Mixed", "Trainable", "Resonant"],
        ),
        hovertemplate=("n=%{x}<br>QAOA depth p=%{y}<br>phase=%{text}<br>"
                       "trainability=%{customdata[4]:.3f}<br>"
                       "variance=%{customdata[0]:.5f}<br>"
                       "max |∇|=%{customdata[1]:.3f}<br>"
                       "ratio range=%{customdata[2]:.4f}<br>"
                       "source=%{customdata[3]}<extra></extra>"),
        showscale=True,
    ))
    # Add score as a fifth hover field without showing a second colour scale.
    fig.data[0].customdata = np.dstack((custom, scores))

    # The black contour is the requested phase boundary. It is separate from
    # colours so its value stays meaningful on a categorical heatmap.
    if scores.size and float(np.nanmin(scores)) <= 0.5 <= float(np.nanmax(scores)):
        fig.add_trace(go.Contour(
            z=scores, x=sizes, y=depths,
            contours=dict(start=0.5, end=0.5, size=0.1, coloring="none", showlabels=True,
                          labelfont=dict(size=10, color="#111111")),
            line=dict(color="#111111", width=3), showscale=False, hoverinfo="skip",
            name="trainability = 0.5",
        ))

    # A sparse interpolation-derived line makes the threshold readable even when
    # the contour has only a few cells to connect.
    bx: List[int] = []
    by: List[float] = []
    for n in sizes:
        item = result["phase_boundary"].get(str(n), {})
        if item.get("status") == "crosses":
            bx.append(n)
            by.append(float(item["critical_depth"]))
    if bx:
        fig.add_trace(go.Scatter(
            x=bx, y=by, mode="lines+markers", name="critical depth",
            line=dict(color="#111111", width=2, dash="dot"),
            marker=dict(size=7, color="#ffffff", line=dict(color="#111111", width=2)),
            hovertemplate="n=%{x}<br>critical p=%{y:.2f}<extra></extra>",
        ))

    label = _ENV_LABELS.get(environment, environment)
    noise = result.get("noise", {})
    fig.update_layout(
        title=dict(
            text=(f"Barren Plateau Phase Diagram — {label} "
                  f"(noise strength {float(noise.get('strength', 0.0)):.4f})"), x=0.5,
        ),
        height=555, margin=dict(l=55, r=30, t=65, b=48),
        xaxis=dict(title="number of qubits  n", dtick=1),
        yaxis=dict(title="QAOA depth  p", tickvals=depths, dtick=1),
        plot_bgcolor="#fcfcfc",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center"),
    )
    fig.add_annotation(
        x=sizes[0], y=max(depths) + 0.33,
        text="black = trainability boundary (score 0.5)", showarrow=False,
        xanchor="left", font=dict(size=11, color="#222222"),
    )
    return fig


def plot_boundary_comparison(results: Dict[str, Dict]) -> go.Figure:
    """Overlay critical QAOA depth versus n for two or three environments."""
    fig = go.Figure()
    for environment, result in results.items():
        sizes = [int(x) for x in result["sizes"]]
        depths = [int(x) for x in result["depths"]]
        y_min, y_max = min(depths), max(depths)
        x_cross: List[int] = []
        y_cross: List[float] = []
        x_above: List[int] = []
        y_above: List[float] = []
        x_below: List[int] = []
        y_below: List[float] = []
        for n in sizes:
            item = result["phase_boundary"].get(str(n), {})
            status = item.get("status")
            depth = float(item.get("critical_depth", y_min))
            if status == "crosses":
                x_cross.append(n); y_cross.append(depth)
            elif status == "above_range":
                x_above.append(n); y_above.append(y_max + 0.22)
            else:  # below range / empty: barren already at smallest displayed p
                x_below.append(n); y_below.append(y_min - 0.22)
        colour = _ENV_COLOUR.get(environment, "#333333")
        label = _ENV_LABELS.get(environment, environment)
        if x_cross:
            fig.add_trace(go.Scatter(
                x=x_cross, y=y_cross, mode="lines+markers", name=label,
                line=dict(color=colour, width=3, shape="spline", smoothing=0.45),
                marker=dict(size=8),
                hovertemplate="n=%{x}<br>critical depth p=%{y:.2f}<extra>" + label + "</extra>",
            ))
        if x_above:
            fig.add_trace(go.Scatter(
                x=x_above, y=y_above, mode="markers", name=f"{label} (above range)",
                marker=dict(symbol="triangle-up", color=colour, size=10),
                hovertemplate="n=%{x}<br>trainable through displayed p range<extra>" + label + "</extra>",
                showlegend=not bool(x_cross),
            ))
        if x_below:
            fig.add_trace(go.Scatter(
                x=x_below, y=y_below, mode="markers", name=f"{label} (below range)",
                marker=dict(symbol="triangle-down", color=colour, size=10),
                hovertemplate="n=%{x}<br>barren from smallest displayed p<extra>" + label + "</extra>",
                showlegend=not bool(x_cross),
            ))
    fig.update_layout(
        title=dict(text="Phase boundary by environment — critical QAOA depth vs problem size", x=0.5),
        height=445, margin=dict(l=55, r=25, t=65, b=48),
        xaxis=dict(title="number of qubits  n", dtick=1),
        yaxis=dict(title="critical depth p at trainability = 0.5", dtick=1),
        plot_bgcolor="#fcfcfc",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center"),
    )
    return fig


def cell_table(result: Dict) -> pd.DataFrame:
    """Long-form, inspectable table of every phase cell and its metrics."""
    rows: List[Dict] = []
    for di, depth in enumerate(result["depths"]):
        for ni, n in enumerate(result["sizes"]):
            rows.append({
                "n": int(n),
                "depth p": int(depth),
                "phase": result["classifications"][di][ni],
                "trainability": round(float(result["scores"][di][ni]), 4),
                "variance": round(float(result["metrics"]["variance"][di][ni]), 6),
                "max |∇|": round(float(result["metrics"]["max_gradient"][di][ni]), 4),
                "ratio range": round(float(result["metrics"]["ratio_range"][di][ni]), 5),
                "source": result["source"]["mode"][di][ni],
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Configuration and page rendering
# --------------------------------------------------------------------------- #
def _noise_descriptor(environment: str, noise_scale: float, elevation: float) -> Optional[Dict]:
    """Build a primitive worker descriptor; never build a Qiskit NoiseModel here."""
    if environment == "ideal":
        return None
    params: Dict[str, float] = {"noise_level": float(REPRESENTATIVE_LEVELS[environment] * noise_scale)}
    if environment == "satellite-ground":
        params["elevation"] = float(elevation)
    return {"environment": environment, "params": params}


def _format_duration(seconds: float) -> str:
    if seconds < 90:
        return f"~{seconds:.0f}s"
    if seconds < 3600:
        return f"~{seconds / 60.0:.1f} min"
    return f"~{seconds / 3600.0:.1f} h"


def _build_configs(
    family: str, n_min: int, n_max: int, p_min: int, p_max: int,
    environments: Sequence[str], noise_scale: float, n_instances: int,
    resolution: int, shots: int, seed_base: int, p_er: float, m_ba: int,
    elevation: float,
) -> List[Dict]:
    configs = []
    for environment in environments:
        configs.append({
            "family": family,
            "n_min": int(n_min), "n_max": int(n_max),
            "p_min": int(p_min), "p_max": int(p_max),
            "n_range": (int(n_min), int(n_max)), "p_range": (int(p_min), int(p_max)),
            "environment": environment,
            "noise_scale": float(noise_scale), "elevation": float(elevation),
            "noise_model": _noise_descriptor(environment, noise_scale, elevation),
            "n_instances": int(n_instances), "resolution": int(resolution),
            "shots": int(shots), "seed_base": int(seed_base),
            "p_er": float(p_er), "m_ba": int(m_ba),
            "execution": "auto", "exact_noise_max_n": int(DEFAULT_EXACT_NOISE_MAX_N),
        })
    return configs


def _render_expected(configs: Sequence[Dict], estimates: Dict[str, Dict], sizes: Sequence[int], depths: Sequence[int]) -> None:
    """Describe the pending computation before the user commits to it."""
    seconds = sum(float(estimates[cache_key(c)]["seconds"]) for c in configs)
    env_names = [f"**{_ENV_LABELS.get(c['environment'], c['environment'])}**" for c in configs]
    with st.expander("ℹ️ What this will compute", expanded=True):
        st.markdown(
            f"- **Graph family:** {configs[0]['family']} · **n:** {list(sizes)} · "
            f"**depths:** {list(depths)}\n"
            f"- **Environment(s):** {', '.join(env_names)} · **instances/cell:** "
            f"{configs[0]['n_instances']} · **landscape grid:** "
            f"{configs[0]['resolution']}×{configs[0]['resolution']}\n"
            f"- **Estimated first run:** {_format_duration(seconds)}; every completed "
            "environment is cached as JSON in `data/barren_diagrams/`.\n\n"
            "Every cell evaluates an exact ideal QAOA landscape. For noisy environments, "
            f"safe small instances (n ≤ {DEFAULT_EXACT_NOISE_MAX_N}) are density-matrix "
            "simulated exactly; larger cells use the documented contrast-transfer protocol "
            "to avoid an O(4ⁿ) memory failure."
        )


def main() -> None:
    st.title("🧊 Barren Plateau Phase Diagram")
    st.caption(
        "Where is a QAOA landscape still navigable? **Green** = trainable, **yellow** = mixed, "
        "**red** = barren, **cyan** = a noise-created structured (resonant) island. "
        "The black contour is the trainability phase boundary at score 0.5."
    )
    st.markdown(
        "> **Trainability score.** Each cell averages actual `(γ, β)` landscapes over graph "
        "instances using variance (40%), maximum gradient (35%) and ratio range (25%). "
        "A flat landscape scores 0; a high-relief landscape approaches 1. The full protocol "
        "is recorded with each export so large-n transfer cells are never mistaken for a "
        "density-matrix calculation."
    )

    # ---------------- sidebar ----------------
    st.sidebar.header("🧊 Phase diagram")
    family = st.sidebar.selectbox("Graph family", ["3-regular", "erdos-renyi", "scale-free"], index=0)
    c1, c2 = st.sidebar.columns(2)
    n_min = c1.number_input("n min", min_value=4, max_value=16, value=4, step=1)
    n_max = c2.number_input("n max", min_value=4, max_value=16, value=16, step=1)
    c3, c4 = st.sidebar.columns(2)
    p_min = c3.number_input("Depth min", min_value=1, max_value=8, value=1, step=1)
    p_max = c4.number_input("Depth max", min_value=1, max_value=8, value=8, step=1)

    environment_label = st.sidebar.selectbox("Environment", _ENV_OPTIONS, index=_ENV_OPTIONS.index("Earth"))
    environment = _LABEL_TO_ENV[environment_label]
    noise_scale = st.sidebar.slider(
        "Environment noise scale", min_value=0.0, max_value=2.0, value=1.0, step=0.05,
        disabled=(environment == "ideal"),
        help="Scales each environment's representative channel probabilities. "
             "0 = idealized channels; 1.0 = its documented Earth/space setting.",
    )

    def _preset_compare() -> None:
        st.session_state["plateau_compare_envs"] = ["Earth", "LEO", "Deep Space"]

    st.sidebar.button("⚖️ Compare Earth / LEO / Deep Space", on_click=_preset_compare,
                      help="Selects three environments and overlays their critical-depth curves.")
    # Seed state before instantiating the keyed widget (avoids Streamlit's
    # default+session-state warning on the preset callback).
    if "plateau_compare_envs" not in st.session_state:
        st.session_state["plateau_compare_envs"] = []
    compare_labels = st.sidebar.multiselect(
        "Compare 2–3 environments", _ENV_OPTIONS, key="plateau_compare_envs", max_selections=3,
        help="The main heatmap stays selectable; below it, critical-depth boundaries overlay.",
    )

    n_instances = st.sidebar.slider(
        "Graph instances averaged per cell", min_value=1, max_value=5, value=2, step=1,
        help="The core default is 5. Two instances gives a first overview sooner; increase it "
             "for smoother phase boundaries.",
    )
    with st.sidebar.expander("⚙️ Landscape & graph options"):
        resolution = st.selectbox("Landscape resolution", [9, 10, 11, 12], index=1,
                                  help="10 avoids the aliasing seen at some coarse periodic grids.")
        shots = st.number_input("Display-equivalent shots", min_value=64, max_value=4096, value=256, step=64,
                                help="Exact landscapes are deterministic; this only scales returned count metadata.")
        seed_base = st.number_input("Instance seed base", min_value=0, max_value=9999, value=42, step=1)
        p_er = st.slider("Erdős–Rényi edge probability", 0.3, 0.8, 0.5, 0.05)
        m_ba = st.slider("Scale-free attachment m", 1, 3, 2, 1)
        elevation = st.slider("Satellite-link elevation (°)", 10, 90, 45, 5,
                              disabled=(environment != "satellite-ground"))

    # The primary environment is always included.  Cap the total at three,
    # honoring the requested comparison scope even if the user selects three
    # extras that omit the primary environment.
    selected_labels = list(dict.fromkeys([environment_label, *compare_labels]))[:3]
    environments = [_LABEL_TO_ENV[label] for label in selected_labels]

    try:
        sizes = diagram_sizes(family, (int(n_min), int(n_max)))
        depths = diagram_depths((int(p_min), int(p_max)))
    except ValueError as exc:
        st.sidebar.error(str(exc))
        st.info("Choose a valid n/depth interval in the sidebar.")
        return

    configs = _build_configs(
        family, int(n_min), int(n_max), int(p_min), int(p_max), environments,
        float(noise_scale), int(n_instances), int(resolution), int(shots), int(seed_base),
        float(p_er), int(m_ba), float(elevation),
    )
    estimates = {
        cache_key(cfg): estimate_barren_cost(
            cfg["family"], cfg["n_range"], cfg["p_range"], cfg["noise_model"],
            cfg["n_instances"], cfg["resolution"], cfg["execution"], cfg["exact_noise_max_n"],
        )
        for cfg in configs
    }
    cached_pairs = [try_load_cached(cfg) for cfg in configs]
    all_cached = all(data is not None for data, _ in cached_pairs)
    estimated_seconds = sum(float(estimates[cache_key(cfg)]["seconds"]) for cfg in configs)

    st.sidebar.divider()
    st.sidebar.metric("Grid × environments × instances",
                      f"{len(sizes)}×{len(depths)}×{len(environments)}×{int(n_instances)}")
    if all_cached:
        st.sidebar.success("✅ This configuration is cached — loads instantly.")
    else:
        st.sidebar.caption(f"⏱️ Estimated first compute: **{_format_duration(estimated_seconds)}**")
        if estimated_seconds > 900:
            st.sidebar.warning("Large sweep: keep this tab open while the worker computes it.")
    compute = st.sidebar.button("🚀 Compute Diagram", type="primary", width="stretch", disabled=all_cached)

    # ---------------- compute / auto-load ----------------
    current_view_key = view_key(configs)
    if compute:
        bar = st.progress(0.0, text="Preparing phase diagram…")
        try:
            with st.spinner("Computing QAOA landscape trainability in the simulator worker…"):
                results, sources = get_diagrams(configs, estimates, bar)
            st.session_state["barren_view"] = {
                "key": current_view_key, "results": results, "sources": sources,
            }
        except Exception as exc:  # noqa: BLE001
            bar.empty()
            st.error(f"Barren-plateau computation failed: {exc}")
            return
        finally:
            time.sleep(0.15)
            bar.empty()

    view = st.session_state.get("barren_view")
    if (view is None or view.get("key") != current_view_key) and all_cached:
        results = {cfg["environment"]: data for cfg, (data, _) in zip(configs, cached_pairs)}
        sources = {cfg["environment"]: str(src) for cfg, (_, src) in zip(configs, cached_pairs)}
        view = {"key": current_view_key, "results": results, "sources": sources}
        st.session_state["barren_view"] = view

    if view is None or view.get("key") != current_view_key:
        st.info("Set the environment(s) and press **🚀 Compute Diagram**. Completed diagrams are "
                "persisted and will auto-load instantly when you revisit the same configuration.")
        _render_expected(configs, estimates, sizes, depths)
        return

    results: Dict[str, Dict] = view["results"]
    sources: Dict[str, str] = view.get("sources", {})
    if any(source == "computed" for source in sources.values()):
        st.toast("Phase diagram computed and cached to data/barren_diagrams/", icon="🧊")

    # ---------------- main plot ----------------
    available = list(results)
    if len(available) > 1:
        heatmap_label = st.selectbox("View phase heatmap for environment",
                                     [_ENV_LABELS.get(e, e) for e in available], index=0)
        shown_env = available[[_ENV_LABELS.get(e, e) for e in available].index(heatmap_label)]
    else:
        shown_env = available[0]
    shown = results[shown_env]
    st.plotly_chart(plot_phase_heatmap(shown, shown_env), width="stretch",
                    key=f"plateau_heat_{current_view_key}_{shown_env}")

    # ---------------- insight panel ----------------
    summary = shown["summary"]
    counts = summary["class_counts"]
    st.subheader("📌 Trainability insights")
    metrics = st.columns(4)
    metrics[0].metric("Trainable cells", f"{counts.get('TRAINABLE', 0)} / {summary['total_cells']}")
    metrics[1].metric("Phase-trainable (≥0.5)", f"{100 * summary['phase_trainable_fraction']:.0f}%")
    metrics[2].metric("Barren cells", f"{counts.get('BARREN', 0)}")
    metrics[3].metric("Resonant islands", f"{counts.get('RESONANT', 0)}")

    if len(results) > 1:
        comparison = compare_diagrams(results)
        leo_comparison = comparison.get("comparison")
        if leo_comparison is not None:
            shrink = float(leo_comparison["shrink_percent"])
            earth_cells = comparison["environments"]["earth-lab"]["phase_trainable_cells"]
            leo_cells = comparison["environments"]["leo-satellite"]["phase_trainable_cells"]
            st.info(
                f"**LEO effect.** In LEO, the phase-trainable region shrinks by "
                f"**~{shrink:.0f}%** compared with Earth ({leo_cells} vs {earth_cells} cells at "
                "trainability ≥ 0.5). The shift comes from larger two-qubit, readout and "
                "cosmic-ray burst channels acting over more QAOA layers."
            )
        else:
            st.info("Compare Earth and LEO together to quantify how the space environment shifts "
                    "the critical-depth boundary.")
    else:
        st.info(
            f"**{_ENV_LABELS.get(shown_env, shown_env)} insight.** Small n / shallow p cells retain "
            "the most variance and gradient. Moving right or upward contracts the landscape; cells "
            "below the black score-0.5 boundary are increasingly hard for a local optimizer to navigate."
        )

    if bool(summary.get("resonance_detected", False)):
        st.success(
            "**Stochastic resonance creates trainable islands at intermediate noise.** Cyan cells have "
            "both an intermediate trainability score and direct noisy-vs-ideal evidence that noise increased "
            "landscape relief and gradient structure."
        )
    else:
        st.caption(
            "No stochastic-resonance island was detected for this deterministic channel setting. Cyan is "
            "reserved for cases where noise measurably creates additional landscape structure, rather than "
            "ordinary depolarising flattening being mislabelled as a benefit."
        )

    # ---------------- boundary comparison ----------------
    st.subheader("📉 Phase boundary")
    if len(results) >= 2:
        st.plotly_chart(plot_boundary_comparison(results), width="stretch",
                        key=f"plateau_boundary_{current_view_key}")
        st.caption("A higher critical depth means the optimizer can tolerate a deeper circuit before the "
                   "score drops through 0.5. ▲ = still trainable beyond the displayed depth range; "
                   "▼ = barren already at the smallest displayed depth.")
    else:
        st.info("Select 2–3 environments in **Compare 2–3 environments** to overlay how space noise "
                "shifts the critical-depth boundary.")

    # ---------------- detail and export ----------------
    st.subheader(f"🔎 Cell detail — {_ENV_LABELS.get(shown_env, shown_env)}")
    st.dataframe(cell_table(shown), width="stretch", hide_index=True)
    with st.expander("📥 Export & protocol metadata"):
        key = cache_key(next(cfg for cfg in configs if cfg["environment"] == shown_env))
        st.download_button(
            "⬇️ Download diagram JSON", data=json.dumps(shown, indent=2),
            file_name=f"{key}.json", mime="application/json", width="stretch",
        )
        st.caption(f"Cached on disk at `data/barren_diagrams/{key}.json`")
        st.json({
            "sizes": shown["sizes"], "depths": shown["depths"], "noise": shown["noise"],
            "protocol": shown["protocol"], "phase_boundary": shown["phase_boundary"],
        }, expanded=False)

    st.divider()
    st.caption(
        "Classification thresholds: TRAINABLE > 0.6; BARREN < 0.1; MIXED in between; RESONANT only "
        "when an intermediate structured candidate has direct noise-created-morphology evidence. "
        "The phase contour itself uses score = 0.5 to provide a stable critical-depth comparison."
    )


main()
