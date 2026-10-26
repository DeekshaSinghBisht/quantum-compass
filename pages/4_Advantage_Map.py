"""Quantum Advantage Boundary — where does QAOA actually beat a classical solver?

This page answers the question the whole app builds toward: *for which problem
sizes and noise levels does QAOA's approximation ratio exceed a classical
baseline?*  It renders a 2-D **advantage map** — problem size ``n`` on the X axis,
noise level ``p`` on the Y axis — coloured by ``QAOA_ratio − classical_ratio``
(green = quantum ahead, red = classical ahead), with the **boundary** where they
are equal drawn on top.

Method (see :mod:`core.advantage_map`): QAOA angles are tuned once on the *ideal*
device (fast statevector optimisation), then evaluated at those fixed angles under
each environment's noise — one cheap noisy evaluation per grid point.  This is the
realistic "tune on a simulator, run on hardware" protocol and is accurate for the
isotropic space environments (their optimal angle is noise-robust).

Feasibility: a noisy density-matrix evaluation costs ``O(4ⁿ)``, so problem size is
capped at n = 12 here (n = 16 would need ~68 GB).  Results are cached to
``data/advantage_maps/*.json`` so a configuration computed once reloads instantly.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from core.advantage_map import (  # noqa: E402
    ENV_LABELS, MAP_ENVIRONMENTS, MEMORY_MAX_N, compute_advantage_map,
    estimate_map_cost, valid_sizes,
)

MAP_DIR = _project_root / "data" / "advantage_maps"
MAP_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="Quantum Advantage Boundary · Quantum Compass",
                   page_icon="🗺️", layout="wide")

# label -> environment key
_LABEL_TO_ENV = {ENV_LABELS[e]: e for e in MAP_ENVIRONMENTS}
_ENV_OPTIONS = [ENV_LABELS[e] for e in MAP_ENVIRONMENTS]

# baseline selector labels
_BASELINE_CHOICES = {
    "GW (Goemans–Williamson guarantee)": "GW",
    "Simulated Annealing": "annealing",
    "Best of all classical": "best",
    "Random sampling": "random",
}


# --------------------------------------------------------------------------- #
# caching: session_state -> durable JSON on disk -> recompute (with progress)
# --------------------------------------------------------------------------- #
def cache_key(cfg: Dict) -> str:
    """Deterministic, filename-safe key for a map configuration."""
    canon = json.dumps(cfg, sort_keys=True, default=str)
    h = hashlib.sha1(canon.encode("utf-8")).hexdigest()[:12]
    fam = re.sub(r"[^A-Za-z0-9]+", "", cfg["family"])[:12]
    envs = "-".join(re.sub(r"[^A-Za-z0-9]", "", e)[:4] for e in cfg["env_labels"])
    raw = f"{fam}_p{cfg['depth']}_{cfg['baseline']}_{envs}_n{cfg['n_min']}-{cfg['n_max']}_{h}"
    return re.sub(r"[^A-Za-z0-9._-]", "_", raw)[:120]


@st.cache_data(show_spinner=False)
def read_map_json(path_str: str) -> Optional[Dict]:
    """Cached read of a persisted map JSON (only ever called when it exists)."""
    try:
        return json.loads(Path(path_str).read_text(encoding="utf-8"))
    except Exception:
        return None


def try_load_cached(cfg: Dict) -> Optional[Dict]:
    """Return a previously-computed map for ``cfg`` from session or disk (no compute)."""
    key = cache_key(cfg)
    store = st.session_state.setdefault("adv_maps", {})
    if key in store:
        return store[key]
    path = MAP_DIR / f"{key}.json"
    if path.exists():
        data = read_map_json(str(path))
        if data is not None:
            store[key] = data
            return data
    return None


def _store(key: str, data: Dict) -> None:
    st.session_state.setdefault("adv_maps", {})[key] = data


def _compute_with_progress(cfg: Dict, bar) -> Dict:
    """Run compute_advantage_map in a worker thread; animate the progress bar."""
    holder: Dict = {"res": None, "done": 0, "total": 1}
    done_evt = threading.Event()
    exc: List[Optional[BaseException]] = [None]

    def cb(d: int, t: int):
        holder["done"] = d
        holder["total"] = t

    def work():
        try:
            holder["res"] = compute_advantage_map(
                family=cfg["family"], sizes=cfg["sizes"], depth=cfg["depth"],
                envs=cfg["envs"], baseline=cfg["baseline"], noise_levels=cfg["noise_levels"],
                n_instances=cfg["n_instances"], seed_base=cfg["seed_base"],
                p_er=cfg["p_er"], m_ba=cfg["m_ba"], opt_restarts=cfg["opt_restarts"],
                opt_max_iter=cfg["opt_max_iter"], shots=cfg["shots"], progress_callback=cb,
            )
        except BaseException as e:  # noqa: BLE001 - re-raised on the main thread
            exc[0] = e
        finally:
            done_evt.set()

    th = threading.Thread(target=work, daemon=True)
    th.start()
    while not done_evt.is_set():
        d, t = holder.get("done", 0), max(1, holder.get("total", 1))
        frac = float(d) / float(t)
        bar.progress(min(1.0, frac), text=f"Computing advantage map — {d}/{t} evaluations "
                                         f"({100*frac:.0f}%)")
        time.sleep(0.2)
    if exc[0] is not None:
        raise exc[0]
    bar.progress(1.0, text="Advantage map complete")
    return holder["res"]


def get_map(cfg: Dict, bar) -> tuple:
    """Cached-or-computed map + its source ('session' | 'disk' | 'computed')."""
    key = cache_key(cfg)
    cached = try_load_cached(cfg)
    if cached is not None:
        src = "session" if key in st.session_state.get("adv_maps", {}) else "disk"
        bar.progress(1.0, text="Loaded from cache")
        return cached, src
    data = _compute_with_progress(cfg, bar)
    _store(key, data)
    try:
        (MAP_DIR / f"{key}.json").write_text(json.dumps(data), encoding="utf-8")
    except Exception:
        pass
    return data, "computed"


# --------------------------------------------------------------------------- #
# plotting
# --------------------------------------------------------------------------- #
def plot_advantage_heatmap(result: Dict, env: str) -> go.Figure:
    """Diverging heatmap of (QAOA − classical) with the zero-crossing boundary."""
    sizes = result["sizes"]
    noise = result["noise_levels"]
    nP = len(noise)
    D = np.asarray(result["difference"][env], dtype=float)   # (n_sizes, nP)
    z = D.T                                                   # (nP, n_sizes): rows=noise, cols=n
    yidx = list(range(nP))

    absmax = float(np.max(np.abs(z))) if z.size else 1.0
    absmax = max(absmax, 1e-3)

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=z, x=[int(n) for n in sizes], y=yidx,
        colorscale="RdYlGn", zmid=0.0, zmin=-absmax, zmax=absmax,
        colorbar=dict(title="QAOA −<br>classical", thickness=14, len=0.8),
        hovertemplate=("n=%{x}<br>p=%{customdata}<br>"
                       "Δ (QAOA−classical)=%{z:.4f}<extra></extra>"),
        customdata=np.array([[f"{p:.4g}" for _ in sizes] for p in noise]),
        showscale=True,
    ))

    # boundary overlay: interpolate each size's crossing noise into index space
    bnd = result["boundary"][env]
    bx, by, btxt = [], [], []
    for n in sizes:
        b = bnd.get(str(n))
        if b is None:                       # never ahead
            continue
        if b == float("inf") or (isinstance(b, float) and np.isinf(b)):
            by.append(nP - 0.5); bx.append(int(n)); btxt.append("whole range")
            continue
        yi = float(np.interp(b, noise, np.arange(nP)))
        bx.append(int(n)); by.append(yi); btxt.append(f"p≈{b:.4g}")
    if bx:
        order = np.argsort(bx)
        bx = [bx[i] for i in order]; by = [by[i] for i in order]; btxt = [btxt[i] for i in order]
        fig.add_trace(go.Scatter(
            x=bx, y=by, mode="lines+markers+text", text=btxt,
            textposition="top center", textfont=dict(size=9, color="#111"),
            line=dict(color="#111111", width=3, shape="spline", smoothing=0.6),
            marker=dict(size=8, color="#111111", line=dict(color="white", width=1)),
            name="QAOA = classical boundary",
            hovertemplate="n=%{x}<br>boundary %{text}<extra></extra>",
        ))

    fig.update_layout(
        title=dict(text=f"Quantum advantage map — {ENV_LABELS.get(env, env)} · "
                        f"depth {result['depth']} · vs {result['baseline']}", x=0.5),
        height=520, margin=dict(l=50, r=20, t=60, b=40),
        xaxis=dict(title="problem size  n  (qubits)", dtick=1, side="bottom"),
        yaxis=dict(title="noise level  p", tickvals=yidx,
                   ticktext=[f"{p:.4g}" for p in noise]),
        plot_bgcolor="#fcfcfc",
    )
    fig.add_annotation(x=sizes[0], y=nP - 0.4, text="▲ green: quantum ahead",
                       showarrow=False, font=dict(size=11, color="#0a7d32"), xanchor="left")
    fig.add_annotation(x=sizes[-1], y=-0.4, text="▼ red: classical ahead",
                       showarrow=False, font=dict(size=11, color="#a01b1b"), xanchor="right")
    return fig


def plot_boundary_comparison(result: Dict, envs: List[str]) -> go.Figure:
    """Overlay the QAOA=classical boundary curves for several environments."""
    sizes = result["sizes"]
    noise = result["noise_levels"]
    nP = len(noise)
    palette = {"earth-lab": "#2e7d32", "leo-satellite": "#1565c0",
               "meo-van-allen": "#ef6c00", "deep-space": "#6a1b9a",
               "satellite-ground": "#c62828", "ideal": "#37474f"}
    fig = go.Figure()
    for e in envs:
        bnd = result["boundary"][e]
        bx, by = [], []
        for n in sizes:
            b = bnd.get(str(n))
            if b is None:
                continue
            yi = (nP - 0.4) if (b == float("inf") or (isinstance(b, float) and np.isinf(b))) \
                else float(np.interp(b, noise, np.arange(nP)))
            bx.append(int(n)); by.append(yi)
        if bx:
            fig.add_trace(go.Scatter(
                x=bx, y=by, mode="lines+markers", name=ENV_LABELS.get(e, e),
                line=dict(color=palette.get(e, "#333"), width=3, shape="spline", smoothing=0.6),
                marker=dict(size=8),
                hovertemplate="n=%{x}<br>boundary p=%{y:.2f} (index)<extra>" + ENV_LABELS.get(e, e) + "</extra>",
            ))
    fig.update_layout(
        title=dict(text=f"Advantage boundary by environment — depth {result['depth']}, "
                        f"vs {result['baseline']} (below each curve = quantum ahead)", x=0.5),
        height=460, margin=dict(l=50, r=20, t=60, b=40),
        xaxis=dict(title="problem size  n  (qubits)", dtick=1),
        yaxis=dict(title="noise level  p", tickvals=list(range(nP)),
                   ticktext=[f"{p:.4g}" for p in noise]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center"),
        plot_bgcolor="#fcfcfc",
    )
    fig.add_hrect(y0=-0.5, y1=nP - 0.5, fillcolor="white", opacity=0.0, line_width=0)
    return fig


def per_size_table(result: Dict, env: str) -> pd.DataFrame:
    sizes = result["sizes"]
    noise = result["noise_levels"]
    bnd = result["boundary"][env]
    rows = []
    for n in sizes:
        b = bnd.get(str(n))
        if b is None:
            btxt = "— (never ahead)"
        elif b == float("inf") or (isinstance(b, float) and np.isinf(b)):
            btxt = f"whole range (≤ {max(noise):g})"
        else:
            btxt = f"p ≈ {b:.4g}"
        rows.append({
            "n": int(n),
            "ideal QAOA ratio": round(result["ideal_qaoa"][str(n)], 4),
            f"classical ({result['baseline']})": round(result["classical"][str(n)], 4),
            "QAOA @ lowest p": round(result["qaoa"][env][str(n)][0], 4),
            "advantage boundary": btxt,
            "quantum ahead @ low noise?": "✓" if result["qaoa"][env][str(n)][0] > result["classical"][str(n)] else "✗",
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# page
# --------------------------------------------------------------------------- #
def main() -> None:
    st.title("🗺️ Quantum Advantage Boundary")
    st.caption(
        "For which problem sizes and noise levels does **QAOA beat a classical "
        "baseline**? Green cells = quantum ahead, red = classical ahead; the black "
        "curve is the boundary where they are equal."
    )
    st.markdown(
        "> **Protocol.** QAOA angles are tuned once on the *ideal* device, then run "
        "at those fixed angles under each environment's noise (the realistic "
        "*tune-on-simulator, run-on-hardware* workflow). Noisy simulation is "
        f"`O(4ⁿ)`, so **n is capped at {MEMORY_MAX_N}** here (n=16 would need ~68 GB). "
        "Maps are cached to `data/advantage_maps/` and reload instantly."
    )

    # ---------------- sidebar configuration ----------------
    st.sidebar.header("🧭 Map configuration")
    family = st.sidebar.selectbox("Graph family", ["3-regular", "erdos-renyi", "scale-free"], index=0)
    c1, c2 = st.sidebar.columns(2)
    n_min = c1.number_input("n min", 4, MEMORY_MAX_N, 4, 1)
    n_max = c2.number_input("n max", 4, MEMORY_MAX_N, 10, 1)
    depth = st.sidebar.selectbox("QAOA depth (p)", [1, 2, 3], index=1)
    baseline_label = st.sidebar.selectbox("Classical baseline", list(_BASELINE_CHOICES.keys()), index=0)
    baseline = _BASELINE_CHOICES[baseline_label]

    def _preset_compare():
        st.session_state["map_envs"] = ["Earth", "LEO", "Deep Space"]
    st.sidebar.button("⚖️ Compare Earth / LEO / Deep Space", on_click=_preset_compare,
                      help="Select three environments and overlay their advantage boundaries.")
    # Seed the keyed widget's state once, before instantiation, so the preset
    # callback can drive it without a "set after instantiation" warning.
    if "map_envs" not in st.session_state:
        st.session_state["map_envs"] = ["Earth"]
    env_labels = st.sidebar.multiselect("Environment(s)", _ENV_OPTIONS, key="map_envs")
    if not env_labels:               # nothing selected -> fall back locally (no state write)
        env_labels = ["Earth"]

    noise_count = st.sidebar.slider("Noise levels (log-spaced 0.001→0.1)", 4, 12, 8)
    n_instances = st.sidebar.slider("Graph instances averaged per cell", 1, 6, 3,
                                    help="The 50-instance ideal is infeasible in this "
                                         "sandbox (hours); 3 gives a smooth map in ~1 min.")
    with st.sidebar.expander("⚙️ Optimiser & family options"):
        opt_restarts = st.number_input("Reference-angle restarts", 1, 6, 3, 1)
        opt_max_iter = st.number_input("Reference-angle iterations", 20, 200, 80, 10)
        shots = st.number_input("Shots per evaluation", 64, 8192, 256, 64)
        p_er = st.slider("Erdős–Rényi p", 0.3, 0.8, 0.5, 0.05)
        m_ba = st.slider("Barabási–Albert m", 1, 3, 2, 1)
        seed_base = st.number_input("Instance seed base", 0, 9999, 42, 1)

    noise_levels = list(np.geomspace(0.001, 0.1, int(noise_count)))
    sizes = valid_sizes(family, int(n_min), int(n_max))
    envs = [_LABEL_TO_ENV[l] for l in env_labels]

    cfg = dict(family=family, sizes=sizes, n_min=int(n_min), n_max=int(n_max),
               depth=int(depth), envs=envs, env_labels=env_labels,
               baseline=baseline, noise_levels=[float(x) for x in noise_levels],
               n_instances=int(n_instances), seed_base=int(seed_base), p_er=float(p_er),
               m_ba=int(m_ba), opt_restarts=int(opt_restarts), opt_max_iter=int(opt_max_iter),
               shots=int(shots))

    if not sizes:
        st.warning("No valid problem sizes for this family/range (3-regular needs even n).")
        return

    eta = estimate_map_cost(sizes, int(depth), envs, noise_levels, int(n_instances),
                            int(opt_restarts), int(opt_max_iter))
    cached = try_load_cached(cfg)
    st.sidebar.divider()
    st.sidebar.metric("Sizes × noise × envs × instances",
                      f"{len(sizes)}×{len(noise_levels)}×{len(envs)}×{n_instances}")
    if cached is not None:
        st.sidebar.success("✅ This configuration is cached — loads instantly.")
        st.sidebar.caption(f"Est. recompute if cleared: ~{eta:.0f}s")
    else:
        st.sidebar.caption(f"⏱️ Estimated compute: **~{eta:.0f}s**"
                           + ("  ⚠️ large" if eta > 300 else ""))

    compute = st.sidebar.button("🚀 Compute Map", type="primary", width="stretch",
                                disabled=bool(cached))

    # ---------------- compute / load ----------------
    if compute:
        bar = st.progress(0.0, text="Preparing advantage map…")
        try:
            with st.spinner("Computing the advantage map (delegated to the simulator worker)…"):
                result, src = get_map(cfg, bar)
            st.session_state["adv_view"] = {"key": cache_key(cfg), "result": result, "src": src,
                                            "env": envs[0]}
        except Exception as e:  # noqa: BLE001
            bar.empty()
            st.error(f"Advantage-map computation failed: {e}")
            return
        finally:
            time.sleep(0.2)
            bar.empty()

    view = st.session_state.get("adv_view")
    key = cache_key(cfg)
    # auto-load a cached result for the current config even without pressing the button
    if (view is None or view.get("key") != key) and cached is not None:
        view = {"key": key, "result": cached, "src": "cache", "env": envs[0]}
        st.session_state["adv_view"] = view

    if view is None or view.get("key") != key:
        st.info("Configure the map on the left and press **🚀 Compute Map**. "
                "Previously-computed configurations reload instantly from cache.")
        _render_expected(cfg, eta)
        return

    result = view["result"]
    src = view.get("src", "")
    if src == "computed":
        st.toast("Advantage map computed and cached to data/advantage_maps/", icon="🗺️")

    # which environment's heatmap to view
    view_envs = result["envs"]
    if len(view_envs) > 1:
        cur = st.selectbox("View heatmap for environment",
                           [ENV_LABELS[e] for e in view_envs], index=0)
        env = view_envs[[ENV_LABELS[e] for e in view_envs].index(cur)]
    else:
        env = view_envs[0]

    st.plotly_chart(plot_advantage_heatmap(result, env), width="stretch",
                    key=f"advmap_{key}_{env}")

    # ---------------- insights ----------------
    adv = result["advantage"][env]
    st.subheader("📌 Advantage window")
    if adv["any_advantage"]:
        st.success(adv["text"])
    else:
        st.error(adv["text"])

    d = int(result["depth"])
    ideal_str = ", ".join(f"n={n}: {result['ideal_qaoa'][str(n)]:.3f}" for n in result["sizes"])
    st.info(
        f"**Depth insight.** At QAOA depth **p={d}**, the *ideal* (noiseless) approximation "
        f"ratios are {ideal_str}. The classical **{result['baseline']}** baseline sits at "
        f"{result['classical'][str(result['sizes'][0])]:.3f}. Quantum advantage appears where "
        f"the ideal ratio clears that baseline *and* the noise is low enough that the "
        f"noisy ratio stays above it — deeper circuits push the advantage to larger n, while "
        f"more noise pulls the boundary back to smaller n and lower p."
    )

    # ---------------- environment comparison ----------------
    if len(view_envs) > 1:
        st.subheader("⚖️ Compare across environments")
        st.plotly_chart(plot_boundary_comparison(result, view_envs), width="stretch",
                        key=f"advcmp_{key}")
        # ordering of boundary heights at the smallest advantaged size
        rows = []
        for e in view_envs:
            fin = [b for b in result["boundary"][e].values()
                   if b is not None and b != float("inf") and np.isfinite(b)]
            rows.append({"environment": ENV_LABELS.get(e, e),
                         "tightest boundary p": (round(min(fin), 4) if fin else None),
                         "sizes with any advantage":
                             sum(1 for b in result["boundary"][e].values() if b is not None)})
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.caption("A **higher** boundary p means the environment tolerates more noise before "
                   "classical takes over — Earth ≳ LEO ≳ MEO ≳ Deep Space, since deep space "
                   "carries the heaviest depolarising + burst load.")

    # ---------------- per-size detail + export ----------------
    st.subheader(f"🔎 Per-size detail — {ENV_LABELS.get(env, env)}")
    st.dataframe(per_size_table(result, env), width="stretch", hide_index=True)

    with st.expander("📥 Export & raw grid"):
        payload = json.dumps(result, indent=2)
        st.download_button("⬇️ Download map JSON", data=payload,
                           file_name=f"{key}.json", mime="application/json",
                           width="stretch")
        st.caption(f"Cached on disk at `data/advantage_maps/{key}.json`")
        st.json({"sizes": result["sizes"], "noise_levels": result["noise_levels"],
                 "depth": result["depth"], "baseline": result["baseline"],
                 "envs": result["envs"], "difference[env0]": result["difference"][result["envs"][0]]},
                expanded=False)

    st.divider()
    st.caption(
        "Baselines: Goemans–Williamson worst-case guarantee (0.8746, a constant reference "
        "line), simulated annealing, and random sampling (see core/classical_baselines.py). "
        "Note that a good classical heuristic (annealing) solves these small instances "
        "*optimally* (ratio ≈ 1), so against it QAOA rarely shows an advantage — the "
        "meaningful quantum-vs-classical boundary here is against the GW guarantee."
    )


def _render_expected(cfg: Dict, eta: float) -> None:
    """Show what the map will contain before computing (manages expectations)."""
    with st.expander("ℹ️ What this will compute", expanded=True):
        st.markdown(
            f"- **Family:** {cfg['family']}  ·  **sizes:** {cfg['sizes']}  ·  **depth:** {cfg['depth']}\n"
            f"- **Baseline:** {cfg['baseline']}  ·  **environments:** {cfg['env_labels']}\n"
            f"- **Noise levels:** {len(cfg['noise_levels'])} (log-spaced 0.001→0.1)  ·  "
            f"**instances/cell:** {cfg['n_instances']}\n"
            f"- **Estimated compute:** ~{eta:.0f}s  ·  cached to `data/advantage_maps/` afterwards\n\n"
            "Expect a green wedge at **small n / low noise** (quantum ahead of the GW "
            "guarantee) that narrows as noise rises, with the black boundary marking where "
            "QAOA = classical."
        )


main()
