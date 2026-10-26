"""🗺️ Landscape View — the interactive QAOA landscape explorer.

Sidebar controls the instance, the display grid, the **noise slider** (the key
feature) and the classical optimizer.  The main area shows a two-column layout:
an interactive Plotly contour (left) and a Plotly 3-D surface plus a results
panel (right).  A NetworkX drawing of the instance lives in the sidebar.

Performance notes
-----------------
The landscape is an *exact* quantum simulation over ``resolution²`` grid points.
Ideal runs use the fast ``statevector`` method (sub-second for the whole grid);
noisy runs use ``density_matrix`` whose cost grows like ``4ⁿ``.  This page
therefore:

* caches every landscape with :func:`st.cache_data` keyed on the primitive
  parameters (graph settings + noise level), so re-visiting a setting is instant;
* probes the per-point cost once and shows a live **ETA**;
* runs live while dragging the noise slider when the ETA is small, and switches
  to an explicit **Recompute** button when a single sweep would be slow — so the
  app never silently hangs.
"""

from __future__ import annotations

import os
import sys
import threading
import time

# --- import bootstrap (works whether launched from root or pages/) ----------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import streamlit as st

from core.graph_generator import (
    GRAPH_FAMILIES,
    compute_graph_properties,
    generate_graph,
    optimal_maxcut,
)
from core.landscape import compute_landscape, detect_local_minima
from core.noise_models import (
    ENVIRONMENTS,
    REPRESENTATIVE_LEVELS,
    effective_params,
    get_environment_info,
)
from core.optimizer import run_optimization
from core.qaoa_evaluator import evaluate_qaoa_exact
from utils.export import (
    counts_to_dataframe,
    export_landscape_csv,
    export_landscape_npz,
    export_results_json,
)
from utils.plotting import (
    plot_graph,
    plot_landscape_3d,
    plot_landscape_contour,
    plot_landscape_contour_plotly,
    plot_trajectories,
)

st.set_page_config(page_title="Landscape View · Quantum Compass", page_icon="🗺️", layout="wide")

# Live-update ETA budget (seconds). Above this, require an explicit Recompute.
_LIVE_BUDGET_S = 12.0
# Optimizer ETA hard cap (seconds). Above this, refuse to run (avoid hangs).
_OPT_BUDGET_S = 120.0

# Module-global progress hook so a @st.cache_data function can drive a bar
# without the (unhashable) bar being part of the cache key.
_PROGRESS = {"cb": None}


def noise_spec(environment: str, noise_args):
    """Primitive, picklable noise descriptor for the worker.

    ``noise_args`` is a hashable tuple
    ``(noise_level, elevation, cust_p1, cust_p2, cust_readout, cust_burst)``
    so it can be part of a ``st.cache_data`` key.

    IMPORTANT: the Streamlit parent must never *construct* a qiskit object
    (NoiseModel / QuantumCircuit) — doing so segfaults Aer's native backend on
    the second script rerun.  So we pass a plain dict; the worker subprocess
    turns it into a real ``NoiseModel`` via ``get_noise_model``.  ``None`` means
    the ideal device.
    """
    noise_level, elevation, cp1, cp2, cread, cburst = noise_args
    if environment == "ideal":
        return None
    if environment == "custom":
        params = {"p1": float(cp1), "p2": float(cp2),
                  "readout": float(cread), "burst_rate": float(cburst)}
        if all(v <= 0.0 for v in (cp1, cp2, cread, cburst)):
            return None  # every channel off → ideal (fast statevector path)
        return {"environment": "custom", "params": params}
    if noise_level <= 0.0:
        return None
    params = {"noise_level": float(noise_level)}
    if environment == "satellite-ground":
        params["elevation"] = float(elevation)
    return {"environment": environment, "params": params}


def is_noisy(environment: str, noise_args) -> bool:
    """True when the spec is a real (non-ideal) noise model."""
    return noise_spec(environment, noise_args) is not None


# ---------------------------------------------------------------------------
# Cached builders
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def build_graph_cached(family, n, seed, p_er, m_ba):
    """Build (and validate) the instance graph from primitive parameters."""
    kwargs = {}
    note = ""
    eff_n = n
    if family == "3-regular":
        if n % 2 != 0:  # a 3-regular graph needs an even number of vertices
            eff_n = max(4, n - 1)
            note = f"3-regular needs an even n → using n = {eff_n}."
        kwargs = {"d": 3}
    elif family == "erdos-renyi":
        kwargs = {"p": p_er}
    else:  # scale-free
        m = min(m_ba, eff_n - 1)
        if m != m_ba:
            note = f"Barabási–Albert needs m < n → using m = {m}."
        kwargs = {"m": m}
    G = generate_graph(family, eff_n, seed=seed, **kwargs)
    return G, eff_n, note


@st.cache_data(show_spinner=False)
def cached_landscape(family, n, seed, p_er, m_ba, resolution, depth,
                     environment, noise_args, shots):
    """Exact landscape sweep. Cached on the primitive parameter tuple."""
    G, eff_n, _ = build_graph_cached(family, n, seed, p_er, m_ba)
    spec = noise_spec(environment, noise_args)
    return compute_landscape(
        G,
        depth=depth,
        resolution=resolution,
        noise_model=spec,
        shots=shots,
        readout_prob=None,  # worker resolves it from the rebuilt NoiseModel
        progress_callback=_PROGRESS["cb"],
    )


@st.cache_data(show_spinner=False)
def cached_best_point(family, n, seed, p_er, m_ba, environment, noise_args,
                      gamma, beta, depth, shots):
    """Exact single-point evaluation at the landscape optimum (for the cut view)."""
    G, eff_n, _ = build_graph_cached(family, n, seed, p_er, m_ba)
    spec = noise_spec(environment, noise_args)
    return evaluate_qaoa_exact(
        G, [gamma] * depth, [beta] * depth,
        noise_model=spec, shots=shots, readout_prob=None,
    )


def estimate_point_cost(G, noisy, environment, noise_args, resolution):
    """Probe the per-grid-point simulation cost once, cached in session_state.

    Returns seconds per point.  Ideal (statevector) is treated as cheap; noisy
    (density_matrix) is measured with a single timed evaluation because its cost
    scales like 4ⁿ and dominates the sweep time.
    """
    n = G.number_of_nodes()
    if not noisy:
        # Ideal statevector is fast and near-linear in 2**n; use a safe upper bound.
        return 0.0025 * (2 ** max(0, (n - 6))) / 8.0 + 0.0006
    key = f"probe_noisy_{n}"
    cache = st.session_state.setdefault("_probe", {})
    if key in cache:
        return cache[key]
    # Probe with a mild, always-noisy spec so the density-matrix cost is measured.
    probe_args = (max(noise_args[0], 0.003), noise_args[1], 0.004, 0.01, 0.02, 0.02)
    spec = noise_spec(environment if environment != "ideal" else "earth-lab", probe_args)
    t0 = time.perf_counter()
    evaluate_qaoa_exact(G, [0.3], [0.3], noise_model=spec, shots=256, readout_prob=None)
    dt = time.perf_counter() - t0
    cache[key] = dt
    return dt


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("🧭 Quantum Compass")
st.sidebar.caption("QAOA landscape controls")

st.sidebar.subheader("1 · Problem instance")
family = st.sidebar.selectbox("Graph family", list(GRAPH_FAMILIES.keys()), index=0,
                              help=GRAPH_FAMILIES["3-regular"])
n_req = st.sidebar.slider("Vertices n", min_value=4, max_value=12, value=6, step=1)
seed = st.sidebar.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1)

p_er = 0.5
m_ba = 2
if family == "erdos-renyi":
    p_er = st.sidebar.slider("Edge probability p", min_value=0.1, max_value=0.9,
                             value=0.5, step=0.05)
elif family == "scale-free":
    m_ba = st.sidebar.slider("Edges per new node m", min_value=1, max_value=4, value=2, step=1)

depth = st.sidebar.selectbox("QAOA depth p", [1, 2, 3], index=0,
                             help="p>1 shows a symmetric slice: the same (γ,β) on every layer.")

st.sidebar.subheader("2 · Display grid")
resolution = st.sidebar.slider("Resolution (grid points per axis)", min_value=15,
                               max_value=35, value=22, step=1)
shots = st.sidebar.select_slider("Shots (for reported counts)", options=[512, 1024, 2048, 4096, 8192],
                                 value=2048)
show_maxima = st.sidebar.checkbox("Also mark local maxima", value=False)

st.sidebar.subheader("3 · ⭐ Noise")


def _on_env_change():
    """Auto-jump the noise slider to the environment's representative level."""
    env = st.session_state.get("env_select", "earth-lab")
    st.session_state["noise_slider"] = REPRESENTATIVE_LEVELS.get(env, 0.003)


environment = st.sidebar.selectbox(
    "Environment", list(ENVIRONMENTS.keys()), index=1, key="env_select",
    on_change=_on_env_change, help=ENVIRONMENTS["earth-lab"],
)

# --- environment info card (description / citation / missions) --------------
_info = get_environment_info(environment)
with st.sidebar.expander(f"ℹ️ {_info['name']}", expanded=False):
    st.markdown(_info["description"])
    if _info.get("missions") and _info["missions"] != ["—"]:
        st.caption("🛰️ " + " · ".join(_info["missions"]))
    if _info.get("citation") and _info["citation"] != "—":
        st.caption("📖 " + _info["citation"])

# --- master noise slider (hidden for 'custom', disabled for 'ideal') --------
elevation = 45.0
c_p1 = c_p2 = c_readout = c_burst = 0.0
if environment == "ideal":
    st.sidebar.info("Ideal device — no noise. Pick a physical environment to use the slider.")
    noise_level = 0.0
elif environment == "custom":
    noise_level = 0.0
    st.sidebar.markdown("**Custom channels** (absolute probabilities)")
    c_p1 = st.sidebar.slider("p₁ (1-qubit depolarizing)", 0.0, 0.2, 0.004, 0.001, format="%.3f")
    c_p2 = st.sidebar.slider("p₂ (2-qubit depolarizing)", 0.0, 0.2, 0.010, 0.001, format="%.3f")
    c_readout = st.sidebar.slider("readout error", 0.0, 0.3, 0.020, 0.001, format="%.3f")
    c_burst = st.sidebar.slider("burst rate (cosmic-ray hits/gate)", 0.0, 0.2, 0.020, 0.001, format="%.3f")
else:
    noise_level = st.sidebar.slider(
        "🎚️ Noise level", min_value=0.0, max_value=0.1,
        value=REPRESENTATIVE_LEVELS.get(environment, 0.003), step=0.001,
        format="%.3f", key="noise_slider",
        help="Master severity — rescales this environment's base channel profile.",
    )
    if environment == "satellite-ground":
        elevation = st.sidebar.slider(
            "📡 Elevation angle (°)", min_value=10.0, max_value=90.0, value=45.0, step=1.0,
            help="Lower elevation → longer slant path → more atmospheric photon loss.",
        )

# Hashable tuple threading every noise parameter into the cache key / worker spec.
noise_args = (round(float(noise_level), 4), round(float(elevation), 2),
              round(float(c_p1), 5), round(float(c_p2), 5),
              round(float(c_readout), 5), round(float(c_burst), 5))

# --- environment-aware effective-channel readout (pure; parent-safe) --------
if environment != "ideal":
    _params = {"noise_level": noise_level, "elevation": elevation}
    if environment == "custom":
        _params = {"p1": c_p1, "p2": c_p2, "readout": c_readout, "burst_rate": c_burst}
    _ep = effective_params(environment, _params)
    if environment == "satellite-ground":
        st.sidebar.caption(
            f"photon loss η={_ep['ad_rate']:.3f} · dephasing={_ep['pd_rate']:.3f} · "
            f"n_th={_ep['n_th']:.3f} @ {_ep['elevation']:.0f}°"
        )
    else:
        _extra = f" · burst={_ep['burst_rate']:.3f}" if _ep["burst_rate"] > 0 else ""
        _therm = f" · thermal +{_ep['thermal_mod']*100:.0f}%" if _ep["thermal_mod"] > 0 else ""
        st.sidebar.caption(
            f"depolarizing p₁={_ep['p1']:.4f} · p₂={_ep['p2']:.4f} · "
            f"readout={_ep['readout']:.3f}{_extra}{_therm}"
        )

st.sidebar.subheader("4 · Optimizer")
opt_method = st.sidebar.selectbox("Method", ["COBYLA", "Nelder-Mead", "Powell", "L-BFGS-B"], index=0)
opt_restarts = st.sidebar.slider("Restarts", min_value=1, max_value=10, value=5, step=1)
opt_maxiter = st.sidebar.slider("Max iterations", min_value=20, max_value=300, value=120, step=10)
run_opt = st.sidebar.button("🧗 Run Optimizer", width="stretch")
clear_opt = st.sidebar.button("Clear optimizer result", width="stretch")

# Signature that identifies "the same problem+device" for optimizer result reuse.
config_sig = (family, n_req, seed, round(float(p_er), 3), m_ba, depth,
              environment, noise_args)

if clear_opt:
    st.session_state.pop("opt_result", None)
    st.session_state.pop("opt_sig", None)

# Invalidate a stored optimizer result when the configuration changes.
if st.session_state.get("opt_sig") != config_sig:
    st.session_state.pop("opt_result", None)
    st.session_state["opt_sig"] = config_sig

# ---------------------------------------------------------------------------
# Build the instance + device (cheap) for the sidebar viz and headers
# ---------------------------------------------------------------------------
try:
    G, eff_n, note = build_graph_cached(family, n_req, seed, p_er, m_ba)
except Exception as exc:  # invalid family/params
    st.error(f"Could not build the graph: {exc}")
    st.stop()

noisy = is_noisy(environment, noise_args)
props = compute_graph_properties(G)
opt_cut, _ = optimal_maxcut(G)

# A short, environment-aware noise label for headers/titles/filenames.
if environment == "ideal" or not noisy:
    noise_label = "ideal"
elif environment == "custom":
    noise_label = f"custom(p2={c_p2:.3f})"
elif environment == "satellite-ground":
    noise_label = f"link(η={effective_params(environment, {'noise_level': noise_level, 'elevation': elevation})['ad_rate']:.2f}@{elevation:.0f}°)"
else:
    noise_label = f"{noise_level:.3f}"

# Filename-safe slug of the noise label (strip non-alphanumerics).
import re as _re
noise_tag = _re.sub(r"[^A-Za-z0-9._-]", "", noise_label.replace("η", "eta").replace("°", "deg")) or "ideal"

# --- sidebar graph drawing -------------------------------------------------
st.sidebar.subheader("Instance")
best_bits_for_viz = None
if "opt_result" in st.session_state:
    _bp = st.session_state["opt_result"].get("_best_point")
    if _bp:
        best_bits_for_viz = _bp.get("best_bitstring")
st.sidebar.pyplot(
    plot_graph(G, best_bitstring=best_bits_for_viz,
               title=f"{family} · n={eff_n} · |E|={G.number_of_edges()}"),
    width="stretch",
)
st.sidebar.caption(
    f"C_max={opt_cut} · density={props['density']:.2f} · "
    f"clustering={props['clustering_coeff']:.2f} · "
    f"spec.gap={props['spectral_gap']:.2f} · tw≈{props['treewidth']}"
)
if note:
    st.sidebar.warning(note)

# ---------------------------------------------------------------------------
# Main header
# ---------------------------------------------------------------------------
st.title("🗺️ QAOA Optimization Landscape")
mode_badge = "🔴 noisy (density-matrix)" if noisy else "🟢 ideal (statevector)"
st.caption(
    f"**{family}** graph · n={eff_n} · |E|={G.number_of_edges()} · depth p={depth} · "
    f"environment **{environment}** · noise **{noise_label}** · {mode_badge}"
)

# ---------------------------------------------------------------------------
# ETA probe + live/button decision
# ---------------------------------------------------------------------------
n_points = resolution * resolution
per_pt = estimate_point_cost(G, noisy, environment, noise_args, resolution)
eta_s = per_pt * n_points

live = eta_s <= _LIVE_BUDGET_S
recompute_pressed = st.sidebar.button("♻️ Recompute landscape", width="stretch")

if not live:
    st.sidebar.warning(
        f"A full sweep is **{eta_s:.0f}s** ({n_points} pts × {per_pt*1000:.0f} ms). "
        f"Lower n/resolution or use ideal mode for live updates — otherwise press "
        f"**Recompute** to run it once (then it's cached)."
    )

need_compute = live or recompute_pressed or "landscape" not in st.session_state \
    or st.session_state.get("landscape_sig") != config_sig or st.session_state.get("landscape_res") != resolution or st.session_state.get("landscape_depth") != depth

# ---------------------------------------------------------------------------
# Compute the landscape (cached) with a live progress bar.
#
# The heavy work runs on a background thread that calls the @st.cache_data
# function; the main thread polls a plain (non-Streamlit) progress holder and
# updates the bar.  No st.* element is ever called *inside* the cached function,
# which avoids Streamlit's CacheReplayClosureError on cache hits while still
# giving a real progress bar on misses.
# ---------------------------------------------------------------------------
if need_compute:
    bar = st.progress(0.0, text="Preparing exact simulation…")
    holder = {"frac": 0.0, "done": False, "result": None, "err": None}

    def _cb(done, total, _h=holder):
        _h["frac"] = (done / total) if total else 1.0

    _PROGRESS["cb"] = _cb

    def _worker():
        try:
            _h = holder
            _h["result"] = cached_landscape(
                family, n_req, seed, p_er, m_ba, resolution, depth,
                environment, noise_args, shots,
            )
        except BaseException as exc:  # noqa: BLE001 - re-raised on the main thread
            holder["err"] = exc
        finally:
            holder["done"] = True

    th = threading.Thread(target=_worker, name="qc-landscape", daemon=True)
    th.start()
    while not holder["done"]:
        frac = min(1.0, max(0.0, holder["frac"]))
        bar.progress(frac, text=f"Computing landscape… {frac * 100:.0f}%")
        time.sleep(0.05)
    th.join()
    _PROGRESS["cb"] = None
    bar.empty()

    if holder["err"] is not None:
        raise holder["err"]
    landscape = holder["result"]

    st.session_state["landscape"] = landscape
    st.session_state["landscape_sig"] = config_sig
    st.session_state["landscape_res"] = resolution
    st.session_state["landscape_depth"] = depth
else:
    landscape = st.session_state.get("landscape")
    if landscape is None:  # first load in button-mode
        st.info("Press **♻️ Recompute landscape** (sidebar) to compute the first sweep.")
        st.stop()

gammas = landscape["gammas"]
betas = landscape["betas"]
ratios = landscape["ratios"]
minima = detect_local_minima(ratios, gammas, betas)

# Best point exact evaluation (for the cut-colored graph + counts panel)
best_point = cached_best_point(family, n_req, seed, p_er, m_ba, environment, noise_args,
                               landscape["best_gamma"], landscape["best_beta"], depth, shots)

# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------
if run_opt:
    spec_opt = noise_spec(environment, noise_args)
    opt_eta = per_pt * opt_maxiter * opt_restarts
    if opt_eta > _OPT_BUDGET_S:
        st.error(
            f"Optimizer would need ≈{opt_eta:.0f}s ({opt_restarts} restarts × {opt_maxiter} "
            f"evals × {per_pt*1000:.0f} ms). Reduce restarts/iterations, lower n, or use "
            f"ideal mode."
        )
    else:
        with st.status(f"Running {opt_method} · {opt_restarts} restarts…", expanded=True) as status:
            pbar = st.progress(0.0)

            def _pcb(done, total, _p=pbar):
                _p.progress(min(1.0, done / total), text=f"restart {done}/{total}")

            t0 = time.perf_counter()
            opt = run_optimization(
                G, depth=depth, noise_model=spec_opt, method=opt_method,
                max_iter=opt_maxiter, n_restarts=opt_restarts,
                record_trajectory=True, readout_prob=None, shots=shots,
                seed=seed, progress_callback=_pcb,
            )
            pbar.empty()
            status.update(label=f"Optimizer finished in {time.time()-t0:.1f}s", state="complete")
        # attach the best-point evaluation for the cut-colored graph
        opt["_best_point"] = evaluate_qaoa_exact(
            G, [opt["best_gamma"]] + [opt["best_gamma"]] * (depth - 1),
            [opt["best_beta"]] + [opt["best_beta"]] * (depth - 1),
            noise_model=spec_opt, shots=shots, readout_prob=None,
        ) if depth == 1 else best_point
        st.session_state["opt_result"] = opt

opt_result = st.session_state.get("opt_result")

# ---------------------------------------------------------------------------
# Two-column main layout
# ---------------------------------------------------------------------------
left, right = st.columns([1.05, 1.0], gap="large")

traj_for_plot = None
if opt_result and depth == 1:
    traj_for_plot = opt_result.get("trajectory")

with left:
    st.markdown("#### Contour landscape")
    fig_contour = plot_landscape_contour_plotly(
        gammas, betas, ratios,
        title=f"Approximation ratio · {environment} · noise {noise_label}",
        trajectory=traj_for_plot, show_maxima=show_maxima,
    )
    st.plotly_chart(fig_contour, width="stretch", config={"displaylogo": False})
    if opt_result and depth != 1:
        st.caption("Optimizer path overlay is shown for depth p = 1 only.")

with right:
    st.markdown("#### 3-D surface")
    fig_3d = plot_landscape_3d(gammas, betas, ratios,
                               title="Landscape surface (ratio vs γ, β)")
    st.plotly_chart(fig_3d, width="stretch", config={"displaylogo": False})

    st.markdown("#### Results")
    m1, m2, m3 = st.columns(3)
    m1.metric("Best ratio (grid)", f"{landscape['best_ratio']:.4f}")
    m2.metric("Local minima", len(minima))
    m3.metric("Optimal cut C_max", landscape["optimal_cut"])
    m4, m5, m6 = st.columns(3)
    m4.metric("γ*", f"{landscape['best_gamma']:.3f}")
    m5.metric("β*", f"{landscape['best_beta']:.3f}")
    m6.metric("⟨C⟩ at optimum", f"{best_point['expected_cut']:.3f}")

    if opt_result:
        st.markdown("**🧗 Optimizer**")
        o1, o2, o3 = st.columns(3)
        o1.metric("Best ratio (opt)", f"{opt_result['best_ratio']:.4f}")
        o2.metric("Method", opt_result["method"])
        o3.metric("Func evals", opt_result["nfev"])
        og, ob = st.columns(2)
        og.metric("γ (optimized)", f"{opt_result['best_gamma']:.4f}")
        ob.metric("β (optimized)", f"{opt_result['best_beta']:.4f}")
        delta = opt_result["best_ratio"] - landscape["best_ratio"]
        st.caption(
            f"Optimizer vs best grid point: **{delta:+.4f}** "
            f"({'finer than grid ✅' if delta >= -1e-9 else 'below grid max — try more restarts'})"
        )

# ---------------------------------------------------------------------------
# Optimizer trajectories over the landscape
# ---------------------------------------------------------------------------
if opt_result and depth == 1 and opt_result.get("all_trajectories"):
    with st.expander("🧗 Optimizer restart trajectories", expanded=False):
        st.plotly_chart(
            plot_trajectories(gammas, betas, ratios, opt_result["all_trajectories"],
                              title=f"{opt_result['method']} · {opt_result['n_restarts']} restarts"),
            width="stretch", config={"displaylogo": False},
        )

# ---------------------------------------------------------------------------
# Bottom panels: measurement, static contour, export
# ---------------------------------------------------------------------------
tab_counts, tab_static, tab_export = st.tabs(["📊 Measurement", "🖼️ Static contour", "💾 Export"])

with tab_counts:
    cc1, cc2 = st.columns([1, 1.4])
    with cc1:
        st.markdown("**Best bitstring at optimum**")
        st.code(best_point["best_bitstring"] or "—", language=None)
        st.caption(
            f"vertex order (char i = side of vertex i) · achieved cut ≈ "
            f"{best_point['expected_cut']:.2f} / {opt_cut}"
        )
        st.dataframe(counts_to_dataframe(best_point["counts"]).head(16),
                     width="stretch", hide_index=True)
    with cc2:
        st.bar_chart(counts_to_dataframe(best_point["counts"]).head(24).set_index("bitstring"))

with tab_static:
    st.markdown("**Matplotlib contour (static, downloadable)**")
    fig_mpl = plot_landscape_contour(gammas, betas, ratios,
                                     title=f"QAOA landscape · {environment} · noise {noise_label}",
                                     show_maxima=show_maxima)
    st.pyplot(fig_mpl, width="stretch")

with tab_export:
    e1, e2, e3 = st.columns(3)
    with e1:
        st.download_button(
            "⬇️ Landscape CSV",
            data=export_landscape_csv(gammas, betas, ratios),
            file_name=f"landscape_{family}_n{eff_n}_{environment}_{noise_tag}.csv",
            mime="text/csv", width="stretch",
        )
    with e2:
        st.download_button(
            "⬇️ Landscape NPZ",
            data=export_landscape_npz(gammas, betas, ratios,
                                      expected_cuts=landscape["expected_cuts"]),
            file_name=f"landscape_{family}_n{eff_n}_{environment}_{noise_tag}.npz",
            mime="application/octet-stream", width="stretch",
        )
    with e3:
        st.download_button(
            "⬇️ Results JSON",
            data=export_results_json(
                landscape=landscape, optimization=opt_result, graph_props=props,
                graph_meta={"family": family, "n": eff_n, "seed": seed,
                            "environment": environment, "noise_level": noise_level,
                            "noise_label": noise_label, "depth": depth,
                            "effective_channels": effective_params(
                                environment,
                                {"noise_level": noise_level, "elevation": elevation}
                                if environment != "custom"
                                else {"p1": c_p1, "p2": c_p2, "readout": c_readout,
                                      "burst_rate": c_burst})},
            ),
            file_name=f"results_{family}_n{eff_n}_{environment}_{noise_tag}.json",
            mime="application/json", width="stretch",
        )
    st.caption("CSV/NPZ contain the γ, β grids and the ratio surface; JSON summarises the run.")

st.divider()
st.caption(
    f"Exact evaluation · {n_points} grid points · "
    f"{'density-matrix (noisy)' if noisy else 'statevector (ideal)'} · "
    f"≈{eta_s:.1f}s per sweep on this device profile."
)
