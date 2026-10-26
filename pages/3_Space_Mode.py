"""🛰️ Space Mode — compare environments side by side.

Select 2–5 noise environments and see, for the **same** problem instance:

* **side-by-side contour landscapes** at a shared noise level,
* a **comparison line chart** of best approximation ratio vs noise level,
* a **robustness ranking** (how much of the noiseless ratio each environment
  retains), and
* an **auto-generated insights panel** that reports, per environment, whether
  stochastic resonance is present, which setting is most/least robust, how fast
  each degrades, and how far the optimal QAOA angles diverge.

All insights are computed from the actual simulated curves — nothing is
hard-coded.  As everywhere in Quantum Compass, Qiskit-Aer execution is delegated
to the worker subprocess via primitive noise specs; the parent constructs no
Qiskit object, and cached sweeps run on a background thread that feeds a plain
progress holder (no ``st.*`` inside ``@st.cache_data``).
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
from core.landscape import compute_landscape
from core.noise_models import ENVIRONMENTS, get_environment_info
from utils.analysis import (
    degradation_stats,
    detect_resonance,
    rank_environments,
)
from utils.plotting import (
    ENV_COLORS,
    plot_env_comparison,
    plot_env_retention_bars,
    plot_graph,
    plot_landscape_contour_plotly,
)

st.set_page_config(page_title="Space Mode · Quantum Compass", page_icon="🛰️", layout="wide")

_PROGRESS = {"cb": None}

# Environments offered for comparison (custom excluded: it does not sweep).
COMPARE_ENVS = [e for e in ENVIRONMENTS if e != "custom"]
DEFAULT_SELECTED = ["earth-lab", "leo-satellite", "deep-space"]

# Shared noise levels for the comparison curves (also the contour snap points).
SWEEP_LEVELS = (0.0, 0.005, 0.01, 0.02, 0.05, 0.1)


def noise_spec(environment: str, noise_level: float, elevation: float = 45.0):
    """Primitive, picklable noise descriptor for the worker (no Qiskit here)."""
    if environment == "ideal" or noise_level <= 0.0:
        return None
    params = {"noise_level": float(noise_level)}
    if environment == "satellite-ground":
        params["elevation"] = float(elevation)
    return {"environment": environment, "params": params}


# ---------------------------------------------------------------------------
# Cached builders
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def build_graph_cached(family, n, seed, p_er, m_ba):
    kwargs = {}
    eff_n = n
    if family == "3-regular":
        if n % 2 != 0:
            eff_n = max(4, n - 1)
        kwargs = {"d": 3}
    elif family == "erdos-renyi":
        kwargs = {"p": p_er}
    else:
        kwargs = {"m": min(m_ba, eff_n - 1)}
    G = generate_graph(family, eff_n, seed=seed, **kwargs)
    return G, eff_n


@st.cache_data(show_spinner=False)
def cached_landscape(family, n, seed, p_er, m_ba, resolution, depth,
                     environment, noise_level, elevation, shots):
    G, eff_n = build_graph_cached(family, n, seed, p_er, m_ba)
    spec = noise_spec(environment, noise_level, elevation)
    return compute_landscape(
        G, depth=depth, resolution=resolution, noise_model=spec,
        shots=shots, readout_prob=None, progress_callback=_PROGRESS["cb"],
    )


def run_batch_multi(jobs, gargs, resolution, depth, elevation, shots, bar):
    """Compute a landscape for every ``(environment, level)`` job with progress.

    Returns ``{(env, level): landscape}``.  Background thread + plain holder so
    no ``st.*`` runs inside the cached function.
    """
    holder = {"frac": 0.0, "done": False, "results": None, "err": None, "label": ""}
    N = max(1, len(jobs))

    def _thread():
        try:
            out = {}
            for i, (env, lv) in enumerate(jobs):
                def _cb(done, total, _i=i, _env=env, _lv=lv):
                    holder["frac"] = (_i + (done / total if total else 1.0)) / N
                    holder["label"] = f"{_env} · p={_lv:.3f}  ({_i + 1}/{N})"
                _PROGRESS["cb"] = _cb
                out[(env, lv)] = cached_landscape(*gargs, resolution, depth, env,
                                                  lv, elevation, shots)
            _PROGRESS["cb"] = None
            holder["frac"] = 1.0
            holder["results"] = out
        except BaseException as exc:  # noqa: BLE001
            holder["err"] = exc
        finally:
            holder["done"] = True

    th = threading.Thread(target=_thread, name="qc-space-sweep", daemon=True)
    th.start()
    while not holder["done"]:
        bar.progress(min(1.0, max(0.0, holder["frac"])),
                     text=holder["label"] or "Preparing comparison sweep…")
        time.sleep(0.05)
    th.join()
    _PROGRESS["cb"] = None
    bar.empty()
    if holder["err"] is not None:
        raise holder["err"]
    return holder["results"]


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("🧭 Quantum Compass")
st.sidebar.caption("Space Mode · environment comparison")

st.sidebar.subheader("1 · Problem instance (shared)")
family = st.sidebar.selectbox("Graph family", list(GRAPH_FAMILIES.keys()), index=0)
n_req = st.sidebar.slider("Vertices n", 4, 8, 6, 1,
                          help="Capped at 8 so the multi-environment sweep stays responsive.")
seed = st.sidebar.number_input("Random seed", 0, 9999, 42, 1)
p_er, m_ba = 0.5, 2
if family == "erdos-renyi":
    p_er = st.sidebar.slider("Edge probability p", 0.1, 0.9, 0.5, 0.05)
elif family == "scale-free":
    m_ba = st.sidebar.slider("Edges per new node m", 1, 4, 2, 1)
depth = st.sidebar.selectbox("QAOA depth p", [1, 2], index=0)
resolution = st.sidebar.slider("Grid resolution", 10, 24, 14, 1)

st.sidebar.subheader("2 · Environments to compare")
selected = []
for env in COMPARE_ENVS:
    if env == "ideal":
        continue  # ideal is the flat p=0 reference; shown implicitly
    on = env in DEFAULT_SELECTED
    if st.sidebar.checkbox(f"{get_environment_info(env)['name']}", value=on, key=f"cmp_{env}"):
        selected.append(env)

elevation = 45.0
if "satellite-ground" in selected:
    elevation = st.sidebar.slider("📡 Satellite-ground elevation (°)", 10.0, 90.0, 45.0, 1.0)

st.sidebar.subheader("3 · Comparison level")
cmp_level = st.sidebar.select_slider("Contour noise level", options=list(SWEEP_LEVELS), value=0.02,
                                     help="Snap level for the side-by-side contours.")

gargs = (family, n_req, seed, p_er, m_ba)
G, eff_n = build_graph_cached(*gargs)
props = compute_graph_properties(G)
opt_cut, _ = optimal_maxcut(G)
st.sidebar.subheader("Instance")
st.sidebar.pyplot(plot_graph(G, title=f"{family} · n={eff_n} · |E|={G.number_of_edges()}"),
                  width="stretch")
st.sidebar.caption(f"C_max={opt_cut} · density={props['density']:.2f} · "
                   f"spec.gap={props['spectral_gap']:.2f}")

shots = 1024

# ---------------------------------------------------------------------------
# Header + selection guard
# ---------------------------------------------------------------------------
st.title("🛰️ Space Mode — Environment Comparison")
if len(selected) < 2:
    st.warning("Select **at least 2 environments** in the sidebar to compare.")
    st.stop()
if len(selected) > 5:
    st.info("More than 5 selected — showing the first 5 to keep the sweep fast.")
    selected = selected[:5]

st.caption(
    f"Comparing **{len(selected)}** environments on the same {family} instance "
    f"(n={eff_n}, |E|={G.number_of_edges()}, depth p={depth}) across noise levels "
    f"{', '.join(f'{x:g}' for x in SWEEP_LEVELS)}."
)

# ---------------------------------------------------------------------------
# Compute the comparison sweeps (memoised per configuration)
# ---------------------------------------------------------------------------
jobs = [(env, lv) for env in selected for lv in SWEEP_LEVELS]
sweep_sig = (family, n_req, seed, round(p_er, 3), m_ba, resolution, depth,
             tuple(selected), round(elevation, 1), SWEEP_LEVELS)

if st.session_state.get("space_sig") != sweep_sig or "space_results" not in st.session_state:
    bar = st.progress(0.0, text="Preparing comparison sweep…")
    try:
        results = run_batch_multi(jobs, gargs, resolution, depth, elevation, shots, bar)
    finally:
        bar.empty()
    st.session_state["space_results"] = results
    st.session_state["space_sig"] = sweep_sig
res = st.session_state["space_results"]

# Per-environment best-ratio curves + resonance/degradation analysis.
curves = {env: [res[(env, lv)]["best_ratio"] for lv in SWEEP_LEVELS] for env in selected}
analyses = {env: detect_resonance(SWEEP_LEVELS, curves[env]) for env in selected}
degrad = {env: degradation_stats(SWEEP_LEVELS, curves[env]) for env in selected}
ranking = rank_environments(curves, SWEEP_LEVELS)

# ---------------------------------------------------------------------------
# 1 · Side-by-side contour landscapes at the comparison level
# ---------------------------------------------------------------------------
st.markdown(f"### 🗺️ Landscapes at noise level p = {cmp_level:g}")
cols = st.columns(len(selected), gap="medium")
for col, env in zip(cols, selected):
    L = res[(env, cmp_level)]
    info = get_environment_info(env)
    with col:
        st.markdown(
            f"<div style='text-align:center;color:{ENV_COLORS.get(env,'#00B4D8')};"
            f"font-weight:600;margin-bottom:2px'>{info['name']}</div>",
            unsafe_allow_html=True,
        )
        fig = plot_landscape_contour_plotly(
            L["gammas"], L["betas"], L["ratios"],
            title=f"best {L['best_ratio']:.3f}", show_maxima=False,
        )
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=34, b=10))
        st.plotly_chart(fig, width="stretch", config={"displaylogo": False})
        st.metric("best ratio", f"{L['best_ratio']:.4f}",
                  help=f"γ*={L['best_gamma']:.2f}, β*={L['best_beta']:.2f}")

st.divider()

# ---------------------------------------------------------------------------
# 2 · Comparison line chart + robustness ranking
# ---------------------------------------------------------------------------
st.markdown("### 📉 Best ratio vs noise level")
left, right = st.columns([1.35, 1.0], gap="large")
with left:
    st.pyplot(plot_env_comparison(SWEEP_LEVELS, curves,
                                  title="Approximation ratio vs noise · all environments"),
              width="stretch")
with right:
    st.pyplot(plot_env_retention_bars(ranking,
                                      title=f"Robustness @ p={SWEEP_LEVELS[-1]:g}"),
              width="stretch")

# Comparison table
st.markdown("**Comparison table** (best approximation ratio at each noise level)")
import pandas as pd

table = pd.DataFrame(
    {env: [round(v, 4) for v in curves[env]] for env in selected},
    index=[f"p={lv:g}" for lv in SWEEP_LEVELS],
).T
table["loss @ max"] = [f"{degrad[e]['pct_loss_max']:.1f}%" for e in selected]
table["resonance"] = ["✅ p*=%.3f" % analyses[e]["p_star"] if analyses[e]["resonance"]
                      else "❌ none" for e in selected]
st.dataframe(table, width="stretch")

st.divider()

# ---------------------------------------------------------------------------
# 3 · Auto-generated insights (data-driven)
# ---------------------------------------------------------------------------
st.markdown("### 💡 Insights (auto-generated)")
insights = []

# (a) resonance verdicts
resonant = [e for e in selected if analyses[e]["resonance"]]
if resonant:
    for e in resonant:
        a = analyses[e]
        insights.append(
            f"✅ **Stochastic resonance detected in {get_environment_info(e)['name']}** at "
            f"p\\* = {a['p_star']:.4f} — a small noise level *improves* the best ratio by "
            f"{a['improvement_pct']:.2f}% over the noiseless case (ΔBIC {a['delta_bic']:+.1f})."
        )
else:
    insights.append(
        "❌ **No stochastic resonance** in any selected environment — the best "
        "approximation ratio falls monotonically with noise (the peak model never "
        "beats the decay model on BIC)."
    )
mono_envs = [e for e in selected if not analyses[e]["resonance"]]
if mono_envs and resonant:
    names = ", ".join(get_environment_info(e)["name"] for e in mono_envs)
    insights.append(f"↘️ {names} show a **monotone decrease** — no resonance.")

# (b) robustness ranking
best_env, worst_env = ranking[0], ranking[-1]
insights.append(
    f"🛡️ **{get_environment_info(best_env['environment'])['name']} is the most robust**, "
    f"retaining **{best_env['retention']*100:.1f}%** of its noiseless ratio at "
    f"p={SWEEP_LEVELS[-1]:g}; **{get_environment_info(worst_env['environment'])['name']} "
    f"is the least**, retaining only {worst_env['retention']*100:.1f}%."
)

# (c) fastest degradation
fastest = max(selected, key=lambda e: degrad[e]["pct_loss_max"])
slowest = min(selected, key=lambda e: degrad[e]["pct_loss_max"])
if fastest != slowest:
    insights.append(
        f"📉 **{get_environment_info(fastest)['name']} degrades fastest** "
        f"({degrad[fastest]['pct_loss_max']:.1f}% loss at p={SWEEP_LEVELS[-1]:g}, "
        f"dr/dp≈{degrad[fastest]['slope']:.2f}) vs "
        f"{get_environment_info(slowest)['name']} ({degrad[slowest]['pct_loss_max']:.1f}%)."
    )

# (d) optimal-angle behaviour at the comparison level
angles = {e: (res[(e, cmp_level)]["best_gamma"], res[(e, cmp_level)]["best_beta"]) for e in selected}
g_spread = max(a[0] for a in angles.values()) - min(a[0] for a in angles.values())
b_spread = max(a[1] for a in angles.values()) - min(a[1] for a in angles.values())
ref_env = selected[0]
divergent = max(selected, key=lambda e: abs(angles[e][0] - angles[ref_env][0]) + abs(angles[e][1] - angles[ref_env][1]))
if cmp_level <= 0.0:
    insights.append(
        "🎯 At **p = 0** every environment collapses to the same noiseless landscape, so the "
        "optimal angles coincide — pick a non-zero comparison level to see them diverge."
    )
elif max(g_spread, b_spread) < 0.16:
    insights.append(
        f"🎯 At p={cmp_level:g} the **optimal QAOA angles are robust** across these environments "
        f"(γ\\* spans {g_spread:.2f} rad, β\\* spans {b_spread:.2f} rad): isotropic depolarising "
        f"and burst noise lower the whole surface but leave its peak in nearly the same place."
    )
else:
    insights.append(
        f"🎯 At p={cmp_level:g} the **optimal QAOA angles diverge across environments**: "
        f"γ\\* spans {g_spread:.2f} rad and β\\* spans {b_spread:.2f} rad. "
        f"{get_environment_info(divergent)['name']} sits farthest from "
        f"{get_environment_info(ref_env)['name']} "
        f"(γ\\*={angles[divergent][0]:.2f}, β\\*={angles[divergent][1]:.2f} vs "
        f"{angles[ref_env][0]:.2f}, {angles[ref_env][1]:.2f}) — an anisotropic channel "
        f"(e.g. photon loss) reshapes the landscape rather than merely flattening it."
    )

# (e) satellite-ground specific (photon loss) if present
if "satellite-ground" in selected:
    sg = degrad["satellite-ground"]
    insights.append(
        f"📡 The **satellite↔ground link** loses {sg['pct_loss_max']:.1f}% of its ratio by "
        f"p={SWEEP_LEVELS[-1]:g} at {elevation:.0f}° elevation — photon loss (amplitude "
        f"damping) dominates, pulling probability toward the uncut |0…0⟩ state."
    )

for line in insights:
    st.markdown(f"- {line}")

# ---------------------------------------------------------------------------
# Export the comparison curves
# ---------------------------------------------------------------------------
st.markdown("### 💾 Export")
# Build a tidy CSV of the comparison curves (columns = environments).
import io as _io

_df = pd.DataFrame(curves, index=[f"{lv}" for lv in SWEEP_LEVELS])
_buf = _io.StringIO()
_df.index.name = "noise_level"
_df.to_csv(_buf)
st.download_button(
    "⬇️ Comparison curves (CSV)", data=_buf.getvalue(),
    file_name=f"space_mode_{family}_n{eff_n}_{'-'.join(selected)}.csv",
    mime="text/csv", width="stretch",
)
st.caption("Columns = environments, rows = noise levels, values = best approximation ratio.")

st.divider()
st.caption(
    f"Exact density-matrix evaluation via the isolated simulation worker · "
    f"{len(jobs)} landscapes ({len(selected)} environments × {len(SWEEP_LEVELS)} levels) · "
    f"insights computed live from the simulated curves."
)
