"""📈 Noise Analysis — resonance detection & noise sweeps.

This page answers one scientific question: **does a little noise ever *help*
QAOA?**  It sweeps the master noise level for a chosen environment and:

* **animates** the landscape morphing from ``p = 0`` to ``p = 0.1`` (▶️ Play),
* draws **β-cross-sections** at the optimal γ for several noise levels overlaid,
* runs a **Resonance Detector**: computes the best approximation ratio at nine
  noise levels, fits a *monotone-decay* model against a *peak* model, compares
  them with AIC/BIC, and reports whether stochastic resonance is present,
* exports the whole analysis as a **PDF** report.

Architecture (identical to the Landscape View): all Qiskit-Aer execution is
delegated to the worker subprocess via primitive noise *specs*; the parent never
constructs a Qiskit object.  Landscapes are ``@st.cache_data``-cached and
computed on a background thread that writes progress to a plain holder, while
the main thread drives the ``st.progress`` bar (no ``st.*`` inside a cached
function → no ``CacheReplayClosureError``).
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
from core.noise_models import (
    ENVIRONMENTS,
    REPRESENTATIVE_LEVELS,
    effective_params,
    get_environment_info,
)
from utils.analysis import (
    RESONANCE_LEVELS,
    degradation_stats,
    detect_resonance,
    slice_at_gamma,
)
from utils.export import export_analysis_pdf
from utils.plotting import (
    plot_cross_sections,
    plot_graph,
    plot_landscape_contour,
    plot_landscape_contour_plotly,
    plot_noise_sweep,
)

st.set_page_config(page_title="Noise Analysis · Quantum Compass", page_icon="📈", layout="wide")

# Module-global progress hook (see Landscape View for the rationale).
_PROGRESS = {"cb": None}

# Environments that respond to the master noise_level sweep. 'custom' uses
# absolute channel values (noise_level is ignored there), so it is excluded.
SWEEP_ENVS = [e for e in ENVIRONMENTS if e != "custom"]

# Noise levels used for the morphing animation and the β-cross-sections.
ANIM_LEVELS = tuple(np.round(np.linspace(0.0, 0.1, 17), 4))
CROSS_LEVELS = (0.0, 0.005, 0.01, 0.02, 0.05, 0.1)


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
    """One exact landscape sweep (cached on the primitive parameter tuple)."""
    G, eff_n = build_graph_cached(family, n, seed, p_er, m_ba)
    spec = noise_spec(environment, noise_level, elevation)
    return compute_landscape(
        G, depth=depth, resolution=resolution, noise_model=spec,
        shots=shots, readout_prob=None, progress_callback=_PROGRESS["cb"],
    )


def run_batch(levels, gargs, resolution, depth, environment, elevation, shots, bar):
    """Compute a landscape for every level with a smooth, thread-driven bar.

    Returns ``{level: landscape}``.  The heavy loop runs on a background thread;
    the cached function's progress callback writes only to a plain holder, and
    this (main) thread polls it to update ``bar`` — so no ``st.*`` call ever
    happens inside ``@st.cache_data``.
    """
    holder = {"frac": 0.0, "done": False, "results": None, "err": None, "label": ""}
    N = len(levels)

    def _thread():
        try:
            out = {}
            for i, lv in enumerate(levels):
                def _cb(done, total, _i=i, _lv=lv):
                    holder["frac"] = (_i + (done / total if total else 1.0)) / N
                    holder["label"] = f"noise p={_lv:.3f}  ({_i + 1}/{N})"
                _PROGRESS["cb"] = _cb
                out[lv] = cached_landscape(*gargs, resolution, depth, environment,
                                           lv, elevation, shots)
            _PROGRESS["cb"] = None
            holder["frac"] = 1.0
            holder["results"] = out
        except BaseException as exc:  # noqa: BLE001
            holder["err"] = exc
        finally:
            holder["done"] = True

    th = threading.Thread(target=_thread, name="qc-sweep", daemon=True)
    th.start()
    while not holder["done"]:
        bar.progress(min(1.0, max(0.0, holder["frac"])),
                     text=holder["label"] or "Preparing exact simulation…")
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
st.sidebar.caption("Noise analysis controls")

st.sidebar.subheader("1 · Problem instance")
family = st.sidebar.selectbox("Graph family", list(GRAPH_FAMILIES.keys()), index=0)
n_req = st.sidebar.slider("Vertices n", 4, 10, 6, 1,
                          help="Kept ≤ 10 here so the multi-level sweep stays fast.")
seed = st.sidebar.number_input("Random seed", 0, 9999, 42, 1)
p_er, m_ba = 0.5, 2
if family == "erdos-renyi":
    p_er = st.sidebar.slider("Edge probability p", 0.1, 0.9, 0.5, 0.05)
elif family == "scale-free":
    m_ba = st.sidebar.slider("Edges per new node m", 1, 4, 2, 1)
depth = st.sidebar.selectbox("QAOA depth p", [1, 2], index=0)

st.sidebar.subheader("2 · Sweep settings")
environment = st.sidebar.selectbox("Environment", SWEEP_ENVS, index=2,
                                   help="leo-satellite by default — a space setting.")
elevation = 45.0
if environment == "satellite-ground":
    elevation = st.sidebar.slider("📡 Elevation angle (°)", 10.0, 90.0, 45.0, 1.0)
resolution = st.sidebar.slider("Grid resolution", 12, 28, 16, 1,
                               help="Coarser grids keep the 9-level sweep snappy.")
n_frames = st.sidebar.slider("Animation frames", 9, 17, 13, 2)

_info = get_environment_info(environment)
with st.sidebar.expander(f"ℹ️ {_info['name']}", expanded=False):
    st.markdown(_info["description"])
    if _info.get("citation", "—") != "—":
        st.caption("📖 " + _info["citation"])

gargs = (family, n_req, seed, p_er, m_ba)
G, eff_n = build_graph_cached(*gargs)
props = compute_graph_properties(G)
opt_cut, _ = optimal_maxcut(G)

st.sidebar.subheader("Instance")
st.sidebar.pyplot(plot_graph(G, title=f"{family} · n={eff_n} · |E|={G.number_of_edges()}"),
                  width="stretch")
st.sidebar.caption(f"C_max={opt_cut} · density={props['density']:.2f} · "
                   f"spec.gap={props['spectral_gap']:.2f}")

shots = 1024  # only scales reported counts; best_ratio is exact & shots-independent

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("📈 Noise Analysis & Resonance Detection")
st.caption(
    f"**{environment}** · {family} graph · n={eff_n} · depth p={depth} · "
    f"resolution {resolution} · sweeping p ∈ [0, 0.1]"
)

sweep_sig = (family, n_req, seed, round(p_er, 3), m_ba, resolution, depth,
             environment, round(elevation, 1), RESONANCE_LEVELS)

# ---------------------------------------------------------------------------
# Resonance sweep (memoised; computed once per configuration)
# ---------------------------------------------------------------------------
if st.session_state.get("sweep_sig") != sweep_sig or "sweep_results" not in st.session_state:
    bar = st.progress(0.0, text="Preparing resonance sweep…")
    try:
        results = run_batch(list(RESONANCE_LEVELS), gargs, resolution, depth,
                            environment, elevation, shots, bar)
    finally:
        bar.empty()
    st.session_state["sweep_results"] = results
    st.session_state["sweep_sig"] = sweep_sig
sweep = st.session_state["sweep_results"]

best_ratios = np.array([sweep[lv]["best_ratio"] for lv in RESONANCE_LEVELS], dtype=float)
result = detect_resonance(RESONANCE_LEVELS, best_ratios)
deg = degradation_stats(RESONANCE_LEVELS, best_ratios)

# ---------------------------------------------------------------------------
# 1 · Resonance Detector
# ---------------------------------------------------------------------------
st.markdown("### 🎯 Resonance Detector")
if result["resonance"]:
    st.success(
        f"✅ **Stochastic resonance detected** at **p\\* = {result['p_star']:.4f}**, "
        f"improvement **{result['improvement_pct']:.2f}%** over the noiseless ratio."
    )
else:
    st.info(
        f"**No resonance observed** — noise monotonically degrades performance in this sweep. "
        f"({result['reason']})"
    )

c1, c2, c3, c4 = st.columns(4)
c1.metric("Best ratio (p=0)", f"{best_ratios[0]:.4f}")
c2.metric("Best ratio (p=0.1)", f"{best_ratios[-1]:.4f}",
          delta=f"{deg['pct_loss_max'] * -1:.1f}%", delta_color="inverse")
c3.metric("Winning model", result["winner"].capitalize())
c4.metric("ΔBIC (mono − peak)", f"{result['delta_bic']:+.1f}")

st.pyplot(
    plot_noise_sweep(RESONANCE_LEVELS, best_ratios, result=result,
                     title=f"Best approximation ratio vs noise · {environment}",
                     env_name=_info["name"]),
    width="stretch",
)

with st.expander("🔬 Model-comparison details (AIC / BIC)"):
    m1, m2 = st.columns(2)
    mono, peak = result["monotone"], result["peak"]
    m1.markdown("**Monotone decay** `r = C + A·e^(−B·p)`")
    m1.write({"A": round(mono["A"], 5), "B": round(mono["B"], 3), "C": round(mono["C"], 5),
              "RSS": f"{mono['rss']:.3e}", "AIC": round(mono["aic"], 1),
              "BIC": round(mono["bic"], 1), "params": mono["n_params"]})
    m2.markdown("**Peak (Gaussian)** `r = C + A·e^(−(p−p*)²/2σ²)`")
    m2.write({"A": round(peak["A"], 5), "p*": round(peak["p_star"], 5),
              "σ": round(peak["sigma"], 5), "C": round(peak["C"], 5),
              "RSS": f"{peak['rss']:.3e}", "AIC": round(peak["aic"], 1),
              "BIC": round(peak["bic"], 1), "params": peak["n_params"]})
    st.caption(
        f"Half-degradation point ≈ **p = {deg['half_point']:.4f}** · linear slope "
        f"dr/dp ≈ **{deg['slope']:.3f}** · data lift over noiseless = "
        f"{result['data_lift_pct']:+.2f}%. A peak is declared only when it wins on "
        f"BIC, sits strictly interior, and lifts the ratio above the p=0 value."
    )

st.divider()

# ---------------------------------------------------------------------------
# 2 · β-cross-sections at the optimal γ
# ---------------------------------------------------------------------------
st.markdown("### 🔪 β-cross-section at γ\\* (overlaid noise levels)")
# Reference γ* from the noiseless landscape so every curve is the same slice.
gamma_ref = float(sweep[0.0]["best_gamma"])
rows = []
for lv in CROSS_LEVELS:
    src = sweep.get(lv)
    if src is None:  # level not in the 9-point resonance grid → compute it
        src = cached_landscape(*gargs, resolution, depth, environment, lv, elevation, shots)
    betas_axis, row = slice_at_gamma(src, gamma_ref)
    rows.append((lv, row))
st.caption(f"Fixed at the noiseless optimum **γ\\* = {gamma_ref:.3f} rad**; each curve is "
           f"ratio vs β at that γ for one noise level.")
st.pyplot(plot_cross_sections(betas_axis, rows,
                              title=f"Ratio vs β at γ*={gamma_ref:.3f} · {environment}"),
          width="stretch")

st.divider()

# ---------------------------------------------------------------------------
# 3 · Morphing animation (Play)
# ---------------------------------------------------------------------------
st.markdown("### ▶️ Landscape morph: noise 0.0 → 0.1")
anim_levels = tuple(np.round(np.linspace(0.0, 0.1, int(n_frames)), 4))
anim_sig = (sweep_sig, anim_levels)

colA, colB = st.columns([1, 3])
with colA:
    play = st.button("▶️ Play sweep", width="stretch")
    loops = st.slider("Loops", 1, 5, 2, 1)
with colB:
    scrub_idx = st.slider("Scrub noise frame", 0, len(anim_levels) - 1, 0, 1,
                          format="frame %d")

# Frames are memoised; computed on first Play/scrub for this configuration.
if st.session_state.get("anim_sig") != anim_sig or "anim_frames" not in st.session_state:
    bar = st.progress(0.0, text="Preparing animation frames…")
    try:
        frames = run_batch(list(anim_levels), gargs, resolution, depth,
                           environment, elevation, shots, bar)
    finally:
        bar.empty()
    st.session_state["anim_frames"] = frames
    st.session_state["anim_sig"] = anim_sig
frames = st.session_state["anim_frames"]

scrub_lv = anim_levels[scrub_idx]
scrub_L = frames[scrub_lv]
frame_slot = st.empty()


def _render_frame(lv, L, slot):
    fig = plot_landscape_contour(
        L["gammas"], L["betas"], L["ratios"],
        title=f"{environment} · noise p={lv:.3f} · best ratio {L['best_ratio']:.4f}")
    slot.pyplot(fig, width="stretch")


_render_frame(scrub_lv, scrub_L, frame_slot)
sm1, sm2, sm3 = st.columns(3)
sm1.metric("Frame noise p", f"{scrub_lv:.3f}")
sm2.metric("Frame best ratio", f"{scrub_L['best_ratio']:.4f}")
sm3.metric("Frame γ*", f"{scrub_L['best_gamma']:.3f}")

if play:
    seq = list(anim_levels) * int(loops)
    for lv in seq:
        _render_frame(lv, frames[lv], frame_slot)
        time.sleep(0.22)
    _render_frame(scrub_lv, scrub_L, frame_slot)

st.divider()

# ---------------------------------------------------------------------------
# 4 · Export the analysis as PDF
# ---------------------------------------------------------------------------
st.markdown("### 💾 Export analysis (PDF)")
verdict_line = (
    f"✅ Stochastic resonance detected at p* = {result['p_star']:.4f}, "
    f"improvement {result['improvement_pct']:.2f}%"
    if result["resonance"] else
    f"❌ No resonance — noise monotonically degrades performance ({result['reason']})"
)
summary_lines = [
    f"Environment      : {environment}  ({_info['name']})",
    f"Graph            : {family}, n={eff_n}, |E|={G.number_of_edges()}, seed={seed}",
    f"QAOA depth       : p={depth}   Grid resolution: {resolution}",
    f"Optimal cut C_max: {opt_cut}",
    "",
    "RESONANCE DETECTOR",
    f"  {verdict_line}",
    f"  Winning model  : {result['winner']}   ΔBIC (mono-peak) = {result['delta_bic']:+.2f}",
    f"  ΔAIC (mono-peak): {result['delta_aic']:+.2f}",
    f"  Best ratio p=0 : {best_ratios[0]:.4f}   p=0.1: {best_ratios[-1]:.4f}",
    f"  Total loss     : {deg['pct_loss_max']:.2f}%   half-point p≈{deg['half_point']:.4f}",
    f"  Linear slope   : dr/dp ≈ {deg['slope']:.3f}",
    "",
    "NOISE SWEEP (best approximation ratio)",
]
for lv, br in zip(RESONANCE_LEVELS, best_ratios):
    summary_lines.append(f"  p={lv:<6.3f}  ratio={br:.4f}")
summary_lines += [
    "",
    f"Reference γ* (noiseless) = {gamma_ref:.4f} rad for the β-cross-sections.",
    "",
    "Model fits:",
    f"  monotone  r = C + A exp(-B p):  A={result['monotone']['A']:.5f} "
    f"B={result['monotone']['B']:.4f} C={result['monotone']['C']:.5f} "
    f"BIC={result['monotone']['bic']:.1f}",
    f"  peak      r = C + A exp(-(p-p*)^2/2s^2):  A={result['peak']['A']:.5f} "
    f"p*={result['peak']['p_star']:.5f} s={result['peak']['sigma']:.5f} "
    f"BIC={result['peak']['bic']:.1f}",
]

if st.button("📄 Build PDF report", width="stretch"):
    with st.spinner("Rendering PDF report…"):
        fig_sweep = plot_noise_sweep(RESONANCE_LEVELS, best_ratios, result=result,
                                     title=f"Best ratio vs noise · {environment}",
                                     env_name=_info["name"], figsize=(7.4, 4.6))
        fig_cross = plot_cross_sections(betas_axis, rows,
                                        title=f"β-cross-section at γ*={gamma_ref:.3f}",
                                        figsize=(7.4, 4.6))
        fig_land0 = plot_landscape_contour(sweep[0.0]["gammas"], sweep[0.0]["betas"],
                                           sweep[0.0]["ratios"],
                                           title=f"Landscape at p=0 · {_info['name']}",
                                           figsize=(6.6, 5.2))
        fig_land1 = plot_landscape_contour(sweep[0.1]["gammas"], sweep[0.1]["betas"],
                                           sweep[0.1]["ratios"],
                                           title=f"Landscape at p=0.1 · {_info['name']}",
                                           figsize=(6.6, 5.2))
        pdf_bytes = export_analysis_pdf(
            "Noise Analysis Report", f"{environment} · {family} n={eff_n} · depth {depth}",
            [
                {"type": "text", "heading": "Analysis summary", "lines": summary_lines},
                {"type": "figure", "fig": fig_sweep, "caption": "Best approximation ratio vs noise level, with monotone/peak model fits and the resonance verdict."},
                {"type": "figure", "fig": fig_cross, "caption": f"β-cross-sections at the noiseless optimum γ*={gamma_ref:.3f} for six noise levels."},
                {"type": "figure", "fig": fig_land0, "caption": "Full (γ, β) landscape at p = 0 (noiseless)."},
                {"type": "figure", "fig": fig_land1, "caption": "Full (γ, β) landscape at p = 0.1 (max noise)."},
            ],
        )
    st.session_state["pdf_bytes"] = pdf_bytes

if st.session_state.get("pdf_bytes"):
    st.download_button(
        "⬇️ Download PDF report",
        data=st.session_state["pdf_bytes"],
        file_name=f"noise_analysis_{environment}_{family}_n{eff_n}.pdf",
        mime="application/pdf", width="stretch",
    )
    st.caption(f"PDF size: {len(st.session_state['pdf_bytes'])/1024:.0f} KB · "
               f"5 pages (summary + 4 figures).")

st.divider()
st.caption(
    "Exact density-matrix evaluation via the isolated simulation worker · "
    "resonance verdict from a BIC-penalised monotone-vs-peak model comparison."
)
