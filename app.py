"""🧭 Quantum Compass — QAOA Optimization Landscape Navigator.

Entry point for the Streamlit multipage app.  Run with::

    streamlit run app.py

The landing page explains the tool and links to the interactive sub-pages.
"""

from __future__ import annotations

import os
import sys

# --- make `core` and `utils` importable regardless of launch directory ------
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import streamlit as st

st.set_page_config(
    page_title="Quantum Compass",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Hero
# ----------------------------------------------------------------------------
st.title("🧭 Quantum Compass")
st.subheader("QAOA Optimization Landscape Navigator")

st.markdown(
    """
    **Quantum Compass** is an interactive explorer for the *optimization
    landscape* of the **Quantum Approximate Optimization Algorithm (QAOA)**
    applied to **MaxCut**.

    QAOA prepares a parameterized quantum state
    `|ψ(γ, β)⟩` and a classical optimizer searches the angles `(γ, β)` to
    maximize the **approximation ratio** `⟨C⟩ / C_max`.  That search happens on
    a rugged surface full of peaks and traps.  This tool *computes and draws
    that surface* so you can see exactly what the optimizer is navigating — and
    watch it deform as you dial hardware **noise** up and down.
    """
)

# ----------------------------------------------------------------------------
# Feature cards
# ----------------------------------------------------------------------------
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(
        """
        #### 🗺️ Landscape View
        Sweep `γ ∈ [0, 2π)`, `β ∈ [0, π)` on a grid and render the
        approximation ratio as a **2-D contour** and **3-D surface**, with the
        global optimum (⭐) and local minima (○) marked.
        """
    )
with c2:
    st.markdown(
        """
        #### 🎚️ The Noise Slider
        Morph the device across **7 environments** — ideal, earth-lab, and five
        space settings (LEO, the satellite↔ground link, MEO/Van Allen belts, and
        deep space) with cosmic-ray bursts, thermal cycling and photon loss.
        Drag one slider and watch the peaks flatten and the traps shift.
        """
    )
with c3:
    st.markdown(
        """
        #### 🧗 Optimizer Paths
        Run multi-start **COBYLA / Nelder-Mead / Powell / L-BFGS-B** and
        overlay every restart's trajectory on the landscape to see how it
        escapes (or falls into) local minima.
        """
    )

st.divider()

# ----------------------------------------------------------------------------
# Navigation
# ----------------------------------------------------------------------------
st.markdown("### Navigate")

_PAGES = [
    ("pages/1_Landscape_View.py",
     "🗺️  Landscape View — compute & explore a QAOA landscape across 7 environments"),
    ("pages/2_Noise_Analysis.py",
     "📈  Noise Analysis — morph animation, β-cross-sections & stochastic-resonance detection"),
    ("pages/3_Space_Mode.py",
     "🛰️  Space Mode — compare Earth, LEO, MEO, deep space & the satellite link side by side"),
    ("pages/4_Advantage_Map.py",
     "🗺️  Advantage Map — where QAOA beats a classical baseline, across size & noise"),
    ("pages/5_Barren_Plateau.py",
     "🧊  Barren Plateau — trainability phase diagram across qubits, depth & environment"),
    ("pages/6_Hardware_Profiles.py",
     "🖥️  Hardware Profiles — calibration-derived QAOA landscapes for IBM fake backends"),
    ("pages/7_Mission_Planner.py",
     "🛰️  Mission Planner — space-QAOA strategy, risk and advantage-window reports"),
    ("pages/8_About.py",
     "📚  About — QAOA methodology, environment citations and reuse guidance"),
    ("pages/9_Fingerprinting.py",
     "🧬  Fingerprinting — infer a simulated environment from landscape shape"),
    ("pages/10_Experimental_Validation.py",
     "🧪  Experimental Validation — preregistered real-hardware versus simulator study"),
]
for path, label in _PAGES:
    try:
        st.page_link(path, label=label)
    except Exception:
        st.markdown(f"➡️ Open **{label}** from the sidebar (`{path}`).")

st.caption(
    "Tip: the first landscape computation runs an exact quantum simulation over "
    "the whole grid — a progress bar shows the sweep. Results are cached, so "
    "revisiting the same graph + noise setting is instant."
)

# ----------------------------------------------------------------------------
# How it works (compact)
# ----------------------------------------------------------------------------
with st.expander("ℹ️ How it works under the hood"):
    st.markdown(
        """
        * **Instance** — a random graph (`3-regular`, `erdos-renyi`, or
          `scale-free`) becomes a MaxCut problem with one qubit per vertex; the
          optimum is found by brute force (`n ≤ 16`).
        * **Ansatz** — `H^{⊗n}` then, for each layer `p`: `RZZ(2γ)` on every
          edge (cost) followed by `RX(2β)` on every qubit (mixer).
        * **Evaluation** — exact output distribution from Qiskit **Aer**
          (`statevector` when ideal, `density_matrix` when noisy) so landscapes
          are smooth rather than shot-noisy. The sampling evaluator
          (`evaluate_qaoa`) reproduces real measurement `counts`.
        * **Noise** — `qiskit_aer.noise` channels on the native `h`/`rx`/`rzz`
          gates plus a symmetric `ReadoutError`, all driven by a single level.
          Space environments add **cosmic-ray burst** errors (a
          `(1−br)·I + br·Depol(0.5)` mixture), an orbital **thermal-cycling**
          modulation, and — for the satellite link — **amplitude damping**
          (elevation-dependent photon loss), turbulence **dephasing** and a
          **thermal-photon** population.

        *Seven environments: `ideal`, `earth-lab`, `leo-satellite`,
        `satellite-ground`, `meo-van-allen`, `deep-space` and `custom`. Each is
        documented in-app with its parameters, citation and relevant missions.*
        """
    )

st.divider()
st.caption("🧭 Quantum Compass · research-validation edition · built with Qiskit, Qiskit-Aer, IBM Runtime, scikit-learn & Streamlit")
