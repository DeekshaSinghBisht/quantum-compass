"""Landscape fingerprinting — infer a simulated noise environment from shape."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from core.fingerprint import (  # noqa: E402
    DEFAULT_MODEL_PATH,
    FINGERPRINT_ENV_LABELS,
    fingerprint_feature_dict,
    generate_mystery_landscape,
    load_default_classifier,
    mystery_landscape_catalog,
    predict_environment,
)
from utils.plotting import plot_landscape_contour_plotly  # noqa: E402

st.set_page_config(page_title="Landscape Fingerprinting · Quantum Compass", page_icon="🧬", layout="wide")


@st.cache_resource(show_spinner=False)
def cached_default_classifier():
    """Load the persisted RandomForest; trains once only if its cache is absent."""
    return load_default_classifier()


@st.cache_data(show_spinner=False)
def cached_mystery(mystery_id: str) -> Dict:
    """Generate/cache a deterministic mystery landscape through the Aer worker."""
    return generate_mystery_landscape(mystery_id, resolution=9)


def parse_landscape_csv(uploaded_file) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate long-form gamma/beta/ratio CSV data and return a dense grid.

    Column matching is case-insensitive and accepts Greek gamma/beta headers.
    Duplicate points are averaged. Missing grid cells, nonnumeric values and
    degenerate axes produce a friendly ValueError for the upload UI.
    """
    try:
        raw = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()
        frame = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        raise ValueError("Could not read the upload as UTF-8/CSV landscape data.") from exc
    normalized = {str(col).strip().lower(): col for col in frame.columns}
    gamma_col = next((normalized[key] for key in ("gamma", "γ") if key in normalized), None)
    beta_col = next((normalized[key] for key in ("beta", "β") if key in normalized), None)
    ratio_col = next((normalized[key] for key in ("ratio", "approximation_ratio", "approximation ratio") if key in normalized), None)
    if not all((gamma_col, beta_col, ratio_col)):
        raise ValueError("CSV needs long-form columns named gamma (or γ), beta (or β), and ratio.")
    data = frame[[gamma_col, beta_col, ratio_col]].copy()
    data.columns = ["gamma", "beta", "ratio"]
    for column in data.columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    if data.empty:
        raise ValueError("CSV contains no finite gamma, beta, ratio rows.")
    pivot = data.groupby(["gamma", "beta"], as_index=False)["ratio"].mean().pivot(
        index="gamma", columns="beta", values="ratio"
    ).sort_index().sort_index(axis=1)
    if pivot.shape[0] < 3 or pivot.shape[1] < 3:
        raise ValueError("Fingerprinting needs at least a 3×3 complete gamma-beta landscape grid.")
    if pivot.isna().any().any():
        raise ValueError("CSV gamma-beta grid has missing cells; upload one ratio for every grid coordinate.")
    return (pivot.index.to_numpy(dtype=float), pivot.columns.to_numpy(dtype=float),
            pivot.to_numpy(dtype=float))


def prediction_bar(prediction: Dict) -> go.Figure:
    """Dark top-three probability chart for one fingerprint prediction."""
    top = list(reversed(prediction["top_3"]))
    labels = [entry["label"] for entry in top]
    values = [entry["confidence_percent"] for entry in top]
    colours = ["#52616B", "#00B4D8", "#FFD700"][-len(top):]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h", marker_color=colours,
        text=[f"{value:.1f}%" for value in values], textposition="outside",
        hovertemplate="%{y}: %{x:.2f}%<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Top environment predictions", x=0.5),
        height=330, margin=dict(l=50, r=60, t=55, b=35),
        xaxis=dict(title="classifier confidence (%)", range=[0, 100], gridcolor="#2A2A3E"),
        yaxis=dict(title=""), paper_bgcolor="#1A1A2E", plot_bgcolor="#0A0A0A",
        font=dict(color="#E0E0E0"),
    )
    return fig


def parameter_table(prediction: Dict) -> pd.DataFrame:
    """Display requested and auxiliary weighted channel estimates."""
    params = prediction["estimated_params"]
    rows = [
        ("Estimated p₂ (2-qubit depolarizing)", params["p2"]),
        ("Estimated burst rate", params["burst_rate"]),
        ("Estimated p₁ (1-qubit depolarizing)", params["p1"]),
        ("Estimated readout error", params["readout"]),
        ("Estimated amplitude damping", params["ad_rate"]),
        ("Estimated phase damping", params["pd_rate"]),
    ]
    return pd.DataFrame(
        [{"Estimated parameter": label, "Value": f"{float(value):.5f}"} for label, value in rows]
    )


def main() -> None:
    st.title("🧬 Landscape Fingerprinting")
    st.caption(
        "Infer which simulated hardware or space environment most likely produced a QAOA landscape from its shape. "
        "This is a model-family diagnostic, not forensic attribution of an arbitrary physical device."
    )
    st.markdown(
        "> **Use cases:** compare calibration runs for **hardware diagnostics**, monitor a landscape stream for "
        "a shift toward a harsher **space environment**, and check whether an observed surface is consistent with "
        "the noise model used in an experiment."
    )

    st.sidebar.header("🧬 Fingerprint input")
    source = st.sidebar.radio("Landscape source", ["Mystery Landscape", "Upload CSV"], index=0)
    catalog = mystery_landscape_catalog()
    source_payload = None
    source_key = ""

    if source == "Mystery Landscape":
        choices = {item["title"]: item for item in catalog}
        title = st.sidebar.selectbox("Choose one of five mystery landscapes", list(choices), index=0)
        choice = choices[title]
        st.sidebar.caption(choice["hint"])
        try:
            with st.spinner("Preparing mystery landscape…"):
                source_payload = cached_mystery(choice["id"])
            landscape = source_payload["landscape"]
            gammas = np.asarray(landscape["gammas"], dtype=float)
            betas = np.asarray(landscape["betas"], dtype=float)
            ratios = np.asarray(landscape["ratios"], dtype=float)
            source_key = f"mystery:{choice['id']}"
            st.plotly_chart(
                plot_landscape_contour_plotly(gammas, betas, ratios, title=choice["title"]),
                width="stretch", key=f"mystery_preview_{choice['id']}",
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not prepare this mystery landscape: {exc}")
            return
    else:
        uploaded = st.sidebar.file_uploader("Upload landscape CSV", type=["csv"],
                                            help="Long form: gamma, beta, ratio columns.")
        if uploaded is not None:
            try:
                gammas, betas, ratios = parse_landscape_csv(uploaded)
                source_payload = {"title": uploaded.name, "landscape": {"gammas": gammas, "betas": betas, "ratios": ratios}}
                source_key = f"upload:{uploaded.name}:{len(uploaded.getvalue())}"
                st.success(f"Loaded a complete {ratios.shape[0]}×{ratios.shape[1]} landscape grid.")
                st.plotly_chart(
                    plot_landscape_contour_plotly(gammas, betas, ratios, title="Uploaded landscape"),
                    width="stretch", key=f"upload_preview_{source_key}",
                )
            except ValueError as exc:
                st.error(f"Invalid landscape CSV: {exc}")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Unexpected upload error: {exc}")

    analyze = st.sidebar.button("🔍 Analyze Fingerprint", type="primary", width="stretch",
                                disabled=(source_payload is None))
    if analyze and source_payload is not None:
        try:
            with st.spinner("Loading the default RandomForest fingerprint model…"):
                classifier = cached_default_classifier()
            prediction = predict_environment(ratios, classifier)
            st.session_state["fingerprint_view"] = {
                "source_key": source_key, "prediction": prediction,
                "title": source_payload["title"],
                "hidden_environment": source_payload.get("_hidden_environment"),
                "model_oob": float(getattr(classifier, "_qc_oob_score", 0.0)),
                "model_training_accuracy": float(getattr(classifier, "_qc_training_accuracy", 0.0)),
            }
        except Exception as exc:  # noqa: BLE001
            st.error(f"Fingerprint analysis failed: {exc}")
            return

    view = st.session_state.get("fingerprint_view")
    if view is None or view.get("source_key") != source_key:
        if source_payload is not None:
            st.info("Press **🔍 Analyze Fingerprint** to classify this landscape.")
        else:
            st.info("Choose a mystery landscape or upload a complete gamma-beta-ratio CSV grid.")
        return

    prediction = view["prediction"]
    st.subheader("Fingerprint result")
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted environment", prediction["predicted_label"])
    c2.metric("Confidence", f"{prediction['confidence']:.1f}%")
    c3.metric("Default model OOB accuracy", f"{100 * view['model_oob']:.1f}%")
    if prediction["confidence"] >= 60.0:
        st.success(
            f"The landscape is most consistent with **{prediction['predicted_label']}** at "
            f"**{prediction['confidence']:.1f}%** confidence."
        )
    else:
        st.warning(
            "The classifier is uncertain. Treat this as a cue to inspect calibration/reference circuits, "
            "not as a definitive environment attribution."
        )

    left, right = st.columns([1.25, 0.75])
    with left:
        st.plotly_chart(prediction_bar(prediction), width="stretch", key=f"fingerprint_bar_{source_key}")
    with right:
        st.markdown("**Estimated noise parameters**")
        st.dataframe(parameter_table(prediction), width="stretch", hide_index=True)

    hidden = view.get("hidden_environment")
    if hidden:
        with st.expander("Reveal mystery source", expanded=False):
            expected = FINGERPRINT_ENV_LABELS.get(hidden, hidden)
            correct = prediction["predicted_environment"] == hidden
            if correct:
                st.success(f"Reference source: **{expected}** — the fingerprint matched this simulated landscape.")
            else:
                st.info(f"Reference source: **{expected}**. This is a useful ambiguous case for diagnostics.")

    with st.expander("Fingerprint features", expanded=False):
        feature_df = pd.DataFrame(
            [{"Feature": name, "Value": f"{value:.6f}"} for name, value in prediction["features"].items()]
        )
        st.dataframe(feature_df, width="stretch", hide_index=True)
        st.caption(
            f"Default model cache: `{DEFAULT_MODEL_PATH.relative_to(_project_root)}` · "
            "features include extrema, width, roughness, gradients, symmetry and distributional moments."
        )

    with st.expander("📥 Export fingerprint result", expanded=False):
        payload = {"source": view["title"], "prediction": prediction}
        st.download_button("⬇️ Download fingerprint JSON", json.dumps(payload, indent=2),
                           file_name="quantum_compass_fingerprint.json", mime="application/json",
                           width="stretch")


main()
