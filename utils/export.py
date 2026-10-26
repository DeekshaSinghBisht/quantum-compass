"""Export helpers — turn landscapes and results into downloadable artefacts.

Everything here returns either a ``str``/``bytes`` payload (for Streamlit
``download_button``) or writes a file to disk (for batch/scripted use).  No
Streamlit import is required, so these functions are also usable from tests and
the CLI.
"""

from __future__ import annotations

import io
import json
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def landscape_to_dataframe(
    gammas: np.ndarray,
    betas: np.ndarray,
    ratios: np.ndarray,
) -> pd.DataFrame:
    """Wide-format table: rows = gamma, columns = beta, values = ratio."""
    ratios = np.asarray(ratios, dtype=float)
    df = pd.DataFrame(
        ratios,
        index=pd.Index(np.round(gammas, 6), name="gamma"),
        columns=pd.Index(np.round(betas, 6), name="beta"),
    )
    return df


def landscape_to_long_dataframe(
    gammas: np.ndarray,
    betas: np.ndarray,
    ratios: np.ndarray,
) -> pd.DataFrame:
    """Long/tidy table with one row per grid point: ``gamma, beta, ratio``."""
    ratios = np.asarray(ratios, dtype=float)
    gi, bj = np.meshgrid(np.arange(len(gammas)), np.arange(len(betas)), indexing="ij")
    return pd.DataFrame(
        {
            "gamma": np.asarray(gammas)[gi.ravel()],
            "beta": np.asarray(betas)[bj.ravel()],
            "ratio": ratios.ravel(),
        }
    )


def export_landscape_csv(
    gammas: np.ndarray,
    betas: np.ndarray,
    ratios: np.ndarray,
    long_format: bool = True,
) -> str:
    """Return the landscape as a CSV string (downloadable)."""
    df = (
        landscape_to_long_dataframe(gammas, betas, ratios)
        if long_format
        else landscape_to_dataframe(gammas, betas, ratios)
    )
    # A long-form file is intentionally upload-ready for the fingerprint page:
    # exactly gamma,beta,ratio columns and no synthetic DataFrame index.
    return df.to_csv(index=not long_format)


def export_landscape_npz(
    gammas: np.ndarray,
    betas: np.ndarray,
    ratios: np.ndarray,
    **extra: np.ndarray,
) -> bytes:
    """Return the landscape as an in-memory ``.npz`` archive (downloadable)."""
    buf = io.BytesIO()
    arrays = {
        "gammas": np.asarray(gammas),
        "betas": np.asarray(betas),
        "ratios": np.asarray(ratios),
    }
    for k, v in extra.items():
        if v is not None:
            arrays[k] = np.asarray(v)
    np.savez_compressed(buf, **arrays)
    return buf.getvalue()


def export_results_json(
    landscape: Optional[Dict] = None,
    optimization: Optional[Dict] = None,
    graph_props: Optional[Dict] = None,
    graph_meta: Optional[Dict] = None,
) -> str:
    """Serialise a run (landscape summary + optimiser result) to pretty JSON.

    Large arrays are reduced to summaries / lists so the payload stays small and
    human-readable.
    """
    payload: Dict = {"graph": graph_meta or {}, "graph_properties": graph_props or {}}

    if landscape is not None:
        payload["landscape"] = {
            "depth": landscape.get("depth"),
            "resolution": landscape.get("resolution"),
            "optimal_cut": landscape.get("optimal_cut"),
            "best_ratio": landscape.get("best_ratio"),
            "best_gamma": landscape.get("best_gamma"),
            "best_beta": landscape.get("best_beta"),
            "gamma_range": [
                float(np.min(landscape["gammas"])),
                float(np.max(landscape["gammas"])),
            ],
            "beta_range": [
                float(np.min(landscape["betas"])),
                float(np.max(landscape["betas"])),
            ],
        }

    if optimization is not None:
        payload["optimization"] = {
            "method": optimization.get("method"),
            "n_restarts": optimization.get("n_restarts"),
            "nfev": optimization.get("nfev"),
            "best_ratio": optimization.get("best_ratio"),
            "best_params": [float(x) for x in np.asarray(optimization.get("best_params", []))],
            "best_gamma": optimization.get("best_gamma"),
            "best_beta": optimization.get("best_beta"),
        }

    return json.dumps(payload, indent=2, default=float)


def save_matplotlib_fig(fig, path: str, dpi: int = 150) -> str:
    """Save a matplotlib figure to ``path`` (PNG/SVG by extension). Returns path."""
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    return path


def save_plotly_html(fig: go.Figure, path: str) -> str:
    """Write a self-contained interactive HTML export of a plotly figure."""
    fig.write_html(path, include_plotlyjs="inline", full_html=True)
    return path


def plotly_fig_to_bytes(fig: go.Figure) -> bytes:
    """Return a plotly figure as standalone HTML bytes (downloadable)."""
    return fig.to_html(include_plotlyjs="inline", full_html=True).encode("utf-8")


def counts_to_dataframe(counts: Dict[str, int]) -> pd.DataFrame:
    """Sort a ``{bitstring: count}`` dict into a descending DataFrame."""
    items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    return pd.DataFrame(items, columns=["bitstring", "count"])


def trajectories_to_dataframe(all_trajectories: Sequence[Sequence]) -> pd.DataFrame:
    """Flatten optimiser trajectories into a tidy DataFrame."""
    rows: List[Dict] = []
    for r, traj in enumerate(all_trajectories or []):
        for step, point in enumerate(traj):
            if len(point) == 3 and np.isscalar(point[0]):
                g, b, ratio = point
                rows.append({"restart": r, "step": step, "gamma": g, "beta": b, "ratio": ratio})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# PDF report export (matplotlib PdfPages — no extra dependency).
#
# Produces a real, downloadable multi-page PDF from a list of text/figure
# blocks.  The analysis pages build one figure per chart plus text summaries and
# hand them here.  Pages use the app's dark theme for visual consistency.
# ---------------------------------------------------------------------------
_BG = "#0A0A0A"
_PANEL = "#1A1A2E"
_TEXT = "#E0E0E0"
_ACCENT = "#00B4D8"


def _text_page(pdf, heading: str, lines: Sequence[str], footer: str = "") -> None:
    """Render one dark-themed text page into an open ``PdfPages``."""
    import textwrap

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
    fig.patch.set_facecolor(_BG)
    fig.text(0.08, 0.93, heading, color=_ACCENT, fontsize=17, weight="bold", va="top")
    y = 0.87
    for raw in lines:
        raw = "" if raw is None else str(raw)
        if raw == "":
            y -= 0.018
            continue
        wrapped = textwrap.wrap(raw, width=88) or [""]
        for wl in wrapped:
            if y < 0.06:
                break
            fig.text(0.08, y, wl, color=_TEXT, fontsize=10.5, va="top", family="monospace")
            y -= 0.021
        y -= 0.006
    if footer:
        fig.text(0.08, 0.03, footer, color="#8A8A9A", fontsize=8, va="bottom")
    pdf.savefig(fig, facecolor=_BG)
    plt.close(fig)


def _figure_page(pdf, fig, caption: str = "", footer: str = "") -> None:
    """Render a matplotlib figure full-page with an optional caption."""
    import matplotlib.pyplot as plt

    if caption:
        # Attach the caption beneath the figure's own axes.
        try:
            fig.text(0.5, 0.015, caption, color=_TEXT, fontsize=9, ha="center", va="bottom")
        except Exception:
            pass
    pdf.savefig(fig, facecolor=fig.get_facecolor() or _BG, bbox_inches="tight")
    plt.close(fig)


def export_analysis_pdf(title: str, subtitle: str, blocks: Sequence[Dict],
                        footer: str = "Generated by Quantum Compass") -> bytes:
    """Build a multi-page PDF report and return it as bytes (downloadable).

    ``blocks`` is an ordered list of either::

        {"type": "text",   "heading": str, "lines": [str, ...]}
        {"type": "figure", "fig": <matplotlib.Figure>, "caption": str}

    The first page is a title/summary page built from ``title``/``subtitle`` plus
    any leading text block; figures each get their own page.
    """
    from matplotlib.backends.backend_pdf import PdfPages

    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        # --- title page ---------------------------------------------------
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(8.27, 11.69))
        fig.patch.set_facecolor(_BG)
        fig.text(0.5, 0.80, "Quantum Compass", color=_ACCENT, fontsize=24,
                 weight="bold", ha="center", va="center")
        fig.text(0.5, 0.735, title, color=_TEXT, fontsize=17, ha="center", va="center")
        fig.text(0.5, 0.685, subtitle, color="#B8B8C8", fontsize=11.5, ha="center", va="center")
        import datetime as _dt

        fig.text(0.5, 0.60, _dt.datetime.now().strftime("%Y-%m-%d  %H:%M"),
                 color="#8A8A9A", fontsize=10, ha="center", va="center")
        fig.add_artist(plt.Line2D([0.15, 0.85], [0.655, 0.655], color=_PANEL, lw=2))
        pdf.savefig(fig, facecolor=_BG)
        plt.close(fig)

        # --- content blocks ------------------------------------------------
        for blk in blocks:
            kind = blk.get("type", "text")
            if kind == "figure" and blk.get("fig") is not None:
                _figure_page(pdf, blk["fig"], blk.get("caption", ""), footer)
            else:
                _text_page(pdf, blk.get("heading", ""), blk.get("lines", []), footer)

        try:  # embed PDF metadata
            d = pdf.infodict()
            d["Title"] = f"Quantum Compass — {title}"
            d["Author"] = "Quantum Compass"
            d["Subject"] = subtitle
        except Exception:
            pass
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Final deliverable exports: robust static images, CSV files and full reports.
# ---------------------------------------------------------------------------
def _output_path(filename: str, suffix: str) -> str:
    """Normalize a writable output filename and create its parent directory."""
    from pathlib import Path

    path = Path(filename).expanduser()
    if path.suffix.lower() != suffix:
        path = path.with_suffix(suffix)
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def _is_plotly_figure(fig) -> bool:
    """Return whether ``fig`` is a Plotly graph object without fragile imports."""
    return isinstance(fig, go.Figure)


def _plotly_to_matplotlib(fig: go.Figure):
    """Best-effort static Matplotlib rendering for common Plotly trace types.

    Kaleido 1.x requires a system Chrome install.  Streamlit Cloud and minimal
    research containers do not always provide one, so this fallback guarantees
    that the export functions still produce a real PNG/SVG/PDF for the charts
    used by Quantum Compass (scatter, bar, heatmap, contour and surface).
    """
    import matplotlib.pyplot as plt

    traces = list(fig.data)
    has_3d = any(getattr(trace, "type", "") in {"surface", "scatter3d"} for trace in traces)
    mpl_fig = plt.figure(figsize=(8.0, 5.2), dpi=110)
    ax = mpl_fig.add_subplot(111, projection="3d" if has_3d else None)
    cmap = "viridis"
    legend_needed = False

    for trace in traces:
        kind = getattr(trace, "type", "")
        name = getattr(trace, "name", None) or kind
        if kind == "scatter":
            x = np.asarray(getattr(trace, "x", []) if getattr(trace, "x", None) is not None else np.arange(len(getattr(trace, "y", []))))
            y = np.asarray(getattr(trace, "y", []), dtype=float)
            mode = str(getattr(trace, "mode", "lines"))
            if "lines" in mode:
                ax.plot(x, y, label=name)
            if "markers" in mode:
                ax.scatter(x, y, s=24, label=None if "lines" in mode else name)
            legend_needed = True
        elif kind == "bar":
            x = np.asarray(getattr(trace, "x", []))
            y = np.asarray(getattr(trace, "y", []))
            orientation = str(getattr(trace, "orientation", "v"))
            if orientation == "h":
                ax.barh(y, x, label=name)
            else:
                ax.bar(x, y, label=name)
            legend_needed = True
        elif kind in {"heatmap", "contour"}:
            z = np.asarray(getattr(trace, "z", []), dtype=float)
            if z.ndim != 2 or z.size == 0:
                continue
            x = getattr(trace, "x", None)
            y = getattr(trace, "y", None)
            extent = None
            try:
                if x is not None and y is not None and len(x) and len(y):
                    extent = [float(np.min(x)), float(np.max(x)), float(np.min(y)), float(np.max(y))]
            except Exception:
                extent = None
            if kind == "contour":
                im = ax.contourf(z, levels=16, cmap=cmap)
            else:
                im = ax.imshow(z, origin="lower", aspect="auto", cmap=cmap, extent=extent)
            mpl_fig.colorbar(im, ax=ax, shrink=0.85)
        elif kind == "surface":
            z = np.asarray(getattr(trace, "z", []), dtype=float)
            if z.ndim != 2 or z.size == 0:
                continue
            raw_x = getattr(trace, "x", None)
            raw_y = getattr(trace, "y", None)
            x = np.asarray(np.arange(z.shape[1]) if raw_x is None else raw_x, dtype=float)
            y = np.asarray(np.arange(z.shape[0]) if raw_y is None else raw_y, dtype=float)
            xx, yy = np.meshgrid(x, y)
            ax.plot_surface(xx, yy, z, cmap=cmap, linewidth=0, antialiased=True, alpha=0.92)
        elif kind == "scatter3d":
            x = np.asarray(getattr(trace, "x", []), dtype=float)
            y = np.asarray(getattr(trace, "y", []), dtype=float)
            z = np.asarray(getattr(trace, "z", []), dtype=float)
            if not (x.size and y.size and z.size):
                continue
            mode = str(getattr(trace, "mode", "markers"))
            if "lines" in mode:
                ax.plot(x, y, z, label=name)
            if "markers" in mode:
                ax.scatter(x, y, z, s=30, label=None if "lines" in mode else name)
            legend_needed = True
        else:
            # Unknown traces do not block export; a textual annotation makes
            # the fallback's scope explicit rather than raising mid-report.
            ax.text(0.02, 0.02, f"Unsupported static trace: {kind}", transform=ax.transAxes,
                    fontsize=8, alpha=0.65)

    try:
        title = fig.layout.title.text
        if title:
            ax.set_title(str(title))
    except Exception:
        pass
    try:
        xlabel = fig.layout.xaxis.title.text
        ylabel = fig.layout.yaxis.title.text
        if xlabel:
            ax.set_xlabel(str(xlabel))
        if ylabel:
            ax.set_ylabel(str(ylabel))
    except Exception:
        pass
    if legend_needed:
        try:
            ax.legend(loc="best", fontsize=8)
        except Exception:
            pass
    mpl_fig.tight_layout()
    return mpl_fig


def _save_static_figure(fig, filename: str, fmt: str, dpi: int = 300) -> str:
    """Write a Matplotlib or Plotly figure, with a Chrome-free Plotly fallback."""
    path = _output_path(filename, f".{fmt}")
    if _is_plotly_figure(fig):
        try:
            # Plotly/Kaleido produces a native vector SVG/PDF where the runtime
            # has Chrome.  `scale` provides approximately 300-DPI PNG quality.
            scale = 3 if fmt == "png" else 1
            fig.write_image(path, format=fmt, scale=scale)
            return path
        except Exception:
            fallback = _plotly_to_matplotlib(fig)
            try:
                fallback.savefig(path, format=fmt, dpi=dpi, bbox_inches="tight")
            finally:
                import matplotlib.pyplot as plt

                plt.close(fallback)
            return path
    if not hasattr(fig, "savefig"):
        raise TypeError("fig must be a matplotlib Figure or plotly.graph_objects.Figure.")
    fig.savefig(path, format=fmt, dpi=dpi, bbox_inches="tight")
    return path


def export_figure_png(fig, filename: str) -> str:
    """Export a figure to ``filename`` as a 300-DPI PNG and return its path."""
    return _save_static_figure(fig, filename, "png", dpi=300)


def export_figure_svg(fig, filename: str) -> str:
    """Export a figure to ``filename`` as an SVG and return its path."""
    return _save_static_figure(fig, filename, "svg", dpi=300)


def export_figure_pdf(fig, filename: str) -> str:
    """Export a figure to ``filename`` as a PDF and return its path."""
    return _save_static_figure(fig, filename, "pdf", dpi=300)


def export_data_csv(gammas, betas, ratios, filename: str) -> str:
    """Write long-form ``gamma,beta,ratio`` landscape data to a CSV file."""
    from pathlib import Path

    path = _output_path(filename, ".csv")
    Path(path).write_text(export_landscape_csv(gammas, betas, ratios, long_format=True), encoding="utf-8")
    return path


def _flatten_report_mapping(value, prefix: str = "") -> List[str]:
    """Turn nested params/results structures into readable deterministic lines."""
    lines: List[str] = []
    if isinstance(value, dict):
        for key in sorted(value, key=str):
            child = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
            lines.extend(_flatten_report_mapping(value[key], child))
    elif isinstance(value, (list, tuple)):
        if len(value) <= 8 and all(np.isscalar(item) or isinstance(item, str) for item in value):
            lines.append(f"{prefix}: {list(value)}")
        else:
            lines.append(f"{prefix}: {len(value)} item(s)")
    elif isinstance(value, np.ndarray):
        array = np.asarray(value)
        if array.size:
            lines.append(
                f"{prefix}: array shape={array.shape}, min={float(np.nanmin(array)):.6g}, "
                f"max={float(np.nanmax(array)):.6g}"
            )
        else:
            lines.append(f"{prefix}: empty array shape={array.shape}")
    else:
        lines.append(f"{prefix}: {value}")
    return lines


def export_report_pdf(params: Dict, results: Dict, figures: Sequence, filename: str) -> str:
    """Write a complete multi-page PDF report and return its output path.

    ``params`` and ``results`` are flattened into readable text pages.  Each
    supplied Matplotlib or Plotly figure gets its own page; Plotly figures use
    the same Chrome-free fallback as the individual static export helpers.
    """
    from pathlib import Path

    blocks: List[Dict] = [
        {"type": "text", "heading": "Run parameters", "lines": _flatten_report_mapping(params or {})},
        {"type": "text", "heading": "Results", "lines": _flatten_report_mapping(results or {})},
    ]
    converted = []
    for index, figure in enumerate(figures or [], start=1):
        if _is_plotly_figure(figure):
            figure = _plotly_to_matplotlib(figure)
        if hasattr(figure, "savefig"):
            converted.append(figure)
            blocks.append({"type": "figure", "fig": figure, "caption": f"Figure {index}"})
        else:
            blocks.append({"type": "text", "heading": f"Figure {index}",
                           "lines": ["Unsupported figure object omitted from static report."]})
    try:
        payload = export_analysis_pdf(
            title="Quantum Compass Report", subtitle="Parameters, results and figures", blocks=blocks,
            footer="Generated by Quantum Compass",
        )
        path = _output_path(filename, ".pdf")
        Path(path).write_bytes(payload)
        return path
    finally:
        # export_analysis_pdf closes figure pages itself; only converted Plotly
        # fallbacks need a defensive close if an exception occurs before save.
        import matplotlib.pyplot as plt

        for figure in converted:
            try:
                plt.close(figure)
            except Exception:
                pass
