"""Plotting helpers for Quantum Compass.

All figures share a dark aesthetic that matches ``.streamlit/config.toml``
(background ``#0A0A0A``, panels ``#1A1A2E``, text ``#E0E0E0``, accent
``#00B4D8``).  Two families of plots are produced:

* **matplotlib** – :func:`plot_landscape_contour` and :func:`plot_graph`
  (rendered with ``st.pyplot`` / shown in the sidebar),
* **plotly** – :func:`plot_landscape_3d`, :func:`plot_landscape_contour_plotly`
  and :func:`plot_trajectories` (interactive, rendered with ``st.plotly_chart``).

Every landscape plot marks the **global optimum with a gold star** and the
**local minima with cyan circles** — the visual signature of the app.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import plotly.graph_objects as go

from core.landscape import detect_local_maxima, detect_local_minima

# --- theme constants -------------------------------------------------------
BG = "#0A0A0A"
PANEL = "#1A1A2E"
TEXT = "#E0E0E0"
ACCENT = "#00B4D8"
GRID = "#2A2A3E"
OPT_COLOR = "#FFD700"      # gold star = global optimum
MIN_COLOR = "#00B4D8"      # cyan circles = local minima
MAX_COLOR = "#FF6B6B"      # red = secondary local maxima
CONTOUR_CMAP = "inferno"
SURFACE_CMAP = "Inferno"


def _dark_layout(fig: go.Figure, title: str) -> go.Figure:
    """Apply the shared dark theme to a plotly figure."""
    fig.update_layout(
        title=dict(text=title, font=dict(color=TEXT, size=16)),
        paper_bgcolor=PANEL,
        plot_bgcolor=BG,
        font=dict(color=TEXT),
        margin=dict(l=10, r=10, t=48, b=10),
    )
    return fig


def plot_landscape_contour(
    gammas: np.ndarray,
    betas: np.ndarray,
    ratios: np.ndarray,
    title: str = "QAOA Approximation-Ratio Landscape (p = 1)",
    figsize=(6.0, 5.0),
    show_maxima: bool = False,
) -> plt.Figure:
    """Filled-contour matplotlib figure of the ratio landscape.

    Gold star = global optimum; cyan circles = local minima; (optional) red
    circles = secondary local maxima.  Axes are labelled in radians with
    pi-based ticks.
    """
    ratios = np.asarray(ratios, dtype=float)
    # ratios[i, j] corresponds to (gammas[i], betas[j]); imshow wants [y, x]
    # so we plot with betas on the horizontal axis and gammas on the vertical.
    fig, ax = plt.subplots(figsize=figsize, dpi=110)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    extent = [betas[0], betas[-1], gammas[0], gammas[-1]]
    # transpose so that x = beta, y = gamma
    im = ax.imshow(
        ratios.T,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap=CONTOUR_CMAP,
        vmin=float(np.min(ratios)),
        vmax=float(np.max(ratios)),
    )
    cs = ax.contour(
        betas, gammas, ratios.T, levels=12, colors="white", linewidths=0.4, alpha=0.35
    )
    ax.clabel(cs, inline=True, fontsize=6, fmt="%.2f")

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("approximation ratio  ⟨C⟩ / C_max", color=TEXT, fontsize=9)
    cbar.ax.yaxis.set_tick_params(color=TEXT)
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color=TEXT)

    # --- extrema markers ---------------------------------------------------
    minima = detect_local_minima(ratios, gammas, betas)
    if minima:
        ax.scatter(
            [m["beta"] for m in minima],
            [m["gamma"] for m in minima],
            s=90, facecolors="none", edgecolors=MIN_COLOR, linewidths=1.6,
            label=f"local minima ({len(minima)})", zorder=5,
        )
    if show_maxima:
        maxima = detect_local_maxima(ratios, gammas, betas)
        if maxima:
            ax.scatter(
                [m["beta"] for m in maxima], [m["gamma"] for m in maxima],
                s=70, facecolors="none", edgecolors=MAX_COLOR, linewidths=1.4,
                label=f"local maxima ({len(maxima)})", zorder=5,
            )
    gi, bj = np.unravel_index(int(np.argmax(ratios)), ratios.shape)
    ax.scatter(
        [betas[bj]], [gammas[gi]], marker="*", s=340, color=OPT_COLOR,
        edgecolors="black", linewidths=0.8, label="global optimum", zorder=6,
    )

    ax.set_xlabel("β  (mixer angle, rad)", color=TEXT, fontsize=10)
    ax.set_ylabel("γ  (cost angle, rad)", color=TEXT, fontsize=10)
    ax.set_title(title, color=TEXT, fontsize=12, pad=10)
    ax.tick_params(colors=TEXT, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    leg = ax.legend(loc="upper right", fontsize=7, framealpha=0.25, facecolor=PANEL)
    for t in leg.get_texts():
        t.set_color(TEXT)
    fig.tight_layout()
    return fig


def plot_graph(
    G: nx.Graph,
    best_bitstring: Optional[str] = None,
    title: Optional[str] = None,
    figsize=(4.2, 3.6),
    seed: int = 7,
) -> plt.Figure:
    """NetworkX drawing of the instance, coloured by a candidate cut.

    If ``best_bitstring`` is given (vertex order: char ``i`` = side of vertex
    ``i``), nodes are coloured by their partition and *cut* edges are drawn as
    solid accent lines while uncut edges are dimmed — so the achieved cut is
    visible at a glance.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=110)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL)

    n = G.number_of_nodes()
    pos = nx.spring_layout(G, seed=seed, k=1.6 / max(1, np.sqrt(n)))

    side = None
    if best_bitstring and len(best_bitstring) == n:
        side = [int(c) for c in best_bitstring]

    if side is not None:
        node_colors = [ACCENT if s == 0 else "#FF6B6B" for s in side]
        cut_edges = [(u, v) for (u, v) in G.edges() if side[u] != side[v]]
        other_edges = [(u, v) for (u, v) in G.edges() if side[u] == side[v]]
        nx.draw_networkx_edges(G, pos, edgelist=other_edges, edge_color=GRID,
                               width=1.0, alpha=0.7, ax=ax)
        nx.draw_networkx_edges(G, pos, edgelist=cut_edges, edge_color=OPT_COLOR,
                               width=2.2, alpha=0.95, ax=ax)
    else:
        node_colors = [ACCENT] * n
        nx.draw_networkx_edges(G, pos, edge_color=GRID, width=1.4, alpha=0.85, ax=ax)

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=340,
                           edgecolors=TEXT, linewidths=0.8, ax=ax)
    nx.draw_networkx_labels(G, pos, font_color="#0A0A0A", font_size=8,
                            font_weight="bold", ax=ax)

    ax.set_axis_off()
    if title:
        ax.set_title(title, color=TEXT, fontsize=10)
    fig.tight_layout()
    return fig


def plot_landscape_contour_plotly(
    gammas: np.ndarray,
    betas: np.ndarray,
    ratios: np.ndarray,
    title: str = "Approximation-Ratio Landscape",
    trajectory: Optional[Sequence] = None,
    show_maxima: bool = False,
) -> go.Figure:
    """Interactive plotly filled-contour of the landscape (page left column).

    ``trajectory`` (optional) is a list of ``(gamma, beta, ratio)`` tuples from
    the optimiser, overlaid as a dashed path with a marker at the final point.
    """
    ratios = np.asarray(ratios, dtype=float)
    fig = go.Figure()

    fig.add_trace(
        go.Contour(
            x=betas,
            y=gammas,
            z=ratios.T,  # x = beta, y = gamma
            colorscale=CONTOUR_CMAP,
            contours=dict(
                coloring="fill",
                showlines=True,
                start=float(ratios.min()),
                end=float(ratios.max()),
                size=(float(ratios.max()) - float(ratios.min())) / 16.0,
            ),
            line=dict(width=0.6, color="rgba(255,255,255,0.25)"),
            colorbar=dict(
                title=dict(text="ratio", font=dict(color=TEXT)),
                tickfont=dict(color=TEXT),
                len=0.9,
            ),
            name="ratio",
        )
    )

    minima = detect_local_minima(ratios, gammas, betas)
    if minima:
        fig.add_trace(
            go.Scatter(
                x=[m["beta"] for m in minima],
                y=[m["gamma"] for m in minima],
                mode="markers",
                marker=dict(size=11, color="rgba(0,0,0,0)", line=dict(color=MIN_COLOR, width=2)),
                name=f"local minima ({len(minima)})",
                hovertemplate="γ=%{y:.3f}<br>β=%{x:.3f}<extra>local min</extra>",
            )
        )
    if show_maxima:
        maxima = detect_local_maxima(ratios, gammas, betas)
        if maxima:
            fig.add_trace(
                go.Scatter(
                    x=[m["beta"] for m in maxima],
                    y=[m["gamma"] for m in maxima],
                    mode="markers",
                    marker=dict(size=9, color="rgba(0,0,0,0)", line=dict(color=MAX_COLOR, width=2)),
                    name=f"local maxima ({len(maxima)})",
                )
            )

    gi, bj = np.unravel_index(int(np.argmax(ratios)), ratios.shape)
    fig.add_trace(
        go.Scatter(
            x=[betas[bj]], y=[gammas[gi]], mode="markers",
            marker=dict(size=20, color=OPT_COLOR, symbol="star",
                        line=dict(color="black", width=1)),
            name="global optimum",
            hovertemplate="γ*=%{y:.3f}<br>β*=%{x:.3f}<br>ratio=%{customdata:.4f}<extra>optimum</extra>",
            customdata=[float(ratios[gi, bj])],
        )
    )

    if trajectory:
        pts = [(t[0], t[1]) for t in trajectory]
        if pts:
            fig.add_trace(
                go.Scatter(
                    x=[p[1] for p in pts],  # beta on x
                    y=[p[0] for p in pts],  # gamma on y
                    mode="lines+markers",
                    line=dict(color="#FFFFFF", width=1.4, dash="dot"),
                    marker=dict(size=3.5, color="#FFFFFF"),
                    name="optimizer path",
                    opacity=0.8,
                    hoverinfo="skip",
                )
            )

    fig.update_layout(
        xaxis=dict(title="β (mixer, rad)", color=TEXT, gridcolor=GRID, zerolinecolor=GRID),
        yaxis=dict(title="γ (cost, rad)", color=TEXT, gridcolor=GRID, zerolinecolor=GRID),
        legend=dict(font=dict(color=TEXT, size=10), bgcolor="rgba(0,0,0,0)"),
        height=520,
    )
    return _dark_layout(fig, title)


def plot_landscape_3d(
    gammas: np.ndarray,
    betas: np.ndarray,
    ratios: np.ndarray,
    title: str = "3-D Landscape Surface",
    show_minima: bool = True,
) -> go.Figure:
    """Interactive plotly 3-D surface of the landscape (page right column).

    Gold star = global optimum; cyan circles = local minima, lifted slightly
    above the surface so they read clearly.
    """
    ratios = np.asarray(ratios, dtype=float)
    fig = go.Figure()

    fig.add_trace(
        go.Surface(
            x=betas,
            y=gammas,
            z=ratios.T,
            colorscale=SURFACE_CMAP,
            opacity=0.95,
            colorbar=dict(title=dict(text="ratio", font=dict(color=TEXT)),
                          tickfont=dict(color=TEXT), len=0.7),
            name="ratio",
            contours=dict(
                z=dict(show=True, usecolormap=True, highlightcolor=OPT_COLOR,
                       project_z=True)
            ),
        )
    )

    zmin = float(ratios.min())
    zspan = float(ratios.max()) - zmin + 1e-9
    if show_minima:
        minima = detect_local_minima(ratios, gammas, betas)
        if minima:
            fig.add_trace(
                go.Scatter3d(
                    x=[m["beta"] for m in minima],
                    y=[m["gamma"] for m in minima],
                    z=[zmin - 0.06 * zspan for _ in minima],
                    mode="markers",
                    marker=dict(size=4.5, color=MIN_COLOR, symbol="circle",
                                line=dict(color="white", width=0.5)),
                    name=f"local minima ({len(minima)})",
                    hovertemplate="γ=%{y:.3f}<br>β=%{x:.3f}<extra>min</extra>",
                )
            )

    gi, bj = np.unravel_index(int(np.argmax(ratios)), ratios.shape)
    fig.add_trace(
        go.Scatter3d(
            x=[betas[bj]], y=[gammas[gi]], z=[float(ratios[gi, bj])],
            mode="markers",
            marker=dict(size=9, color=OPT_COLOR, symbol="diamond",
                        line=dict(color="black", width=1)),
            name="global optimum",
            hovertemplate="γ*=%{y:.3f}<br>β*=%{x:.3f}<br>ratio=%{z:.4f}<extra>optimum</extra>",
        )
    )

    fig.update_layout(
        scene=dict(
            xaxis=dict(title="β", color=TEXT, gridcolor=GRID, backgroundcolor=PANEL,
                       showbackground=True),
            yaxis=dict(title="γ", color=TEXT, gridcolor=GRID, backgroundcolor=PANEL,
                       showbackground=True),
            zaxis=dict(title="ratio", color=TEXT, gridcolor=GRID, backgroundcolor=PANEL,
                       showbackground=True),
            camera=dict(eye=dict(x=1.5, y=-1.6, z=0.9)),
        ),
        legend=dict(font=dict(color=TEXT, size=10), bgcolor="rgba(0,0,0,0)"),
        height=520,
    )
    return _dark_layout(fig, title)


def plot_trajectories(
    gammas: np.ndarray,
    betas: np.ndarray,
    ratios: np.ndarray,
    all_trajectories: List[Sequence],
    title: str = "Optimizer restarts over the landscape",
) -> go.Figure:
    """Contour of the landscape with every optimiser restart path overlaid.

    ``all_trajectories`` is a list (one per restart) of ``(gamma, beta, ratio)``
    sequences.  Each restart is a distinct translucent line; its final point is
    marked.  Helps show how multi-start escapes local minima.
    """
    fig = plot_landscape_contour_plotly(
        gammas, betas, ratios, title=title, show_maxima=False
    )
    palette = ["#00B4D8", "#FF6B6B", "#FFD700", "#90BE6D", "#F9C74F", "#B388EB", "#43AA8B"]
    for r, traj in enumerate(all_trajectories or []):
        if not traj:
            continue
        pts = [(t[0], t[1]) for t in traj]
        col = palette[r % len(palette)]
        fig.add_trace(
            go.Scatter(
                x=[p[1] for p in pts], y=[p[0] for p in pts],
                mode="lines",
                line=dict(color=col, width=1.6),
                opacity=0.75,
                name=f"restart {r + 1}",
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[pts[-1][1]], y=[pts[-1][0]],
                mode="markers",
                marker=dict(size=8, color=col, symbol="x", line=dict(color="white", width=1)),
                name=f"end {r + 1}",
                showlegend=False,
                hoverinfo="skip",
            )
        )
    fig.update_layout(height=520)
    return fig


# ---------------------------------------------------------------------------
# Analysis line charts (matplotlib — used on-screen AND embedded in the PDF
# report, so a single figure serves both).  Dark theme to match the app.
# ---------------------------------------------------------------------------
PALETTE = ["#00B4D8", "#FF6B6B", "#FFD700", "#90BE6D", "#F9C74F",
           "#B388EB", "#43AA8B", "#FF9F1C", "#6EC1E4", "#E86FA9"]

#: Environment → colour, so an environment keeps one colour across every chart.
ENV_COLORS = {
    "ideal": "#9AA0A6",
    "earth-lab": "#00B4D8",
    "leo-satellite": "#FFD700",
    "satellite-ground": "#90BE6D",
    "meo-van-allen": "#FF9F1C",
    "deep-space": "#FF6B6B",
    "custom": "#B388EB",
}


def _style_ax(ax, xlabel: str, ylabel: str, title: str):
    ax.set_facecolor(BG)
    ax.set_xlabel(xlabel, color=TEXT, fontsize=10)
    ax.set_ylabel(ylabel, color=TEXT, fontsize=10)
    ax.set_title(title, color=TEXT, fontsize=12, pad=10)
    ax.tick_params(colors=TEXT, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.grid(True, color=GRID, alpha=0.35, linewidth=0.6)
    return ax


def _dark_fig(figsize=(7.0, 4.2)):
    fig, ax = plt.subplots(figsize=figsize, dpi=110)
    fig.patch.set_facecolor(BG)
    return fig, ax


def plot_noise_sweep(noise_levels, best_ratios, result=None, title="Best ratio vs noise level",
                     env_name="", figsize=(7.2, 4.4)):
    """Resonance-detector chart: data + monotone/peak model fits + verdict.

    ``result`` is the dict from :func:`utils.analysis.detect_resonance` (optional).
    """
    noise_levels = np.asarray(noise_levels, dtype=float)
    best_ratios = np.asarray(best_ratios, dtype=float)
    fig, ax = _dark_fig(figsize)
    _style_ax(ax, "noise level  p", "best approximation ratio", title)

    ax.plot(noise_levels, best_ratios, "o-", color=ACCENT, lw=2.0, ms=5,
            label="measured best ratio", zorder=4)

    if result is not None:
        xs = np.linspace(noise_levels.min(), noise_levels.max(), 200)
        mono, peak = result.get("monotone"), result.get("peak")
        if mono is not None:
            ax.plot(xs, mono["predict"](xs), "--", color="#9AA0A6", lw=1.5,
                    label=f"monotone fit (BIC {mono['bic']:.0f})", zorder=3)
        if peak is not None:
            ax.plot(xs, peak["predict"](xs), "-.", color=MAX_COLOR, lw=1.6,
                    label=f"peak fit (BIC {peak['bic']:.0f})", zorder=3)
        if result.get("resonance"):
            ps = result["p_star"]
            ax.axvline(ps, color=OPT_COLOR, ls=":", lw=1.6, zorder=2)
            ax.scatter([ps], [peak["predict"](ps)], marker="*", s=320, color=OPT_COLOR,
                       edgecolors="black", linewidths=0.8, zorder=6,
                       label=f"resonance p*={ps:.4f}")
            verdict = (f"Stochastic resonance at p*={ps:.4f}\n"
                       f"improvement {result['improvement_pct']:.2f}%  (ΔBIC {result['delta_bic']:+.1f})")
            vcol = OPT_COLOR
        else:
            verdict = f"No resonance — monotone degradation\n(ΔBIC {result['delta_bic']:+.1f} favours decay)"
            vcol = "#FF6B6B"
        ax.text(0.025, 0.045, verdict, transform=ax.transAxes, color=vcol,
                fontsize=9, va="bottom", ha="left",
                bbox=dict(boxstyle="round,pad=0.4", facecolor=PANEL, edgecolor=vcol, alpha=0.9))

    if env_name:
        ax.text(0.985, 0.97, env_name, transform=ax.transAxes, color=TEXT, fontsize=9,
                ha="right", va="top", alpha=0.8)
    leg = ax.legend(loc="upper right", fontsize=7.5, framealpha=0.3, facecolor=PANEL)
    for t in leg.get_texts():
        t.set_color(TEXT)
    fig.tight_layout()
    return fig


def plot_cross_sections(betas, rows_by_level, title="β-cross-section at γ*", figsize=(7.2, 4.4)):
    """Overlaid ratio-vs-β curves, one per noise level (coloured by level).

    ``rows_by_level`` is a list of ``(noise_level, ratio_row)`` tuples sharing the
    same ``betas`` axis.
    """
    betas = np.asarray(betas, dtype=float)
    fig, ax = _dark_fig(figsize)
    _style_ax(ax, "β  (mixer angle, rad)", "approximation ratio", title)

    levels = [float(lv) for lv, _ in rows_by_level]
    lmin, lmax = (min(levels), max(levels)) if levels else (0.0, 1.0)
    cmap = plt.get_cmap("plasma")
    for (lv, row) in rows_by_level:
        t = 0.0 if lmax <= lmin else (float(lv) - lmin) / (lmax - lmin)
        ax.plot(betas, np.asarray(row, dtype=float), color=cmap(0.15 + 0.75 * t),
                lw=1.9, label=f"p={lv:.3f}")
    leg = ax.legend(loc="best", fontsize=7.5, ncol=2, framealpha=0.3, facecolor=PANEL,
                    title="noise level")
    for t in leg.get_texts():
        t.set_color(TEXT)
    leg.get_title().set_color(TEXT)
    fig.tight_layout()
    return fig


def plot_env_comparison(noise_levels, curves_by_env, title="Best ratio vs noise · environments",
                        figsize=(7.6, 4.6)):
    """One line per environment: best approximation ratio vs noise level."""
    noise_levels = np.asarray(noise_levels, dtype=float)
    fig, ax = _dark_fig(figsize)
    _style_ax(ax, "noise level  p", "best approximation ratio", title)

    for i, (env, ys) in enumerate(curves_by_env.items()):
        ys = np.asarray(ys, dtype=float)
        col = ENV_COLORS.get(env, PALETTE[i % len(PALETTE)])
        ax.plot(noise_levels[:len(ys)], ys, "o-", color=col, lw=2.0, ms=4.5, label=env)
    leg = ax.legend(loc="best", fontsize=8, framealpha=0.3, facecolor=PANEL)
    for t in leg.get_texts():
        t.set_color(TEXT)
    fig.tight_layout()
    return fig


def plot_env_retention_bars(ranking, title="Robustness ranking (ratio retained at max noise)",
                            figsize=(7.4, 4.2)):
    """Horizontal bars of the fraction of the noiseless ratio each env retains."""
    fig, ax = _dark_fig(figsize)
    envs = [r["environment"] for r in ranking][::-1]
    vals = [r["retention"] * 100.0 for r in ranking][::-1]
    cols = [ENV_COLORS.get(e, ACCENT) for e in envs]
    ax.set_facecolor(BG)
    bars = ax.barh(envs, vals, color=cols, edgecolor=GRID, alpha=0.92)
    ax.set_xlabel("% of noiseless best ratio retained", color=TEXT, fontsize=10)
    ax.set_title(title, color=TEXT, fontsize=12, pad=10)
    ax.tick_params(colors=TEXT, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.grid(True, axis="x", color=GRID, alpha=0.35, linewidth=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_width() + 0.6, b.get_y() + b.get_height() / 2, f"{v:.1f}%",
                va="center", color=TEXT, fontsize=8)
    ax.set_xlim(0, max(105.0, max(vals, default=100.0) + 8))
    fig.tight_layout()
    return fig
