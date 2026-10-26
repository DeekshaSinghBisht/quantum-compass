"""Pure analysis helpers for the Noise Analysis and Space Mode pages.

Everything here is deterministic numeric post-processing of already-computed
landscapes — **no Qiskit, no Streamlit, no simulation** — so it is safe to call
from the Streamlit parent process and easy to unit-test.

The headline tool is :func:`detect_resonance`, which decides whether the best
approximation ratio has a genuine interior *peak* as a function of noise level
("stochastic resonance": a little noise helps) or simply decays monotonically
(noise only hurts).  It does this by fitting two competing models and comparing
them with AIC/BIC:

* **monotone** – ``r(p) = C + A·exp(-B·p)``  (A, B > 0): a smooth decay,
* **peak**     – ``r(p) = C + A·exp(-(p-p*)² / 2σ²)``: a Gaussian bump with an
  interior optimum ``p*``.

Whichever model explains the data better *after penalising its extra parameter*
wins.  A peak is only declared when it wins on BIC, its centre ``p*`` is strictly
interior, and it actually lifts the ratio above the noiseless value.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Noise levels used by the resonance detector (spec-defined sweep).
RESONANCE_LEVELS: Tuple[float, ...] = (
    0.0, 0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1,
)


# ---------------------------------------------------------------------------
# Landscape slicing
# ---------------------------------------------------------------------------
def slice_at_gamma(landscape: Dict, gamma_target: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(betas, ratio_row)``: the β-cross-section at a fixed γ.

    The landscape stores ``ratios[i, j]`` at ``(gammas[i], betas[j])``.  We
    linearly interpolate along the γ axis at ``gamma_target`` for every β column,
    so the cut need not land exactly on a grid line.
    """
    gammas = np.asarray(landscape["gammas"], dtype=float)
    betas = np.asarray(landscape["betas"], dtype=float)
    ratios = np.asarray(landscape["ratios"], dtype=float)
    row = np.empty(len(betas), dtype=float)
    for j in range(len(betas)):
        row[j] = np.interp(gamma_target, gammas, ratios[:, j])
    return betas, row


def best_point(landscape: Dict) -> Tuple[float, float, float]:
    """``(best_gamma, best_beta, best_ratio)`` of a landscape dict."""
    return (
        float(landscape["best_gamma"]),
        float(landscape["best_beta"]),
        float(landscape["best_ratio"]),
    )


# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------
def _aic_bic(n: int, rss: float, k: int) -> Tuple[float, float]:
    """Gaussian-likelihood AIC/BIC from a residual sum of squares."""
    rss = max(float(rss), 1e-18)
    core = n * np.log(rss / n)
    return core + 2.0 * k, core + k * np.log(n)


def _fit(fn, x, y, p0, bounds):
    """Least-squares fit via scipy; returns (popt, rss) or (None, inf) on failure."""
    from scipy.optimize import curve_fit

    try:
        popt, _ = curve_fit(fn, x, y, p0=p0, bounds=bounds, maxfev=20000)
        resid = y - fn(x, *popt)
        return popt, float(np.sum(resid ** 2))
    except Exception:
        return None, float("inf")


def fit_monotone(x: np.ndarray, y: np.ndarray) -> Dict:
    """Fit ``r(p) = C + A·exp(-B·p)`` (monotone decay). 3 parameters."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)

    def model(xx, A, B, C):
        return C + A * np.exp(-B * xx)

    ymin, ymax = float(y.min()), float(y.max())
    span = max(ymax - ymin, 1e-6)
    p0 = [span, 1.0 / max(x[-1], 1e-3), ymin]
    bounds = ([0.0, 0.0, ymin - span], [10 * span + 1, 1e4, ymax + span])
    popt, rss = _fit(model, x, y, p0, bounds)
    if popt is None:  # fall back to a straight line through the data
        popt = np.array([span, 1.0 / max(x[-1], 1e-3), ymin])
        rss = float(np.sum((y - model(x, *popt)) ** 2))
    aic, bic = _aic_bic(n, rss, 3)
    A, B, C = (float(v) for v in popt)
    return {
        "model": "monotone", "A": A, "B": B, "C": C, "rss": rss,
        "aic": aic, "bic": bic, "n_params": 3,
        "predict": lambda xx: C + A * np.exp(-B * np.asarray(xx, dtype=float)),
        "r_at_zero": C + A,
    }


def fit_peak(x: np.ndarray, y: np.ndarray) -> Dict:
    """Fit ``r(p) = C + A·exp(-(p-p*)²/2σ²)`` (Gaussian bump). 4 parameters."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)

    def model(xx, A, pstar, sigma, C):
        return C + A * np.exp(-((xx - pstar) ** 2) / (2.0 * sigma ** 2))

    ymin, ymax = float(y.min()), float(y.max())
    span = max(ymax - ymin, 1e-6)
    xr = float(x[-1] - x[0]) or 1.0
    p0 = [span, float(x[int(np.argmax(y))]), xr / 4.0, ymin]
    bounds = ([0.0, float(x[0]), xr / 50.0, ymin - span],
              [10 * span + 1, float(x[-1]), xr * 2.0, ymax + span])
    popt, rss = _fit(model, x, y, p0, bounds)
    if popt is None:
        popt = np.array(p0)
        rss = float(np.sum((y - model(x, *popt)) ** 2))
    aic, bic = _aic_bic(n, rss, 4)
    A, pstar, sigma, C = (float(v) for v in popt)
    return {
        "model": "peak", "A": A, "p_star": pstar, "sigma": sigma, "C": C,
        "rss": rss, "aic": aic, "bic": bic, "n_params": 4,
        "predict": lambda xx: C + A * np.exp(
            -((np.asarray(xx, dtype=float) - pstar) ** 2) / (2.0 * sigma ** 2)),
        "r_at_peak": C + A,
        "r_at_zero": C + A * np.exp(-(pstar ** 2) / (2.0 * sigma ** 2)),
    }


def detect_resonance(noise_levels: Sequence[float], best_ratios: Sequence[float],
                     bic_margin: float = 2.0) -> Dict:
    """Decide whether best-ratio vs noise shows a genuine interior peak.

    Returns a dict with the fitted models, the AIC/BIC comparison, and a verdict::

        {
          'resonance': bool,          # True if a real interior peak wins
          'p_star': float,            # resonant noise level (0.0 if none)
          'improvement_pct': float,   # (r(p*) - r(0)) / r(0) * 100
          'delta_bic': float,         # bic_mono - bic_peak (>0 favours peak)
          'delta_aic': float,
          'winner': 'peak'|'monotone',
          'monotone': {...}, 'peak': {...},
          'reason': str,
        }
    """
    x = np.asarray(noise_levels, dtype=float)
    y = np.asarray(best_ratios, dtype=float)
    mono = fit_monotone(x, y)
    peak = fit_peak(x, y)

    delta_bic = mono["bic"] - peak["bic"]   # > 0 → peak model preferred
    delta_aic = mono["aic"] - peak["aic"]

    interior = float(x[0]) < peak["p_star"] < float(x[-1])
    r0 = float(mono["r_at_zero"])
    rpeak = float(peak["r_at_peak"])
    # Data-driven improvement of the best interior point over the noiseless one.
    y0 = float(y[0])
    ymax_interior = float(np.max(y[1:-1])) if len(y) >= 3 else float(y.max())
    data_lift = (ymax_interior - y0) / y0 * 100.0 if y0 > 0 else 0.0

    resonance = bool(
        delta_bic > bic_margin
        and interior
        and peak["A"] > 1e-4
        and rpeak > mono["predict"](peak["p_star"]) - 1e-9
        and data_lift > 0.25           # the data itself must rise above noiseless
    )

    if resonance:
        reason = (
            f"peak model wins on BIC (Δ={delta_bic:.1f}) with an interior optimum "
            f"p*={peak['p_star']:.4f}"
        )
        improvement = data_lift if data_lift > 0 else (rpeak - r0) / r0 * 100.0
        p_star = float(peak["p_star"])
    else:
        if not interior:
            reason = "best-fit peak sits at the boundary (no interior optimum)"
        elif delta_bic <= bic_margin:
            reason = f"monotone decay fits at least as well (ΔBIC={delta_bic:.1f} ≤ {bic_margin})"
        else:
            reason = "peak does not lift the ratio above the noiseless value"
        improvement = 0.0
        p_star = 0.0

    return {
        "resonance": resonance,
        "p_star": p_star,
        "improvement_pct": float(improvement),
        "delta_bic": float(delta_bic),
        "delta_aic": float(delta_aic),
        "winner": "peak" if resonance else "monotone",
        "monotone": mono,
        "peak": peak,
        "reason": reason,
        "data_lift_pct": float(data_lift),
    }


def degradation_stats(noise_levels: Sequence[float], best_ratios: Sequence[float]) -> Dict:
    """Summarise how fast an environment degrades with noise (monotone case)."""
    x = np.asarray(noise_levels, dtype=float)
    y = np.asarray(best_ratios, dtype=float)
    if len(x) < 2:
        return {"pct_loss_max": 0.0, "slope": 0.0, "half_point": float("nan"),
                "r0": float(y[0]) if len(y) else 0.0, "rmax_noise": float(y[-1]) if len(y) else 0.0}
    r0, rlast = float(y[0]), float(y[-1])
    pct_loss = (r0 - rlast) / r0 * 100.0 if r0 > 0 else 0.0
    slope = float(np.polyfit(x, y, 1)[0])           # dratio/dnoise (linear approx)
    half = float("nan")
    if r0 > rlast:
        target = r0 - 0.5 * (r0 - rlast)
        # noise level where the ratio has fallen halfway to its worst value
        idx = np.where(y <= target)[0]
        if len(idx):
            half = float(x[idx[0]])
    return {
        "pct_loss_max": float(pct_loss), "slope": slope, "half_point": half,
        "r0": r0, "rmax_noise": rlast,
        "best_ratio": float(y.max()), "worst_ratio": float(y.min()),
    }


def rank_environments(curves: Dict[str, Sequence[float]],
                      noise_levels: Sequence[float]) -> List[Dict]:
    """Rank environments by robustness at the highest shared noise level.

    ``curves`` maps ``environment -> best_ratios`` (aligned to ``noise_levels``).
    Returns a list sorted best-first with the retention fraction and total loss.
    """
    rows: List[Dict] = []
    x = np.asarray(noise_levels, dtype=float)
    for env, ys in curves.items():
        y = np.asarray(ys, dtype=float)
        if len(y) == 0:
            continue
        r0 = float(y[0]) if y[0] > 0 else 1e-9
        retention = float(y[-1] / r0)
        rows.append({
            "environment": env,
            "r0": r0,
            "r_final": float(y[-1]),
            "retention": retention,
            "pct_loss": float((1.0 - retention) * 100.0),
            "area": float(np.trapezoid(y, x)) if len(x) == len(y) else float(y.mean()),
        })
    rows.sort(key=lambda d: d["retention"], reverse=True)
    return rows
