#!/usr/bin/env python3
"""Shared fixed-window observables for the off-gate measurement protocol.

The experimental estimator is a ratio of *window-averaged* densities.  Keeping
that definition in one module prevents a point-value spectral comparator from
being mixed with histogram counts.  The two reference windows used in the
paper are disjoint, so their counting uncertainty follows directly from a
multinomial delta-method calculation.
"""
from __future__ import annotations

from collections.abc import Callable
import math

import numpy as np


Window = tuple[float, float]


def _validated_window(window: Window) -> tuple[float, float]:
    lo, hi = map(float, window)
    if not (math.isfinite(lo) and math.isfinite(hi) and 0.0 <= lo < hi):
        raise ValueError(f"invalid observation window: {window!r}")
    return lo, hi


def window_probability(
    density: Callable[[np.ndarray], np.ndarray],
    window: Window,
    *,
    quadrature_order: int = 160,
) -> float:
    """Integrate a vectorized density over ``window`` by Gauss--Legendre quadrature."""
    lo, hi = _validated_window(window)
    if quadrature_order < 8:
        raise ValueError("quadrature_order must be at least 8")
    nodes, weights = np.polynomial.legendre.leggauss(int(quadrature_order))
    tau = 0.5 * (hi - lo) * nodes + 0.5 * (hi + lo)
    values = np.asarray(density(tau), dtype=float)
    if values.shape != tau.shape or not np.all(np.isfinite(values)):
        raise ValueError("density must return one finite value per quadrature node")
    return float(0.5 * (hi - lo) * np.dot(weights, values))


def window_average(
    density: Callable[[np.ndarray], np.ndarray],
    window: Window,
    *,
    quadrature_order: int = 160,
) -> float:
    """Return the mean density over a fixed observation window."""
    lo, hi = _validated_window(window)
    return window_probability(density, (lo, hi), quadrature_order=quadrature_order) / (hi - lo)


def exact_window_ratio(
    density: Callable[[np.ndarray], np.ndarray],
    peak_window: Window,
    valley_window: Window,
    *,
    quadrature_order: int = 160,
) -> tuple[float, float, float]:
    """Return ``(R, p_peak, p_valley)`` for the same statistic used by a histogram."""
    plo, phi = _validated_window(peak_window)
    vlo, vhi = _validated_window(valley_window)
    if max(plo, vlo) < min(phi, vhi):
        raise ValueError("peak and valley windows must be disjoint for the stated uncertainty law")
    p_peak = window_probability(density, (plo, phi), quadrature_order=quadrature_order)
    p_valley = window_probability(density, (vlo, vhi), quadrature_order=quadrature_order)
    if p_peak <= 0.0 or p_valley <= 0.0:
        raise ValueError("window probabilities must be positive")
    ratio = (p_peak / (phi - plo)) / (p_valley / (vhi - vlo))
    return float(ratio), float(p_peak), float(p_valley)


def sample_window_ratio(
    first_passage_times: np.ndarray,
    peak_window: Window,
    valley_window: Window,
) -> float:
    """Apply the fixed-window density-ratio estimator to simulated times."""
    p_lo, p_hi = _validated_window(peak_window)
    v_lo, v_hi = _validated_window(valley_window)
    times = np.asarray(first_passage_times, dtype=float)
    p_density = np.count_nonzero((times >= p_lo) & (times <= p_hi)) / (p_hi - p_lo)
    v_density = np.count_nonzero((times >= v_lo) & (times <= v_hi)) / (v_hi - v_lo)
    return float(p_density / v_density) if v_density > 0 else float("nan")


def multinomial_ratio_sd(ratio: float, p_peak: float, p_valley: float, n_trials: int) -> float:
    r"""Delta-method s.d. for two disjoint multinomial windows.

    For ``R=(p_peak/width_peak)/(p_valley/width_valley)``, the covariance
    between the two counts cancels the two binomial ``-1/N`` terms, leaving

    ``Var(R) = R^2 (1/p_peak + 1/p_valley) / N``.
    """
    if not (ratio > 0.0 and p_peak > 0.0 and p_valley > 0.0 and n_trials > 0):
        raise ValueError("ratio, probabilities, and n_trials must be positive")
    return float(ratio * math.sqrt((1.0 / p_peak + 1.0 / p_valley) / int(n_trials)))


def trials_for_one_sigma(
    ratio: float,
    p_peak: float,
    p_valley: float,
    slope: float,
    target_delta_b: float,
) -> float:
    """Trials needed for one-standard-deviation precision ``target_delta_b``."""
    if slope == 0.0 or target_delta_b <= 0.0:
        raise ValueError("slope and target_delta_b must be nonzero/positive")
    target_sd = abs(float(slope)) * float(target_delta_b)
    return float(ratio**2 * (1.0 / p_peak + 1.0 / p_valley) / target_sd**2)
