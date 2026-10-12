#!/usr/bin/env python3
"""Continuum spectrum for a midpoint sink (theta = 1/2) and release at the sink (xi = 1/2).

The sink-coupled wavenumbers k = 2w solve tan(w) = -2w/b; modes with a node at the sink carry
zero weight for this release point.  Phi(tau; b) = sum_j G_j exp(-2 w_j^2 tau).
"""
from __future__ import annotations

import math

import numpy as np


Q_DEFAULT = 2.0 / 3.0


def roots_w(b: float, n_modes: int) -> np.ndarray:
    """Roots of ``2 w cos(w) + b sin(w)=0`` in ``((j-1/2)pi,j pi)``."""
    if b <= 0.0 or n_modes <= 0:
        raise ValueError("b and n_modes must be positive")
    roots = np.empty(int(n_modes), dtype=float)

    def secular(w: float) -> float:
        return 2.0 * w * math.cos(w) + b * math.sin(w)

    for j in range(1, int(n_modes) + 1):
        lo = (j - 0.5) * math.pi + 1e-12
        hi = j * math.pi - 1e-12
        f_lo = secular(lo)
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            f_mid = secular(mid)
            if f_lo * f_mid <= 0.0:
                hi = mid
            else:
                lo, f_lo = mid, f_mid
        roots[j - 1] = 0.5 * (lo + hi)
    return roots


def amplitudes(ws: np.ndarray, b: float) -> np.ndarray:
    """Source-started antipodal FPT amplitudes in continuum-density normalization."""
    ws = np.asarray(ws, dtype=float)
    return 4.0 * ws * (1.0 - np.cos(ws)) / (
        np.sin(ws) * (1.0 + b * (b + 2.0) / (4.0 * ws**2))
    )


def density(tau: np.ndarray, b: float, n_modes: int = 400) -> np.ndarray:
    """Evaluate ``Phi(tau;b)=sum_j G_j exp(-2 w_j^2 tau)``."""
    values = np.asarray(tau, dtype=float)
    ws = roots_w(b, n_modes)
    G = amplitudes(ws, b)
    return np.sum(G[None, :] * np.exp(-np.outer(values.reshape(-1), 2.0 * ws**2)), axis=1).reshape(values.shape)
