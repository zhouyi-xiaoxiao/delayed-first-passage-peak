"""Evaluation of the exact lattice first-passage probabilities F(t) = sum_j A_j s_j^(t-1).

The eigenvalues s_j of the symmetric substochastic matrix P_lambda are real and may be negative
(lazy walk with q > 1/2), so powers are evaluated through |s_j|^(t-1) and the parity of t-1.
Terms whose magnitude is below the double-precision underflow threshold at the earliest
requested time are dropped; they cannot contribute at any later time either.
"""
from __future__ import annotations

import numpy as np


def evaluate_geometric_mixture(s, amplitudes, times, chunk: int = 4096) -> np.ndarray:
    """Return sum_j amplitudes[j] * s[j]**(t-1) for every integer t in ``times`` (t >= 1)."""
    s = np.asarray(s, dtype=float)
    a = np.asarray(amplitudes, dtype=float)
    t = np.asarray(times, dtype=np.int64)
    if t.ndim != 1 or t.size == 0 or np.any(t < 1):
        raise ValueError("times must be a non-empty 1-D array of integers >= 1")
    keep = (a != 0.0) & (s != 0.0)
    s, a = s[keep], a[keep]
    log_abs = np.log(np.abs(s))
    n0 = float(t.min() - 1)
    # drop terms that are already below ~1e-320 relative to their amplitude at the first time
    alive = log_abs * n0 > -737.0
    s, a, log_abs = s[alive], a[alive], log_abs[alive]
    negative = s < 0.0
    out = np.empty(t.size, dtype=float)
    for start in range(0, t.size, chunk):
        n = (t[start:start + chunk] - 1).astype(float)
        magnitude = np.exp(np.outer(n, log_abs))
        sign = np.where(negative[None, :] & (np.mod(n, 2.0)[:, None] == 1.0), -1.0, 1.0)
        with np.errstate(all="ignore"):   # some BLAS builds raise spurious flags on underflow
            out[start:start + chunk] = (magnitude * sign) @ a
    if not np.all(np.isfinite(out)):
        raise FloatingPointError("non-finite lattice first-passage probability")
    return out
