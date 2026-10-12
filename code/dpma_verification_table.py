#!/usr/bin/env python3
"""First three rows of Table II: point-sink jump, closed forms of I and J, normalization identity.

The checks use the first six affected roots for three (theta,b) pairs. Closed-form I and J are
compared with 50-digit numerical quadrature; the other residuals use float64, as in the spectral
code. Writes outputs/tables/dpma_verification_table.txt.
"""
from __future__ import annotations

import math

from mpmath import mp
import numpy as np

from paths import TABLES


def roots_D(theta, b, wmax=40.0, n=80001):
    k = np.linspace(1e-6, 2 * wmax, n)
    values = k * np.sin(k) + 2 * b * np.sin(k * theta) * np.sin(k * (1 - theta))
    indices = np.where(np.sign(values[:-1]) * np.sign(values[1:]) < 0)[0]
    lo, hi, flo = k[indices], k[indices + 1], values[indices]
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        fmid = mid * np.sin(mid) + 2 * b * np.sin(mid * theta) * np.sin(mid * (1 - theta))
        left = flo * fmid <= 0
        hi = np.where(left, mid, hi)
        lo = np.where(left, lo, mid)
        flo = np.where(left, flo, fmid)
    return 0.5 * (lo + hi)


def main():
    mp.dps = 50
    worst_jump = 0.0
    worst_quadrature = 0.0
    worst_normalization = 0.0
    checked = 0

    for theta, b in ((0.5, 3.0), (0.38, 2.0), (0.27, 4.0)):
        for k in roots_D(theta, b)[:6]:
            A = math.sin(k * theta)
            B = math.sin(k * (1 - theta))
            worst_jump = max(worst_jump, abs(-k * math.sin(k) - 2 * b * A * B))

            I = (B * (1 - math.cos(k * theta)) + A * (1 - math.cos(k * (1 - theta)))) / k
            J = 0.5 * (theta * B**2 + (1 - theta) * A**2) - A * B * math.sin(k) / (2 * k)
            kk = mp.mpf(repr(float(k)))
            th = mp.mpf(repr(float(theta)))

            def eigenfunction(x):
                if x <= th:
                    return mp.sin(kk * (1 - th)) * mp.sin(kk * x)
                return mp.sin(kk * th) * mp.sin(kk * (1 - x))

            I_quad = mp.quad(eigenfunction, [0, th, 1])
            J_quad = mp.quad(lambda x: eigenfunction(x) ** 2, [0, th, 1])
            worst_quadrature = max(
                worst_quadrature,
                float(abs(mp.mpf(I) - I_quad)),
                float(abs(mp.mpf(J) - J_quad)),
            )

            Dk = math.sin(k) + k * math.cos(k) + 2 * b * (
                theta * math.cos(k * theta) * B
                + (1 - theta) * A * math.cos(k * (1 - theta))
            )
            worst_normalization = max(
                worst_normalization,
                abs(J - math.sin(k) * Dk / (4 * b)),
            )
            checked += 1

    lines = [
        "Analytic verification-table generator",
        f"parameter/mode cases checked: {checked}",
        f"max |delta-sink jump - secular rhs| = {worst_jump:.16e}",
        f"max |closed I,J - 50-dps quadrature| = {worst_quadrature:.16e}",
        f"max |J - sin(k) D_k/(4b)| = {worst_normalization:.16e}",
    ]
    if not (checked == 18 and worst_jump < 6e-14 and worst_quadrature < 3e-16
            and worst_normalization < 6e-15):
        raise RuntimeError(
            "analytic verification regression failed: "
            f"checked={checked}, jump={worst_jump}, quadrature={worst_quadrature}, "
            f"normalization={worst_normalization}"
        )
    print("\n".join(lines), flush=True)
    (TABLES / "dpma_verification_table.txt").write_text("\n".join(lines) + "\n")
    print("wrote outputs/tables/dpma_verification_table.txt", flush=True)


if __name__ == "__main__":
    main()
