#!/usr/bin/env python3
"""Fold of the K-mode truncation of the midpoint release-at-sink density (theta = xi = 1/2).

Retaining the first K sink-coupled modes: no fold for K = 2, b_c^(3) = 3.046, b_c^(4) = 3.0766 and
b_c^(5) = 3.0764 (the converged value).  Also prints the number of sign changes of S1 (critical
points of Phi) on tau in [1e-3, 0.3] for several b, and checks whether the truncated sum becomes
negative at short times.  mpmath, 30 digits.
Writes outputs/tables/mode_truncation_check.txt.
"""
from __future__ import annotations

import mpmath as mp
import numpy as np

from paths import TABLES

mp.mp.dps = 30
LINES: list[str] = []


def say(s=""):
    print(s, flush=True)
    LINES.append(s)


def modes(b, K):
    ks = []
    for j in range(1, K + 1):
        f = lambda k: k * mp.cos(k / 2) + b * mp.sin(k / 2)
        ks.append(mp.findroot(f, ((2 * j - 1) * mp.pi + 1e-12, 2 * j * mp.pi - 1e-12), solver="anderson"))
    out = []
    for k in ks:
        # xi = 1/2: sin(k xi) = sin(k/2) in the midpoint amplitude
        G = 2 * k * (1 - mp.cos(k / 2)) * mp.sin(k / 2) / (mp.sin(k / 2) ** 2 * (1 + b * (b + 2) / k**2))
        out.append((k**2 / 2, G))
    return out


def S(n, tau, md):
    return sum(G * mu**n * mp.e ** (-mu * tau) for mu, G in md)


def crit(b, K, lo=1e-3, hi=0.3, n=3000):
    md = modes(b, K)
    t = np.exp(np.linspace(np.log(lo), np.log(hi), n))
    s = [float(S(1, x, md)) for x in t]
    return sum(1 for i in range(n - 1) if s[i] * s[i + 1] < 0)


def min_density(b, K, lo=1e-4, hi=0.3, n=400):
    md = modes(b, K)
    t = np.exp(np.linspace(np.log(lo), np.log(hi), n))
    return min(float(S(0, x, md)) for x in t)


def main():
    say("K-mode truncation of the midpoint density (theta = xi = 1/2)")
    for K in (2, 3, 4, 5):
        try:
            sol = mp.findroot(lambda tau, b: [S(1, tau, modes(b, K)), S(2, tau, modes(b, K))],
                              (0.0384, 3.07))
            say(f"K={K}: fold near the midpoint threshold: tau={float(sol[0]):.6f} b={float(sol[1]):.6f}")
        except Exception as exc:  # no root near the converged fold
            say(f"K={K}: no fold found near (tau, b) = (0.038, 3.07) ({type(exc).__name__})")
        counts = [crit(b, K) for b in (0.5, 1.0, 2.0, 3.0, 3.1)]
        say(f"   sign changes of S1 on [1e-3, 0.3] for b = 0.5, 1, 2, 3, 3.1: {counts}")
        say(f"   minimum of the truncated density on [1e-4, 0.3] at b = 3: {min_density(3.0, K):.4g}")
    (TABLES / "mode_truncation_check.txt").write_text("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
