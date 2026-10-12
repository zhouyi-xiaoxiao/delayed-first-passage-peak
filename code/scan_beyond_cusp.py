#!/usr/bin/env python3
"""Scan for two interior maxima for release farther from a midpoint sink than the cusp (App. D).

Grid: 39 release points xi in [0.01, 0.377] (0.01, 0.02, ..., 0.36 and 0.372, 0.375, 0.377) and
400 log-spaced sink strengths b in [0.03, 300]; 2x10^4 log-spaced time points per density, from
tau_min = min(xi, 1/2 - xi)^2/40 to tau = 2, with the mode sum truncated where mu tau_min > 45.
A density with two interior maxima would show up as two sign changes + -> - of Phi'.
Controls inside the two-peak region (midpoints between the stored fold lines,
data/dpma_offgate_dense.csv) must be detected.
Writes outputs/tables/scan_beyond_cusp.txt.  About 1-2 minutes.
"""
from __future__ import annotations

import csv
import math
import time

import numpy as np
from scipy.optimize import brentq

from paths import DATA, TABLES

LINES: list[str] = []


def say(s=""):
    print(s, flush=True)
    LINES.append(s)


def roots_midpoint(b, kmax, step=0.004):
    """All roots k <= kmax of k sin k + 2 b sin^2(k/2) (theta = 1/2), node modes included."""
    ks = np.arange(1e-7, kmax, step)
    D = ks * np.sin(ks) + 2 * b * np.sin(ks / 2) ** 2
    idx = np.where(np.sign(D[:-1]) * np.sign(D[1:]) < 0)[0]
    f = lambda k: k * math.sin(k) + 2 * b * math.sin(k / 2) ** 2
    return np.array([brentq(f, ks[i], ks[i + 1], xtol=1e-14, rtol=1e-15) for i in idx])


def nmax_scan(xi, b, k, ntau=20000):
    """Number of interior maxima of Phi(tau) for release at xi (theta = 1/2)."""
    tmin = min(xi, 0.5 - xi) ** 2 / 40
    kmax = math.sqrt(2 * 45 / tmin)
    kk = k[k <= kmax]
    kk = kk[np.abs(np.sin(kk / 2)) > 1e-9]          # node modes carry no weight
    G = 2 * kk * (1 - np.cos(kk / 2)) * np.sin(kk * xi) / (np.sin(kk / 2) ** 2 * (1 + b * (b + 2) / kk ** 2))
    mm = kk ** 2 / 2
    taus = np.geomspace(tmin, 2.0, ntau)
    E = np.exp(-np.outer(taus, mm))
    with np.errstate(all="ignore"):   # some BLAS builds raise spurious flags on underflow
        ph = E @ G
        d1 = -(E @ (G * mm))
    good = ph > 1e-8 * ph.max()
    d1 = d1[good]
    sc = np.where(np.sign(d1[:-1]) * np.sign(d1[1:]) < 0)[0]
    return sum(1 for i in sc if d1[i] > 0)


def controls():
    """Midpoints between the stored fold lines at a few release points inside the region."""
    rows = list(csv.DictReader(open(DATA / "dpma_offgate_dense.csv", newline="")))
    out = []
    for target in (0.381, 0.385, 0.39, 0.40, 0.42, 0.44):
        pair = {}
        for branch in ("lower", "upper"):
            cand = [(abs(float(r["xi"]) - target), float(r["xi"]), float(r["b"]))
                    for r in rows if r["branch"] == branch and r["param"] != "mirror"]
            _, x, b = min(cand)
            pair[branch] = (x, b)
        xl, bl = pair["lower"]
        xu, bu = pair["upper"]
        if abs(xl - xu) < 1e-9 and abs(xl - target) < 1e-6:
            out.append((xl, bl, bu, math.sqrt(bl * bu)))
    return out


def main():
    t0 = time.time()
    k_cache = {}

    def k_for(b):
        if b not in k_cache:
            k_cache[b] = roots_midpoint(b, 4500.0)
        return k_cache[b]

    say("Controls inside the two-peak region (theta = 1/2; geometric mean of the stored fold lines):")
    for xi, bl, bu, bm in controls():
        n = nmax_scan(xi, bm, roots_midpoint(bm, 4500.0))
        say(f"   xi={xi:.3f}: b_-={bl:.5g}, b_+={bu:.5f}; at b={bm:.5g}: interior maxima = {n}")
        if n != 2:
            raise RuntimeError(f"control inside the region not detected at xi={xi}")
    bs = np.geomspace(0.03, 300, 400)
    xis = np.round(np.concatenate([np.arange(0.01, 0.37, 0.01), [0.372, 0.375, 0.377]]), 4)
    hits = []
    for b in bs:
        k = k_for(b)
        for xi in xis:
            if nmax_scan(xi, b, k) >= 2:
                hits.append((xi, b))
        k_cache.pop(b, None)
    say(f"scan: {len(xis)} xi values in [{xis.min():g},{xis.max():g}] x {len(bs)} b values in "
        f"[0.03,300], 20000 tau points: cases with >= 2 interior maxima = {len(hits)}")
    if hits:
        say(f"   first cases: {hits[:10]}")
    say(f"time {time.time() - t0:.0f} s")
    (TABLES / "scan_beyond_cusp.txt").write_text("\n".join(LINES) + "\n")
    if hits:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
