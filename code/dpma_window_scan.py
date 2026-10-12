#!/usr/bin/env python3
"""Dense scans in b for release at the sink: one interval 0 < b < b_c(theta), no re-entrance.

Spike, valley and delayed peak are expected to coexist on a single interval 0 < b < b_c(theta)
(at b = 0 the single exit-time peak remains).  A bisection on the valley/peak count presupposes a
single transition, so it cannot by itself exclude re-entrance.  This script scans a dense b grid
(step 0.02, from b = 0.05 to b_c + 1.5, i.e. 1.4--1.7 b_c) at the seven sink positions
theta = 0.20, 0.25, ..., 0.50 and counts the interior extrema of the continuum density Phi(tau; b).

Numerical care (both matter; naive versions produce spurious counts):
  (i)  DEEP spectrum: roots of D_theta(k;b) up to w = k/2 = 120 (vectorized bisection), so the
       signed-mode sum is converged for all tau >= 1.5e-3 (mu_max ~ 2.9e4).  The wmax=40 fold
       solver is already converged at its fold times (dpma_fold_branch_margins.py bounds the
       cutoff sensitivity); the deeper sum is needed here because this scan also probes earlier
       times and small b.
  (ii) Extremum counting prunes adjacent extremum pairs with amplitude < 1e-6 max Phi
       (floating-point wiggles of the signed cancellation).
At the explicitly stated 1e-6 prominence resolution, the expected pattern per theta is exactly one
transition in b -- two interior extrema (valley+peak) for
all 0 < b < b_c, none above, no re-entrance -- with the transition matching the bisected b_c.
Writes outputs/tables/dpma_window_scan.txt
"""
from __future__ import annotations
import math
import numpy as np
from dpma_saddle_node_bc_theta import G_affected, bc
from paths import TABLES as TAB
OUT = []
def say(s=""):
    print(s, flush=True); OUT.append(s)

WMAX = 120.0
TAUS = np.exp(np.linspace(math.log(1.5e-3), math.log(1.2), 4000))

def roots_deep(th, b):
    """all roots k of D_theta(k;b)=k sin k + 2b sin(k th) sin(k(1-th)) with k <= 2*WMAX (vectorized)."""
    k = np.linspace(1e-6, 2 * WMAX, 240001)
    f = k * np.sin(k) + 2 * b * np.sin(k * th) * np.sin(k * (1 - th))
    s = np.sign(f)
    i = np.where(s[:-1] * s[1:] < 0)[0]
    lo, hi = k[i], k[i + 1]
    flo = f[i]
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        fm = mid * np.sin(mid) + 2 * b * np.sin(mid * th) * np.sin(mid * (1 - th))
        left = flo * fm <= 0
        hi = np.where(left, mid, hi)
        lo = np.where(left, lo, mid)
        flo = np.where(left, flo, fm)
    return 0.5 * (lo + hi)

def phi_deep(th, b):
    """Phi(tau; b) at xi=theta from the deep affected spectrum (node modes vanish at xi=theta)."""
    ks = roots_deep(th, b)
    G, mu = [], []
    for k in ks:
        w = k / 2.0
        if abs(math.sin(k * th)) < 1e-12:      # node mode: amplitude ~ sin(n pi xi) = 0 at xi=theta
            continue
        G.append(G_affected(w, th, th, b))
        mu.append(2 * w * w)
    G = np.array(G); mu = np.array(mu)
    return (np.exp(-np.outer(TAUS, mu)) * G).sum(axis=1)

def n_extrema(th, b, eps=1e-6):
    v = phi_deep(th, b)
    d1 = np.diff(v)
    sgn = np.sign(d1)
    idx = [i + 1 for i in range(len(sgn) - 1) if sgn[i] != 0 and sgn[i + 1] != 0 and sgn[i] != sgn[i + 1]]
    vmax = float(np.max(np.abs(v)))
    changed = True                              # prune tiny numerical wiggles pairwise
    while changed and len(idx) >= 2:
        changed = False
        for j in range(len(idx) - 1):
            if abs(v[idx[j + 1]] - v[idx[j]]) < eps * vmax:
                del idx[j + 1]; del idx[j]; changed = True; break
    return len(idx)

say("Dense scan in b, release at the sink (xi=theta, db=0.02, w<=120 i.e. k<=240, tau>=1.5e-3)")
say(f"  {'theta':>6} {'b-range':>13} {'transitions':>11} {'pattern':>24} {'b_c(scan)':>10} {'b_c(bisect)':>11}")
ok_all = True
for th in (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50):
    bcv = bc(th, th, lo=0.3, hi=8.0)
    bs = np.arange(0.05, bcv + 1.5, 0.02)
    counts = np.array([n_extrema(th, b) for b in bs])
    trans = np.where(counts[:-1] != counts[1:])[0]
    below = sorted(set(int(c) for c in counts[bs < bcv - 0.03]))
    above = sorted(set(int(c) for c in counts[bs > bcv + 0.03]))
    pattern = f"below={below} above={above}"
    b_scan = float(bs[trans[0]] + 0.01) if len(trans) else float("nan")
    ok = (len(trans) == 1) and below == [2] and above == [0] and abs(b_scan - bcv) <= 0.02
    ok_all &= ok
    say(f"  {th:>6.2f} [0.05,{bs[-1]:.2f}] {len(trans):>11d} {pattern:>24} {b_scan:>10.3f} {bcv:>11.4f}"
        + ("" if ok else "   <-- UNEXPECTED"))
say("")
say("Result: " + ("no re-entrance at any of the seven theta (exactly one transition at the stated "
                  "b, tau, cutoff, and 1e-6 prominence resolution; below b_c two interior critical "
                  "points = valley + delayed peak, above b_c none)."
                  if ok_all else "unexpected structure found -- see rows above."))
(TAB / "dpma_window_scan.txt").write_text("\n".join(OUT) + "\n")
