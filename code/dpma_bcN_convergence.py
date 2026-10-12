#!/usr/bin/env python3
"""Finite-N fold-threshold convergence ``b_{c,N} -> b_c`` on the exact ring.

The density reconstruction converges as O(N^-2); this script checks that the fold threshold itself
converges. It computes b_{c,N}(theta=1/2, xi=theta) on the exact lazy ring (q=2/3, u=N/2, start
r=u) without a prominence cutoff.  At every integer time
in the fold window we evaluate the exact increment

    Delta F_N(t;b) = sum_j B_j (s_j - 1) s_j**(t-1)

and define ``H_N(b)=max_t Delta F_N(t;b)``.  The window endpoints have negative increments, so
``H_N>0`` exactly when a valley--peak pair exists.  Bisection on ``H_N=0`` therefore locates the
discrete loss of local extrema rather than an arbitrary finite-prominence contour.

The calculation uses the spectral decomposition of the transient block (``eigh``),
for N = 100...1200, and fits the convergence exponent of |b_{c,N} - b_c| with
b_c = 3.0764323604 (continuum, 40-digit Newton solve in dpma_normal_form.py).
Writes outputs/tables/dpma_bcN_convergence.{txt,csv}.
"""
from __future__ import annotations
import math
import numpy as np

from lattice_mixture import evaluate_geometric_mixture
from paths import TABLES as TAB
OUT = []
def say(s=""):
    print(s, flush=True); OUT.append(s)

q = 2.0 / 3.0
BC_CONT = 3.0764323604

def modes(N, b):
    """Exact spectral modes of the ring with a midpoint shortcut: s_j (eigenvalues), B_j (residues)."""
    u = N // 2
    lam = b * q / N                      # lambda = beta(1-q) = b q / N
    M = np.zeros((N - 1, N - 1))
    for i in range(N - 1):
        M[i, i] = 1 - q
        if i > 0: M[i, i - 1] = q / 2
        if i < N - 2: M[i, i + 1] = q / 2
    M[u - 1, u - 1] -= lam
    b_abs = np.zeros(N - 1)
    b_abs[0] += q / 2; b_abs[-1] += q / 2; b_abs[u - 1] += lam
    s, V = np.linalg.eigh(M)
    er = np.zeros(N - 1); er[u - 1] = 1.0   # start at the source r = u
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        B = (er @ V) * (V.T @ b_abs)
    B = np.where(np.isfinite(B), B, 0.0)    # extreme modes are irrelevant on the fold window
    return s, B

def fold_indicator(N, b):
    """Return ``H_N=max_t [F(t+1)-F(t)]`` over every integer time in the fold window."""
    s, B = modes(N, b)
    scale = N * N / q
    ts = np.arange(math.ceil(0.03 * scale), math.floor(0.05 * scale) + 1, dtype=np.int64)
    increments = evaluate_geometric_mixture(s, B * (s - 1.0), ts)
    # A positive interior increment bracketed by negative endpoint increments is equivalent to
    # a valley followed by a peak.  The endpoint check also guards against a drifting search window.
    if not (increments[0] < 0.0 and increments[-1] < 0.0):
        raise RuntimeError(
            f"finite-N fold window no longer brackets the pair: N={N}, b={b}, "
            f"end increments=({increments[0]}, {increments[-1]})"
        )
    j = int(np.argmax(increments))
    return float(increments[j]), int(ts[j])


def has_pair(N, b):
    return fold_indicator(N, b)[0] > 0.0

def bc_N(N, lo=3.0, hi=3.15, iters=30):
    if not has_pair(N, lo):
        return float("nan")
    if has_pair(N, hi):
        raise RuntimeError(f"upper bisection bracket still has a pair: N={N}, b={hi}")
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if has_pair(N, mid): lo = mid
        else: hi = mid
    return 0.5 * (lo + hi)

say("Finite-N fold-threshold convergence, antipodal (theta=xi=1/2, q=2/3)")
say("  definition: H_N(b)=max_t [F_N(t+1)-F_N(t;b)]=0, all integer t in tau=[0.03,0.05]")
say(f"  continuum b_c = {BC_CONT}")
say(f"  {'N':>6} {'b_c,N':>14} {'b_c,N - b_c':>13} {'N^2 error':>11}")
rows = []
for N in (100, 200, 300, 400, 600, 800, 1200):
    b = bc_N(N)
    rows.append((N, b))
    say(f"  {N:>6d} {b:>14.10f} {b - BC_CONT:>+13.3e} {N*N*(b-BC_CONT):>11.3f}")
Ns = np.array([r[0] for r in rows], float)
dev = np.array([abs(r[1] - BC_CONT) for r in rows])
mask = dev > 1e-9
p, c = np.polyfit(np.log(Ns[mask]), np.log(dev[mask]), 1)
# The standard error is only a descriptive regression diagnostic for deterministic truncation data.
lx = np.log(Ns[mask]); ly = np.log(dev[mask]); fitted = p * lx + c
p_se = math.sqrt(np.sum((ly - fitted)**2) / (len(lx) - 2) / np.sum((lx - np.mean(lx))**2))
expected = np.array([
    3.0811742822, 3.0776148194, 3.0769575579, 3.0767277166,
    3.0765636137, 3.0765061866, 3.0764651707,
])
observed = np.array([row[1] for row in rows])
if np.max(np.abs(observed - expected)) > 5e-7 or abs(p + 2.0) > 0.01:
    raise RuntimeError(f"finite-N threshold regression failed: max drift={np.max(np.abs(observed-expected))}, p={p}")
say("")
say(f"  descriptive log-log fit: |b_c,N - b_c| ~ N^({p:.5f}) "
    f"(slope OLS diagnostic {p_se:.5f}; prefactor {math.exp(c):.3g})")
# Limit of N^2 (b_c,N - b_c): the lattice corrections are even in 1/N, so fit A + C/N^2 to the
# largest sizes (bisection resolution 1.4e-10 in b, i.e. <= 2e-4 in N^2 (b_c,N - b_c) at N = 1200).
n2err = Ns**2 * (observed - BC_CONT)
limits = []
for first in range(2, len(Ns) - 1):           # windows N >= 300, ..., N >= 800
    design = np.column_stack([np.ones(len(Ns) - first), 1.0 / Ns[first:]**2])
    limits.append(float(np.linalg.lstsq(design, n2err[first:], rcond=None)[0][0]))
say(f"  N^2(b_c,N - b_c) at N=1200: {n2err[-1]:.3f}; extrapolation A + C/N^2: A = {limits[1]:.3f} "
    f"(fit over N >= {int(Ns[3])}; fits over N >= {int(Ns[2])}...{int(Ns[-2])}: {min(limits):.3f}-{max(limits):.3f})")
say(f"  => the exact discrete fold threshold converges as O(N^-2); N^2(b_c,N-b_c) tends to {limits[1]:.2f}.")
(TAB / "dpma_bcN_convergence.txt").write_text("\n".join(OUT) + "\n")
np.savetxt(
    TAB / "dpma_bcN_convergence.csv",
    np.column_stack([Ns, observed, observed - BC_CONT, Ns**2 * (observed - BC_CONT)]),
    delimiter=",",
    header="N,b_c_N,b_c_N_minus_b_c,N_squared_error",
    comments="",
    fmt=["%.0f", "%.12f", "%.12g", "%.9f"],
)
