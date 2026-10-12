#!/usr/bin/env python3
"""Direct check that b_c = 3.0764 (theta = xi = 1/2) is a fold of the density's critical points.

A sum of decaying exponentials with positive coefficients is completely monotone and cannot
produce an interior fold; the exact modal amplitudes here are signed.  The script checks:
  (A) amplitude signs on the exact lattice: the residues B_j take both signs;
  (B) the continuum density Phi(tau;b) = sum_j G_j e^{-2 w_j^2 tau}: as b -> b_c the interior
      valley and delayed peak approach each other -- both the time gap (tau_max - tau_min) and the
      height difference (Phi_max - Phi_min) tend to zero, i.e. Phi'(tau_c) = Phi''(tau_c) = 0 --
      and both critical points are absent for b > b_c.  The approach follows the fold exponents
      1/2 and 3/2 (seven points just below b_c; these are the data of Fig. 2(c)).

Writes outputs/tables/dpma_saddle_node_certification.txt and
outputs/tables/dpma_fold_scaling_generated.csv.
"""
from __future__ import annotations
import math
import numpy as np

from dpma_antipodal_spectrum import Q_DEFAULT, amplitudes, roots_w
from paths import TABLES

q = Q_DEFAULT
OUT = []
def say(s=""):
    print(s, flush=True); OUT.append(s)

def transient_and_absorb(N, beta, u):
    """Symmetric transient block M (sites 1..N-1) + one-step absorption vector b_abs."""
    n = N - 1
    M = np.zeros((n, n)); b_abs = np.zeros(n)
    lam = beta * (1 - q)
    for i in range(1, N):
        k = i - 1
        M[k, k] = (1 - q) * (1 - beta) if i == u else (1 - q)
        if i == u:
            b_abs[k] += lam
        for j in (i - 1, i + 1):
            if j % N == 0:
                b_abs[k] += q / 2.0       # step onto target v=0 (also site N == 0)
            elif 1 <= j <= N - 1:
                M[k, (j % N) - 1] += q / 2.0
    return M, b_abs

def amplitude_signs(N, b, d=3):
    beta = q * b / ((1 - q) * N); u = N // 2; r = u - d
    M, b_abs = transient_and_absorb(N, beta, u)
    s, V = np.linalg.eigh(M)              # symmetric -> real eigenpairs
    er = np.zeros(N - 1); er[r - 1] = 1.0
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        B = (er @ V) * (V.T @ b_abs)      # B_j = (e_r.v_j)(v_j.b_abs); F(t)=sum B_j s_j^{t-1}
    order = np.argsort(s)[::-1]
    B = B[order]; s = s[order]
    npos = int(np.sum(B > 1e-15)); nneg = int(np.sum(B < -1e-15))
    # the modes with largest |B| and their signs
    idx = np.argsort(-np.abs(B))[:8]
    return npos, nneg, [(int(i), float(s[i]), float(B[i])) for i in idx], float(np.sum(B))

def phi_pair(b, xlo=1e-4, xhi=3.0, n=120000):
    """Locate the valley--peak pair by bracketing S1 and refining its zeros.

    The grid is used only to find brackets. Bisection on the analytic spectral derivative removes
    the mesh error that otherwise biases finite-window exponent fits near the fold.
    """
    ws = roots_w(b, 400)
    G = amplitudes(ws, b)
    mu = 2 * ws**2

    def moment(tau, order):
        return float(np.sum(G * mu**order * np.exp(-mu * tau)))

    xs = np.exp(np.linspace(math.log(xlo), math.log(xhi), n))
    # Chunked evaluation bounds peak memory while retaining a dense, deterministic bracket grid.
    s1 = np.empty_like(xs)
    for start in range(0, len(xs), 5000):
        stop = min(start + 5000, len(xs))
        block = xs[start:stop]
        s1[start:stop] = (
            (G * mu)[None, :] * np.exp(-np.outer(block, mu))
        ).sum(axis=1)
    crossings = np.where(s1[:-1] * s1[1:] < 0)[0]
    roots = []
    for index in crossings:
        lo, hi = float(xs[index]), float(xs[index + 1])
        flo = moment(lo, 1)
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            fmid = moment(mid, 1)
            if flo * fmid <= 0:
                hi = mid
            else:
                lo, flo = mid, fmid
        tau = 0.5 * (lo + hi)
        # S1 changes + to - at a minimum of Phi; - to + at a maximum.
        roots.append((tau, np.sign(s1[index]), np.sign(s1[index + 1])))
    minima = [tau for tau, left, right in roots if left > 0 and right < 0]
    maxima = [tau for tau, left, right in roots if left < 0 and right > 0]
    pairs = [(tmin, tmax) for tmin in minima for tmax in maxima if tmin < tmax]
    if pairs:
        tmin, tmax = pairs[-1]
        vmin, vmax = moment(tmin, 0), moment(tmax, 0)
        return True, tmin, tmax, vmin, vmax
    return False, None, None, None, None

def main():
    say("=" * 78)
    say("Fold of the critical points at b_c (theta = xi = 1/2)")
    say("=" * 78)

    say("\n(A) Amplitude signs of the exact lattice residues:")
    for b in (1.0, 2.0, 3.0):
        npos, nneg, top, ssum = amplitude_signs(400, b)
        say(f"  N=400 b={b}: #(B_j>0)={npos}, #(B_j<0)={nneg}, sum B_j={ssum:.2e}; "
            f"top-|B| signs = {['%+.3f'%t[2] for t in top[:6]]}")
    say("  => the amplitudes take both signs; the density is not a positive mixture of")
    say("     exponentials.")

    say("\n(B) Continuum density Phi(tau;b): the (valley, delayed peak) pair versus b")
    say(f"  {'b':>7} {'pair?':>6} {'tau_min':>9} {'tau_max':>9} {'gap=tmax-tmin':>13} {'prominence':>11}")
    for b in (2.80, 2.95, 3.00, 3.04, 3.06, 3.07, 3.074, 3.076, 3.078, 3.085, 3.10, 3.20):
        ok, tmin, tmax, vmin, vmax = phi_pair(b)
        if ok:
            say(f"  {b:>7.3f} {'YES':>6} {tmin:>9.5f} {tmax:>9.5f} {tmax-tmin:>13.5f} {vmax-vmin:>11.3e}")
        else:
            say(f"  {b:>7.3f} {'no':>6} {'-':>9} {'-':>9} {'-':>13} {'-':>11}")
    say("  => gap and height difference both tend to zero as b -> b_c and the pair is absent")
    say("     above: the two critical points merge, Phi'(tau_c)=Phi''(tau_c)=0 (a fold).")

    # quantify the approach to the fold + normal-form scaling exponents
    say("\n  Approach to the fold (fine grid just below b_c) + normal-form scaling:")
    bc = 3.0764323604
    sampled_b, deltas, gaps, proms = [], [], [], []
    for b in (3.040, 3.050, 3.060, 3.065, 3.070, 3.073, 3.075):
        ok, tmin, tmax, vmin, vmax = phi_pair(b)
        if ok:
            d = bc - b
            sampled_b.append(b)
            deltas.append(d); gaps.append(tmax - tmin); proms.append(vmax - vmin)
            say(f"    b={b:.4f} (b_c-b={d:.4f}): gap={tmax-tmin:.6f}  prominence={vmax-vmin:.3e}")

    def _slope_with_se(xs, ys):
        lx = [math.log(x) for x in xs]; ly = [math.log(y) for y in ys]
        n = len(lx); mx = sum(lx) / n; my = sum(ly) / n
        sxx = sum((a - mx) ** 2 for a in lx)
        slope = sum((a - mx) * (c - my) for a, c in zip(lx, ly)) / sxx
        intercept = my - slope * mx
        rss = sum((c - intercept - slope * a) ** 2 for a, c in zip(lx, ly))
        se = math.sqrt((rss / (n - 2)) / sxx)
        return slope, se
    if len(deltas) >= 3:
        pg, pg_se = _slope_with_se(deltas, gaps)
        pp, pp_se = _slope_with_se(deltas, proms)
        ok_nf = abs(pg - 0.5) < 0.06 and abs(pp - 1.5) < 0.12
        say(f"    => OLS log-log slopes (descriptive fit diagnostics): gap = {pg:.6f} +/- {pg_se:.6f} "
            f"(fold predicts 1/2); prominence = {pp:.6f} +/- {pp_se:.6f} "
            f"(fold predicts 3/2)")
        gap_subwindow_slopes, prominence_subwindow_slopes = [], []
        for start in range(len(deltas)):
            for stop in range(start + 4, len(deltas) + 1):
                gap_subwindow_slopes.append(_slope_with_se(deltas[start:stop], gaps[start:stop])[0])
                prominence_subwindow_slopes.append(
                    _slope_with_se(deltas[start:stop], proms[start:stop])[0]
                )
        gap_range = (min(gap_subwindow_slopes), max(gap_subwindow_slopes))
        prominence_range = (
            min(prominence_subwindow_slopes), max(prominence_subwindow_slopes)
        )
        if not (0.5004 <= gap_range[0] <= 0.5007 and 0.5020 <= gap_range[1] <= 0.5024):
            raise RuntimeError(f"gap fit-subwindow range drifted: {gap_range}")
        if not (1.5007 <= prominence_range[0] <= 1.5011
                and 1.5041 <= prominence_range[1] <= 1.5045):
            raise RuntimeError(f"prominence fit-subwindow range drifted: {prominence_range}")
        say("    => all contiguous fit subwindows with >=4 points: "
            f"gap slopes {gap_range[0]:.4f}--{gap_range[1]:.4f}; "
            f"prominence slopes {prominence_range[0]:.4f}--{prominence_range[1]:.4f}")
        say(f"    => fold exponents {'reproduced' if ok_nf else 'NOT reproduced'}: the two critical "
            "points merge with gap ~ (b_c-b)^(1/2) and height difference ~ (b_c-b)^(3/2).")

    p = TABLES / "dpma_saddle_node_certification.txt"
    p.write_text("\n".join(OUT) + "\n")
    say("\nwrote outputs/tables/dpma_saddle_node_certification.txt")
    csv_path = p.parent / "dpma_fold_scaling_generated.csv"
    np.savetxt(
        csv_path,
        np.column_stack([sampled_b, deltas, gaps, proms]),
        delimiter=",",
        header="b,b_c_minus_b,tau_gap,prominence",
        comments="",
        fmt="%.12g",
    )
    say("wrote outputs/tables/dpma_fold_scaling_generated.csv")

if __name__ == "__main__":
    main()
