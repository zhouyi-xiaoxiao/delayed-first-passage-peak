#!/usr/bin/env python3
"""Normal-form prefactors of the midpoint fold and the curvature of b_c(theta) at theta = 1/2.

Expanding S1 about the fold (S1=S2=0), with S3=d^2 S1/d tau^2 and S1b = dS1/db both nonzero:

    S1(tau,b) ~ (1/2) S3 (tau-tau_c)^2 - S1b (b_c - b)
    tau_pm - tau_c = -+ delta,   delta = sqrt(2 S1b (b_c-b)/S3)
    gap:        tau_+ - tau_-        = a_gap  (b_c-b)^{1/2},  a_gap  = 2 sqrt(2 S1b/S3)
    prominence: Phi(tau_+)-Phi(tau_-) = (2/3) S3 delta^3
                                      = a_prom (b_c-b)^{3/2},  a_prom = (4 sqrt2/3) S1b^{3/2}/sqrt(S3)

so the prefactors follow from (S1b, S3) at the fold. This script (i) locates the midpoint fold
to high precision with mpmath (roots of tan w = -2w/b, midpoint amplitudes), (ii) evaluates S3
and S1b there (S1b by 40-digit central differences, checked against the implicit-function formula
dk_j/db = -D_b/D_k), (iii) prints a_gap and a_prom, (iv) cross-checks them against directly
measured gap/prominence at small b_c-b, and (v) fits b_c(1/2+eps) ~ b_c - c2 eps^2.
Writes outputs/tables/dpma_normal_form.txt
"""
from __future__ import annotations
import numpy as np
from mpmath import mp, mpf, cos, sin, exp, sqrt, findroot
from dpma_saddle_node_bc_theta import bc
from paths import TABLES as TAB

mp.dps = 40
OUT = []
def say(s=""):
    print(s, flush=True); OUT.append(s)

NMODES = 60

_ROOT_CACHE = {}

def roots_antip(b):
    """w_j solving 2 w cos w + b sin w = 0, one per ((j-1/2)pi, j pi)."""
    key = mp.nstr(b, 35)
    if key in _ROOT_CACHE:
        return _ROOT_CACHE[key]
    ws = []
    for j in range(1, NMODES + 1):
        lo = mpf(2 * j - 1) * mp.pi / 2 + mpf("1e-30")
        hi = mpf(j) * mp.pi - mpf("1e-30")
        f = lambda w: 2 * w * cos(w) + b * sin(w)
        flo = f(lo)
        for _ in range(140):
            mid = (lo + hi) / 2
            if flo * f(mid) <= 0: hi = mid
            else: lo, flo = mid, f(mid)
        ws.append((lo + hi) / 2)
    _ROOT_CACHE[key] = ws
    return ws

def S(n, tau, b):
    """S_n(tau;b) = sum_j G_j mu_j^n exp(-mu_j tau), antipodal xi=theta=1/2."""
    tot = mpf(0)
    for w in roots_antip(b):
        mu = 2 * w * w
        G = 4 * w * (1 - cos(w)) / (sin(w) * (1 + b * (b + 2) / (4 * w * w)))
        tot += G * mu ** n * exp(-mu * tau)
    return tot

# --- (i) locate the fold to high precision ------------------------------------------------
db = mpf("1e-14")
def newton_fold(tau, b):
    for _ in range(30):
        s1, s2, s3 = S(1, tau, b), S(2, tau, b), S(3, tau, b)
        s1b = (S(1, tau, b + db) - S(1, tau, b - db)) / (2 * db)
        s2b = (S(2, tau, b + db) - S(2, tau, b - db)) / (2 * db)
        det = (-s2) * s2b - s1b * (-s3)
        dtau = (-s1 * s2b + s2 * s1b) / det
        dbv = (-s2 * (-s2) + s1 * (-s3)) / det   # cramer for [ -s2 s1b; -s3 s2b ] [dtau,db]^T = -[s1,s2]
        tau += dtau; b += dbv
        if abs(dtau) < mpf("1e-25") and abs(dbv) < mpf("1e-25"):
            break
    return tau, b

tau_c, b_c = newton_fold(mpf("0.038363"), mpf("3.07643"))
say("Antipodal fold (theta=xi=1/2), mpmath 40 dps:")
say(f"  b_c   = {mp.nstr(b_c, 15)}")
say(f"  tau_c = {mp.nstr(tau_c, 15)}")
s3 = S(3, tau_c, b_c)
s1b = (S(1, tau_c, b_c + db) - S(1, tau_c, b_c - db)) / (2 * db)
say(f"  S3    = {mp.nstr(s3, 10)}   dS1/db = {mp.nstr(s1b, 10)}   (both nonzero: nondegenerate)")


def s1b_implicit(tau, b):
    """dS1/db from dk_j/db = -D_b/D_k (theta = 1/2) and partial derivatives of G_j(k, b)."""
    total = mpf(0)
    for w in roots_antip(b):
        k = 2 * w
        D_b = 2 * sin(k / 2) ** 2
        D_k = sin(k) + k * cos(k) + b * sin(k)
        dk_db = -D_b / D_k
        G = lambda kk, bb: 2 * kk * (1 - cos(kk / 2)) / (sin(kk / 2) * (1 + bb * (bb + 2) / kk**2))
        dG = mp.diff(lambda kk: G(kk, b), k) * dk_db + mp.diff(lambda bb: G(k, bb), b)
        mu = k * k / 2
        dmu = k * dk_db
        e = exp(-mu * tau)
        total += dG * mu * e + G(k, b) * dmu * (1 - mu * tau) * e
    return total


s1b_ift = s1b_implicit(tau_c, b_c)
say(f"  dS1/db from dk/db = -D_b/D_k: {mp.nstr(s1b_ift, 10)}  "
    f"(relative difference to the central difference {mp.nstr(abs(s1b_ift / s1b - 1), 3)})")

# --- (ii)-(iii) analytic prefactors --------------------------------------------------------
a_gap = 2 * sqrt(2 * s1b / s3)
a_prom = (4 * sqrt(mpf(2)) / 3) * s1b ** mpf("1.5") / sqrt(s3)
say("\nAnalytic normal-form prefactors:")
say(f"  gap:        tau_+ - tau_-        = {mp.nstr(a_gap, 8)} (b_c-b)^(1/2)")
say(f"  prominence: Phi(tau_+)-Phi(tau_-) = {mp.nstr(a_prom, 8)} (b_c-b)^(3/2)")

# --- (iv) direct cross-check near the fold ------------------------------------------------
say("\nDirect measurement vs analytic prediction:")
say(f"  {'b_c-b':>9} {'gap meas':>12} {'gap pred':>12} {'rel':>9} {'prom meas':>12} {'prom pred':>12} {'rel':>9}")
for eps in (mpf("1e-3"), mpf("1e-4"), mpf("1e-5")):
    b = b_c - eps
    delta0 = sqrt(2 * s1b * eps / s3)
    lo_m, hi_m = tau_c - 3 * delta0, tau_c - delta0 / 10
    lo_p, hi_p = tau_c + delta0 / 10, tau_c + 3 * delta0
    tm = findroot(lambda t: S(1, t, b), (lo_m, hi_m), solver="anderson")
    tp = findroot(lambda t: S(1, t, b), (lo_p, hi_p), solver="anderson")
    gap_m = tp - tm
    prom_m = S(0, tp, b) - S(0, tm, b)
    gap_p = a_gap * sqrt(eps)
    prom_p = a_prom * eps ** mpf("1.5")
    say(f"  {mp.nstr(eps,2):>9} {mp.nstr(gap_m,6):>12} {mp.nstr(gap_p,6):>12} "
        f"{mp.nstr(abs(gap_m/gap_p-1),2):>9} {mp.nstr(prom_m,6):>12} {mp.nstr(prom_p,6):>12} "
        f"{mp.nstr(abs(prom_m/prom_p-1),2):>9}")

# --- (v) curvature of b_c(theta) at theta = 1/2 ----------------------------------------------
say("\nExpansion b_c(1/2+eps) = b_c - c2 eps^2 (float bisection, general theta):")
eps_list = [0.010, 0.015, 0.020, 0.030, 0.040]
rows = []
for e in eps_list:
    bce = bc(0.5 + e, 0.5 + e, lo=0.3, hi=6.0)
    rows.append((e, bce))
    say(f"  eps={e:.3f}: b_c={bce:.8f}   (b_c(0)-b_c)/eps^2 = {(float(b_c)-bce)/e**2:.2f}")
E = np.array([r[0] for r in rows]); Y = np.array([r[1] for r in rows])
# fit b_c(eps) = c0 + c2 eps^2 + c4 eps^4
A = np.vstack([np.ones_like(E), E**2, E**4]).T
coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
say(f"  quartic-in-eps fit: c0={coef[0]:.6f} (vs {mp.nstr(b_c, 7)}), c2={coef[1]:.1f}, c4={coef[2]:.0f}")
say(f"  => quoted coefficient -133: fitted c2 = {coef[1]:.1f}")

(TAB / "dpma_normal_form.txt").write_text("\n".join(OUT) + "\n")
