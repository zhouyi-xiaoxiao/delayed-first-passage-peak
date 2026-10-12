#!/usr/bin/env python3
"""Fig. 3: near-end (half-line) limit for release at the sink.

Model.  Brownian motion with generator (1/2) d^2/dx^2 on [0,1], absorbing ends, an interior
delta-sink of strength b at theta, started at xi = theta (source-started).  Near an endpoint,
d = min(theta, 1-theta) -> 0, the rescaling y = x/d, sigma = tau/d^2, B = b d maps the problem onto
the half-line y > 0 (absorbing wall at 0, delta-sink of strength B at y = 1, start y = 1), whose
first-passage density has the explicit branch-cut form [Sec. III D, Eq. (phihlsource)]

    Phi_hl(sigma; B) = (1/pi) int_0^inf v e^{-sigma v^2/2} (1+g) sin v / (1 + 2 g cos v + g^2) dv,
    g(v) = 2 B sin(v) / v.

Panels
  (a) Collapse of the release-at-sink threshold onto the near-end law: b_c(theta) d -> B*,
      tau_c/d^2 -> c_*.
      Filled points: continuation (data/dpma_boundary_generated.csv);
      crosses: independent bisection (data/dpma_endpoint_law.txt, from dpma_endpoint_law.py);
      open points: direct Newton solutions of S1 = S2 = 0 at small d with a converged spectrum.
  (b) Phi_hl(sigma; B) for B = 0.5, B*, 1.2 overlaid with the rescaled finite-interval densities
      d^2 Phi_{theta=d, xi=d}(d^2 sigma; b = B/d), d = 0.2, 0.1, 0.05 (exact continuum mode sums with
      the amplitudes G_affected); lower strip: relative collapse error.
  (c) Direct Brownian dynamics on the half-line (Euler-Maruyama with exact Brownian-bridge wall test,
      killing at rate B/ell inside a gate of width ell around y = 1), histogram normalised by all
      trials, against Phi_hl; lower strip: relative residual, with the exact finite-gate bias
      (Talbot inversion of the closed-form square-gate transform) for comparison.

Brownian-dynamics samples
  By default the stored samples data/dpma_endpoint_bd.npz (2x10^5 walkers per B, seeds 20261012
  and 20261013) are reused.  With --rerun-bd the simulation is repeated with the same seeds
  (about one minute) and written to outputs/data/dpma_endpoint_bd.npz.

Outputs
  outputs/figures/fig3_near_end.{pdf,png}
  outputs/tables/fig3_near_end.txt

Run from anywhere:  python code/fig3_near_end.py [--rerun-bd]
"""
from __future__ import annotations

import argparse
import csv
import math
import time

import numpy as np
import mpmath as mp
from scipy.optimize import brentq
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullFormatter, FixedLocator

from dpma_saddle_node_bc_theta import G_affected, M_theta, roots as roots_scan
import figstyle as S
from paths import DATA, FIGURES as FIG, GENERATED, TABLES as TAB

B_STAR = 0.7890261736       # half-line fold (dpma_halfline_bstar.py)
C_STAR = 0.1579221011
B_VALUES = (0.5, B_STAR, 1.2)
D_VALUES = (0.2, 0.1, 0.05)
SIG_MIN, SIG_MAX = 1.0e-3, 3.0

OUT: list[str] = []


def say(s: str = "") -> None:
    print(s, flush=True)
    OUT.append(s)


# ----------------------------------------------------------------------------------------------
# Half-line density: branch-cut integral (Simpson), vectorised over sigma
# ----------------------------------------------------------------------------------------------
def _hl_kernel(B: float, sigma_min: float, npts: int = 400_001):
    vmax = math.sqrt(2.0 * 85.0 / sigma_min)      # exp(-sigma v^2/2) < e^-85 beyond vmax
    v = np.linspace(0.0, vmax, npts)
    sv = np.sin(v)
    g = np.empty_like(v)
    g[0] = 2.0 * B
    g[1:] = 2.0 * B * sv[1:] / v[1:]
    h = (1.0 + g) * sv / (1.0 + 2.0 * g * np.cos(v) + g * g)
    wts = np.ones(npts)
    wts[1:-1:2] = 4.0
    wts[2:-1:2] = 2.0
    wts *= (v[1] - v[0]) / 3.0
    return v, h, wts


def phi_hl(sigmas, B, deriv: int = 0):
    """(d/dsigma)^deriv Phi_hl(sigma; B, X=1)."""
    sig = np.atleast_1d(np.asarray(sigmas, dtype=float))
    v, h, wts = _hl_kernel(B, float(sig.min()))
    base = wts * v * h * (-0.5 * v * v) ** deriv / math.pi
    return np.array([float(np.dot(base, np.exp(-0.5 * s * v * v))) for s in sig])


def surv_hl(sigmas, B):
    """Survival S(sigma) = int_sigma^inf Phi_hl = (2/pi) int h(v)/v e^{-sigma v^2/2} dv."""
    sig = np.atleast_1d(np.asarray(sigmas, dtype=float))
    v, h, wts = _hl_kernel(B, float(sig.min()))
    hv = np.empty_like(v)
    hv[0] = 1.0 / (1.0 + 2.0 * B)
    hv[1:] = h[1:] / v[1:]
    base = wts * hv * 2.0 / math.pi
    return np.array([float(np.dot(base, np.exp(-0.5 * s * v * v))) for s in sig])


# ----------------------------------------------------------------------------------------------
# Closed-form transforms (mpmath) for independent checks and the finite-gate bias
# ----------------------------------------------------------------------------------------------
mp.mp.dps = 30


def fhat_delta(s, B):
    k = mp.sqrt(2 * s)
    r = (B / k) * (1 - mp.exp(-2 * k))
    return (mp.exp(-k) + r) / (1 + r)


def fhat_gate(s, B, ell):
    """Arrival-time transform for the square gate: killing rate B/ell on |y-1| < ell/2, start y=1.

    u(y) = Laplace transform of survival solves (1/2)u'' - (s + V)u = -1, u(0)=0, u bounded;
    f_hat = 1 - s u(1).  Region bases are normalised so that the 4x4 system stays O(1).
    """
    h = mp.mpf(ell) / 2
    kap = mp.mpf(B) / ell
    k = mp.sqrt(2 * s)
    q = mp.sqrt(2 * (s + kap))
    a = 1 - h
    th = mp.tanh(q * h)
    cth = mp.coth(k * a)
    A = mp.matrix([[1, -1, th, 0],
                   [k * cth, q * th, -q, 0],
                   [0, 1, th, -1],
                   [0, q * th, q, k]])
    rhs = mp.matrix([1 / (s + kap) - (1 - mp.exp(-k * a)) / s,
                     -(k / s) * mp.exp(-k * a),
                     1 / s - 1 / (s + kap),
                     0])
    x = mp.lu_solve(A, rhs)
    return kap / (s + kap) - s * x[1] / mp.cosh(q * h)


def talbot_density(fhat, sigma):
    return float(mp.invertlaplace(fhat, sigma, method="talbot"))


def talbot_survival(fhat, sigma):
    return float(mp.invertlaplace(lambda s: (1 - fhat(s)) / s, sigma, method="talbot"))


# ----------------------------------------------------------------------------------------------
# Finite interval: affected spectrum by interlacing brackets (no missed roots by construction)
# ----------------------------------------------------------------------------------------------
def _F(k, th, b):            # = M_theta(k/2) : (k/2) sin k + b sin(k th) sin(k(1-th))
    return 0.5 * k * math.sin(k) + b * math.sin(k * th) * math.sin(k * (1.0 - th))


def _F1(k, th, b):           # = _F / sin(k th); node root removed (used only in node brackets)
    st = math.sin(k * th)
    if abs(st) < 1e-9:
        return 0.5 * k * math.cos(k) / (th * math.cos(k * th)) + b * math.sin(k * (1.0 - th))
    return 0.5 * k * math.sin(k) / st + b * math.sin(k * (1.0 - th))


def affected_spectrum(th: float, b: float, kmax: float):
    """Roots k_j = 2 w_j of M_theta with nonzero coupling, k_j <= kmax, and G_j at xi = theta.

    The delta-sink is a rank-one positive perturbation of the Dirichlet Laplacian, so exactly one
    affected eigenvalue lies between consecutive coupled Dirichlet levels m pi < m' pi
    (sin(m pi theta) != 0).  Node levels (sin(n pi theta) = 0) are unperturbed and carry
    G_node(n, xi=theta) = 0, so they are omitted.
    """
    mmax = int(kmax / math.pi) + 3
    m = np.arange(1, mmax + 1)
    coupled = m[np.abs(np.sin(m * math.pi * th)) > 1e-9]
    ks = []
    for m0, m1 in zip(coupled[:-1], coupled[1:]):
        lo, hi = m0 * math.pi, m1 * math.pi
        if lo > kmax:
            break
        f = _F if m1 == m0 + 1 else _F1
        ks.append(brentq(f, lo, hi, args=(th, b), xtol=1e-13, rtol=1e-15, maxiter=200))
    ks = np.array([k for k in ks if k <= kmax])
    w = 0.5 * ks
    G = np.array([G_affected(float(wi), th, th, b) for wi in w])
    return 2.0 * w * w, G, w


def mode_sum(taus, mu, G):
    taus = np.atleast_1d(taus)
    with np.errstate(all="ignore"):   # Accelerate BLAS raises spurious FP flags on underflowed terms
        out = np.exp(-np.outer(taus, mu)) @ G
    if not np.all(np.isfinite(out)):
        raise FloatingPointError("non-finite mode sum")
    return out


def rescaled_interval_density(sig, d, B, kmax_factor):
    kmax = kmax_factor / d
    mu, G, w = affected_spectrum(d, B / d, kmax)
    full = d * d * mode_sum(d * d * sig, mu, G)
    keep = w <= 0.8 * w.max()
    trunc = d * d * mode_sum(d * d * sig, mu[keep], G[keep])
    return full, trunc, w


def fold_point(d, guess=None, kfac=40.0, tol=1e-12):
    """Direct Newton solve of S1 = S2 = 0 at theta = xi = d (log coordinates, FD Jacobian)."""
    kmax = kfac / d

    def resid(x):
        b, tau = np.exp(x)
        mu, G, _ = affected_spectrum(d, b, kmax)
        e = G * np.exp(-mu * tau)
        ph = e.sum()
        return np.array([tau * (e * mu).sum() / ph, tau * tau * (e * mu * mu).sum() / ph])

    x = np.log(np.array(guess if guess is not None else (B_STAR / d, C_STAR * d * d)))
    hstep = 1e-6
    for it in range(30):
        r = resid(x)
        J = np.column_stack([(resid(x + hstep * e) - resid(x - hstep * e)) / (2 * hstep)
                             for e in np.eye(2)])
        step = np.linalg.solve(J, -r)
        lam = 1.0
        while lam > 1e-3:
            xn = x + lam * step
            if np.max(np.abs(resid(xn))) < np.max(np.abs(r)) or np.max(np.abs(lam * step)) < 1e-10:
                break
            lam *= 0.5
        x = xn
        if np.max(np.abs(lam * step)) < tol:
            break
    b, tau = np.exp(x)
    return float(b), float(tau), float(np.max(np.abs(resid(x))))


# ----------------------------------------------------------------------------------------------
# Brownian dynamics on the half-line with a finite killing gate
# ----------------------------------------------------------------------------------------------
def bd_run(B, ell, n, seed, sigma_max=SIG_MAX, c=4.0, dtmin_frac=1.0 / 8.0, dt_max=1e-2, rel=0.05):
    """Euler-Maruyama for dY = dW (generator (1/2) d^2/dy^2) on y > 0, start y = 1.

    Adaptive per-walker step dt = clip((dist/c)^2, dt_min, dt_max), dist = distance to the gate,
    dt <= max(rel t, dt_min); dt_min = (ell/8)^2 resolves the gate.  Wall at 0: exact
    Brownian-bridge crossing probability exp(-2 y0 y1/dt).  Gate: killing probability
    1 - exp(-(B/ell) dt * occupancy), occupancy by the trapezoid rule on the step end points.
    Event time = t + U dt.  Returns first-passage times (inf if not absorbed by sigma_max) and type
    (1 wall, 2 gate).
    """
    rng = np.random.default_rng(seed)
    h = 0.5 * ell
    kappa = B / ell
    dt_min = (ell * dtmin_frac) ** 2
    y = np.ones(n)
    t = np.zeros(n)
    idx = np.arange(n)
    T = np.full(n, np.inf)
    kind = np.zeros(n, np.int8)
    steps = 0
    while idx.size:
        m = idx.size
        steps += m
        dist = np.abs(y - 1.0) - h
        np.maximum(dist, 0.0, out=dist)
        dt = (dist / c) ** 2
        np.minimum(dt, np.maximum(rel * t, dt_min), out=dt)
        np.clip(dt, dt_min, dt_max, out=dt)
        y1 = y + np.sqrt(dt) * rng.standard_normal(m)
        with np.errstate(over="ignore"):
            pw = np.where(y1 > 0.0, np.exp(-2.0 * y * np.maximum(y1, 0.0) / dt), 1.0)
        occ = 0.5 * ((np.abs(y - 1.0) < h).astype(float) + (np.abs(y1 - 1.0) < h))
        pk = -np.expm1(-kappa * dt * occ)
        u = rng.random(m)
        ev = u < pw + (1.0 - pw) * pk
        if ev.any():
            e = np.nonzero(ev)[0]
            T[idx[e]] = t[e] + dt[e] * rng.random(e.size)
            kind[idx[e]] = np.where(u[e] < pw[e], 1, 2)
        t = t + dt
        keep = (~ev) & (t < sigma_max)
        y = y1[keep]
        t = t[keep]
        idx = idx[keep]
    T[T > sigma_max] = np.inf
    kind[~np.isfinite(T)] = 0
    return T, kind, steps / n


# ----------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerun-bd", action="store_true",
                    help="repeat the Brownian dynamics instead of reusing data/dpma_endpoint_bd.npz")
    ap.add_argument("--n-walkers", type=int, default=200_000)
    ap.add_argument("--ell", type=float, default=0.02)
    args = ap.parse_args()
    t_start = time.time()

    say("Fig. 3: near-end (half-line) limit for release at the sink")
    say(f"B* = {B_STAR:.10f}, c_* = {C_STAR:.10f} (half-line fold, dpma_halfline_bstar.py)")
    say("")

    # ---------------- half-line density ----------------
    sig = np.exp(np.linspace(math.log(SIG_MIN), math.log(SIG_MAX), 420))
    PHI = {B: phi_hl(sig, B) for B in B_VALUES}
    say("[1] Half-line density Phi_hl (branch-cut Simpson, 400001 nodes)")
    lev = np.exp(-1 / (2 * sig)) / np.sqrt(2 * math.pi * sig ** 3)
    p0 = phi_hl(sig, 0.0)
    say(f"  B=0 check vs Levy density, sigma in [0.05,3]: max rel err = "
        f"{np.max(np.abs(p0[sig > 0.05] / lev[sig > 0.05] - 1)):.2e}")
    for B in (0.5, 1.2):
        for sg in (1e-3, 0.03, 0.3, 3.0):
            tal = talbot_density(lambda s: fhat_delta(s, B), sg)
            say(f"  B={B:<4} sigma={sg:<6}: branch-cut={phi_hl([sg], B)[0]:.10f}  "
                f"Talbot(closed form)={tal:.10f}  rel={abs(phi_hl([sg], B)[0] / tal - 1):.1e}")
    d1 = phi_hl([C_STAR], B_STAR, 1)[0]
    d2 = phi_hl([C_STAR], B_STAR, 2)[0]
    p_ = phi_hl([C_STAR], B_STAR)[0]
    say(f"  fold residuals at (c_*, B*): sigma Phi'/Phi = {C_STAR * d1 / p_:+.2e}, "
        f"sigma^2 Phi''/Phi = {C_STAR**2 * d2 / p_:+.2e}")
    # extrema of Phi_hl for the plotted B
    fine = np.exp(np.linspace(math.log(0.01), math.log(SIG_MAX), 1500))
    EXT = {}
    for B in B_VALUES:
        dp = phi_hl(fine, B, 1)
        sc = np.nonzero(np.diff(np.sign(dp)) != 0)[0]
        ext = []
        for i in sc:
            r = brentq(lambda s: phi_hl([s], B, 1)[0], fine[i], fine[i + 1], xtol=1e-12)
            ext.append((r, "valley" if dp[i] < 0 else "peak", phi_hl([r], B)[0]))
        EXT[B] = ext
        txt = ", ".join(f"{k} at sigma={r:.5f} (Phi={v:.5f})" for r, k, v in ext) or "none (monotone)"
        say(f"  B={B:.7f}: interior extrema in [0.01,3]: {txt}")
    S3 = {B: surv_hl([SIG_MAX], B)[0] for B in (0.5, 1.2)}
    for B in (0.5, 1.2):
        say(f"  exact truncated mass P(T>{SIG_MAX:g}) at B={B}: {S3[B]:.6f}; "
            f"total sink-capture probability 2B/(1+2B) = {2 * B / (1 + 2 * B):.6f}")
    say("")

    # ---------------- interval spectrum cross-check against the grid-scan root finder ----------
    say("[2] Interval spectrum: interlacing-bracket roots vs grid scan (dpma_saddle_node_bc_theta.roots)")
    for d in (0.2, 0.3):
        b = 0.5 / d
        _, _, w_new = affected_spectrum(d, b, 2 * 40.0)
        w_old = np.array([w for w in roots_scan(d, b, wmax=40.0)
                          if abs(math.sin(2 * d * w)) >= 1e-3])
        n = min(len(w_old), np.sum(w_new <= 40.0))
        say(f"  theta=d={d}, b={b:.3f}: affected roots w<=40: new={np.sum(w_new <= 40.0)}, "
            f"grid scan={len(w_old)}, max|dw|={np.max(np.abs(w_new[:n] - w_old[:n])):.1e}, "
            f"max|M_theta(w_new)|={max(abs(M_theta(float(x), d, b)) for x in w_new[:n]):.1e}")
    say("")

    # ---------------- rescaled interval densities ----------------
    kfac = math.sqrt(90.0 / SIG_MIN)        # mu_max * tau_min = (kmax^2/2) d^2 sigma_min = 45
    say(f"[3] Rescaled interval densities d^2 Phi_(theta=xi=d)(d^2 sigma; b=B/d), "
        f"kmax = {kfac:.1f}/d (mu_max tau_min = 45)")
    INT, CONV = {}, {}
    for d in D_VALUES:
        for B in B_VALUES:
            full, trunc, w = rescaled_interval_density(sig, d, B, kfac)
            INT[(d, B)] = full
            CONV[(d, B)] = np.max(np.abs(full - trunc) / np.abs(full))
            say(f"  d={d:<5} B={B:.7f}: {len(w)} modes (w_max={w.max():.0f}); "
                f"cutoff 0.8 w_max changes Phi by <= {CONV[(d, B)]:.1e} (rel) over sigma>={SIG_MIN:g}")
    say("")
    say("  Collapse error  max_sigma |d^2 Phi_int / Phi_hl - 1|")
    say(f"  {'d':>5} {'B':>9}  {'sigma in [1e-3,1]':>18} {'sigma in [1e-3,3]':>18}")
    COLL = {}
    for d in D_VALUES:
        for B in B_VALUES:
            rel = np.abs(INT[(d, B)] / PHI[B] - 1)
            COLL[(d, B)] = rel
            say(f"  {d:>5} {B:>9.6f}  {np.max(rel[sig <= 1]):>18.2e} {np.max(rel):>18.2e}")
    say("")

    # ---------------- fold collapse (panel a) ----------------
    say("[4] Fold line in endpoint variables: b_c d and tau_c/d^2 (d = theta <= 1/2)")
    rows = list(csv.DictReader(open(DATA / "dpma_boundary_generated.csv")))
    csv_d = np.array([float(r["theta"]) for r in rows])
    csv_b = np.array([float(r["b_c"]) for r in rows])
    csv_t = np.array([float(r["tau_c"]) for r in rows])
    bis = []
    for line in open(DATA / "dpma_endpoint_law.txt"):
        parts = line.split()
        if len(parts) == 3:
            try:
                bis.append((float(parts[0]), float(parts[1])))
            except ValueError:
                pass
    bis = np.array(bis)
    say(f"  stored continuation: {len(csv_d)} points, d in [{csv_d.min():.2f},{csv_d.max():.2f}];"
        f" at d=0.2: b_c d = {csv_b[0] * csv_d[0]:.9f}, tau_c/d^2 = {csv_t[0] / csv_d[0]**2:.9f}")
    NEW = []
    say(f"  {'d':>6} {'b_c':>14} {'b_c d':>13} {'b_c d - B*':>11} {'tau_c/d^2':>13} "
        f"{'tau/d^2 - c*':>12} {'resid':>8}")
    checks = {0.2: (csv_b[0], csv_t[0])}
    i5 = int(np.argmin(np.abs(csv_d - 0.5)))
    checks[0.5] = (csv_b[i5], csv_t[i5])
    for d in (0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.5):
        guess = checks.get(d)
        b, tau, res = fold_point(d, guess=guess)
        NEW.append((d, b, tau))
        say(f"  {d:>6.3f} {b:>14.9f} {b * d:>13.10f} {b * d - B_STAR:>+11.1e} {tau / d**2:>13.10f} "
            f"{tau / d**2 - C_STAR:>+12.1e} {res:>8.1e}")
    for d in (0.2, 0.5):
        bn, tn = [(b, t) for (dd, b, t) in NEW if dd == d][0]
        say(f"  check vs stored continuation at d={d}: |db_c|={abs(bn - checks[d][0]):.1e}, "
            f"|dtau_c|={abs(tn - checks[d][1]):.1e}")
    NEW = np.array(NEW)
    say("")

    # ---------------- Brownian dynamics (panel c) ----------------
    ell = args.ell
    n = args.n_walkers
    cache = DATA / "dpma_endpoint_bd.npz"
    BD = {}
    reuse_ok = False
    if not args.rerun_bd and cache.exists():
        z = np.load(cache)
        if float(z["ell"]) == ell and int(z["n"]) == n:
            reuse_ok = True
            for i, B in enumerate((0.5, 1.2)):
                BD[B] = dict(T=z[f"T{i}"], kind=z[f"kind{i}"], steps=float(z[f"steps{i}"]),
                             runtime=float(z[f"runtime{i}"]), seed=int(z[f"seed{i}"]))
    if not reuse_ok:
        for i, B in enumerate((0.5, 1.2)):
            seed = 20261012 + i
            t0 = time.time()
            T, kind, spw = bd_run(B, ell, n, seed)
            BD[B] = dict(T=T, kind=kind, steps=spw, runtime=time.time() - t0, seed=seed)
            print(f"  BD B={B}: {BD[B]['runtime']:.1f}s", flush=True)
        np.savez_compressed(GENERATED / "dpma_endpoint_bd.npz", ell=ell, n=n,
                            **{f"T{i}": BD[B]["T"] for i, B in enumerate((0.5, 1.2))},
                            **{f"kind{i}": BD[B]["kind"] for i, B in enumerate((0.5, 1.2))},
                            **{f"steps{i}": BD[B]["steps"] for i, B in enumerate((0.5, 1.2))},
                            **{f"runtime{i}": BD[B]["runtime"] for i, B in enumerate((0.5, 1.2))},
                            **{f"seed{i}": BD[B]["seed"] for i, B in enumerate((0.5, 1.2))})

    edges = np.exp(np.linspace(math.log(1e-3), math.log(SIG_MAX), 43))
    width = np.diff(edges)
    ctr = np.sqrt(edges[:-1] * edges[1:])
    say(f"[5] Brownian dynamics on the half-line: N={n} walkers per B, gate width ell={ell}, "
        f"kill rate B/ell, dt_min=(ell/8)^2={(ell / 8) ** 2:.2e}, dt_max=1e-2, truncation sigma={SIG_MAX:g}")
    say("    histogram: 42 log bins on [1e-3, 3], normalised by ALL trials; exact bin masses from the")
    say("    survival function (delta sink: branch-cut; finite gate: Talbot inversion of the square-gate transform)")
    HIST = {}
    for B in (0.5, 1.2):
        T = BD[B]["T"]
        kind = BD[B]["kind"]
        cnt, _ = np.histogram(T[np.isfinite(T)], bins=edges)
        Sd = surv_hl(edges, B)
        pd_ = -np.diff(Sd)
        Sg = np.array([talbot_survival(lambda s: fhat_gate(s, B, ell), e) for e in edges])
        pg = -np.diff(Sg)
        dens = cnt / (n * width)
        err = np.sqrt(cnt) / (n * width)
        HIST[B] = dict(cnt=cnt, dens=dens, err=err, pd=pd_, pg=pg)
        sel = cnt >= 25
        zd = (cnt - n * pd_) / np.sqrt(n * pd_ * (1 - pd_))
        zg = (cnt - n * pg) / np.sqrt(n * pg * (1 - pg))
        trunc_bd = float(np.mean(~np.isfinite(T)))
        early = float(np.mean(T < edges[0]))
        say(f"  B={B}: runtime {BD[B]['runtime']:.1f}s, {BD[B]['steps']:.0f} steps/walker, seed {BD[B]['seed']}")
        say(f"    truncated mass P(T>{SIG_MAX:g}): BD {trunc_bd:.5f} +- {math.sqrt(trunc_bd * (1 - trunc_bd) / n):.5f}"
            f" | exact delta {S3[B]:.5f} | exact gate {Sg[-1]:.5f}")
        say(f"    mass before sigma=1e-3: BD {early:.5f} | exact delta {1 - Sd[0]:.5f} | exact gate {1 - Sg[0]:.5f}")
        say(f"    absorbed by sigma={SIG_MAX:g}: wall {np.mean(kind == 1):.5f}, gate {np.mean(kind == 2):.5f}")
        say(f"    chi2/dof vs delta-sink density: {np.sum(zd[sel]**2):.1f}/{sel.sum()} "
            f"(max|z|={np.max(np.abs(zd[sel])):.2f}); vs exact finite gate: {np.sum(zg[sel]**2):.1f}/{sel.sum()} "
            f"(max|z|={np.max(np.abs(zg[sel])):.2f})")
        rb = dens[sel] / (pd_[sel] / width[sel]) - 1
        say(f"    max |BD/exact_delta - 1| over bins = {np.max(np.abs(rb)):.3f}; "
            f"median relative stat. error per bin = {np.median(err[sel] / dens[sel]):.3f}")
        gb = pg / pd_ - 1
        say(f"    finite-gate bias |Phi_gate/Phi_delta - 1| (bin-averaged): max {np.max(np.abs(gb)):.4f} "
            f"over [1e-3,3]; max {np.max(np.abs(gb[edges[:-1] >= 1e-2])):.4f} over [1e-2,3]")
    say("")

    # ------------------------------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------------------------------
    S.apply_style()                  # TrueType (Type 42) fonts, sizes at printed size
    INK, MUTED = "#1a1a1a", "#666666"
    COL = {0.5: "#0072B2", B_STAR: "#1a1a1a", 1.2: "#D55E00"}
    LAB = {0.5: r"$B=0.5$", B_STAR: r"$B=B^*$", 1.2: r"$B=1.2$"}
    MK = {0.2: "o", 0.1: "s", 0.05: "^"}
    DCOL = {0.2: "#1a1a1a", 0.1: "#666666", 0.05: "#9a9a9a"}     # one grey level per d
    DLS = {0.2: "-", 0.1: "--", 0.05: ":"}                        # one line style per d

    fig = plt.figure(figsize=(S.FULL_WIDTH, 3.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[2.05, 1.0], width_ratios=[0.86, 1.12, 1.12],
                          left=0.072, right=0.992, bottom=0.118, top=0.93, wspace=0.36, hspace=0.07)

    # ---- (a) ----
    a1 = fig.add_subplot(gs[0, 0])
    a2 = fig.add_subplot(gs[1, 0], sharex=a1)
    a1.axhline(B_STAR, color=MUTED, ls="--", lw=0.9, zorder=1)
    a1.plot(csv_d, csv_b * csv_d, "o", ms=3.0, color="#0072B2", zorder=3)
    a1.plot(bis[:, 0], bis[:, 0] * bis[:, 1], "x", ms=4.2, mew=0.9, color=INK, zorder=4)
    small = NEW[NEW[:, 0] < 0.199]
    a1.plot(small[:, 0], small[:, 0] * small[:, 1], "o", ms=3.6, mfc="white", mec="#0072B2",
            mew=0.9, zorder=4)
    a1.set_ylabel(r"rescaled threshold $b_c\,d$")
    a1.set_ylim(0.74, 1.62)
    a1.text(0.03, B_STAR + 0.02, r"$B^*=0.78903$", fontsize=7, color=MUTED, va="bottom")
    S.panel_title(a1, "a", "threshold near an end")
    hnd = [Line2D([], [], ls="", marker="o", ms=3.0, color="#0072B2"),
           Line2D([], [], ls="", marker="x", ms=4.2, mew=0.9, color=INK),
           Line2D([], [], ls="", marker="o", ms=3.6, mfc="white", mec="#0072B2", mew=0.9)]
    a1.legend(hnd, ["continuation", "bisection", "direct solve"], loc="center left",
              bbox_to_anchor=(0.0, 0.42), frameon=False, handletextpad=0.2, borderaxespad=0.3,
              fontsize=6.8, labelspacing=0.25)
    # d = 1/2 is the midpoint of the interval (theta = 1/2)
    i_mid = int(np.argmin(np.abs(csv_d - 0.5)))
    assert abs(csv_d[i_mid] - 0.5) < 1e-12
    a1.annotate(r"midpoint $\theta=1/2$", xy=(csv_d[i_mid], csv_d[i_mid] * csv_b[i_mid]),
                xytext=(0.36, csv_d[i_mid] * csv_b[i_mid]), ha="right", va="center",
                fontsize=6.8, color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6, shrinkA=1, shrinkB=3))
    a2.axhline(C_STAR, color=MUTED, ls="--", lw=0.9, zorder=1)
    a2.plot(csv_d, csv_t / csv_d ** 2, "o", ms=3.0, color="#0072B2", zorder=3)
    a2.plot(small[:, 0], small[:, 2] / small[:, 0] ** 2, "o", ms=3.6, mfc="white", mec="#0072B2",
            mew=0.9, zorder=4)
    a2.set_ylabel(r"time $\tau_c/d^{2}$")
    a2.set_xlabel(r"distance to nearer end $d$")
    a2.set_xlim(0.0, 0.52)
    # y-limits from the data: every plotted tau_c/d^2 point lies strictly inside the strip
    yall = np.concatenate([csv_t / csv_d ** 2, small[:, 2] / small[:, 0] ** 2, [C_STAR]])
    span = yall.max() - yall.min()
    ylo, yhi = yall.min() - 0.16 * span, yall.max() + 0.16 * span
    a2.set_ylim(ylo, yhi)
    assert np.all((yall > ylo) & (yall < yhi))
    say(f"  lower (a) strip: tau_c/d^2 in [{yall.min():.5f}, {yall.max():.5f}] -> "
        f"y-limits [{ylo:.5f}, {yhi:.5f}] (all {yall.size} points visible)")
    a2.yaxis.set_major_locator(FixedLocator([0.15, 0.17, 0.19]))
    a2.text(0.03, C_STAR + 0.0022, r"$c_*=0.15792$", fontsize=7, color=MUTED, va="bottom")
    plt.setp(a1.get_xticklabels(), visible=False)

    # ---- (b) ----
    b1 = fig.add_subplot(gs[0, 1])
    b2 = fig.add_subplot(gs[1, 1], sharex=b1)
    nmk = 11
    for B in B_VALUES:
        b1.loglog(sig, PHI[B], "-", color=COL[B], lw=1.5, zorder=3, label=LAB[B])
        for j, d in enumerate(D_VALUES):
            # sparse markers, staggered so that coinciding curves remain distinguishable
            lg = np.linspace(math.log(2.2e-3), math.log(2.6), nmk) + (j - 1) * 0.16
            ii = np.searchsorted(sig, np.exp(lg))
            ii = ii[(ii > 0) & (ii < len(sig))]
            b1.plot(sig[ii], INT[(d, B)][ii], MK[d], ms=3.2, mfc="none", mec=COL[B], mew=0.8,
                    zorder=4)
    for B in B_VALUES:
        for r, kind_, v in EXT[B]:
            if B == 0.5:
                b1.plot(r, v, marker="v" if kind_ == "valley" else "^", ms=4.0, color=COL[B],
                        zorder=5, mec="white", mew=0.4)
    pf = phi_hl([C_STAR], B_STAR)[0]
    b1.plot(C_STAR, pf, "*", ms=7.5, color=INK, mec="white", mew=0.5, zorder=6)
    b1.annotate("valley", xy=(EXT[0.5][0][0], EXT[0.5][0][2]), xytext=(0.030, 0.26), fontsize=6.8,
                color=COL[0.5], arrowprops=dict(arrowstyle="-", color=COL[0.5], lw=0.6, shrinkB=2))
    b1.annotate("delayed\npeak", xy=(EXT[0.5][1][0], EXT[0.5][1][2]), xytext=(0.32, 0.125),
                fontsize=6.8, color=COL[0.5], ha="center", va="center", linespacing=1.05,
                arrowprops=dict(arrowstyle="-", color=COL[0.5], lw=0.6, shrinkB=2))
    b1.annotate(r"merger $(c_*,B^*)$", xy=(C_STAR, pf), xytext=(0.10, 4.6), fontsize=6.8,
                color=INK, ha="center", va="center",
                arrowprops=dict(arrowstyle="-", color=INK, lw=0.6, shrinkA=1, shrinkB=3.5))
    b1.set_ylim(0.03, 70)
    b1.set_xlim(SIG_MIN, SIG_MAX)
    b1.set_ylabel(r"rescaled density $\Phi_{\rm hl}$, $d^{2}\Phi$")
    S.panel_title(b1, "b", "half-line limit")
    leg1 = b1.legend(loc="upper right", frameon=False, handlelength=1.6, borderaxespad=0.3,
                     labelspacing=0.25)
    b1.add_artist(leg1)
    hd = [Line2D([], [], ls="", marker=MK[d], ms=3.2, mfc="none", mec=DCOL[d], mew=0.8) for d in D_VALUES]
    b1.legend(hd, [rf"$d={d:g}$" for d in D_VALUES], loc="lower left", frameon=False,
              handletextpad=0.1, borderaxespad=0.3, labelspacing=0.2, title="interval",
              title_fontsize=6.8, fontsize=6.8)
    for d in D_VALUES:
        env = np.max(np.vstack([COLL[(d, B)] for B in B_VALUES]), axis=0)
        b2.loglog(sig, np.maximum(env, 1e-16), DLS[d], color=DCOL[d], lw=1.1)
        b2.plot(sig[19::38], np.maximum(env, 1e-16)[19::38], MK[d], ms=3.2, mfc="white",
                mec=DCOL[d], mew=0.8)
    hb2 = [Line2D([], [], color=DCOL[d], ls=DLS[d], lw=1.1, marker=MK[d], ms=3.2, mfc="white",
                  mec=DCOL[d], mew=0.8) for d in D_VALUES]
    b2.legend(hb2, [rf"$d={d:g}$" for d in D_VALUES], loc="upper left", ncol=3, frameon=False,
              handlelength=2.0, handletextpad=0.3, columnspacing=0.6, borderaxespad=0.25,
              fontsize=6.8)
    b2.text(1.3e-3, 1e-11, r"round-off floor (all $d$)", fontsize=6.8, color=MUTED, ha="left")
    b2.set_ylim(1e-15, 3e3)
    b2.yaxis.set_major_locator(FixedLocator([1e-14, 1e-9, 1e-4]))
    b2.yaxis.set_minor_formatter(NullFormatter())
    b2.set_ylabel("rel. deviation", labelpad=1)
    b2.set_xlabel(r"rescaled time $\sigma=\tau/d^{2}$")
    plt.setp(b1.get_xticklabels(), visible=False)

    # ---- (c) ----
    c1 = fig.add_subplot(gs[0, 2])
    c2 = fig.add_subplot(gs[1, 2], sharex=c1)
    for B in (0.5, 1.2):
        Hd = HIST[B]
        sel = Hd["cnt"] > 0
        c1.loglog(sig, PHI[B], "-", color=COL[B], lw=1.2, zorder=3, label=LAB[B] + " exact")
        c1.errorbar(ctr[sel], Hd["dens"][sel], yerr=Hd["err"][sel], fmt="o", ms=3.0, mfc="none",
                    mec=COL[B], mew=0.7, ecolor=COL[B], elinewidth=0.6, capsize=0, zorder=4)
        exact_bin = Hd["pd"] / width
        gate_bin = Hd["pg"] / width
        off = 0.97 if B == 0.5 else 1.03
        c2.errorbar(ctr[sel] * off, Hd["dens"][sel] / exact_bin[sel] - 1,
                    yerr=Hd["err"][sel] / exact_bin[sel], fmt="o", ms=2.9, mfc="none", mec=COL[B],
                    mew=0.6, ecolor=COL[B], elinewidth=0.55, capsize=0, zorder=4)
        c2.plot(ctr, gate_bin / exact_bin - 1, "-", color=COL[B], lw=0.9, alpha=0.9, zorder=3)
    c2.axhline(0, color=MUTED, lw=0.6, zorder=1)
    c1.set_xlim(SIG_MIN, SIG_MAX)
    c1.set_ylim(0.03, 70)
    c1.set_ylabel("absorption-time density")
    S.panel_title(c1, "c", "simulation on the half-line")
    hc = [Line2D([], [], color=COL[0.5], lw=1.2), Line2D([], [], color=COL[1.2], lw=1.2),
          Line2D([], [], ls="", marker="o", ms=3.0, mfc="none", mec=MUTED, mew=0.7)]
    c1.legend(hc, [r"$\Phi_{\rm hl}$, $B=0.5$", r"$\Phi_{\rm hl}$, $B=1.2$",
                   "Brownian dynamics\n" + rf"(gate $\ell={ell:g}$)"], loc="lower left", frameon=False,
              handlelength=1.5, borderaxespad=0.3, labelspacing=0.25, fontsize=6.8)
    c1.text(0.97, 0.96, "mass at $\\sigma>3$\nBD | exact", transform=c1.transAxes, ha="right",
            va="top", fontsize=6.6, color=MUTED, linespacing=1.1)
    for B in (0.5, 1.2):
        tb = float(np.mean(~np.isfinite(BD[B]["T"])))
        c1.text(0.97, 0.80 - (0 if B == 0.5 else 0.075), f"{tb:.3f} | {S3[B]:.3f}",
                transform=c1.transAxes, ha="right", va="top", fontsize=6.6, color=COL[B])
    c2.set_ylim(-0.145, 0.145)
    c2.yaxis.set_major_locator(FixedLocator([-0.1, 0.0, 0.1]))
    c2.set_ylabel("rel. residual", labelpad=1)
    c2.set_xlabel(r"rescaled time $\sigma$")
    plt.setp(c1.get_xticklabels(), visible=False)
    for ax in (b1, b2, c1, c2):
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=6))
        ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=20))
        ax.xaxis.set_minor_formatter(NullFormatter())
    for ax in (b1, c1):
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=6))
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=20))
        ax.yaxis.set_minor_formatter(NullFormatter())
    for ax in (a1, a2, b1, b2, c1, c2):
        ax.tick_params(which="both", top=True, right=True)
    fig.align_ylabels([a1, a2])

    info = S.save(fig, FIG, "fig3_near_end")   # layout/font checks, PDF + PNG (150 dpi)
    plt.close(fig)
    say(f"Figure: outputs/figures/fig3_near_end.pdf (+ .png, {S.PNG_DPI} dpi; "
        f"{info['width_in']:.2f} x {info['height_in']:.2f} in); total runtime {time.time() - t_start:.0f}s")
    (TAB / "fig3_near_end.txt").write_text("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
