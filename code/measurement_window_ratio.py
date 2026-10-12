#!/usr/bin/env python3
"""Window-ratio numbers of the proposed measurement (Sec. VI and App. D, "Measurement estimate").

Model: Brownian motion with D = 1/2 on [0, 1], absorbing ends, a delivery-and-terminate gate at
theta = 1/2, release point xi = 0.44.  Phi(tau; b) is the density of the trial-termination time
(delivery by the gate OR absorption at an end).

Independence from the other solver modules
------------------------------------------
* The delta-sink spectrum is not taken from ``dpma_saddle_node_bc_theta.roots/G_affected``.  At
  theta = 1/2 the antisymmetric modes have zero overlap with the constant function, so they carry
  zero weight in S and Phi.  The symmetric modes obey  k cos(k/2) + b sin(k/2) = 0  with k in
  ((2j-1)pi, 2j pi) and mu = k^2/2; amplitudes are re-derived here from the matched eigenfunction.
* Window probabilities are integrated mode-by-mode in closed form,
  P[a, c] = sum_j c_j (exp(-mu_j a) - exp(-mu_j c)), not by Gauss-Legendre quadrature.
* The fold is found by a nested root solve (inflection Phi''=0, then Phi'=0), not by a
  bisection on the valley/peak count.
* The finite-width gate (killing rate b/ell on |x-1/2| <= ell/2) is solved exactly (piecewise
  sin / cosh eigenfunctions, transcendental matching condition), with no spatial discretization,
  as an independent check of the finite-difference calculation ``dpma_finite_gate_check.py``.
* The delta-method variance is checked against direct multinomial sampling (no Brownian dynamics).

``dpma_saddle_node_bc_theta.phi_vals`` and ``solve_fold`` are imported only for cross-checks.
Writes outputs/tables/measurement_window_ratio.{txt,json}.  Light (< 1 min).
"""
from __future__ import annotations

import json
import math

import numpy as np
from scipy.optimize import brentq

from paths import TABLES

THETA = 0.5
XI = 0.44
B_REF = 2.5                  # reference (Fig. bd left panel) used to locate the window centres
TAU_P_QUOTED, TAU_V_QUOTED = 0.0462, 0.0253
FRAC = 0.20

QUOTED = {
    "b_plus": 3.13660, "b_plus_bisection": 3.136602575910443,
    "R_c": 0.9699549, "p": 0.1045292, "v": 0.0590153, "dR_db": -0.0883849,
    "R_centre_point": 0.9785, "sd_R_N4000": 0.07897,
    "N_025": 51086, "N_010": 319287, "factor_95": 3.84,
    "b_plus_ell": {0.04: 3.023350870, 0.02: 3.075533280, 0.01: 3.104943944},
    "R_c_ell": {0.04: 0.970136652, 0.02: 0.970000032, 0.01: 0.969966008},
    "b_plus_ell_to_0": 3.13657,
    "R_exact_table": {2.0: 1.106923, 2.5: 1.035862, 3.0: 0.982413, 3.5: 0.940240, 4.0: 0.905821},
}

LINES: list[str] = []


def say(s: str = "") -> None:
    print(s, flush=True)
    LINES.append(s)


# ----------------------------------------------------------------------------------------------
# Spectral representation shared by the delta sink and the finite gate
# ----------------------------------------------------------------------------------------------
class Spectrum:
    """Phi(tau) = sum_j mu_j c_j exp(-mu_j tau); S(tau) = sum_j c_j exp(-mu_j tau)."""

    def __init__(self, mu: np.ndarray, c: np.ndarray):
        self.mu = np.asarray(mu, float)
        self.c = np.asarray(c, float)

    def _m(self, tau, power):
        tau = np.atleast_1d(np.asarray(tau, float))
        w = self.c * self.mu**power
        # explicit row sums (macOS Accelerate matmul raises spurious FP warnings on underflow)
        return np.array([float(np.sum(w * np.exp(-self.mu * t))) for t in tau])

    def S(self, tau):
        return self._m(tau, 0)

    def phi(self, tau):
        return self._m(tau, 1)

    def dphi(self, tau):
        return -self._m(tau, 2)

    def d2phi(self, tau):
        return self._m(tau, 3)

    def window_prob(self, lo, hi):
        return float(np.sum(self.c * (np.exp(-self.mu * lo) - np.exp(-self.mu * hi))))


def delta_spectrum(b: float, nmodes: int = 400) -> Spectrum:
    """Symmetric modes of -(1/2)psi'' + b delta(x-1/2) psi = mu psi, psi(0)=psi(1)=0."""
    j = np.arange(1, nmodes + 1, dtype=float)
    lo, hi = (2 * j - 1) * np.pi, 2 * j * np.pi

    def f(k):
        return k * np.cos(k / 2) + b * np.sin(k / 2)

    flo = f(lo)
    for _ in range(110):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        same = np.sign(fm) == np.sign(flo)
        lo = np.where(same, mid, lo)
        flo = np.where(same, fm, flo)
        hi = np.where(same, hi, mid)
    k = 0.5 * (lo + hi)
    mu = 0.5 * k**2
    psi_xi = np.sin(k * XI)                     # psi = sin(kx) on [0,1/2], mirrored
    overlap = 2.0 * (1.0 - np.cos(k / 2)) / k   # int_0^1 psi
    norm2 = 0.5 - np.sin(k) / (2.0 * k)         # int_0^1 psi^2
    return Spectrum(mu, psi_xi * overlap / norm2)


def gate_spectrum(b: float, ell: float, kmax: float = 200.0, dk: float = 0.004) -> Spectrum:
    """Exact symmetric modes for killing rate V=b/ell on |x-1/2|<=ell/2 (no discretization)."""
    x0 = 0.5 - ell / 2
    V = b / ell

    def inner(k):
        s = k * k - 2.0 * V                       # psi'' = -s psi inside the gate
        q = np.sqrt(np.abs(s))
        h = q * ell / 2
        osc = s >= 0
        psi_in = np.where(osc, np.cos(h), np.cosh(h))
        dpsi_in = np.where(osc, q * np.sin(h), -q * np.sinh(h))     # derivative at x0 (y=-ell/2)
        safe_q = np.where(q > 1e-12, q, 1.0)
        g1 = np.where(q > 1e-12, np.where(osc, np.sin(h), np.sinh(h)) / safe_q, ell / 2)
        g2 = np.where(q > 1e-12,
                      ell / 4 + np.where(osc, np.sin(2 * h), np.sinh(2 * h)) / (4 * safe_q),
                      ell / 2)
        return psi_in, dpsi_in, g1, g2

    def F(k):
        psi_in, dpsi_in, _, _ = inner(k)
        return k * np.cos(k * x0) * psi_in - np.sin(k * x0) * dpsi_in

    grid = np.arange(0.01, kmax, dk)
    fg = F(grid)
    idx = np.where(np.sign(fg[:-1]) * np.sign(fg[1:]) < 0)[0]
    lo, hi = grid[idx].copy(), grid[idx + 1].copy()
    flo = fg[idx].copy()
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        fm = F(mid)
        same = np.sign(fm) == np.sign(flo)
        lo = np.where(same, mid, lo)
        flo = np.where(same, fm, flo)
        hi = np.where(same, hi, mid)
    k = 0.5 * (lo + hi)
    psi_in, dpsi_in, g1, g2 = inner(k)
    sx0 = np.sin(k * x0)
    alpha = np.where(np.abs(sx0) > 1e-3, psi_in / np.where(np.abs(sx0) > 1e-3, sx0, 1.0),
                     dpsi_in / (k * np.cos(k * x0)))
    overlap = 2.0 * (alpha * (1.0 - np.cos(k * x0)) / k + g1)
    norm2 = 2.0 * (alpha**2 * (x0 / 2 - np.sin(2 * k * x0) / (4 * k)) + g2)
    psi_xi = alpha * np.sin(k * XI)
    return Spectrum(0.5 * k**2, psi_xi * overlap / norm2)


# ----------------------------------------------------------------------------------------------
# Observables
# ----------------------------------------------------------------------------------------------
def windows(tp, tv, frac=FRAC):
    return (tp * (1 - frac), tp * (1 + frac)), (tv * (1 - frac), tv * (1 + frac))


def window_ratio(sp: Spectrum, pwin, vwin):
    p = sp.window_prob(*pwin)
    v = sp.window_prob(*vwin)
    R = (p / (pwin[1] - pwin[0])) / (v / (vwin[1] - vwin[0]))
    return R, p, v


def critical_points(sp: Spectrum, lo=1.5e-3, hi=0.5, n=6000):
    t = np.exp(np.linspace(math.log(lo), math.log(hi), n))
    d = sp.dphi(t)
    out = []
    for i in np.where(np.sign(d[:-1]) * np.sign(d[1:]) < 0)[0]:
        r = brentq(lambda s: sp.dphi(s)[0], t[i], t[i + 1], xtol=1e-15, rtol=1e-14)
        kind = "max" if d[i] > 0 else "min"
        out.append((kind, r, float(sp.phi(r)[0])))
    return out


def inflection(sp: Spectrum, lo=0.022, hi=0.06, n=800):
    """tau where Phi' has its local max between the valley and the late peak (Phi'' from + to -)."""
    t = np.linspace(lo, hi, n)
    d2 = sp.d2phi(t)
    cand = [i for i in np.where((d2[:-1] > 0) & (d2[1:] <= 0))[0]]
    if not cand:
        raise RuntimeError("no + to - inflection in bracket")
    # the one with largest Phi'
    best = max(cand, key=lambda i: sp.dphi(t[i])[0])
    return brentq(lambda s: sp.d2phi(s)[0], t[best], t[best + 1], xtol=1e-15, rtol=1e-14)


def fold(make_spectrum, blo, bhi):
    def g(b):
        sp = make_spectrum(b)
        return float(sp.dphi(inflection(sp))[0])
    gl, gh = g(blo), g(bhi)
    if not (gl > 0 > gh):
        raise RuntimeError(f"fold not bracketed: g({blo})={gl}, g({bhi})={gh}")
    b = brentq(g, blo, bhi, xtol=1e-13, rtol=1e-14)
    sp = make_spectrum(b)
    return b, inflection(sp)


def slope(make_spectrum, b, pwin, vwin):
    vals = {}
    for h in (1e-3, 1e-4, 1e-5):
        rp = window_ratio(make_spectrum(b + h), pwin, vwin)[0]
        rm = window_ratio(make_spectrum(b - h), pwin, vwin)[0]
        vals[h] = (rp - rm) / (2 * h)
    return vals


def n_trials(R, p, v, dRdb, db):
    return R**2 * (1 / p + 1 / v) / (dRdb * db) ** 2


def rel(a, b):
    return abs(a - b) / abs(b)


def main() -> None:
    results: dict = {}
    say("Window-ratio numbers of the proposed measurement (Sec. VI, App. D)")
    say(f"model: D=1/2 on [0,1], absorbing ends, gate at theta={THETA}, start xi={XI}")
    say("")

    # 0. cross-check the density against dpma_saddle_node_bc_theta.phi_vals
    from dpma_saddle_node_bc_theta import phi_vals  # noqa: E402
    taus = np.array([0.01, 0.02, 0.0253, 0.0346, 0.0462, 0.08, 0.2])
    for b in (B_REF, 3.1366):
        mine = delta_spectrum(b).phi(taus)
        theirs = phi_vals(taus, XI, THETA, b)
        say(f"[0] Phi cross-check vs phi_vals at b={b}: max rel diff = "
            f"{np.max(np.abs(mine - theirs) / np.abs(theirs)):.2e}")
    sp_ref = delta_spectrum(B_REF, 4000)
    say(f"    normalization S(0.002) with 4000 modes at b=2.5: {sp_ref.S(0.002)[0]:.10f} "
        f"(should be ~1, small-tau mass not yet absorbed)")
    say("")

    # 1. where (tau_p, tau_v) come from
    say("[1] Critical points of the exact delta-sink Phi at the reference strength b=2.5:")
    cps = critical_points(delta_spectrum(B_REF))
    for kind, t, ph in cps:
        say(f"    {kind}: tau={t:.7f}  Phi={ph:.6f}")
    tv_exact = [t for k, t, _ in cps if k == "min"][0]
    tp_exact = [t for k, t, _ in cps if k == "max"][-1]
    say(f"    => valley tau_v={tv_exact:.6f} -> 0.0253 ; late peak tau_p={tp_exact:.6f} -> 0.0462 "
        f"(4-decimal rounding reproduces the quoted centres: "
        f"{round(tv_exact, 4) == TAU_V_QUOTED and round(tp_exact, 4) == TAU_P_QUOTED})")
    results["b_ref"] = B_REF
    results["tau_v_exact"], results["tau_p_exact"] = tv_exact, tp_exact
    results["critical_points_b2.5"] = cps
    for b in (2.0, 3.0):
        c = critical_points(delta_spectrum(b))
        say(f"    (for comparison, b={b}: " + ", ".join(f"{k} {t:.5f}" for k, t, _ in c) + ")")
    say("")

    pwin, vwin = windows(TAU_P_QUOTED, TAU_V_QUOTED)
    say(f"[2] Windows: peak [{pwin[0]:.5f}, {pwin[1]:.5f}], valley [{vwin[0]:.5f}, {vwin[1]:.5f}] "
        f"(disjoint: {vwin[1] < pwin[0]})")
    results["peak_window"], results["valley_window"] = pwin, vwin

    # 3. the upper fold b_+ (delta sink)
    bplus, tauc = fold(lambda b: delta_spectrum(b), 3.0, 3.3)
    say(f"[3] delta-sink upper fold: b_+ = {bplus:.10f} (quoted 3.13660; bisection value "
        f"{QUOTED['b_plus_bisection']:.10f}; diff {bplus - QUOTED['b_plus_bisection']:+.2e}); "
        f"tau_c = {tauc:.7f}")
    say(f"    fold time lies in the gap between the windows: {vwin[1] < tauc < pwin[0]}")
    for b in (bplus - 0.01, bplus + 0.01):
        n = len(critical_points(delta_spectrum(b)))
        say(f"    critical points at b={b:.4f}: {n}")
    from dpma_saddle_node_bc_theta import solve_fold  # Newton solver of S1=S2=0
    b_newton, t_newton = solve_fold(THETA, (3.1366, 0.0346), xi=XI, tol=1e-13, max_iter=30)
    say(f"    cross-check, Newton solver of S1=S2=0: b_+={b_newton:.10f}, tau_c={t_newton:.7f}; "
        f"the bisection value 3.1366025759 (valley/peak count on a log grid) is "
        f"{QUOTED['b_plus_bisection'] - bplus:+.1e} off (grid resolution; immaterial)")
    results["b_plus"], results["tau_c"] = bplus, tauc
    results["b_plus_newton"] = b_newton
    say("")

    # 4. R_c, p, v, centre-point ratio, slope
    sp_c = delta_spectrum(bplus)
    Rc, p, v = window_ratio(sp_c, pwin, vwin)
    centre = float(sp_c.phi(TAU_P_QUOTED)[0] / sp_c.phi(TAU_V_QUOTED)[0])
    sl = slope(lambda b: delta_spectrum(b), bplus, pwin, vwin)
    dRdb = sl[1e-4]
    say("[4] Fixed-window observable at b_+ (delta sink):")
    say(f"    R_c = {Rc:.7f}  (quoted 0.9699549; rel diff {rel(Rc, QUOTED['R_c']):.1e})")
    say(f"    p   = {p:.7f}  (quoted 0.1045292; rel diff {rel(p, QUOTED['p']):.1e})")
    say(f"    v   = {v:.7f}  (quoted 0.0590153; rel diff {rel(v, QUOTED['v']):.1e})")
    say(f"    centre-point ratio Phi(tau_p)/Phi(tau_v) = {centre:.5f}  (quoted 0.9785)")
    say("    dR/db (central differences): " + ", ".join(f"h={h:g}: {s:.8f}" for h, s in sl.items())
        + f"  (quoted -0.0883849)")
    results.update(R_c=Rc, p=p, v=v, R_centre_point=centre, dR_db=dRdb)
    say("    R(b) table vs reference values:")
    for b, q in QUOTED["R_exact_table"].items():
        r = window_ratio(delta_spectrum(b), pwin, vwin)[0]
        say(f"      b={b:.2f}: R={r:.6f} (quoted {q:.6f}, diff {r - q:+.1e})")
    for db in (-0.25, 0.25, -0.1, 0.1):
        r = window_ratio(delta_spectrum(bplus + db), pwin, vwin)[0]
        say(f"      linearity: R(b_+{db:+.2f}) = {r:.6f}; linear prediction {Rc + dRdb * db:.6f}")
    say("")

    # 5. variance and trial counts
    sd4000 = Rc * math.sqrt((1 / p + 1 / v) / 4000)
    n25 = n_trials(Rc, p, v, dRdb, 0.25)
    n10 = n_trials(Rc, p, v, dRdb, 0.10)
    say("[5] Delta method: Var(R) = R^2 (1/p + 1/v) / N_t  (two disjoint multinomial bins)")
    say(f"    s.d.(R) at N_t=4000: {sd4000:.5f} (quoted 0.07897)")
    say(f"    one-sigma |delta b|=0.25: N_t = {n25:,.0f} (quoted 51,086 ~ 5.1e4); expected counts "
        f"peak {n25 * p:,.0f}, valley {n25 * v:,.0f}")
    say(f"    one-sigma |delta b|=0.10: N_t = {n10:,.0f} (quoted 319,287 ~ 3.2e5); expected counts "
        f"peak {n10 * p:,.0f}, valley {n10 * v:,.0f}")
    say(f"    95% factor 1.95996^2 = {1.959964 ** 2:.4f} (quoted 3.84)")
    results.update(sd_R_N4000=sd4000, N_025=n25, N_010=n10)

    rng = np.random.default_rng(20261012)
    wp, wv = pwin[1] - pwin[0], vwin[1] - vwin[0]
    for N in (4000, int(round(n25))):
        cnt = rng.multinomial(N, [p, v, 1 - p - v], size=200_000)
        R_s = (cnt[:, 0] / wp) / (cnt[:, 1] / wv)
        b_hat = bplus + (R_s - Rc) / dRdb
        say(f"    multinomial sampling, N_t={N}: s.d.(R)={np.std(R_s):.5f} "
            f"(delta {Rc * math.sqrt((1 / p + 1 / v) / N):.5f}); s.d.(b_hat)={np.std(b_hat):.4f}")
    say("")

    # 6. design sensitivity: reference b for the centres and window half-width
    say("[6] Design sensitivity of the trial count (delta sink, one-sigma, delta b = 0.25 / 0.1):")
    sens = []
    for bref in (2.0, 2.5, 3.0):
        c = critical_points(delta_spectrum(bref))
        tv_ = [t for k, t, _ in c if k == "min"][0]
        tp_ = [t for k, t, _ in c if k == "max"][-1]
        for frac in (0.1, 0.2, 0.3):
            pw, vw = windows(tp_, tv_, frac)
            if vw[1] >= pw[0]:
                say(f"    b_ref={bref}, +/-{int(frac * 100)}%: windows overlap -> not admissible")
                continue
            R_, p_, v_ = window_ratio(sp_c, pw, vw)
            s_ = slope(lambda b: delta_spectrum(b), bplus, pw, vw)[1e-4]
            row = (bref, frac, tp_, tv_, R_, p_, v_, s_, n_trials(R_, p_, v_, s_, 0.25),
                   n_trials(R_, p_, v_, s_, 0.1))
            sens.append(row)
            say(f"    b_ref={bref}, +/-{int(frac * 100)}%: centres ({tp_:.4f},{tv_:.4f}) R_c={R_:.4f} "
                f"p={p_:.4f} v={v_:.4f} dR/db={s_:.4f}  N(0.25)={row[8]:,.0f}  N(0.1)={row[9]:,.0f}")
    results["design_sensitivity"] = sens
    say("")

    # 7. exact finite-width gate (independent of the finite-difference calculation)
    say("[7] Exact finite-width gate (piecewise-analytic eigenfunctions, no mesh):")
    sp_test = gate_spectrum(B_REF, 1e-4)
    say(f"    ell=1e-4 vs delta sink at b=2.5, Phi rel diff at tau=0.0253/0.0462: "
        + ", ".join(f"{abs(a / b_ - 1):.1e}" for a, b_ in zip(sp_test.phi([0.0253, 0.0462]),
                                                            delta_spectrum(B_REF).phi([0.0253, 0.0462]))))
    fin = {}
    for ell in (0.04, 0.02, 0.01, 0.005, 0.0025):
        bl, tl = fold(lambda b, e=ell: gate_spectrum(b, e), 2.8, 3.3)
        Rl = window_ratio(gate_spectrum(bl, ell), pwin, vwin)[0]
        fin[ell] = (bl, tl, Rl)
        extra = ""
        if ell in QUOTED["b_plus_ell"]:
            extra = (f"  [FD-Richardson {QUOTED['b_plus_ell'][ell]:.6f} (diff {bl - QUOTED['b_plus_ell'][ell]:+.1e});"
                     f" R_c^FD {QUOTED['R_c_ell'][ell]:.6f}]")
        say(f"    ell={ell:<6}: b_+^(ell)={bl:.7f}  tau_c={tl:.6f}  R at own fold={Rl:.6f}{extra}")
    e3 = np.array([0.04, 0.02, 0.01])
    b0 = float(np.polyfit(e3, [fin[e][0] for e in e3], 2)[2])
    say(f"    quadratic ell->0 extrapolation from ell=0.04,0.02,0.01: {b0:.6f} "
        f"(paper 3.13657; delta sink {bplus:.6f})")
    shift = bplus - fin[0.02][0]
    say(f"    finite-gate systematic at ell=0.02: b_+ - b_+^(ell) = {shift:.4f} "
        f"(= {shift / 0.25:.2f} of the 0.25 target, {shift / 0.1:.2f} of the 0.1 target)")
    # what a delta-sink calibration would infer if the real gate has ell = 0.02
    Rcl = fin[0.02][2]
    say(f"    R at the ell=0.02 fold ({Rcl:.6f}) differs from the delta-sink R_c by {Rcl - Rc:+.2e}; "
        f"the fold-point ratio is nearly width-independent, but the b at which it occurs shifts.")
    results["finite_gate"] = {str(k): v for k, v in fin.items()}
    results["b_plus_ell_to_0_quadratic"] = b0

    (TABLES / "measurement_window_ratio.txt").write_text("\n".join(LINES) + "\n")
    (TABLES / "measurement_window_ratio.json").write_text(
        json.dumps(results, indent=2, default=lambda o: float(o) if hasattr(o, "__float__") else str(o))
        + "\n")
    print("\nwrote outputs/tables/measurement_window_ratio.txt and .json")


if __name__ == "__main__":
    main()
