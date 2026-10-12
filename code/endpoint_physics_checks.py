#!/usr/bin/env python3
"""Physical reading of the near-end law b_c ~ B*/d, tau_c ~ c_* d^2 (Sec. III D).

Run from anywhere:  python code/endpoint_physics_checks.py

Conventions (Eq. (1) and App. B): generator (1/2) d^2/dx^2 on [0,1], absorbing
ends, delta-sink  -b delta(x-theta) p.  Boundary layer: y = x/d, sigma = tau/d^2, B = b d, half-line
(0,inf), absorbing at 0, sink at y=1, source-started X=1.  Half-line arrival-time transform
(kappa = sqrt(2s), r0(1,1) = (1-e^{-2 kappa})/kappa):
    escape  E(s) = e^{-kappa} / (1 + B r0),      capture C(s) = B r0 / (1 + B r0),
    total   T(s) = E + C   (= Eq. (fhl) at X = 1).
On the branch cut kappa = i v, B r0 = g e^{-iv} with g = 2B sin(v)/v, so
    Phi_X(sigma) = (1/pi) int_0^inf v e^{-sigma v^2/2} K_X(v) dv,
    K_T = (1+g) sin v / den,  K_E = sin v / den,  K_C = g sin v / den,  den = 1 + 2 g cos v + g^2.

Sections
  1  units, jump condition, absorption length, capture odds (lattice-exact and continuum)
  2  valley / inherited-peak tracking of Phi_hl(sigma;B,1) for B in [0, B*]
  3  capture probability at the fold (endpoint and interval)
  4  heuristics (clearly labelled) for why the inherited peak dies near B ~ 0.8
  5  Phi' = -S1, Phi'' = S2: same fold criterion, two representations
  6  small lattice Monte Carlo realisation of the half-line density (2e5 walkers)
Output is printed and mirrored to outputs/tables/endpoint_physics_checks.txt.
"""
from __future__ import annotations

import csv
import math
import time

import numpy as np
import mpmath as mp
from scipy.optimize import brentq, minimize_scalar
from scipy.special import erfcx

from dpma_saddle_node_bc_theta import M_theta, roots, G_affected, G_node
from paths import DATA, TABLES

OUT: list[str] = []


def say(s: str = "") -> None:
    print(s, flush=True)
    OUT.append(s)


BSTAR_REF, CSTAR_REF = 0.7890261736, 0.1579221011   # dpma_halfline_bstar.py

# ----------------------------------------------------------------------------------------------
# half-line branch-cut integrals (same quadrature as dpma_halfline_bstar.phi_derivs,
# extended to the escape/capture channels and to cumulative probabilities)
# ----------------------------------------------------------------------------------------------

def _simpson_w(n):
    w = np.ones(n)
    w[1:-1:2] = 4.0
    w[2:-1:2] = 2.0
    return w


def _kernel(v, B, which):
    sv = np.sin(v)
    with np.errstate(divide="ignore", invalid="ignore"):
        g = np.where(v > 0, 2.0 * B * sv / v, 2.0 * B)
    den = 1.0 + 2.0 * g * np.cos(v) + g * g
    if which == "T":
        return (1.0 + g) * sv / den
    if which == "E":
        return sv / den
    if which == "C":
        return g * sv / den
    raise ValueError(which)


def phi_hl(sigma, B, m=0, which="T", npts=200_001):
    """(d/dsigma)^m of the half-line density (channel 'T','E','C'), source-started X=1."""
    vmax = math.sqrt(2.0 * 85.0 / sigma)
    v = np.linspace(0.0, vmax, npts)
    base = v * np.exp(-0.5 * sigma * v * v) * _kernel(v, B, which) * (-0.5 * v * v) ** m
    return float(np.sum(_simpson_w(npts) * base) * (v[1] - v[0]) / 3.0 / math.pi)


def cum_hl(sigma, B, which, npts=400_001):
    """Cumulative probability absorbed by time sigma through channel 'C' or 'E'.

    L^{-1}[F(s)/s] = F(0) + (2/pi) int_0^inf e^{-sigma v^2/2} Im F(-v^2/2+i0)/v dv,
    with Im C = -g sin v/den, Im E = -sin v/den, C(0) = 2B/(1+2B), E(0) = 1/(1+2B).
    """
    vmax = math.sqrt(2.0 * 85.0 / sigma)
    v = np.linspace(1e-12, vmax, npts)
    sv = np.sin(v)
    g = 2.0 * B * sv / v
    den = 1.0 + 2.0 * g * np.cos(v) + g * g
    k = (g if which == "C" else 1.0) * sv / (v * den)
    integral = float(np.sum(_simpson_w(npts) * np.exp(-0.5 * sigma * v * v) * k) * (v[1] - v[0]) / 3.0)
    f0 = 2.0 * B / (1.0 + 2.0 * B) if which == "C" else 1.0 / (1.0 + 2.0 * B)
    return f0 - 2.0 / math.pi * integral


def halfline_fold(sigma=0.158, B=0.789, dB=1e-6):
    """2D Newton on Phi'=Phi''=0 (same scheme as dpma_halfline_bstar.py)."""
    for _ in range(12):
        F = np.array([phi_hl(sigma, B, 1, npts=400_001), phi_hl(sigma, B, 2, npts=400_001)])
        J = np.array([
            [phi_hl(sigma, B, 2, npts=400_001),
             (phi_hl(sigma, B + dB, 1, npts=400_001) - phi_hl(sigma, B - dB, 1, npts=400_001)) / (2 * dB)],
            [phi_hl(sigma, B, 3, npts=400_001),
             (phi_hl(sigma, B + dB, 2, npts=400_001) - phi_hl(sigma, B - dB, 2, npts=400_001)) / (2 * dB)],
        ])
        step = np.linalg.solve(J, -F)
        sigma += float(step[0])
        B += float(step[1])
        if max(abs(step)) < 1e-11:
            break
    return sigma, B


def levy(s):
    return np.exp(-1.0 / (2.0 * s)) / np.sqrt(2.0 * np.pi * s ** 3)


# ----------------------------------------------------------------------------------------------
# interval mode sums (source-started xi = theta), with the capture channel b p(theta,tau)
# ----------------------------------------------------------------------------------------------

def interval_modes(th, b, xi=None, wmax=40.0):
    if xi is None:
        xi = th
    G, mu, cw = [], [], []
    for w in roots(th, b, wmax=wmax):
        k = 2.0 * w
        mu.append(2.0 * w * w)
        if abs(math.sin(2.0 * th * w)) < 1e-3:
            n = int(round(2.0 * w / math.pi))
            G.append(G_node(n, xi))
            cw.append(0.0)
        else:
            G.append(G_affected(w, xi, th, b))
            A, Bb = math.sin(k * th), math.sin(k * (1.0 - th))
            J = 0.5 * (th * Bb * Bb + (1.0 - th) * A * A) - A * Bb * math.sin(k) / (2.0 * k)
            cw.append(b * (A * Bb) ** 2 / J)       # capture current weights b psi(theta)^2/||psi||^2
    return np.array(G), np.array(mu), np.array(cw)


def interval_moments(tau, G, mu):
    e = G * np.exp(-mu * tau)
    return [float(np.sum(e * mu ** n)) for n in range(4)]   # Phi, S1, S2, S3


# ==============================================================================================
def main() -> None:
    t0 = time.time()
    SS, BS = halfline_fold()
    say("=" * 92)
    say("Near-end law: physical checks")
    say("=" * 92)
    say(f"Half-line fold re-solved here: B* = {BS:.10f}, sigma* = c_* = {SS:.10f}")
    say(f"  (reference dpma_halfline_bstar.py: B* = {BSTAR_REF}, c_* = {CSTAR_REF};"
        f" |dB| = {abs(BS-BSTAR_REF):.1e}, |dsigma| = {abs(SS-CSTAR_REF):.1e})")

    # ------------------------------------------------------------------------------------------
    say("\n[1] Units, jump condition, absorption length, capture odds")
    say("-" * 92)
    say("1a. Diffusive clock.  Lazy walk: per-step displacement variance q (sites^2); with x=r/N,")
    say("    tau = q t/N^2 the variance of x is exactly q t/N^2 = tau, i.e. generator (1/2)d^2/dx^2,")
    say("    <x^2> = tau.  Free first-passage density over a distance d (B=0 limit, Levy-Smirnov):")
    for sg in (0.05, 1.0 / 3.0, 0.8):
        num = phi_hl(sg, 0.0, npts=400_001)
        say(f"    sigma={sg:.4f}: branch-cut={num:.12f}  sigma^-3/2 e^-1/(2sigma)/sqrt(2pi)={levy(sg):.12f}"
            f"  rel={abs(num/levy(sg)-1):.1e}")
    pk = brentq(lambda s: phi_hl(s, 0.0, 1, npts=400_001), 0.2, 0.5, xtol=1e-14)
    say(f"    Levy mode (most probable crossing time) = {pk:.10f} d^2  (exact 1/3)")

    say("1b. Jump condition on interval eigenfunctions (Eq. (1): d_tau p = p''/2 - b delta p).")
    say("    Integrating -psi''/2 + b delta psi = mu psi across theta gives psi'(th+)-psi'(th-) = 2b psi(th).")
    th, b = 0.3, 2.5
    ws = [w for w in roots(th, b) if abs(math.sin(2 * th * w)) > 1e-3][:3]
    for w in ws:
        k = 2 * w
        psi = lambda x: (math.sin(k * (1 - th)) * math.sin(k * x) if x <= th
                         else math.sin(k * th) * math.sin(k * (1 - x)))
        h = 1e-6
        dp = (psi(th + h) - psi(th)) / h - (psi(th) - psi(th - h)) / h
        exact_jump = -k * math.sin(k)                       # App. B
        say(f"    theta={th}, b={b}, k={k:.6f}: FD jump={dp:+.6f}, -k sin k={exact_jump:+.6f},"
            f" 2b psi(theta)={2*b*psi(th):+.6f}, D_theta(k;b)={k*math.sin(k)+2*b*math.sin(k*th)*math.sin(k*(1-th)):+.1e}")
    say("    -> sign/convention: jump = +2b psi(theta) (a downward kink, local depletion at the sink),")
    say("       consistent with D_theta = k sin k + 2b sin(k th) sin(k(1-th)) = 0 (Eq. (Dtheta)) and App. B.")

    say("1c. Absorption length.  Sink current b p(theta) equals the diffusive current (1/2) p(theta)/l into")
    say("    a perfectly absorbing wall at distance l = l_abs = 1/(2b) [equivalently, a profile approaching")
    say("    the sink from one side extrapolates to zero a distance 1/(2b) beyond it, as for a Robin wall].")
    say(f"    Hence B = b d = d/(2 l_abs); at the endpoint fold d/l_abs = 2B* = {2*BS:.7f}.")
    say("    Physical units (dp/dt = D p'' - kappa delta p): l_abs = D/kappa, 2B = kappa d/D,")
    say(f"    t_c = c_* d^2/(2D) = {SS/2:.7f} d^2/D  (free Levy mode d^2/(6D) = {1/6:.7f} d^2/D).")

    say("1d. Capture odds = 2B.  Current balance: sink strength b vs the diffusive current 1/(2d) through the gap.")
    say("    Continuum: pi_sc = lim_{s->0} B r0(1,1)/(1+B r0(1,1)) with r0(1,1)->2  => 2B/(1+2B).")
    for Bt in (0.1, 0.5, BS, 2.0):
        s = 1e-14
        kap = math.sqrt(2 * s)
        r0 = -math.expm1(-2 * kap) / kap
        say(f"    B={Bt:.7f}: C(s=1e-14)={Bt*r0/(1+Bt*r0):.10f}  2B/(1+2B)={2*Bt/(1+2*Bt):.10f}")
    say("    Lattice (ring convention: hop q/2 each way, shortcut lambda at the sink, B = lambda M/q):")
    say("    per step at the sink: capture lambda vs leave-and-reach-wall-before-return (q/2)(1/M);")
    say("    odds = 2 lambda M/q = 2B for EVERY M.  Numerical check (truncated half-line, reflecting far end):")
    for M, q in ((5, 0.5), (20, 0.5), (20, 0.2)):
        lam = BS * q / M
        L = 6 * M
        A = np.zeros((L, L)); rhs = np.zeros(L)            # unknowns u(1..L)
        for n in range(1, L + 1):
            i = n - 1
            stay = 1 - q - (lam if n == M else 0.0)
            A[i, i] += 1 - stay
            for nb in (n - 1, n + 1):
                if nb == 0:
                    continue                                   # absorbing wall: u(0)=0
                nb = min(nb, L)                                # reflecting: attempted-outside-stays
                A[i, nb - 1] -= q / 2
            if n == M:
                rhs[i] = lam
        u = np.linalg.solve(A, rhs)
        say(f"    M={M:3d}, q={q}: u(M) = {u[M-1]:.12f}   2B/(1+2B) = {2*BS/(1+2*BS):.12f}")

    # ------------------------------------------------------------------------------------------
    say("\n[2] Valley / inherited peak of Phi_hl(sigma;B,1) as B increases to B*")
    say("-" * 92)

    def extrema(B, lo=0.02, hi=3.0, n=600):
        s = np.exp(np.linspace(math.log(lo), math.log(hi), n))
        d = np.array([phi_hl(x, B, 1) for x in s])
        out = []
        for i in range(n - 1):
            if max(abs(d[i]), abs(d[i + 1])) < 1e-6:          # B=0: e^{-1/(2 sigma)} underflow noise
                continue
            if d[i] * d[i + 1] < 0:
                r = brentq(lambda x: phi_hl(x, B, 1), s[i], s[i + 1], xtol=1e-13)
                out.append(("valley" if d[i] < 0 else "peak", r))
        return out

    def smallB_valley(B):                                      # small-B valley, y = 1/(2 sigma_v)
        y = brentq(lambda y: y - 2 * math.log(y) - math.log(4 / B), 2.0001, 400)
        return 1 / (2 * y)

    say(f"  {'B':>9} {'sigma_v':>9} {'sigma_p':>9} {'Phi(sig_v)':>11} {'Phi(sig_p)':>11} {'prominence':>11}"
        f" {'#extrema':>8} {'small-B sig_v':>12}")
    for B in (0.0, 0.1, 0.2, 0.4, 0.6, 0.7, 0.75, 0.78, 0.785, 0.788):
        ex = extrema(B)
        sv = [r for k_, r in ex if k_ == "valley"]
        sp = [r for k_, r in ex if k_ == "peak"]
        v_, p_ = (sv[0] if sv else float("nan")), sp[0]
        fv = phi_hl(v_, B) if sv else float("nan")
        fp = phi_hl(p_, B)
        d5 = smallB_valley(B) if B > 0 else float("nan")
        say(f"  {B:9.4f} {v_:9.6f} {p_:9.6f} {fv:11.6f} {fp:11.6f} {fp-fv:11.3e} {len(ex):8d} {d5:12.4f}")
    say(f"  {BS:9.7f} {SS:9.6f} {SS:9.6f} {phi_hl(SS, BS):11.6f} {phi_hl(SS, BS):11.6f} {0.0:11.3e}"
        f" {'merge':>8} {'':>12}")
    for B in (0.8, 0.9):
        say(f"  {B:9.4f}  extrema in sigma in [0.02,3]: {extrema(B)}  (pair gone)")
    say("  B=0: no valley (no capture singularity); single Levy peak at 1/3.  For 0<B<B* the valley")
    say("  rises (0.067 -> 0.158) and the inherited peak moves earlier (0.333 -> 0.158) until they merge.")
    say("  Prominence vs B*-B near the fold (normal form predicts exponent 3/2):")
    pr = []
    for dB in (4e-3, 2e-3, 1e-3):
        B = BS - dB
        ex = extrema(B, lo=0.12, hi=0.2, n=200)
        sv = [r for k_, r in ex if k_ == "valley"][0]
        sp = [r for k_, r in ex if k_ == "peak"][0]
        pr.append((dB, phi_hl(sp, B) - phi_hl(sv, B), sp - sv))
    for (d1, p1, g1), (d2, p2, g2) in zip(pr[:-1], pr[1:]):
        say(f"    B*-B {d1:.0e}->{d2:.0e}: prominence slope {math.log(p1/p2)/math.log(d1/d2):.4f},"
            f" gap slope {math.log(g1/g2)/math.log(d1/d2):.4f}")

    # ------------------------------------------------------------------------------------------
    say("\n[3] Capture (splitting) probability at the fold")
    say("-" * 92)
    pis = 2 * BS / (1 + 2 * BS)
    say(f"  endpoint: pi_sc = 2B*/(1+2B*) = {pis:.7f}; escape (near wall) = {1-pis:.7f}; odds 2B* = {2*BS:.7f}")
    say("  numerical check of C(0): cumulative capture from the branch cut vanishes as sigma->0:")
    for sg in (1e-4, 1e-3):
        say(f"    sigma={sg:.0e}: Cap_cum={cum_hl(sg, BS, 'C'):.7e}  local B sqrt(2s/pi)-B^2 s/2="
            f"{BS*math.sqrt(2*sg/math.pi)-BS*BS*sg/2:.7e}")
    say("  interval fold points (data/dpma_boundary_generated.csv; xi=theta):")
    say(f"  {'theta':>7} {'b_c':>9} {'b_c d':>9} {'tau_c/d^2':>9} {'pi_sc':>8} {'odds cap:near':>13}"
        f" {'far-wall factor':>15}")
    rows = list(csv.DictReader(open(DATA / "dpma_boundary_generated.csv")))
    sel = []
    for r in rows:
        th_ = float(r["theta"])
        if abs(th_ * 20 - round(th_ * 20)) < 1e-9 or r["is_min"] == "true":
            sel.append((th_, float(r["b_c"]), float(r["tau_c"])))
    for th_, bc, tc in sel:
        d = min(th_, 1 - th_)
        P = th_ * (1 - th_)
        pisc = 2 * bc * P / (1 + 2 * bc * P)
        far = math.exp(-((1 - d) ** 2 - d ** 2) / (2 * tc))    # Levy far/near wall weight at tau_c
        say(f"  {th_:7.4f} {bc:9.5f} {bc*d:9.6f} {tc/d/d:9.6f} {pisc:8.5f} {2*bc*d:13.6f} {far:15.2e}")
    say("  pi_sc at the fold is NOT universal: 0.505 (theta~0.381) .. 0.606 (theta=1/2) .. 0.612 (endpoint).")
    say("  For theta<=0.30 the fold obeys b_c d = B* to <2e-5 although pi_sc = 0.525-0.558 != 0.612: the far")
    say("  wall changes the eventual capture probability but is out of diffusive reach at tau_c (Levy far/near")
    say("  weight <1e-6).  The local odds capture:near-wall escape = 2 b_c d is the endpoint-controlling number.")
    say(f"  check theta=1/2: 2*3.07643*0.25/(1+2*3.07643*0.25) = {2*3.07643*.25/(1+2*3.07643*.25):.5f};"
        f" theta=0.381: 2*2.165*0.381*0.619/(1+...) = {2*2.165*.381*.619/(1+2*2.165*.381*.619):.5f}")

    # ------------------------------------------------------------------------------------------
    say("\n[4] Why the inherited peak dies near B ~ 0.8")
    say("-" * 92)
    say("4a (exact).  Phi = Phi_cap + Phi_esc.  Phi_cap = B p(1,sigma) is a diagonal heat kernel of a")
    say("    self-adjoint nonnegative operator => completely monotone (strictly decreasing, convex).")
    G_, mu_, cw_ = interval_modes(0.3, 2.63014)
    say(f"    interval check (theta=0.3,b=2.63014): capture weights b psi_j(theta)^2/||psi_j||^2 all > 0:"
        f" {bool(np.all(cw_[cw_ != 0] > 0))} (min {cw_[cw_ != 0].min():.3e}, {int(np.sum(cw_ != 0))} affected modes)")
    grid = np.exp(np.linspace(math.log(0.03), math.log(3.0), 300))
    for B in (0.3, BS, 1.5):
        cp = np.array([phi_hl(s, B, 1, "C") for s in grid])
        cpp = np.array([phi_hl(s, B, 2, "C") for s in grid])
        ep = np.array([phi_hl(s, B, 1, "E") for s in grid])
        say(f"    half-line B={B:.4f}: max Phi_cap' = {cp.max():+.3e} (<0), min Phi_cap'' = {cpp.min():+.3e} (>0),"
            f" sign changes of Phi_esc' = {int(np.sum(np.diff(np.sign(ep)) != 0))} (unimodal escape flux)")
    split = max(abs(phi_hl(s, B, 0, "T") - phi_hl(s, B, 0, "C") - phi_hl(s, B, 0, "E"))
                for s in (0.05, 0.158, 1 / 3, 0.8) for B in (0.3, BS, 1.5))
    say(f"    channel split closes: max |Phi - Phi_cap - Phi_esc| over 12 test points = {split:.1e}")
    say("    => all interior structure comes from the escape channel; a valley/peak pair exists iff")
    say("       max_sigma [Phi_esc'(sigma) + Phi_cap'(sigma)] > 0  (rise of escape flux beats decay of capture).")
    T_ = phi_hl(SS, BS); C_ = phi_hl(SS, BS, which="C"); E_ = phi_hl(SS, BS, which="E")
    Cp_ = phi_hl(SS, BS, 1, "C"); Ep_ = phi_hl(SS, BS, 1, "E")
    say(f"    at the fold: Phi={T_:.6f}, Phi_cap={C_:.6f}, Phi_esc={E_:.6f}; Phi_cap'={Cp_:+.6f}, Phi_esc'={Ep_:+.6f}")
    say(f"    log-slopes sigma d ln/d sigma: capture {SS*Cp_/C_:+.4f} (free sink: -1/2), escape {SS*Ep_/E_:+.4f}")

    say("4b (controlled approximation).  Drop terms in which the walker returns from the wall to the sink")
    say("    (r0(1,1) = (1-e^{-2kappa})/kappa -> 1/kappa; error O(e^{-2/sigma}) = "
        f"{math.exp(-2/SS):.1e} at sigma*).  Then T = (kappa e^-kappa + B)/(kappa + B) inverts in closed form:")
    say("      Phi_cap ~ B/sqrt(2 pi s) - (B^2/2) e^{B^2 s/2} erfc(B sqrt(s/2))        [= B K_b, Eq. (localheat)]")
    say("      Phi_esc ~ Levy(s) - B e^{-1/(2s)}/sqrt(2 pi s) + (B^2/2) e^{B+B^2 s/2} erfc(1/sqrt(2s)+B sqrt(s/2))")

    def cap_loc(s, B):
        return B / np.sqrt(2 * np.pi * s) - 0.5 * B * B * erfcx(B * np.sqrt(s / 2))

    def esc_loc(s, B):
        z = 1 / np.sqrt(2 * s) + B * np.sqrt(s / 2)
        return levy(s) - B * np.exp(-1 / (2 * s)) / np.sqrt(2 * np.pi * s) + 0.5 * B * B * erfcx(z) * np.exp(-1 / (2 * s))

    for s in (0.05, SS, 1 / 3, 0.8):
        ex_ = phi_hl(s, BS)
        say(f"    sigma={s:.4f}: (cap+esc)_closed/Phi_hl - 1 = {(cap_loc(s, BS)+esc_loc(s, BS))/ex_-1:+.2e}")
    mp.mp.dps = 40

    def Phi_loc_mp(s, B):
        s = mp.mpf(s); B = mp.mpf(B)
        cap = B / mp.sqrt(2 * mp.pi * s) - B * B / 2 * mp.exp(B * B * s / 2) * mp.erfc(B * mp.sqrt(s / 2))
        esc = (mp.exp(-1 / (2 * s)) / mp.sqrt(2 * mp.pi * s ** 3) - B * mp.exp(-1 / (2 * s)) / mp.sqrt(2 * mp.pi * s)
               + B * B / 2 * mp.exp(B + B * B * s / 2) * mp.erfc(1 / mp.sqrt(2 * s) + B * mp.sqrt(s / 2)))
        return cap + esc

    sol = mp.findroot(lambda s, B: [mp.diff(lambda x: Phi_loc_mp(x, B), s, 1),
                                    mp.diff(lambda x: Phi_loc_mp(x, B), s, 2)],
                      (mp.mpf("0.158"), mp.mpf("0.789")))
    sl, Bl = float(sol[0]), float(sol[1])
    say(f"    fold of the closed form: B = {Bl:.7f} (rel {Bl/BS-1:+.1e}), sigma = {sl:.7f} (rel {sl/SS-1:+.1e})")

    say("4c HEURISTIC (non-interacting channels, leading order in B sqrt(sigma)): Phi ~ Levy(s) + B/sqrt(2 pi s).")
    say("    Pair exists iff B < max_s Levy'(s)/|d/ds (2 pi s)^{-1/2}| = max_s e^{-1/(2s)}(1/s^2 - 3/s).")
    R0 = lambda s: math.exp(-1 / (2 * s)) * (1 / s ** 2 - 3 / s)
    r = minimize_scalar(lambda s: -R0(s), bounds=(0.05, 0.5), method="bounded", options={"xatol": 1e-12})
    say(f"    maximum at s = {r.x:.8f} (exact 1/6), value {-r.fun:.8f} = 18 e^-3 = {18*math.exp(-3):.8f}")
    say(f"    vs exact B* = {BS:.6f} ({18*math.exp(-3)/BS-1:+.1%}), c_* = {SS:.6f} ({(1/6)/SS-1:+.1%}).")
    say(f"    at s=1/6 the heuristic capture share of the outflux is 3/4 (exact half-line fold: {C_/T_:.4f}).")
    h1 = lambda s: (1 - mp.exp(-1 / (2 * s))) / mp.sqrt(2 * mp.pi * s)
    Lv = lambda s: mp.exp(-1 / (2 * s)) / mp.sqrt(2 * mp.pi * s ** 3)
    ratio1 = lambda s: float(mp.diff(Lv, s) / (-mp.diff(h1, s)))
    opt1 = minimize_scalar(lambda s: -ratio1(s), bounds=(0.05, 0.5), method="bounded", options={"xatol": 1e-10})
    say(f"    adding only the O(B) escape depletion gives {-opt1.fun:.4f} at s={opt1.x:.4f} ({-opt1.fun/BS-1:+.1%}): the")
    say("    O(B^2) self-depletion terms of 4b matter; the leading-order estimate is a ~14% heuristic only.")

    say("4d HEURISTIC (current balance at the free Levy mode s=1/3): free shapes give B = 3 e^{-3/2} ="
        f" {3*math.exp(-1.5):.4f} ({3*math.exp(-1.5)/BS-1:+.1%});")
    Bx = brentq(lambda B: phi_hl(1 / 3, B, which="C") - phi_hl(1 / 3, B, which="E"), 0.3, 2.0, xtol=1e-10)
    say(f"    exact channels: Phi_cap(1/3)=Phi_esc(1/3) at B = {Bx:.4f} ({Bx/BS-1:+.1%}); at B*:"
        f" Phi_esc/Phi_cap(1/3) = {phi_hl(1/3, BS, which='E')/phi_hl(1/3, BS, which='C'):.4f}.  Weaker than 4c.")

    say("4e Channel bookkeeping at the fold (endpoint vs interval folds):")
    cC, cE = cum_hl(SS, BS, "C"), cum_hl(SS, BS, "E")
    say(f"    half-line: survival S(sigma*) = {1-cC-cE:.6f}; absorbed so far: capture {cC:.6f}, escape {cE:.6f};"
        f" fraction of eventual captures done {cC/pis:.4f}, of eventual escapes {cE/(1-pis):.4f};"
        f" capture share of outflux {C_/T_:.6f}")
    say(f"  {'theta':>7} {'cap share':>10} {'survival':>9} {'pi_sc':>8}")
    for th_, bc, tc in sel:
        G_, mu_, cw_ = interval_modes(th_, bc)
        e = np.exp(-mu_ * tc)
        Phi_i = float(np.sum(G_ * e)); cap_i = float(np.sum(cw_ * e)); S_i = float(np.sum(G_ / mu_ * e))
        P = th_ * (1 - th_)
        say(f"  {th_:7.4f} {cap_i/Phi_i:10.5f} {S_i:9.5f} {2*bc*P/(1+2*bc*P):8.5f}")
    say("    The capture share of the instantaneous outflux at the fold is the most nearly constant of these")
    say("    (0.646-0.696 over theta in [0.2,0.5]); it is an observation, not a criterion.  The fold is decided")
    say("    early: ~78% of walkers are still alive at the endpoint fold and <3% of eventual escapes have occurred.")

    # ------------------------------------------------------------------------------------------
    say("\n[5] Phi' = -S1, Phi'' = S2: one fold criterion, two representations")
    say("-" * 92)
    for th, b, tc in ((0.3, 2.0, 0.02), (0.5, 3.0, 0.03)):          # generic (non-fold) points
        G_, mu_, _ = interval_modes(th, b, wmax=60.0)
        Phi0, S1, S2, S3 = interval_moments(tc, G_, mu_)
        h1s, h2s = 1e-6 * tc, 1e-3 * tc
        fd1 = (interval_moments(tc + h1s, G_, mu_)[0] - interval_moments(tc - h1s, G_, mu_)[0]) / (2 * h1s)
        fd2 = (interval_moments(tc + h2s, G_, mu_)[0] - 2 * Phi0 + interval_moments(tc - h2s, G_, mu_)[0]) / h2s ** 2
        say(f"  interval mode sum (theta={th}, b={b}, tau={tc}): FD Phi' = {fd1:+.8e}, -S1 = {-S1:+.8e}"
            f" (rel {abs(fd1+S1)/abs(S1):.1e}); FD Phi'' = {fd2:+.6e}, S2 = {S2:+.6e} (rel {abs(fd2-S2)/abs(S2):.1e})")
    for s, B in ((0.12, 0.7), (0.3, 0.5)):
        hs = 1e-3 * s
        f0 = phi_hl(s, B, npts=400_001)
        fp, fm = phi_hl(s + hs, B, npts=400_001), phi_hl(s - hs, B, npts=400_001)
        a1, a2 = phi_hl(s, B, 1, npts=400_001), phi_hl(s, B, 2, npts=400_001)
        say(f"  half-line branch cut (sigma={s}, B={B}): FD Phi' = {(fp-fm)/(2*hs):+.8f} vs v-weighted integral"
            f" {a1:+.8f}; FD Phi'' = {(fp-2*f0+fm)/hs**2:+.6f} vs {a2:+.6f}")
    say("  d->0 correspondence: d^2 Phi_int(d^2 s; b=B/d) -> Phi_hl, -d^4 S1 -> Phi_hl', d^6 S2 -> Phi_hl''")
    for th, s, B in ((0.2, 0.3, 0.5), (0.25, 0.12, 0.7), (0.3, 0.2, 1.0), (0.2, SS, BS)):
        d = th
        G_, mu_, _ = interval_modes(th, B / d, wmax=70.0)
        P0, P1, P2, _ = interval_moments(d * d * s, G_, mu_)
        h0, h1_, h2 = phi_hl(s, B, 0, npts=400_001), phi_hl(s, B, 1, npts=400_001), phi_hl(s, B, 2, npts=400_001)
        say(f"    theta={th}, B={B:.6f}, sigma={s:.6f}: d^2Phi={d**2*P0:.10f} vs {h0:.10f};"
            f" -d^4S1={-d**4*P1:+.8f} vs {h1_:+.8f}; d^6S2={d**6*P2:+.6f} vs {h2:+.6f}")
    say("  => the criterion is literally the same (Phi'=Phi''=0 <=> S1=S2=0); only the representation of Phi")
    say("     changes (discrete mode sum on the interval, branch-cut integral on the half-line).")

    # ------------------------------------------------------------------------------------------
    say("\n[6] Lattice Monte Carlo realisation of the half-line density")
    say("-" * 92)
    rng = np.random.default_rng(20261012)
    M, q, nw, smax = 20, 0.5, 200_000, 1.0
    tmax = int(round(smax * M * M / q))
    nbin_steps = 10
    for B in (0.4, BS):
        lam = B * q / M
        pos = np.full(nw, M, dtype=np.int32)
        alive = np.ones(nw, dtype=bool)
        t_abs = np.zeros(nw, dtype=np.int32)
        chan = np.zeros(nw, dtype=np.int8)                 # 1 capture, 2 escape
        idx = np.arange(nw)
        for t in range(1, tmax + 1):
            a = idx[alive]
            if a.size == 0:
                break
            u = rng.random(a.size)
            x = pos[a]
            at = x == M
            cap = at & (u < lam)
            uu = np.where(at, u - lam, u)                  # shortcut diverted from the self-loop
            step = np.where(uu < q / 2, -1, np.where(uu < q, 1, 0))
            step[cap] = 0
            x = x + step
            pos[a] = x
            esc = (x == 0) & ~cap
            done = cap | esc
            t_abs[a[done]] = t
            chan[a[cap]] = 1
            chan[a[esc]] = 2
            alive[a[done]] = False
        edges_t = np.arange(0, tmax + 1, nbin_steps)
        edges_s = edges_t * q / M ** 2
        z = []
        for ch, lab in ((1, "C"), (2, "E")):
            cnt, _ = np.histogram(t_abs[chan == ch], bins=edges_t + 0.5)
            cum = np.array([0.0] + [cum_hl(s, B, lab) for s in edges_s[1:]])
            expct = nw * np.diff(cum)
            ok = expct > 20
            z.append((cnt[ok] - expct[ok]) / np.sqrt(expct[ok]))
            say(f"  B={B:.6f} channel {lab}: MC absorbed by sigma=1: {np.mean(chan == ch):.5f}"
                f"  vs branch cut {cum[-1]:.5f}")
        zz = np.concatenate(z)
        tot_cnt, _ = np.histogram(t_abs[chan > 0], bins=edges_t + 0.5)
        dens = tot_cnt / nw / (nbin_steps * q / M ** 2)
        mids = 0.5 * (edges_s[1:] + edges_s[:-1])
        say(f"  B={B:.6f}: per-bin z-scores (both channels, {zz.size} bins): chi2/dof = {np.mean(zz**2):.3f},"
            f" max|z| = {np.max(np.abs(zz)):.2f};  M={M}, q={q}, {nw} walkers, censored at sigma=1")
        for s_ in (0.095, SS, 0.25, 0.5):
            i = int(np.argmin(np.abs(mids - s_)))
            pred = (cum_hl(edges_s[i + 1], B, "C") + cum_hl(edges_s[i + 1], B, "E")
                    - (cum_hl(edges_s[i], B, "C") + cum_hl(edges_s[i], B, "E") if i > 0 else 0.0)) \
                / (edges_s[i + 1] - edges_s[i])
            say(f"    bin centred {mids[i]:.4f}: MC density {dens[i]:.4f} +- {np.sqrt(tot_cnt[i])/nw/(edges_s[i+1]-edges_s[i]):.4f},"
                f" bin-averaged Phi_hl {pred:.4f}")
    say(f"\nruntime {time.time()-t0:.1f} s")

    (TABLES / "endpoint_physics_checks.txt").write_text("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
