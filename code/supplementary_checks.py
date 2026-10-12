#!/usr/bin/env python3
"""Supplementary checks of physical statements in the text.

Independent of the other solver modules (own root finder, own amplitudes, own half-line kernel).
  A  interval mode sums: midpoint fold; signs of G_j; exit-time density without a sink; b_c(theta)
     near its minimum; fold without the far-end flux; capture share at the valley and the delayed
     peak (theta = xi = 1/2, b = 2.6); Hellmann-Feynman slope d mu_1/db at b_c; the shoulder of
     Fig. 1(d); fold lines at xi = 0.40; mean absorption time near an end.
  B  half-line: branch-cut kernel vs Talbot inversion; the fold; bookkeeping at the merger
     (surviving, captured, reached the wall); crude estimates with and without depletion.
  C  lattice capture current lambda <u|P^(t-1)|u> (q = 2/3, N = 100).
  D  lower fold line b_-(xi) near the sink from a weak-sink expansion.
  E  release at the sink: at most one interior maximum over b in [0.02, 300] (40 values) at
     theta = 0.05, 0.1, 0.2, 0.3, 0.381, 0.45, 0.5; fold margins along b_c(theta).
Usage:  python code/supplementary_checks.py [A|B|C|D|E|all]   (default all)
Writes outputs/tables/supplementary_checks.txt (or supplementary_checks_<part>.txt).
"""
from __future__ import annotations

import io
import math
import sys
from contextlib import redirect_stdout

import mpmath as mp
import numpy as np
from scipy.optimize import brentq, fsolve
from scipy.special import erfcx

from paths import TABLES

# ----------------------------------------------------------------------------------------------
# Interval: point sink of strength b at theta, absorbing ends, generator (1/2) d^2/dx^2
# ----------------------------------------------------------------------------------------------


def roots_theta(th, b, kmax, step=0.004):
    ks = np.arange(1e-7, kmax, step)
    D = ks * np.sin(ks) + 2 * b * np.sin(ks * th) * np.sin(ks * (1 - th))
    idx = np.where(np.sign(D[:-1]) * np.sign(D[1:]) < 0)[0]
    f = lambda k: k * math.sin(k) + 2 * b * math.sin(k * th) * math.sin(k * (1 - th))
    return np.array([brentq(f, ks[i], ks[i + 1], xtol=1e-14, rtol=1e-15) for i in idx])


class Interval:
    """Mode data for release at xi; channel-resolved coefficients (near end 0, far end 1, sink)."""

    def __init__(self, th, b, xi, kmax=400.0):
        self.th, self.b, self.xi = th, b, xi
        ks = roots_theta(th, b, kmax)
        rows = []
        for k in ks:
            al, be = math.sin(k * th), math.sin(k * (1 - th))
            if abs(al) < 1e-9 or abs(be) < 1e-9:      # node mode sin(n pi x)
                n = int(round(k / math.pi))
                psi = lambda x: math.sin(n * math.pi * x)
                J = 0.5
                dpsi0, dpsi1, psith = n * math.pi, n * math.pi * (-1) ** n, 0.0
                k = n * math.pi
            else:
                psi = (lambda x, k=k, al=al, be=be: be * math.sin(k * x) if x <= th
                       else al * math.sin(k * (1 - x)))
                J = 0.5 * (th * be ** 2 + (1 - th) * al ** 2) - al * be * math.sin(k) / (2 * k)
                dpsi0 = k * be                      # psi'(0)
                dpsi1 = -k * al                     # psi'(1)
                psith = al * be
            px = psi(xi)
            c_near = 0.5 * dpsi0 * px / J          # flux into x=0
            c_far = -0.5 * dpsi1 * px / J          # flux into x=1
            c_cap = b * psith * px / J             # capture current
            rows.append((k, k * k / 2, c_near, c_far, c_cap, psith ** 2 / J))
        a = np.array(rows)
        self.k, self.mu = a[:, 0], a[:, 1]
        self.cn, self.cf, self.cc, self.hf = a[:, 2], a[:, 3], a[:, 4], a[:, 5]
        self.G = self.cn + self.cf + self.cc

    def S(self, n, tau, which="all", kcut=None):
        c = {"all": self.G, "near": self.cn, "far": self.cf, "cap": self.cc,
             "nofar": self.cn + self.cc}[which]
        m = slice(None) if kcut is None else (self.k <= kcut)
        tau = np.atleast_1d(tau)
        E = np.exp(-np.outer(tau, self.mu[m]))
        with np.errstate(all="ignore"):   # some BLAS builds raise spurious flags on underflow
            return E @ (c[m] * self.mu[m] ** n)

    def phi(self, tau, which="all", kcut=None):
        return self.S(0, tau, which, kcut)


def fold_solve(th, xi, guess, which="all", kmax=400.0):
    """Solve S1=S2=0 for (tau,b) (Phi'=-S1, Phi''=S2); variables in logs."""
    def F(v):
        tau, b = math.exp(v[0]), math.exp(v[1])
        I = Interval(th, b, xi, kmax)
        s1 = I.S(1, tau, which)[0]
        s2 = I.S(2, tau, which)[0]
        s0 = I.S(0, tau, which)[0]
        return [tau * s1 / s0, tau * tau * s2 / s0]
    v = fsolve(F, [math.log(guess[0]), math.log(guess[1])], xtol=1e-12, full_output=True)
    tau, b = math.exp(v[0][0]), math.exp(v[0][1])
    return tau, b, np.max(np.abs(v[1]["fvec"]))


def count_extrema(I, taus, which="all", rel_floor=1e-9):
    s1 = I.S(1, taus, which)
    ph = I.S(0, taus, which)
    good = ph > rel_floor * ph.max()
    d = -s1[good]
    t = taus[good]
    sc = np.where(np.sign(d[:-1]) * np.sign(d[1:]) < 0)[0]
    maxima = [t[i] for i in sc if d[i] > 0]
    minima = [t[i] for i in sc if d[i] < 0]
    return maxima, minima


def partA():
    print("=" * 100)
    print("PART A: interval mode sums (independent root finder and amplitudes)")
    # A0 sanity: midpoint fold
    tc, bc, res = fold_solve(0.5, 0.5, (0.0384, 3.07))
    print(f"[A0] midpoint fold: b_c={bc:.7f} tau_c={tc:.7f} (resid {res:.1e}); paper 3.0764, 0.03836")

    # A1 sign pattern of G_j for release at the sink
    print("[A1] sign pattern of G_j (release at the sink), first 16 modes")
    for th, b in ((0.5, 2.0), (0.381, 2.0), (0.3, 2.6), (0.2, 3.9), (0.1, 7.9)):
        I = Interval(th, b, th, kmax=120)
        sg = np.sign(I.G[:16]).astype(int)
        alt = all(sg[i] * sg[i + 1] < 0 for i in range(len(sg) - 1))
        print(f"   theta={th:5.3f} b={b:4.1f}: {''.join('+' if s > 0 else '-' for s in sg)}  strictly alternating={alt}")

    # A2 b=0 exit-time density unimodality (image series, exact)
    print("[A2] b=0 exit-time density from theta: number of interior maxima (image series)")
    def exitdens(th, tau, nimg=30):
        out = np.zeros_like(tau)
        dout = np.zeros_like(tau)
        for m in range(-nimg, nimg + 1):
            for a in (th + 2 * m, (1 - th) + 2 * m):
                e = np.exp(-a * a / (2 * tau))
                out += a * e / np.sqrt(2 * np.pi * tau ** 3)
                dout += a * e * tau ** -3.5 * (a * a / 2 - 1.5 * tau) / np.sqrt(2 * np.pi)
        return out, dout
    for th in (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.381, 0.45, 0.5):
        taus = np.geomspace(th * th / 40, 3.0, 20000)
        f, df = exitdens(th, taus)
        sc = np.where(np.sign(df[:-1]) * np.sign(df[1:]) < 0)[0]
        print(f"   theta={th:6.3f}: sign changes of Phi0'={len(sc)}, at tau={[round(taus[i], 5) for i in sc]}")

    # A3 minimum of b_c(theta)
    print("[A3] b_c(theta) near its minimum (fold solve, k<=400)")
    g = (0.0384, 3.0764)
    rows = []
    for th in (0.5, 0.45, 0.42, 0.40, 0.39, 0.385, 0.382, 0.381, 0.380, 0.375, 0.37, 0.35, 0.30, 0.25, 0.2):
        tc, bc, res = fold_solve(th, th, g)
        g = (tc, bc)
        d = min(th, 1 - th)
        rows.append((th, bc, tc))
        print(f"   theta={th:5.3f}: b_c={bc:.6f} tau_c={tc:.6f} b_c*d={bc * d:.6f} tau_c/d^2={tc / d / d:.6f} resid={res:.1e}")

    # A3b: is the rise of b_c*d above B* caused by far-end arrivals? Fold with the far-end flux removed
    print("[A3b] fold of Phi minus the far-end flux (capture + near-end only), same interval")
    lookup = {round(r[0], 3): (r[2], r[1]) for r in rows}
    for th in (0.30, 0.35, 0.381, 0.42, 0.5):
        tc0, bc0 = lookup[th]
        tc1, bc1, res = fold_solve(th, th, (tc0, bc0), which="nofar")
        d = min(th, 1 - th)
        print(f"   theta={th:5.3f}: full b_c*d={bc0 * d:.5f}; without far-end flux b_c*d={bc1 * d:.5f}"
              f" (tau_c/d^2 {tc0 / d / d:.4f} -> {tc1 / d / d:.4f}); B*=0.78903; resid {res:.1e}")

    # A4 channel shares at valley / delayed peak, midpoint, b=2.6
    print("[A4] channel shares, theta=xi=1/2, b=2.6 (continuum)")
    I = Interval(0.5, 2.6, 0.5, kmax=600)
    taus = np.geomspace(2e-4, 0.5, 40000)
    mx, mn = count_extrema(I, taus)
    print(f"   maxima at {mx}, minima at {mn}")
    for name, t in (("valley", mn[0]), ("delayed peak", mx[-1])):
        tot = I.phi(t)[0]
        cap = I.phi(t, "cap")[0]
        print(f"   {name:13s} tau={t:.5f}: Phi={tot:.5f}, capture share={cap / tot:.3f}, ends share={1 - cap / tot:.3f}")
    dcap = -I.S(1, taus, "cap")
    print(f"   capture current monotone decreasing on [2e-4,0.5]: {bool(np.all(dcap < 0))}")
    # same at theta=0.38 lattice-figure parameters
    I38 = Interval(0.38, 1.5, 0.38, kmax=600)
    mx, mn = count_extrema(I38, taus)
    t = mx[-1]
    print(f"   theta=xi=0.38, b=1.5: delayed peak tau={t:.5f}, capture share={I38.phi(t, 'cap')[0] / I38.phi(t)[0]:.3f}")

    # A5 Hellmann-Feynman slope at the midpoint fold
    I = Interval(0.5, 3.0764323604, 0.5, kmax=50)
    print(f"[A5] d mu_1/db at midpoint b_c = psi(theta)^2/J = {I.hf[0]:.4f} (paper 1.10);"
          f" check by finite difference: {(Interval(0.5, 3.0774323604, 0.5, 20).mu[0] - Interval(0.5, 3.0754323604, 0.5, 20).mu[0]) / 0.002:.4f}")

    # A6 shoulder at b=4.5 (Fig. 1d): zeros of Phi'' after the spike
    print("[A6] Fig. 1(d) 'shoulder' at theta=xi=1/2, b=4.5: inflection points (zeros of Phi''=S2)")
    for b in (3.3, 4.5, 6.0, 8.0):
        I = Interval(0.5, b, 0.5, kmax=600)
        taus = np.geomspace(5e-4, 0.5, 30000)
        s2 = I.S(2, taus)
        s1 = I.S(1, taus)
        sc = np.where(np.sign(s2[:-1]) * np.sign(s2[1:]) < 0)[0]
        # flatness: min over tau of |Phi'|/Phi * tau in the shoulder region
        lg = -s1 * taus / I.S(0, taus)
        win = (taus > 0.01) & (taus < 0.1)
        print(f"   b={b:4.1f}: inflections at tau={[round(taus[i], 4) for i in sc]};"
              f" max of tau*Phi'/Phi on [0.01,0.1] = {lg[win].max():.3f}")

    # A7 off-gate xi=0.40: b_- and b_+
    print("[A7] release at xi=0.40, midpoint sink: fold lines")
    taus = np.geomspace(1e-4, 1.0, 30000)
    for b in (1.0, 1.8, 1.85, 2.5, 3.85, 3.87, 4.5):
        I = Interval(0.5, b, 0.40, kmax=600)
        mx, mn = count_extrema(I, taus)
        print(f"   b={b:5.2f}: maxima at {[round(x, 5) for x in mx]}, minima at {[round(x, 5) for x in mn]}")
    for guess in ((0.006, 1.84), (0.035, 3.86)):
        tc, bc, res = fold_solve(0.5, 0.40, guess, kmax=600)
        print(f"   fold from guess {guess}: b={bc:.4f} tau={tc:.5f} resid={res:.1e}")
    # late-peak channel shares at xi=0.44, b=2.5
    I = Interval(0.5, 2.5, 0.44, kmax=600)
    mx, mn = count_extrema(I, np.geomspace(5e-4, 1, 30000))
    for t in mx:
        print(f"   xi=0.44,b=2.5 maximum tau={t:.5f}: capture share {I.phi(t, 'cap')[0] / I.phi(t)[0]:.3f}")

    # A9 mean: share from far-end exits, small d, b=B*/d
    print("[A9] mean absorption time near an end (theta=d, b=B*/d): channel probabilities and mean shares")
    Bs = 0.7890261736
    for d in (0.1, 0.05, 0.02):
        b = Bs / d
        I = Interval(d, b, d, kmax=3000.0 if d < 0.03 else 1500.0)
        P = {w: float(np.sum(c / I.mu)) for w, c in (("near", I.cn), ("far", I.cf), ("cap", I.cc))}
        M = {w: float(np.sum(c / I.mu ** 2)) for w, c in (("near", I.cn), ("far", I.cf), ("cap", I.cc))}
        mean = sum(M.values())
        exact = d * (1 - d) / (1 + 2 * b * d * (1 - d))
        den = 1 + 2 * b * d * (1 - d)          # closed forms, Eqs. (pisccont) and (pnear), release at the sink
        Pc = {"cap": 2 * b * d * (1 - d) / den, "near": (1 - d) / den, "far": d / den}
        print(f"   d={d}: closed forms P(cap)={Pc['cap']:.5f} P(near)={Pc['near']:.5f} P(far)={Pc['far']:.5f}; "
              f"odds cap:near={Pc['cap'] / Pc['near']:.5f} = 2bd={2 * b * d:.5f}; P(far)/d={Pc['far'] / d:.3f}")
        print(f"          (truncated mode sums, which converge slowly for the probabilities: P(cap)={P['cap']:.4f} "
              f"P(near)={P['near']:.4f} P(far)={P['far']:.4f}, total {sum(P.values()):.4f})")
        print(f"          mean={mean:.6f} (Eq.: {exact:.6f}; d/(1+2B*)={d / (1 + 2 * Bs):.6f}); shares of mean: far-end exits {M['far'] / mean:.3f},"
              f" captures {M['cap'] / mean:.3f}, near-end exits {M['near'] / mean:.3f}")


# ----------------------------------------------------------------------------------------------
# Half-line (wall at y=0, sink at y=1, release at the sink)
# ----------------------------------------------------------------------------------------------
BSTAR, CSTAR = 0.7890261736, 0.1579221011


def kernel_phi(sig, B, nder=0):
    f = lambda v: (v * mp.e ** (-sig * v * v / 2) * (1 + 2 * B * mp.sin(v) / v) * mp.sin(v)
                   / (1 + 4 * B * mp.sin(v) * mp.cos(v) / v + (2 * B * mp.sin(v) / v) ** 2) * (-v * v / 2) ** nder)
    vmax = mp.sqrt(2 * 60 / sig)
    pts = [0] + [mp.pi * j for j in range(1, int(vmax / mp.pi) + 2)]
    return mp.quad(f, pts) / mp.pi


def fhat(s, B):
    k = mp.sqrt(2 * s)
    return (k * mp.e ** (-k) + B * (1 - mp.e ** (-2 * k))) / (k + B * (1 - mp.e ** (-2 * k)))


def partB():
    mp.mp.dps = 25
    print("=" * 100)
    print("PART B: half-line")
    print("[B1] branch-cut kernel (Eq. phihlsource) vs Talbot inversion of Eq. (fhlsource)")
    for B in (0.5, BSTAR, 1.2):
        for sig in (0.05, CSTAR, 0.5, 2.0):
            a = kernel_phi(sig, B)
            t = mp.invertlaplace(lambda s: fhat(s, B), sig, method="talbot")
            print(f"   B={float(B):.4f} sigma={sig:.4f}: kernel={float(a):.10f} talbot={float(t):.10f} rel={float(abs(a / t - 1)):.1e}")
    print("[B2] fold of the kernel")
    mp.mp.dps = 20
    def F(s, B):
        return [kernel_phi(s, B, 1), kernel_phi(s, B, 2)]
    sol = mp.findroot(lambda s, B: F(s, B), (mp.mpf("0.158"), mp.mpf("0.789")))
    print(f"   sigma*={float(sol[0]):.9f}, B*={float(sol[1]):.9f}  (paper c*=0.15792, B*=0.78903)")
    mp.mp.dps = 25
    print("[B3] bookkeeping at the fold (Talbot on transforms / s)")
    B = mp.mpf(BSTAR)
    k = lambda s: mp.sqrt(2 * s)
    C = lambda s: B * (1 - mp.e ** (-2 * k(s))) / (k(s) + B * (1 - mp.e ** (-2 * k(s))))
    W = lambda s: k(s) * mp.e ** (-k(s)) / (k(s) + B * (1 - mp.e ** (-2 * k(s))))
    capt = mp.invertlaplace(lambda s: C(s) / s, CSTAR, method="talbot")
    esc = mp.invertlaplace(lambda s: W(s) / s, CSTAR, method="talbot")
    print(f"   captured by sigma*: {float(capt):.6f}; reached wall: {float(esc):.6f}; surviving: {float(1 - capt - esc):.6f}")
    print(f"   eventual capture prob 2B/(1+2B)={float(2 * B / (1 + 2 * B)):.6f}; odds 2B*={float(2 * B):.6f}")
    capd = mp.invertlaplace(C, CSTAR, method="talbot")
    escd = mp.invertlaplace(W, CSTAR, method="talbot")
    print(f"   at sigma*: capture current {float(capd):.6f}, wall flux {float(escd):.6f}, capture share {float(capd / (capd + escd)):.4f}")

    print("[B4] crude estimates: which depletion lowers 0.896 to 0.789? (closed forms, wall->sink returns dropped)")
    def levy(s):
        return s ** -1.5 * np.exp(-1 / (2 * s)) / math.sqrt(2 * math.pi)
    def cap_free(s, B):
        return B / math.sqrt(2 * math.pi * s)
    def cap_self(s, B):   # free point-sink capture current B K_B (self-depletion)
        return B / math.sqrt(2 * math.pi * s) - 0.5 * B * B * erfcx(B * math.sqrt(s / 2))
    def esc_dep(s, B):    # Levy flux depleted by the sink
        z = 1 / math.sqrt(2 * s) + B * math.sqrt(s / 2)
        return levy(s) - B * math.exp(-1 / (2 * s)) / math.sqrt(2 * math.pi * s) + 0.5 * B * B * math.exp(-1 / (2 * s)) * erfcx(z)
    variants = {
        "independent (free capture + free Levy)": lambda s, B: cap_free(s, B) + levy(s),
        "capture self-depleted only": lambda s, B: cap_self(s, B) + levy(s),
        "wall flux depleted by sink only": lambda s, B: cap_free(s, B) + esc_dep(s, B),
        "both (= exact up to O(e^{-2/sigma}))": lambda s, B: cap_self(s, B) + esc_dep(s, B),
    }
    for name, fn in variants.items():
        def eqs(v, fn=fn):
            s, B = v
            h = 1e-4 * s
            f0, fp, fm = fn(s, B), fn(s + h, B), fn(s - h, B)
            fpp, fmm = fn(s + 2 * h, B), fn(s - 2 * h, B)
            d1 = (fp - fm) / (2 * h)
            d2 = (fp - 2 * f0 + fm) / h ** 2
            return [s * d1 / f0, s * s * d2 / f0]
        s, Bf = fsolve(eqs, [0.16, 0.85], xtol=1e-12)
        print(f"   {name:42s}: sigma={s:.5f}, B={Bf:.5f}")
    # wall effect on the capture current at the fold (relative size of the wall image term)
    print(f"   relative wall correction to the capture current at sigma*: e^(-2/sigma*) = {math.exp(-2 / CSTAR):.1e}")


# ----------------------------------------------------------------------------------------------
# Lattice: capture current monotonicity (q=2/3, N=100)
# ----------------------------------------------------------------------------------------------
def partC():
    print("=" * 100)
    print("PART C: lattice capture current f_sc(t)=lambda <u|P^(t-1)|u>, q=2/3")
    q, N = 2 / 3, 100
    for u, b in ((50, 2.6), (38, 1.5), (38, 2.16)):
        lam = b * q / N
        n = N - 1
        P = np.zeros((n, n))
        for i in range(n):
            P[i, i] = 1 - q
            if i > 0:
                P[i, i - 1] = q / 2
            if i < n - 1:
                P[i, i + 1] = q / 2
        P[u - 1, u - 1] -= lam
        ev = np.linalg.eigvalsh(P)
        v = np.zeros(n); v[u - 1] = 1.0
        fsc, fend, F = [], [], []
        for t in range(1, 4001):
            fsc.append(lam * v[u - 1]); fend.append(q / 2 * (v[0] + v[-1])); F.append(fsc[-1] + fend[-1])
            with np.errstate(all="ignore"):
                v = P @ v
        fsc = np.array(fsc); F = np.array(F)
        inc = np.where(np.diff(fsc) > 0)[0] + 1
        dF = np.diff(F)
        locmax = [t + 1 for t in range(1, len(F) - 1) if F[t] > F[t - 1] and F[t] >= F[t + 1]]
        print(f"   u={u}, b={b}: min eigenvalue of P_lambda={ev.min():.4f}; steps t with f_sc(t)>f_sc(t-1): {list(inc[:12] + 1)}"
              f" (tau=qt/N^2 up to {q * (inc.max() + 1) / N ** 2 if len(inc) else 0:.1e})")
        print(f"            local maxima of the total F(t) at t={locmax[:6]} (tau={[round(q * t / N ** 2, 5) for t in locmax[:6]]})")


# ----------------------------------------------------------------------------------------------
# b_- for release near the midpoint sink: weak-sink linearisation with exact image series
# ----------------------------------------------------------------------------------------------
def partD():
    mp.mp.dps = 40
    print("=" * 100)
    print("PART D: lower fold b_- (theta=1/2) from Phi = Phi0_xi + b p0(1/2,tau|xi) + O(b^2), image series")
    def Phi0(t, xi, nder):
        s = mp.mpf(0)
        for m in range(-6, 7):
            for a in (xi + 2 * m, 1 - xi + 2 * m):
                s += mp.diff(lambda tt: a * mp.e ** (-a * a / (2 * tt)) / mp.sqrt(2 * mp.pi * tt ** 3), t, nder)
        return s
    def p0(t, xi, nder):
        s = mp.mpf(0)
        for m in range(-6, 7):
            for z, sg in ((mp.mpf(1) / 2 - xi + 2 * m, 1), (mp.mpf(1) / 2 + xi + 2 * m, -1)):
                s += sg * mp.diff(lambda tt: mp.e ** (-z * z / (2 * tt)) / mp.sqrt(2 * mp.pi * tt), t, nder)
        return s
    for xi in (mp.mpf("0.42"), mp.mpf("0.43"), mp.mpf("0.44")):
        delta = mp.mpf(1) / 2 - xi
        a = xi; A = a * a - delta * delta
        t0 = delta ** 2 * (1 + 2 * delta ** 2 / A)
        g = lambda t: Phi0(t, xi, 1) / Phi0(t, xi, 2) - p0(t, xi, 1) / p0(t, xi, 2)
        t = mp.findroot(g, (t0 * mp.mpf('0.97'), t0 * mp.mpf('1.03')), solver='secant', tol=mp.mpf(10) ** -30)
        b = -Phi0(t, xi, 1) / p0(t, xi, 1)
        asym = mp.e * a * A ** 2 / (2 * delta ** 6) * mp.e ** (-A / (2 * delta ** 2))
        print(f"   xi={float(xi):.2f} (delta={float(delta):.2f}): tau_-={float(t):.6f} (asymptote {float(t0):.6f}), b_-={float(b):.4e},"
              f" b_-/asymptote={float(b / asym):.3f}")
        if float(xi) == 0.44:
            # how much probability does the early capture peak carry just above b_-?
            for bb in (2 * b, 10 * b):
                ratio = (bb * p0(t, xi, 0)) / (Phi0(t, xi, 0) + bb * p0(t, xi, 0))
                print(f"      b={float(bb):.1e}: density at tau_- = {float(Phi0(t, xi, 0) + bb * p0(t, xi, 0)):.2e}"
                      f" (late-peak density at b=2.5 is ~5.39); probability captured by the sink ~ b*E[local time] <= {float(bb) * 2 * 0.44 * 0.5:.1e}")


# ----------------------------------------------------------------------------------------------
# E: release at the sink -- number of interior maxima; fold margins along b_c(theta)
# ----------------------------------------------------------------------------------------------
def partE():
    print("=" * 100)
    print("PART E: release at the sink")
    print("[E1] number of interior maxima over b in [0.02,300] (40 values, log-spaced)")
    bs = np.geomspace(0.02, 300, 40)
    for th in (0.05, 0.1, 0.2, 0.3, 0.381, 0.45, 0.5):
        d = min(th, 1 - th)
        counts = []
        for b in bs:
            tmin = d * d / 200
            kmax = math.sqrt(2 * 45 / tmin)
            I = Interval(th, b, th, kmax=kmax)
            taus = np.geomspace(tmin, 3.0, 8000)
            mx, mn = count_extrema(I, taus)
            counts.append(len(mx))
        largest = max([b for b, c in zip(bs, counts) if c >= 1], default=0)
        print(f"   theta={th}: max interior maxima={max(counts)}; b values with a delayed peak: "
              f"{sum(1 for c in counts if c >= 1)} (largest such b={largest:.3f})")
    print("[E2] fold margins along b_c(theta): tau^3 Phi'''/Phi, tau dPhi'/db/Phi, b tau dPhi'/db/Phi")
    g = (0.0384, 3.0764)
    for th in (0.5, 0.45, 0.4, 0.381, 0.35, 0.3, 0.25, 0.2):
        tc, bc, _ = fold_solve(th, th, g)
        g = (tc, bc)
        I = Interval(th, bc, th, 400.0)
        phi = I.S(0, tc)[0]
        s3 = I.S(3, tc)[0]
        h = 1e-5 * bc
        s1p = Interval(th, bc + h, th, 400.0).S(1, tc)[0]
        s1m = Interval(th, bc - h, th, 400.0).S(1, tc)[0]
        dbphi1 = -(s1p - s1m) / (2 * h)       # d/db of Phi' = -S1
        print(f"   theta={th}: |tau^3 Phi'''/Phi|={abs(tc ** 3 * (-s3) / phi):.3f}  "
              f"|tau dPhi'/db/Phi|={abs(tc * dbphi1 / phi):.4f}  "
              f"|b tau dPhi'/db/Phi|={abs(bc * tc * dbphi1 / phi):.4f}")


class _Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)


if __name__ == "__main__":
    part = sys.argv[1] if len(sys.argv) > 1 else "all"
    buffer = io.StringIO()
    with redirect_stdout(_Tee(sys.stdout, buffer)):
        if part in ("A", "all"):
            partA()
        if part in ("B", "all"):
            partB()
        if part in ("C", "all"):
            partC()
        if part in ("D", "all"):
            partD()
        if part in ("E", "all"):
            partE()
    name = "supplementary_checks.txt" if part == "all" else f"supplementary_checks_{part}.txt"
    (TABLES / name).write_text(buffer.getvalue())
