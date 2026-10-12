#!/usr/bin/env python3
"""Fig. 1: the model and the three shapes of the absorption-time density.

Top row (illustration, no numerics):
  (a) lazy ring of N sites with an absorbing target (site 0) and a one-way shortcut u->0;
  (b) continuum limit: Brownian motion on [0,1] (generator (1/2) d^2/dx^2) with absorbing
      ends and a point (delta) sink of strength b at theta; release point xi.
Bottom row: the exact continuum first-passage density Phi(tau) on log-log axes for the three
morphologies discussed in the text (theta = 1/2 throughout):
  (c) release at the sink (xi = 1/2), weak sink b = 1.5 -> capture spike, valley, delayed peak;
  (d) release at the sink (xi = 1/2), strong sink b = 4.5 (> b_c = 3.0764) -> no delayed peak;
  (e) release away from the sink (xi = 0.40), b = 2.5 (inside 1.8391 < b < 3.8621) -> early and
      late peak (two interior peaks).
Linear-scale insets magnify the valley/peak region.

Numerics: signed spectral sum Phi(tau) = sum_j G_j exp(-mu_j tau), mu_j = 2 w_j^2, over the
sink-coupled modes 2 w cos w + b sin w = 0, w in ((j+1/2)pi, (j+1)pi), with the closed-form
amplitude G_j = 4 w (1-cos w) sin(2 w min(xi,1-xi)) / [sin^2 w (1 + b(b+2)/(4w^2))]
(identical to dpma_offgate_phase.antipodal_modes; the node modes sin w = 0 carry zero
first-passage weight at theta = 1/2).  Checks performed and printed:
  * amplitudes agree with the general-theta formula dpma_saddle_node_bc_theta.G_affected and with
    dpma_offgate_phase.antipodal_modes (to round-off);
  * mode-number doubling at every plotted point (J chosen so that 2 w_J^2 tau_min >= 60);
  * an independent route: Talbot inversion (mpmath) of the Laplace transform obtained from the
    Dirichlet resolvent by Sherman-Morrison, written with decaying exponentials only;
  * critical points of Phi: sign changes of Phi' = -sum G mu e^{-mu tau}, refined by Brent,
    classified by Phi''; repeated with 2J modes; short-time side closed by the capture law
    (start at sink) or by Talbot inversion of L[Phi'] = s*Lhat(s) (start off sink, Phi(0+)=0);
    long-time side closed by a one-mode dominance bound.
Style: shared figure style (figstyle.py); PNG at 150 dpi.
Outputs: outputs/figures/fig1_model_morphology.{pdf,png},
         outputs/tables/fig1_model_morphology.txt
"""
from __future__ import annotations

import math

import numpy as np
import mpmath as mp
from scipy.optimize import brentq
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle, Arc
from matplotlib.ticker import LogLocator, NullFormatter, FixedLocator, FixedFormatter

from dpma_saddle_node_bc_theta import G_affected            # general-theta amplitude
from dpma_offgate_phase import antipodal_modes              # theta=1/2 amplitude
import figstyle as S                                         # shared figure style
from paths import FIGURES as FIG, TABLES as TAB

LOG = []


def say(text=""):
    print(text, flush=True)
    LOG.append(text)


# ============================================================== numerics
THETA = 0.5
B_C = 3.0764323604            # release-at-sink threshold at theta = 1/2
B_MINUS, B_PLUS = 1.8390559, 3.8620786   # fold lines at xi = 0.40

CASES = [
    dict(key="c", b=1.5, xi=0.5, title="release at the sink, weak sink",
         xc=0.47, params=r"$\xi=\theta$,  $b=1.5$", expect=["min", "max"]),
    dict(key="d", b=4.5, xi=0.5, title="release at the sink, strong sink",
         xc=0.44, params=r"$\xi=\theta$,  $b=4.5>b_c\approx3.08$", expect=[]),
    dict(key="e", b=2.5, xi=0.40, title="release away from the sink",
         xc=0.56, params=r"$\xi=0.40$,  $b=2.5$", expect=["max", "min", "max"]),
]

TAU_PLOT = (1.0e-4, 1.0)      # plotted time window
Y_PLOT = (0.08, 400.0)        # plotted density window (10^-1 label clear of the corner)
TAU_SCAN_HI = 5.0             # upper end of the critical-point scan (tail bound beyond)


def sink_roots(b: float, n_modes: int) -> np.ndarray:
    """First n_modes roots of 2 w cos w + b sin w = 0, one per ((j+1/2)pi, (j+1)pi)."""
    j = np.arange(n_modes, dtype=float)
    lo = (j + 0.5) * np.pi
    hi = (j + 1.0) * np.pi
    g = lambda z: 2.0 * z * np.cos(z) + b * np.sin(z)
    glo = g(lo)
    for _ in range(90):
        mid = 0.5 * (lo + hi)
        gm = g(mid)
        same = np.sign(gm) == np.sign(glo)
        lo = np.where(same, mid, lo)
        glo = np.where(same, gm, glo)
        hi = np.where(same, hi, mid)
    w = 0.5 * (lo + hi)
    for _ in range(2):  # Newton polish
        f = 2.0 * w * np.cos(w) + b * np.sin(w)
        fp = (2.0 + b) * np.cos(w) - 2.0 * w * np.sin(w)
        w = w - f / fp
    return w


def modes(b: float, xi: float, n_modes: int):
    w = sink_roots(b, n_modes)
    xs = min(xi, 1.0 - xi)
    G = 4.0 * w * (1.0 - np.cos(w)) * np.sin(2.0 * w * xs) / (
        np.sin(w) ** 2 * (1.0 + b * (b + 2.0) / (4.0 * w * w)))
    return 2.0 * w * w, G, w


def moment(tau, mu, G, order=0, absolute=False):
    """sum_j G_j mu_j^order exp(-mu_j tau) (absolute=True: sum of |terms|), chunked."""
    tau = np.atleast_1d(np.asarray(tau, dtype=float))
    coef = G * mu ** order
    if absolute:
        coef = np.abs(coef)
    out = np.empty(tau.size)
    for i in range(0, tau.size, 200):
        E = np.exp(-np.outer(tau[i:i + 200], mu))
        out[i:i + 200] = (E * coef).sum(axis=1)
    return out


def n_modes_for(tau_min: float) -> int:
    """Smallest J with 2 w_J^2 tau_min >= 60 (truncation e^-60), plus margin."""
    return int(math.ceil(math.sqrt(30.0 / tau_min) / math.pi)) + 10


# ---------------------------------------------------- independent Laplace route
def laplace_fpt(s, b, xi, th=mp.mpf(1) / 2):
    """E[e^{-sT}] for the interval with absorbing ends and a delta sink (Sherman-Morrison).

    u0(x): exit transform without sink; g(x,y): Dirichlet resolvent of s - (1/2)d^2/dx^2.
    Lhat(s) = u0(xi) + b g(xi,th) (1 - u0(th)) / (1 + b g(th,th)).  Only e^{-k a}, a >= 0.
    """
    k = mp.sqrt(2 * s)
    E = lambda a: mp.exp(-k * a)
    den = 1 - E(2)
    u0 = lambda x: (E(1 - x) - E(1 + x) + E(x) - E(2 - x)) / den

    def g(x, y):
        a, c = min(x, y), max(x, y)
        return (E(c - a) - E(2 - c - a) - E(c + a) + E(2 - c + a)) / (k * den)

    return u0(xi) + b * g(xi, th) * (1 - u0(th)) / (1 + b * g(th, th))


def talbot_phi(tau, b, xi, derivative=False, dps=40):
    mp.mp.dps = dps
    bb, xx = mp.mpf(b), mp.mpf(xi)
    if derivative:  # valid only when Phi(0+) = 0 (start off the sink)
        f = lambda s: s * laplace_fpt(s, bb, xx)
    else:
        f = lambda s: laplace_fpt(s, bb, xx)
    return float(mp.invertlaplace(f, mp.mpf(tau), method="talbot"))


# ---------------------------------------------------------- per-case analysis
def critical_points(mu, G, tau_lo, tau_hi, n_grid=6000):
    t = np.exp(np.linspace(math.log(tau_lo), math.log(tau_hi), n_grid))
    s1 = moment(t, mu, G, 1)
    idx = np.flatnonzero(np.sign(s1[:-1]) != np.sign(s1[1:]))
    out = []
    for i in idx:
        r = brentq(lambda x: moment(x, mu, G, 1)[0], t[i], t[i + 1], xtol=1e-16, rtol=1e-13)
        s2 = moment(r, mu, G, 2)[0]            # Phi'' = +S2
        out.append((r, "max" if s2 < 0 else "min", moment(r, mu, G, 0)[0]))
    return out


def analyse(case):
    b, xi = case["b"], case["xi"]
    say(f"--- panel ({case['key']}): theta=1/2, xi={xi}, b={b}  [{case['title']}]")

    # amplitude cross-checks against the solver modules
    mu50, G50, w50 = modes(b, xi, 50)
    Gref = np.array([G_affected(w, xi, THETA, b) for w in w50])
    mu_s, G_s = antipodal_modes(b, xi, 160.0)
    n = min(len(mu_s), 50)
    say(f"  amplitude check: max rel |G - G_affected| = "
        f"{np.max(np.abs(G50 - Gref) / np.abs(Gref)):.2e}; "
        f"vs antipodal_modes (first {n}): max |dmu| = {np.max(np.abs(mu50[:n] - mu_s[:n])):.1e}, "
        f"max |dG| = {np.max(np.abs(G50[:n] - G_s[:n])):.1e}")

    # plotted density, converged by doubling
    tau_scan_lo = 1.0e-6 if xi == THETA else 3.0e-4
    J = n_modes_for(min(TAU_PLOT[0], tau_scan_lo))
    mu, G, _ = modes(b, xi, J)
    mu2, G2, _ = modes(b, xi, 2 * J)
    tau = np.exp(np.linspace(math.log(TAU_PLOT[0]), math.log(TAU_PLOT[1]), 2400))
    phi = moment(tau, mu, G)
    phi2 = moment(tau, mu2, G2)
    floor = 1e3 * np.finfo(float).eps * moment(tau, mu2, G2, absolute=True)
    ok = phi2 > 1e3 * floor                       # above the cancellation floor
    rel = np.max(np.abs(phi[ok] - phi2[ok]) / phi2[ok])
    say(f"  modes: J={J} vs 2J={2*J}; max rel. change on plotted window "
        f"[{tau[ok][0]:.2e},{TAU_PLOT[1]}] = {rel:.1e}")

    # independent route (Talbot inversion of the Sherman-Morrison transform)
    checks = []
    for t in (1e-4, 1e-3, 1e-2, 0.1, 0.5):
        m = moment(t, mu2, G2)[0]
        if m > 1e-6:
            checks.append(abs(talbot_phi(t, b, xi) - m) / m)
    say(f"  Talbot vs mode sum at tau in (1e-4,1e-3,1e-2,0.1,0.5): max rel. diff = {max(checks):.1e}")

    # critical points
    cps = critical_points(mu, G, tau_scan_lo, TAU_SCAN_HI)
    cps2 = critical_points(mu2, G2, tau_scan_lo, TAU_SCAN_HI)
    assert [c[1] for c in cps] == [c[1] for c in cps2], "critical points change with mode doubling"
    shift = max([abs(a[0] - c[0]) / a[0] for a, c in zip(cps, cps2)], default=0.0)
    say(f"  critical points on [{tau_scan_lo:g}, {TAU_SCAN_HI:g}] (J and 2J agree; max rel. shift {shift:.1e}):")
    for r, kind, val in cps2:
        # independent confirmation: Talbot values bracket an extremum of the stated type
        h = 2e-3
        lft, mid, rgt = (talbot_phi(r * (1 - h), b, xi), talbot_phi(r, b, xi),
                         talbot_phi(r * (1 + h), b, xi))
        conf = (mid > lft and mid > rgt) if kind == "max" else (mid < lft and mid < rgt)
        say(f"    {kind:3s}  tau = {r:.8f}   Phi = {val:.6f}   (Talbot: Phi = {mid:.6f}, "
            f"extremum confirmed: {conf})")
        assert conf
    if not cps2:
        say("    none: Phi strictly decreasing on the scanned interval")

    # short-time side
    if xi == THETA:
        s1_lo = moment(tau_scan_lo, mu2, G2, 1)[0]
        say(f"  short-time side: Phi'({tau_scan_lo:g}) = {-s1_lo:.4e} vs capture law "
            f"-b/(2 sqrt(2 pi)) tau^-3/2 = {-b / (2 * math.sqrt(2 * math.pi)) * tau_scan_lo**-1.5:.4e};"
            f" Phi = b/sqrt(2 pi tau) - b^2/2 + O(b^3 tau^1/2) is decreasing below")
    else:
        # Talbot inversion of L[Phi'] = s*Lhat(s) (Phi(0+) = 0), precision raised with the
        # Gaussian suppression exp(-delta^2/2tau) so that the tiny values are resolved
        delta = abs(THETA - xi)
        tt = np.exp(np.linspace(math.log(1e-5), math.log(tau_scan_lo), 16))
        d, lead = [], []
        for t in tt:
            dps = 30 + int(math.ceil(delta**2 / (2 * t) / math.log(10)))
            d.append(talbot_phi(t, b, xi, derivative=True, dps=dps))
        # leading capture term b exp(-delta^2/2tau)/sqrt(2 pi tau): derivative ratio at 1e-5
        t0 = tt[0]
        lead0 = b * math.exp(-delta**2 / (2 * t0)) / math.sqrt(2 * math.pi * t0) * (
            delta**2 / (2 * t0**2) - 1 / (2 * t0))
        say(f"  short-time side (Talbot of s*Lhat = L[Phi'], Phi(0+)=0, adaptive precision): "
            f"Phi' > 0 at all {len(tt)} points in [1e-5, {tau_scan_lo:g}]: {all(x > 0 for x in d)}")
        say(f"    below 1e-5: Phi ~ b exp(-d^2/2tau)/sqrt(2 pi tau) (d=|xi-theta|={delta:g}; "
            f"Phi'/leading-term derivative at 1e-5 = {d[0] / lead0:.6f}), increasing for tau < d^2")
        assert all(x > 0 for x in d) and abs(d[0] / lead0 - 1) < 1e-2

    # long-time side: one-mode dominance bound, monotone decreasing in tau
    R = np.sum(np.abs(G2[1:]) * mu2[1:] * np.exp(-(mu2[1:] - mu2[0]) * TAU_SCAN_HI)) / (G2[0] * mu2[0])
    say(f"  long-time side: G_1 = {G2[0]:.6f} > 0, mu_1 = {mu2[0]:.6f}; "
        f"sum_(j>=2)|G_j|mu_j e^-(mu_j-mu_1)tau / (G_1 mu_1) at tau={TAU_SCAN_HI:g} = {R:.1e} < 1 "
        f"=> Phi' < 0 for all tau >= {TAU_SCAN_HI:g}")
    assert G2[0] > 0 and R < 1
    kinds = [c[1] for c in cps2]
    assert kinds == case["expect"], (kinds, case["expect"])
    say(f"  => {len(cps2)} critical point(s) on (0, inf): {kinds if kinds else 'monotone'}")
    say()

    plot_mask = phi2 > max(1e3 * floor.max(), 1e-3)
    return dict(tau=tau[plot_mask], phi=phi2[plot_mask], cps=cps2, mu=mu2, G=G2)


# ============================================================== figure
S.apply_style()                  # TrueType (Type 42) fonts, sizes at printed size

INK = "#1a1a1a"
MUTED = "#555555"
C_ABS = "#D55E00"     # absorbing ends / target (Okabe-Ito vermillion)
C_SINK = "#009E73"    # sink / shortcut / capture (Okabe-Ito bluish green)
C_PEAK = "#0072B2"    # critical-point markers (Okabe-Ito blue)
C_START = INK


def panel_ring(ax):
    ax.set_aspect("equal")
    ax.axis("off")
    N, R = 16, 1.0
    ang = [math.pi / 2 - 2 * math.pi * i / N for i in range(N)]
    xs = [R * math.cos(a) for a in ang]
    ys = [R * math.sin(a) for a in ang]
    u = N // 2
    for i in range(N):
        j = (i + 1) % N
        ax.plot([xs[i], xs[j]], [ys[i], ys[j]], color="0.6", lw=0.8, zorder=1)
    for i in range(N):
        if i == 0:
            ax.add_patch(Circle((xs[i], ys[i]), 0.11, fc=C_ABS, ec=INK, lw=0.8, zorder=5))
        elif i == u:
            ax.add_patch(Circle((xs[i], ys[i]), 0.11, fc=C_SINK, ec=INK, lw=0.8, zorder=5))
        else:
            ax.add_patch(Circle((xs[i], ys[i]), 0.06, fc="white", ec=INK, lw=0.7, zorder=4))
    ax.add_patch(FancyArrowPatch((xs[u], ys[u] + 0.12), (xs[0], ys[0] - 0.12),
                                 arrowstyle="-|>", mutation_scale=11, lw=1.6, color=C_SINK,
                                 zorder=3, connectionstyle="arc3,rad=0.13"))
    ax.text(xs[0], ys[0] + 0.17, "target, site 0\n(absorbing)", ha="center", va="bottom",
            color=C_ABS, fontsize=7.5, linespacing=1.05)
    ax.text(xs[u], ys[u] - 0.17, r"shortcut site $u$", ha="center", va="top",
            color=C_SINK, fontsize=7.5)
    # shortcut label outside the ring (left), with a thin leader to the arrow
    ax.annotate("one-way\nshortcut " + r"$u\!\to\!0$" + "\n" + r"(prob. $\lambda$" + "\nper step)",
                xy=(0.06, -0.19), xytext=(-1.2, -0.19), ha="right", va="center", color=C_SINK,
                fontsize=6.8, linespacing=1.1,
                arrowprops=dict(arrowstyle="-", color=C_SINK, lw=0.6, shrinkA=2, shrinkB=0))
    gi = 3
    ax.add_patch(Arc((xs[gi] + 0.13, ys[gi] + 0.13), 0.24, 0.24, theta1=-30, theta2=210,
                     color=MUTED, lw=0.8, zorder=2))
    ax.text(xs[gi] + 0.28, ys[gi] + 0.22, r"stay $1-q$", fontsize=7, color=MUTED, va="bottom")
    ax.text((xs[gi] + xs[gi + 1]) / 2 + 0.12, (ys[gi] + ys[gi + 1]) / 2 - 0.02,
            r"hop $q/2$", fontsize=7, color=MUTED, va="center")
    ax.set_xlim(-2.4, 2.05)
    ax.set_ylim(-1.42, 1.66)
    ax.set_anchor("W")           # flush left, aligned with panel (c) below
    S.panel_title(ax, "a", "lazy ring with a one-way shortcut")


def panel_interval(ax):
    th, xi = 0.5, 0.40
    ax.plot([0, 1], [0, 0], color=INK, lw=1.3, zorder=2)
    for xw, lab in [(0.0, r"$x=0$"), (1.0, r"$x=1$")]:
        ax.plot([xw, xw], [-0.26, 0.26], color=C_ABS, lw=2.4, zorder=3, solid_capstyle="butt")
        ax.text(xw, -0.33, lab, ha="center", va="top", fontsize=7.5)
        ax.text(xw, 0.32, "absorbing\nend", ha="center", va="bottom", fontsize=7, color=C_ABS,
                linespacing=1.05)
    ax.annotate("", xy=(th, -0.5), xytext=(th, 0.0),
                arrowprops=dict(arrowstyle="-|>", color=C_SINK, lw=2.0, mutation_scale=10))
    ax.text(th + 0.025, -0.3, "point sink at " + r"$\theta$" + "\n(absorbs with strength " + r"$b$)",
            ha="left", va="center", fontsize=7, color=C_SINK, linespacing=1.1)
    ax.plot([xi], [0.0], marker="o", ms=5.5, mfc=C_START, mec=C_START, zorder=5)
    ax.text(xi, 0.09, r"release point $\xi$", ha="center", va="bottom", fontsize=7.5,
            color=C_START)
    for x0 in (0.18, 0.8):
        ax.annotate("", xy=(x0 + 0.07, 0.07), xytext=(x0 - 0.07, 0.07),
                    arrowprops=dict(arrowstyle="<->", color="0.5", lw=0.8))
        ax.text(x0, 0.13, "diffusion", ha="center", va="bottom", fontsize=6.5, color="0.45")
    ax.text(0.5, 0.78, r"$\partial_\tau p=\frac{1}{2}\,\partial_x^2 p-b\,\delta(x-\theta)\,p$",
            ha="center", va="center", fontsize=8.5)
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.72, 0.98)
    ax.axis("off")
    S.panel_title(ax, "b", "continuum limit: an interval with a point sink")


def interval_glyph(ax, rect, xi, b):
    """Tiny pictogram: interval, absorbing ends, sink at 1/2 (arrow length ~ b), start dot."""
    g = ax.inset_axes(rect)
    g.axis("off")
    g.set_xlim(-0.06, 1.06)
    g.set_ylim(-1.15, 0.4)
    g.plot([0, 1], [0, 0], color=INK, lw=0.9, zorder=2)
    for xw in (0, 1):
        g.plot([xw, xw], [-0.3, 0.3], color=C_ABS, lw=1.8, solid_capstyle="butt")
    L = 0.25 + 0.17 * b          # arrow length grows with the sink strength
    g.annotate("", xy=(0.5, -L), xytext=(0.5, -0.06),
               arrowprops=dict(arrowstyle="-|>", color=C_SINK, lw=1.5, mutation_scale=7,
                               shrinkA=0, shrinkB=0))
    g.plot([xi], [0.0], "o", ms=3.8, color=C_START, zorder=5)
    return g


def style_loglog(ax):
    ax.set_xlim(*TAU_PLOT)
    ax.set_ylim(*Y_PLOT)
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=6))
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=6))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=20))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=20))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())


def mark_cps(ax, cps, ms=4.2):
    for r, kind, val in cps:
        ax.plot([r], [val], "o", ms=ms, mfc=C_PEAK if kind == "max" else "white",
                mec=C_PEAK, mew=1.0, zorder=5)


ANN = dict(fontsize=7, color=INK, ha="center")
ARR = dict(arrowstyle="-", color=MUTED, lw=0.6, shrinkA=1, shrinkB=3)


def morphology_panel(ax, case, res, show_ylabel):
    tau, phi = res["tau"], res["phi"]
    ax.loglog(tau, phi, color=INK, lw=1.4, zorder=3)
    mark_cps(ax, res["cps"])
    style_loglog(ax)
    ax.set_xlabel(r"time $\tau$")
    if show_ylabel:
        ax.set_ylabel(r"absorption-time density $\Phi(\tau)$")
    S.panel_title(ax, case["key"], case["title"])
    # pictogram of the set-up + parameters (same place in every panel)
    xc = case["xc"]
    interval_glyph(ax, [xc - 0.16, 0.03, 0.32, 0.15], case["xi"], case["b"])
    ax.text(xc, 0.185, case["params"], transform=ax.transAxes, ha="center", va="bottom",
            fontsize=7, color=INK)

    def phi_at(t):
        return float(moment(t, res["mu"], res["G"])[0])

    key = case["key"]
    if key == "c":
        t0 = 4e-4
        ax.annotate("capture spike\n(\u221d " + r"$\tau^{-1/2}$)", xy=(t0, phi_at(t0)),
                    xytext=(2.4e-3, 3.0 * phi_at(2.4e-3)), va="bottom", arrowprops=ARR, **ANN)
    if key == "d":
        t0 = 3e-4
        ax.annotate("capture spike\n(\u221d " + r"$\tau^{-1/2}$)", xy=(t0, phi_at(t0)),
                    xytext=(1.0e-3, 9.0), va="top", arrowprops=ARR, **ANN)
    if key == "c":
        (tmin, _, vmin), (tmax, _, vmax) = res["cps"]
        ax.annotate("valley", xy=(tmin, vmin), xytext=(tmin * 0.62, vmin * 0.42), va="top",
                    arrowprops=ARR, **ANN)
        ax.annotate("delayed\npeak", xy=(tmax, vmax), xytext=(tmax * 1.9, vmax * 3.0),
                    va="bottom", arrowprops=ARR, **ANN)
    if key == "d":
        t1 = 0.045
        ax.annotate("no delayed peak\n(shoulder)", xy=(t1, phi_at(t1)),
                    xytext=(0.12, 3.4 * phi_at(t1)), va="bottom", arrowprops=ARR, **ANN)
    if key == "e":
        ax.annotate("particle must\nfirst reach\nthe sink", xy=(1.08e-3, phi_at(1.08e-3)),
                    xytext=(5.2e-4, 2.2), va="bottom", arrowprops=ARR, fontsize=6.8,
                    color=INK, ha="center", linespacing=1.1)
        inset_two_peaks(ax, res)


def inset_two_peaks(ax, res):
    """Linear-scale magnification of the two-peak region of panel (e)."""
    lo, hi = 3.0e-3, 0.12
    ylo, yhi = 4.75, 6.45
    ins = ax.inset_axes([0.40, 0.66, 0.57, 0.31])
    t = np.exp(np.linspace(math.log(lo), math.log(hi), 600))
    ins.plot(t, moment(t, res["mu"], res["G"]), color=INK, lw=1.2)
    mark_cps(ins, res["cps"], ms=3.6)
    ins.set_xscale("log")
    ins.set_xlim(lo, hi)
    ins.set_ylim(ylo, yhi)
    ins.tick_params(labelsize=6, length=2, pad=1.5)
    ins.xaxis.set_major_locator(FixedLocator([0.01, 0.1]))      # plain decimals: no tiny
    ins.xaxis.set_major_formatter(FixedFormatter(["0.01", "0.1"]))  # superscripts in the inset
    ins.xaxis.set_minor_formatter(NullFormatter())
    ins.set_yticks([5.0, 5.5, 6.0])
    (t1, _, v1), (t2, _, v2), (t3, _, v3) = res["cps"]
    a = dict(fontsize=6.4, ha="center", color=INK)
    ins.text(t1, v1 + 0.12, "early\npeak", va="bottom", linespacing=1.05, **a)
    ins.text(t2, v2 - 0.1, "valley", va="top", **a)
    ins.text(t3, v3 + 0.1, "late peak", va="bottom", **a)
    ax.indicate_inset_zoom(ins, edgecolor="0.45", alpha=0.9, lw=0.6)


def main():
    say("Fig. 1: model + continuum density morphology (theta = 1/2)")
    say(f"reference values: b_c(1/2) = {B_C}; off-sink window at xi=0.40: {B_MINUS} < b < {B_PLUS}")
    say()
    results = {c["key"]: analyse(c) for c in CASES}

    fig = plt.figure(figsize=(7.05, 4.6))
    top = fig.add_gridspec(1, 2, left=0.085, right=0.995, top=0.952, bottom=0.62,
                           width_ratios=[0.95, 1.55], wspace=0.02)
    bot = fig.add_gridspec(1, 3, left=0.085, right=0.985, top=0.548, bottom=0.09, wspace=0.13)
    panel_ring(fig.add_subplot(top[0]))
    panel_interval(fig.add_subplot(top[1]))
    axes = [fig.add_subplot(bot[i]) for i in range(3)]
    for i, (ax, case) in enumerate(zip(axes, CASES)):
        morphology_panel(ax, case, results[case["key"]], show_ylabel=(i == 0))
        if i:
            ax.tick_params(labelleft=False)
    info = S.save(fig, FIG, "fig1_model_morphology")
    say(f"wrote outputs/figures/fig1_model_morphology.pdf and .png ({S.PNG_DPI} dpi); "
        f"{info['width_in']:.2f} x {info['height_in']:.2f} in")
    (TAB / "fig1_model_morphology.txt").write_text("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
