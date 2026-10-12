#!/usr/bin/env python3
"""Half-line transform for an arbitrary release point, Eqs. (fhl) and (phihl) of App. C.

    fhat(s;B,X) = exp(-w X) + B (1-exp(-w)) Delta_X(w) / D_B(w),      w = sqrt(2 s),
    Delta_X(w) = exp(-w|1-X|) - exp(-w(1+X)),   D_B(w) = w + B (1-exp(-2w)).

Checks (40-digit mpmath unless stated otherwise)
  (i)   fhat(s) -> 1 as s -> 0 (absorption is certain) for several (B, X);
  (ii)  B = 0 gives the Levy-Smirnov transform exp(-w X);
  (iii) X = 1 reproduces the release-at-sink transform, Eq. (fhlsource);
  (iv)  the branch-cut density, Eq. (phihl), evaluated with the Gauss-Legendre quadrature used for
        the half-line cusps (cusp_endpoint_continuation.py, double precision), agrees with Talbot
        inversion of fhat at release points on both sides of the sink, including the two cusps.
Writes outputs/tables/halfline_transform_checks.txt.
"""
from __future__ import annotations

import mpmath as mp

from cusp_endpoint_continuation import HalfLineCusp
from paths import TABLES

mp.mp.dps = 40
LINES: list[str] = []


def say(s=""):
    print(s, flush=True)
    LINES.append(s)


def fhat(s, B, X):
    w = mp.sqrt(2 * s)
    delta = mp.exp(-w * abs(1 - X)) - mp.exp(-w * (1 + X))
    return mp.exp(-w * X) + B * (1 - mp.exp(-w)) * delta / (w + B * (1 - mp.exp(-2 * w)))


def fhat_source(s, B):
    w = mp.sqrt(2 * s)
    return (w * mp.exp(-w) + B * (1 - mp.exp(-2 * w))) / (w + B * (1 - mp.exp(-2 * w)))


def main():
    Bs = (mp.mpf("0.5"), mp.mpf("1.085683"), mp.mpf("2.868772"), mp.mpf(5))
    Xs = (mp.mpf("0.2"), mp.mpf("0.758314"), mp.mpf(1), mp.mpf("1.424897"), mp.mpf(3))
    say("Half-line transform for arbitrary release X [Eq. (fhl)]")
    worst_limit = max(abs(1 - fhat(mp.mpf(10) ** -60, B, X)) for B in Bs for X in Xs)
    say(f"  (i)   max over B in {{0.5, 1.09, 2.87, 5}}, X in {{0.2, 0.76, 1, 1.42, 3}} of |1 - fhat(1e-60)| = "
        f"{mp.nstr(worst_limit, 3)} (the deviation scales as sqrt(s): absorption is certain)")
    points = [mp.mpf("0.3"), mp.mpf("2.5"), mp.mpc("0.7", "1.9"), mp.mpc("-0.4", "3.1"), mp.mpf(40)]
    worst_levy = max(abs(fhat(s, 0, X) - mp.exp(-mp.sqrt(2 * s) * X)) for s in points for X in Xs)
    worst_source = max(abs(fhat(s, B, 1) - fhat_source(s, B)) / abs(fhat_source(s, B)) for s in points for B in Bs)
    say(f"  (ii)  B = 0: max |fhat - exp(-w X)| over complex s and X = {mp.nstr(worst_levy, 3)}")
    say(f"  (iii) X = 1: max relative |fhat - Eq. (fhlsource)| over complex s and B = {mp.nstr(worst_source, 3)}")

    cusp = HalfLineCusp(order=112, vmax=60.0)
    say("  (iv)  branch-cut density (Gauss-Legendre, double precision) vs Talbot inversion of fhat:")
    worst_density = 0.0
    for B, X in ((0.5, 0.5), (2.868772, 0.758314), (1.085683, 1.424897), (1.2, 2.0), (0.78903, 1.0)):
        rels = []
        for sigma in (0.05, 0.2, 1.0):
            branch = float(cusp.moments(B, sigma, X, order=0)[0])
            talbot = float(mp.invertlaplace(lambda s: fhat(s, mp.mpf(B), mp.mpf(X)), sigma, method="talbot"))
            rels.append(abs(branch / talbot - 1.0))
        worst_density = max(worst_density, max(rels))
        say(f"        B={B:<8} X={X:<8}: max relative difference over sigma = 0.05, 0.2, 1 = {max(rels):.1e}")
    ok = worst_limit < 1e-25 and worst_levy == 0 and worst_source < 1e-35 and worst_density < 1e-9
    say(f"  worst relative difference of the density: {worst_density:.1e}")
    say(f"result: {'PASS' if ok else 'FAIL'}")
    (TABLES / "halfline_transform_checks.txt").write_text("\n".join(LINES) + "\n")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
