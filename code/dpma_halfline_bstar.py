#!/usr/bin/env python3
"""Near-end constants B* and c_* from the explicit half-line transform.

The endpoint law b_c(theta) ~ B*/d comes from the d->0 boundary layer: rescaling x=d*y,
tau=d^2*sigma, B=b*d turns the interval delta-sink problem into Brownian motion (generator
(1/2) d^2/dy^2) on (0,inf), absorbing at y=0, delta-sink of strength B at y=1, started at y=1.
This problem has a CLOSED-FORM arrival-time Laplace transform. With kappa=sqrt(2s), the
half-line Dirichlet resolvent is r0(y,y';s) = 2 sinh(kappa y_<) exp(-kappa y_>)/kappa, and
Sherman-Morrison for the rank-one sink gives (flux at 0 plus sink capture, source at 1):

    fhat(s;B) = [exp(-kappa) + (B/kappa)(1-exp(-2 kappa))] / [1 + (B/kappa)(1-exp(-2 kappa))].

Checks: fhat(0)=1 (certain absorption); B=0 gives exp(-kappa), the Levy transform.
Bromwich inversion along the branch cut s=-v^2/2 (kappa=iv) gives the REAL integral

    Phi_half(sigma;B) = (1/pi) Int_0^inf v exp(-sigma v^2/2) (1+g) sin(v)
                          / (1 + 2 g cos(v) + g^2) dv,      g(v) = 2B sin(v)/v,

(the denominator equals (g+cos v)^2 + sin^2 v >= 0, vanishing nowhere on the support).
The fold Phi_half' = Phi_half'' = 0 in sigma then selects (sigma*, B*) with NO interval
input, no lattice, no bisection on theta: b_c(theta) ~ B*/theta and tau_c ~ sigma* theta^2.
Independent targets: B* ~ 0.789026 (interval bisection + extrapolation, dpma_endpoint_law.py)
and c_* ~ 0.158. Writes outputs/tables/dpma_halfline_bstar.txt
"""
from __future__ import annotations
import math
import numpy as np

from paths import TABLES as TAB
OUT = []
def say(s=""):
    print(s, flush=True); OUT.append(s)

def phi_derivs(sigma, B, orders=(0, 1, 2, 3), npts=400_001):
    """(d/dsigma)^m Phi_half(sigma;B) for m in orders, by Simpson on the branch-cut integral."""
    vmax = math.sqrt(2.0 * 85.0 / sigma)          # exp(-sigma v^2/2) < 1e-36 beyond
    v = np.linspace(0.0, vmax, npts)
    sv = np.sin(v)
    with np.errstate(divide="ignore", invalid="ignore"):
        g = np.where(v > 0, 2.0 * B * sv / v, 2.0 * B)
    h = (1.0 + g) * sv / (1.0 + 2.0 * g * np.cos(v) + g * g)
    base = v * np.exp(-0.5 * sigma * v * v) * h
    w = np.ones(npts); w[1:-1:2] = 4.0; w[2:-1:2] = 2.0
    dv = v[1] - v[0]
    out = {}
    for m in orders:
        out[m] = float(np.sum(w * base * (-0.5 * v * v) ** m) * dv / 3.0 / math.pi)
    return out

# --- validation of the inversion at B=0 against the exact Levy density -------------------
say("Half-line boundary-layer problem: closed-form transform, branch-cut inversion")
say("Check (B=0) vs exact Levy density exp(-1/(2 sigma))/sqrt(2 pi sigma^3):")
for sg in (0.05, 0.158, 0.5):
    num = phi_derivs(sg, 0.0, orders=(0,))[0]
    exact = math.exp(-1.0 / (2.0 * sg)) / math.sqrt(2.0 * math.pi * sg**3)
    say(f"  sigma={sg:6.3f}: inverted={num:.12f}  exact={exact:.12f}  rel={abs(num/exact-1):.2e}")

# --- fold: 2D Newton on (Phi', Phi'') in (sigma, B) ---------------------------------------
def F_and_J(sigma, B, dB=1e-6):
    d = phi_derivs(sigma, B, orders=(1, 2, 3))
    dp = phi_derivs(sigma, B + dB, orders=(1, 2))
    dm = phi_derivs(sigma, B - dB, orders=(1, 2))
    F = np.array([d[1], d[2]])
    J = np.array([[d[2], (dp[1] - dm[1]) / (2 * dB)],
                  [d[3], (dp[2] - dm[2]) / (2 * dB)]])
    return F, J

sigma, B = 0.158, 0.789
say("\n2D Newton on Phi_half'=Phi_half''=0:")
for it in range(12):
    F, J = F_and_J(sigma, B)
    step = np.linalg.solve(J, -F)
    sigma += float(step[0]); B += float(step[1])
    say(f"  it{it}: sigma={sigma:.10f}  B={B:.10f}  |F|=({F[0]:+.3e},{F[1]:+.3e})")
    if max(abs(step)) < 1e-11:
        break

say(f"\n=> half-line fold: B* = {B:.8f},  sigma* (= c_*) = {sigma:.8f}")
say("   independent targets: B* = 0.789026(1) [interval bisection, dpma_endpoint_law.py],")
say("   c_* ~ 0.158 [interval fold times]. No lattice, no interval, no theta scan is used here.")

# The half-line root must be a nondegenerate fold, rather than merely a zero of
# the two numerical residuals.  Report dimensionless margins and the full
# (Phi', Phi'') Jacobian used by the implicit-function theorem.
fold_derivatives = phi_derivs(sigma, B, orders=(0, 1, 2, 3))
fold_residual, fold_jacobian = F_and_J(sigma, B)
third_margin = sigma**3 * fold_derivatives[3] / fold_derivatives[0]
mixed_margin = B * sigma * fold_jacobian[0, 1] / fold_derivatives[0]
jacobian_determinant = float(np.linalg.det(fold_jacobian))
if (
    max(abs(fold_residual)) > 1.0e-8
    or abs(third_margin) < 1.0
    or abs(mixed_margin) < 0.1
    or abs(jacobian_determinant) < 100.0
):
    raise RuntimeError(
        "half-line fold nondegeneracy failed: "
        f"residual={fold_residual}, third={third_margin}, mixed={mixed_margin}, "
        f"det={jacobian_determinant}"
    )
say("\nHalf-line fold nondegeneracy (dimensionless where applicable):")
say(f"  sigma^3 Phi'''/Phi = {third_margin:+.10f}")
say(f"  B sigma (partial_B Phi')/Phi = {mixed_margin:+.10f}")
say(f"  det d(Phi',Phi'')/d(sigma,B) = {jacobian_determinant:+.10e}")

# --- sanity: pair existence straddles B* --------------------------------------------------
say("\nExtrema-pair check (interior valley+peak in sigma in [0.02, 2]):")
for Btest in (0.70, 0.75, B - 0.01, B + 0.01, 0.85):
    sgs = np.exp(np.linspace(math.log(0.02), math.log(2.0), 900))
    vals = np.array([phi_derivs(s, Btest, orders=(1,), npts=100_001)[1] for s in sgs])
    signflips = int(np.sum(np.diff(np.sign(vals)) != 0))
    say(f"  B={Btest:.5f}: sign changes of Phi' = {signflips}  ({'pair' if signflips >= 2 else 'no pair'})")

(TAB / "dpma_halfline_bstar.txt").write_text("\n".join(OUT) + "\n")
