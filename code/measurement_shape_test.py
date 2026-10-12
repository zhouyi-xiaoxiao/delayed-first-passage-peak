#!/usr/bin/env python3
"""Shape test, mean-based calibration and colloid example of the proposed measurement (Sec. VI, App. D).

Window ratio R(b) (late-peak window over valley window), theta = 1/2, xi = 0.44, windows +-20%
around tau_p = 0.0462 and tau_v = 0.0253 (spectrum and windows from measurement_window_ratio.py).
Computes, for the point sink: the b at which R = 1; trials for a two-standard-deviation test of
R > 1; the probability that an experiment of this size finds R > 1 at the one-sided 5% level and
the factor in the number of trials for probability 0.95 (normal distribution); the early-peak
time; the relative depth of the valley just below b_+; mean, standard deviation and b-derivative
of the absorption time at b_+ and the trials needed for a mean-based calibration of b; the capture
rates kappa for the two thresholds.
For the gate of the colloid example, ell/L = 0.05 (exact finite-gate spectrum of
measurement_window_ratio.py, no mesh, same windows): the upper fold b_+^(ell), the b at which
R = 1, and at b = 2 the ratio R, the trials for the two-standard-deviation test and the mean
absorption time; then the numbers of the colloid example, evaluated with this finite-gate model.
Writes outputs/tables/measurement_shape_test.txt.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr, ndtri

import measurement_window_ratio as m
from paths import TABLES

LINES: list[str] = []


def say(s=""):
    print(s, flush=True)
    LINES.append(s)


THETA, XI = 0.5, 0.44
B_PLUS = 3.1366036780          # upper fold line at xi = 0.44 (dpma_offgate_phase.py)
B_C_MIN = 2.16451853           # minimum of b_c(theta), at theta = 0.381 (dpma_saddle_node_bc_theta.py)

# colloid example: 1 um bead in water, D ~ 0.44 um^2/s, L = 5 um, ell = 0.25 um, 5 ms frames
D_UM2_S, L_UM, ELL_UM, FRAME_S = 0.44, 5.0, 0.25, 0.005
ELL = ELL_UM / L_UM            # gate width of the example in units of L (0.05)

pw, vw = m.windows(0.0462, 0.0253)


def R(b):
    return m.window_ratio(m.delta_spectrum(b), pw, vw)


def R_gate(b):
    return m.window_ratio(m.gate_spectrum(b, ELL), pw, vw)


def trials_two_sd(r, p, v):
    """Trials for which the expected excess R - 1 equals two standard deviations of R."""
    return 4 * r**2 * (1 / p + 1 / v) / (r - 1) ** 2


def mfpt(b):
    return XI * (1 - XI) - 2 * b * THETA * (1 - THETA) * min(XI, THETA) * (1 - max(XI, THETA)) / (
        1 + 2 * b * THETA * (1 - THETA))


def mfpt_gate(b, ell):
    """Mean absorption time for the gate |x - 1/2| <= ell/2 with killing rate b/ell (closed form).

    T'' = -2 outside the gate and T'' = 2 (b/ell) T - 2 inside, T(0) = T(1) = 0, T and T'
    continuous; valid for a release point outside the gate (XI <= 1/2 - ell/2).
    """
    x0 = 0.5 - ell / 2
    rate = b / ell
    g = math.sqrt(2 * rate)
    h = g * ell / 2
    c = (x0**2 - 1 / rate) / (math.cosh(h) + g * x0 * math.sinh(h))
    a = 2 * x0 - c * g * math.sinh(h)
    return XI * (a - XI)


def main():
    say("Shape test and calibration numbers (theta = 1/2, xi = 0.44)")
    b1 = brentq(lambda b: R(b)[0] - 1.0, 2.0, 3.2, xtol=1e-10)
    say(f"R = 1 at b = {b1:.4f}")
    for b in (1.5, 2.0, 2.5):
        r, p, v = R(b)
        say(f"b = {b}: R = {r:.5f}, f_p = {p:.5f}, f_v = {v:.5f};  "
            f"N_t(2 s.d., R > 1) = {trials_two_sd(r, p, v):.3g}")
    # an expected excess of n_sd standard deviations, tested one-sided at level alpha (normal law)
    n_sd, alpha, target = 2.0, 0.05, 0.95
    z_alpha = float(ndtri(1 - alpha))
    power = float(ndtr(n_sd - z_alpha))
    factor = ((z_alpha + float(ndtri(target))) / n_sd) ** 2
    say(f"   expected excess of {n_sd:g} s.d.: probability of R > 1 at the one-sided "
        f"{100 * alpha:g}% level = {power:.3f}; trials factor for probability {target:g} = {factor:.2f}")

    # early peak and valley depth
    for b in (2.5, B_PLUS - 0.25, B_PLUS - 0.1):
        cps = m.critical_points(m.delta_spectrum(b), lo=1e-3, hi=0.5)
        maxima = [(t, ph) for kind, t, ph in cps if kind == "max"]
        minima = [(t, ph) for kind, t, ph in cps if kind == "min"]
        text = ", ".join(f"{kind} {t:.5f} ({ph:.4f})" for kind, t, ph in cps)
        say(f"b = {b:.4f}: critical points {text}")
        if len(maxima) == 2 and len(minima) == 1:
            depth = (maxima[1][1] - minima[0][1]) / maxima[1][1]
            say(f"   early peak at tau = {maxima[0][0]:.4f}; late peak exceeds the valley by "
                f"{100 * depth:.2f}% of its height")

    sp = m.delta_spectrum(B_PLUS)
    mean = float((sp.c / sp.mu).sum())
    m2 = float(2 * (sp.c / sp.mu**2).sum())
    sd = math.sqrt(m2 - mean**2)
    h = 1e-5
    dm = (mfpt(B_PLUS + h) - mfpt(B_PLUS - h)) / (2 * h)
    say(f"at b_+ = {B_PLUS:.5f}: mean = {mean:.5f} (closed form {mfpt(B_PLUS):.5f}), "
        f"s.d. = {sd:.5f}, d<tau>/db = {dm:.5f}")
    for db in (0.25, 0.1):
        say(f"   mean-based calibration of b to +-{db}: {sd**2 / (dm * db) ** 2:.3g} trials")
    say(f"kappa ell L / D = 2 b: {2 * B_C_MIN:.3f} for b_c(0.381) = {B_C_MIN:.4f}; "
        f"{2 * B_PLUS:.3f} for b_+ = {B_PLUS:.4f}")

    # gate of the example: exact finite-gate spectrum (no mesh), same windows
    say(f"finite gate of the example, ell/L = {ELL:g} (exact gate spectrum, no mesh; same windows):")
    b_plus_gate, tau_c_gate = m.fold(lambda b: m.gate_spectrum(b, ELL), 2.8, 3.3)
    say(f"   upper fold b_+^(ell) = {b_plus_gate:.7f} at tau_c = {tau_c_gate:.5f}; "
        f"R^(ell) at this fold = {R_gate(b_plus_gate)[0]:.5f}")
    b1_gate = brentq(lambda b: R_gate(b)[0] - 1.0, 2.0, 3.2, xtol=1e-10)
    say(f"   R^(ell) = 1 at b = {b1_gate:.4f}")
    r, p, v = R_gate(2.0)
    n_gate = trials_two_sd(r, p, v)
    say(f"   b = 2.0: R^(ell) = {r:.5f}, f_p = {p:.5f}, f_v = {v:.5f};  "
        f"N_t^(ell)(2 s.d., R > 1) = {n_gate:.3g}")
    sp = m.gate_spectrum(2.0, ELL)
    mean_gate = float((sp.c / sp.mu).sum())
    say(f"   mean absorption time at b = 2.0: {mean_gate:.5f} (closed form {mfpt_gate(2.0, ELL):.5f}; "
        f"point sink {mfpt(2.0):.5f})")

    # colloid example, evaluated with the finite-gate model
    t_diff = L_UM**2 / (2 * D_UM2_S)
    kappa = 2 * D_UM2_S * b_plus_gate / (ELL_UM * L_UM)
    say(f"colloid example (D = {D_UM2_S:g} um^2/s, L = {L_UM:g} um, ell = {ELL_UM:g} um, "
        f"{1e3 * FRAME_S:g} ms frames), finite-gate model:")
    say(f"   diffusion time L^2/(2D) = {t_diff:.1f} s; gate crossing ell^2/(2D) = "
        f"{ELL_UM**2 / (2 * D_UM2_S):.3f} s")
    say(f"   the finite-gate fold b_+^(ell) needs kappa = {kappa:.2f} 1/s; trigger probability per "
        f"frame 1 - exp(-kappa dt) = {-math.expm1(-kappa * FRAME_S):.4f}")
    mean_b2 = mean_gate * t_diff
    say(f"   mean trial duration at b = 2: {mean_b2:.2f} s; {n_gate:.3g} trials take "
        f"{n_gate * mean_b2 / 3600:.1f} h of trajectory time")
    (TABLES / "measurement_shape_test.txt").write_text("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
