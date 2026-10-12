#!/usr/bin/env python3
"""Shape test, mean-based calibration and colloid example of the proposed measurement (Sec. VI).

Window ratio R(b) (late-peak window over valley window), theta = 1/2, xi = 0.44, windows +-20%
around tau_p = 0.0462 and tau_v = 0.0253 (spectrum and windows from measurement_window_ratio.py).
Computes: the b at which R = 1; trials for a two-standard-deviation test of R > 1; the early-peak
time; the relative depth of the valley just below b_+; mean, standard deviation and b-derivative
of the absorption time at b_+ and the trials needed for a mean-based calibration of b; the capture
rates kappa for the two thresholds; and the numbers of the colloid example.
Writes outputs/tables/measurement_shape_test.txt.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq

import measurement_window_ratio as m
from paths import TABLES

LINES: list[str] = []


def say(s=""):
    print(s, flush=True)
    LINES.append(s)


THETA, XI = 0.5, 0.44
B_PLUS = 3.1366036780          # upper fold line at xi = 0.44 (dpma_offgate_phase.py)
B_C_MIN = 2.16451853           # minimum of b_c(theta), at theta = 0.381 (dpma_saddle_node_bc_theta.py)

pw, vw = m.windows(0.0462, 0.0253)


def R(b):
    return m.window_ratio(m.delta_spectrum(b), pw, vw)


def mfpt(b):
    return XI * (1 - XI) - 2 * b * THETA * (1 - THETA) * min(XI, THETA) * (1 - max(XI, THETA)) / (
        1 + 2 * b * THETA * (1 - THETA))


def main():
    say("Shape test and calibration numbers (theta = 1/2, xi = 0.44)")
    b1 = brentq(lambda b: R(b)[0] - 1.0, 2.0, 3.2, xtol=1e-10)
    say(f"R = 1 at b = {b1:.4f}")
    for b in (1.5, 2.0, 2.5):
        r, p, v = R(b)
        n2 = 4 * r**2 * (1 / p + 1 / v) / (r - 1) ** 2
        say(f"b = {b}: R = {r:.5f}, f_p = {p:.5f}, f_v = {v:.5f};  N_t(2 s.d., R > 1) = {n2:.3g}")

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

    # colloid example: 1 um bead in water, D ~ 0.44 um^2/s, L = 5 um, ell = 0.25 um, 5 ms frames
    D, L, ell, frame = 0.44, 5.0, 0.25, 0.005
    t_diff = L**2 / (2 * D)
    kappa = 2 * D * B_PLUS / (ell * L)
    say("colloid example (D = 0.44 um^2/s, L = 5 um, ell = 0.25 um, 5 ms frames):")
    say(f"   diffusion time L^2/(2D) = {t_diff:.1f} s; gate crossing ell^2/(2D) = {ell**2 / (2 * D):.3f} s")
    say(f"   kappa at b_+ = {kappa:.2f} 1/s; trigger probability per frame 1 - exp(-kappa dt) = "
        f"{-math.expm1(-kappa * frame):.4f}")
    mean_b2 = mfpt(2.0) * t_diff
    say(f"   mean trial duration at b = 2: {mean_b2:.2f} s; 1.4e4 trials take "
        f"{1.4e4 * mean_b2 / 3600:.1f} h of trajectory time; ell/L = {ell / L:.2f}")
    (TABLES / "measurement_shape_test.txt").write_text("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
