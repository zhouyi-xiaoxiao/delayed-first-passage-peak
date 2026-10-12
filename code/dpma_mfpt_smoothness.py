#!/usr/bin/env python3
"""Exact mean/splitting laws and spectral smoothness through the morphology fold.

The fold changes the number of extrema of the full first-passage density.  This
script verifies that it is not a singularity of either the mean first-passage
time or the lowest relaxation eigenvalue.  It also checks the continuum laws
against exact finite-ring linear algebra; the scaled mean identity is exact at
finite N when the lattice positions are integer representations of xi/theta.
"""
from __future__ import annotations

import json
import math
import numpy as np

from dpma_saddle_node_bc_theta import roots
from paths import TABLES


Q = 2.0 / 3.0
BC_ANTIPODAL = 3.0764323604
B_ENDPOINT = 0.7890262


def continuum_mfpt(xi: float, theta: float, b: float) -> float:
    overlap = min(xi, theta) * (1.0 - max(xi, theta))
    return xi * (1.0 - xi) - (
        2.0 * b * theta * (1.0 - theta) * overlap
        / (1.0 + 2.0 * b * theta * (1.0 - theta))
    )


def continuum_shortcut_probability(xi: float, theta: float, b: float) -> float:
    overlap = min(xi, theta) * (1.0 - max(xi, theta))
    return 2.0 * b * overlap / (1.0 + 2.0 * b * theta * (1.0 - theta))


def finite_chain(N: int, r: int, u: int, b: float) -> tuple[np.ndarray, float]:
    """Transient matrix and per-step shortcut rate for ``lambda=b q/N``."""
    lam = b * Q / N
    matrix = np.zeros((N - 1, N - 1))
    for i in range(N - 1):
        matrix[i, i] = 1.0 - Q
        if i > 0:
            matrix[i, i - 1] = Q / 2.0
        if i < N - 2:
            matrix[i, i + 1] = Q / 2.0
    matrix[u - 1, u - 1] -= lam
    return matrix, lam


def direct_finite_observables(N: int, r: int, u: int, b: float) -> tuple[float, float]:
    matrix, lam = finite_chain(N, r, u, b)
    fundamental_start = np.linalg.solve(np.eye(N - 1) - matrix, np.eye(N - 1)[:, r - 1])
    mfpt = float(np.sum(fundamental_start))
    p_shortcut = float(lam * fundamental_start[u - 1])
    return Q * mfpt / N**2, p_shortcut


def first_continuum_eigenvalue(theta: float, b: float) -> tuple[float, float]:
    """Return ``mu_1`` and its analytic b derivative from the secular determinant."""
    w = roots(theta, b, wmax=8.0)[0]
    k = 2.0 * w
    A = math.sin(k * theta)
    B = math.sin(k * (1.0 - theta))
    D_b = 2.0 * A * B
    D_k = math.sin(k) + k * math.cos(k) + 2.0 * b * (
        theta * math.cos(k * theta) * B
        + (1.0 - theta) * A * math.cos(k * (1.0 - theta))
    )
    dk_db = -D_b / D_k
    return 0.5 * k * k, k * dk_db


def main() -> None:
    lines: list[str] = []

    def say(message: str = "") -> None:
        print(message, flush=True)
        lines.append(message)

    say("Exact finite-N and continuum mean/splitting laws")
    say("  scaled mean: (q/N^2) E[T] = xi(1-xi) - 2b theta(1-theta)m/[1+2b theta(1-theta)]")
    say("  shortcut mass: pi_sc = 2b m/[1+2b theta(1-theta)], m=min(xi,theta)[1-max(xi,theta)]")
    say(f"  {'N':>5} {'xi':>6} {'theta':>6} {'b':>6} {'MFPT error':>12} {'pi_sc error':>12}")
    checks = []
    for N, r, u, b in ((100, 50, 50, BC_ANTIPODAL), (200, 88, 100, 2.5),
                        (300, 72, 114, 2.1645), (400, 244, 152, 1.7)):
        xi, theta = r / N, u / N
        direct_mean, direct_pi = direct_finite_observables(N, r, u, b)
        formula_mean = continuum_mfpt(xi, theta, b)
        formula_pi = continuum_shortcut_probability(xi, theta, b)
        mean_error = abs(direct_mean - formula_mean)
        pi_error = abs(direct_pi - formula_pi)
        checks.append((N, xi, theta, b, mean_error, pi_error))
        say(f"  {N:5d} {xi:6.3f} {theta:6.3f} {b:6.3f} {mean_error:12.3e} {pi_error:12.3e}")
    if max(max(row[-2:]) for row in checks) > 2e-12:
        raise RuntimeError("finite-N mean/splitting identity regression failed")

    theta = xi = 0.5
    mfpt_c = continuum_mfpt(xi, theta, BC_ANTIPODAL)
    pi_c = continuum_shortcut_probability(xi, theta, BC_ANTIPODAL)
    say("\nAntipodal morphology fold (no moment or spectral singularity)")
    say(f"  b_c={BC_ANTIPODAL:.10f}: <tau>={mfpt_c:.10f}, pi_sc={pi_c:.10f}")
    for b in (BC_ANTIPODAL - 0.05, BC_ANTIPODAL, BC_ANTIPODAL + 0.05):
        mu, dmu = first_continuum_eigenvalue(theta, b)
        say(f"  b={b:.10f}: mu_1={mu:.10f}, dmu_1/db={dmu:.10f} (>0)")
        if dmu <= 0.0:
            raise RuntimeError("lowest continuum eigenvalue is not monotone")
    h = 1e-5
    mu_minus = first_continuum_eigenvalue(theta, BC_ANTIPODAL - h)[0]
    mu_plus = first_continuum_eigenvalue(theta, BC_ANTIPODAL + h)[0]
    mu_analytic, dmu_analytic = first_continuum_eigenvalue(theta, BC_ANTIPODAL)
    dmu_fd = (mu_plus - mu_minus) / (2.0 * h)
    if abs(dmu_fd / dmu_analytic - 1.0) > 2e-7:
        raise RuntimeError("Hellmann--Feynman/secular eigenvalue derivative check failed")
    say(f"  derivative cross-check: analytic={dmu_analytic:.10f}, finite-difference={dmu_fd:.10f}")

    endpoint_pi = 2.0 * B_ENDPOINT / (1.0 + 2.0 * B_ENDPOINT)
    endpoint_mean_coefficient = 1.0 / (1.0 + 2.0 * B_ENDPOINT)
    say("\nEndpoint predictions along the source-started fold")
    say(f"  pi_sc,c -> 2B*/(1+2B*) = {endpoint_pi:.7f}")
    say(f"  <tau>_c/d -> 1/(1+2B*) = {endpoint_mean_coefficient:.7f}")
    say("  whereas tau_c/d^2 -> 0.1579221: the mean and fold time have different endpoint scales")

    (TABLES / "dpma_mfpt_smoothness.txt").write_text("\n".join(lines) + "\n")
    (TABLES / "dpma_mfpt_smoothness.json").write_text(json.dumps({
        "finite_identity_max_abs_error": max(max(row[-2:]) for row in checks),
        "antipodal": {
            "b_c": BC_ANTIPODAL,
            "mfpt_at_fold": mfpt_c,
            "shortcut_probability_at_fold": pi_c,
            "mu_1_at_fold": mu_analytic,
            "dmu_1_db": dmu_analytic,
        },
        "endpoint": {
            "B_star": B_ENDPOINT,
            "shortcut_probability_limit": endpoint_pi,
            "mfpt_over_d_limit": endpoint_mean_coefficient,
            "fold_time_over_d_squared_limit": 0.1579221,
        },
    }, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
