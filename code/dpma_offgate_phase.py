#!/usr/bin/env python3
"""Two-peak region for release away from the sink (fold lines b_-(xi), b_+(xi) and the cusp).

For xi != theta the early (capture) feature is an interior maximum only on a bounded
interval of sink strengths.  This script follows both fold lines directly from

    S_1(tau,b;theta,xi) = S_2(tau,b;theta,xi) = 0,

without any grid classification of the density.  At theta=1/2 it also solves the
codimension-two end point of the interval, S_1=S_2=S_3=0, and checks the A_3 (cusp)
nondegeneracy.

Numerical safeguards
--------------------
* eigenvalues are recomputed from the continuum characteristic equation up to
  k=160 (mu=k^2/2), with the antipodal spectrum bracketed mode by mode;
* folds are solved in log(b), log(tau) by damped Newton continuation;
* critical-point counts are cross-checked from signed-moment roots and from
  positive-density peak prominence on a converged time interval;
* selected folds are re-solved at k=120 and k=160;
* previously computed reference values are asserted, not merely printed.

Outputs
-------
outputs/tables/dpma_offgate_phase.{txt,csv,json}
(the figure of the two-peak region is drawn by fig4_two_peak_region.py from the dense
continuation written by dpma_offgate_dense.py)
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import numpy as np

from paths import TABLES as TAB

K_DEEP = 160.0
K_CHECK = 120.0
ROOT_STEP = 0.002
# Near xi=theta the lower sheet is exponentially small; double precision then
# limits the normalized cancellation before it limits b or tau.  The cusp and
# all O(1) folds converge much more tightly than this tolerance.
NEWTON_TOL = 2.0e-10
SMALL_B_NEWTON_TOL = 5.0e-8


@dataclass(frozen=True)
class Fold:
    theta: float
    xi: float
    branch: str
    b: float
    tau: float
    residual: float
    kmax: float


def bisect_scalar(function, lo: float, hi: float, xtol=2.0e-14, max_iter=100):
    """Dependency-free bracketed scalar root solve with a hard error bound."""
    flo = float(function(lo))
    fhi = float(function(hi))
    if not (math.isfinite(flo) and math.isfinite(fhi)):
        raise RuntimeError("non-finite endpoint in scalar root bracket")
    if flo == 0.0:
        return float(lo)
    if fhi == 0.0:
        return float(hi)
    if (flo < 0.0) == (fhi < 0.0):
        raise RuntimeError(f"scalar root is not bracketed: f(lo)={flo}, f(hi)={fhi}")
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        fmid = float(function(mid))
        if not math.isfinite(fmid):
            raise RuntimeError("non-finite midpoint in scalar root bracket")
        if fmid == 0.0 or hi - lo <= xtol + 2.0 * np.finfo(float).eps * abs(mid):
            return float(mid)
        if (flo < 0.0) != (fmid < 0.0):
            hi = mid
        else:
            lo, flo = mid, fmid
    raise RuntimeError(f"scalar bisection did not converge on [{lo}, {hi}]")


def sampled_extrema(values: np.ndarray, minimum_prominence: float):
    """Local-extremum count with pairwise pruning of sub-resolution wiggles."""
    differences = np.diff(values)
    signs = np.sign(differences)
    # Plateaus are absent in the curves considered here, but deterministic sign filling
    # prevents a single rounded-zero difference from splitting an extremum.
    for index in range(1, len(signs)):
        if signs[index] == 0.0:
            signs[index] = signs[index - 1]
    for index in range(len(signs) - 2, -1, -1):
        if signs[index] == 0.0:
            signs[index] = signs[index + 1]
    extrema = []
    for index in range(len(signs) - 1):
        if signs[index] > 0.0 and signs[index + 1] < 0.0:
            extrema.append([index + 1, "max"])
        elif signs[index] < 0.0 and signs[index + 1] > 0.0:
            extrema.append([index + 1, "min"])
    changed = True
    while changed and len(extrema) >= 2:
        changed = False
        for index in range(len(extrema) - 1):
            left, right = extrema[index], extrema[index + 1]
            if abs(values[right[0]] - values[left[0]]) < minimum_prominence:
                del extrema[index : index + 2]
                changed = True
                break
    maxima = [index for index, kind in extrema if kind == "max" and values[index] > 0.0]
    minima = [index for index, kind in extrema if kind == "min" and values[index] > 0.0]
    return maxima, minima


def characteristic(k: np.ndarray | float, theta: float, b: float):
    return (
        k * np.sin(k)
        + 2.0 * b * np.sin(theta * k) * np.sin((1.0 - theta) * k)
    )


def antipodal_modes(b: float, xi: float, kmax: float = K_DEEP):
    """Affected antipodal modes; node modes have exactly zero FPT residue."""
    ws = []
    j = 0
    while True:
        lo = (j + 0.5) * math.pi + 2.0e-13
        hi = (j + 1.0) * math.pi - 2.0e-13
        if 2.0 * lo >= kmax:
            break
        w = bisect_scalar(
            lambda z: 2.0 * z * math.cos(z) + b * math.sin(z),
            lo,
            min(hi, 0.5 * kmax),
            xtol=5.0e-15,
        )
        ws.append(w)
        j += 1
    w = np.asarray(ws, dtype=float)
    xstar = min(xi, 1.0 - xi)
    denominator = np.sin(w) ** 2 * (
        1.0 + b * (b + 2.0) / (4.0 * w * w)
    )
    G = 4.0 * w * (1.0 - np.cos(w)) * np.sin(2.0 * w * xstar) / denominator
    mu = 2.0 * w * w
    return mu, G


def general_roots(theta: float, b: float, kmax: float = K_DEEP):
    """All simple positive roots of the continuum characteristic function."""
    grid = np.arange(1.0e-8, kmax + ROOT_STEP, ROOT_STEP)
    values = characteristic(grid, theta, b)
    changes = np.flatnonzero(np.signbit(values[:-1]) != np.signbit(values[1:]))
    roots = []
    for index in changes:
        lo, hi = float(grid[index]), float(grid[index + 1])
        root = bisect_scalar(
            lambda z: float(characteristic(z, theta, b)),
            lo,
            hi,
            xtol=8.0e-14,
        )
        if not roots or abs(root - roots[-1]) > 2.0e-7:
            roots.append(root)
    return np.asarray(roots, dtype=float)


def general_modes(theta: float, b: float, xi: float, kmax: float = K_DEEP):
    k = general_roots(theta, b, kmax)
    G = np.empty_like(k)
    for index, kval in enumerate(k):
        n = int(round(kval / math.pi))
        is_node = (
            abs(kval / math.pi - n) < 2.0e-7
            and abs(math.sin(n * math.pi * theta)) < 2.0e-7
        )
        if is_node:
            G[index] = n * math.pi * (1 - (-1) ** n) * math.sin(n * math.pi * xi)
            continue
        phi = (
            math.sin(kval * (1.0 - theta)) * math.sin(kval * xi)
            if xi <= theta
            else math.sin(kval * theta) * math.sin(kval * (1.0 - xi))
        )
        I = (
            math.sin(kval * (1.0 - theta)) * (1.0 - math.cos(kval * theta))
            + math.sin(kval * theta) * (1.0 - math.cos(kval * (1.0 - theta)))
        ) / kval
        J = (
            math.sin(kval * (1.0 - theta)) ** 2
            * (theta / 2.0 - math.sin(2.0 * kval * theta) / (4.0 * kval))
            + math.sin(kval * theta) ** 2
            * (
                (1.0 - theta) / 2.0
                - math.sin(2.0 * kval * (1.0 - theta)) / (4.0 * kval)
            )
        )
        G[index] = 0.5 * kval * kval * phi * I / J
    return 0.5 * k * k, G


def modes(theta: float, b: float, xi: float, kmax: float = K_DEEP):
    if abs(theta - 0.5) < 5.0e-15:
        return antipodal_modes(b, xi, kmax)
    return general_modes(theta, b, xi, kmax)


def moments(
    tau: float, theta: float, xi: float, b: float, kmax: float = K_DEEP, order: int = 4
):
    mu, G = modes(theta, b, xi, kmax)
    weighted = G * np.exp(-mu * tau)
    return np.asarray([np.sum(weighted * mu**power) for power in range(order + 1)])


def fold_residual(z: np.ndarray, theta: float, xi: float, kmax: float):
    b, tau = np.exp(z)
    m = moments(tau, theta, xi, b, kmax, order=2)
    if not np.all(np.isfinite(m)) or m[0] <= 0.0:
        raise RuntimeError("invalid density while solving fold")
    return np.asarray([tau * m[1] / m[0], tau * tau * m[2] / m[0]])


def cusp_residual(z: np.ndarray, theta: float, kmax: float):
    b, tau = np.exp(z[:2])
    xi = float(z[2])
    m = moments(tau, theta, xi, b, kmax, order=3)
    if not np.all(np.isfinite(m)) or m[0] <= 0.0 or not (0.0 < xi < 1.0):
        raise RuntimeError("invalid density while solving cusp")
    return np.asarray(
        [tau * m[1] / m[0], tau**2 * m[2] / m[0], tau**3 * m[3] / m[0]]
    )


def damped_newton(function, initial: Iterable[float], tolerance=NEWTON_TOL, iterations=18):
    z = np.asarray(initial, dtype=float)
    eye = np.eye(len(z))
    h = 1.5e-4
    for _ in range(iterations):
        residual = function(z)
        norm = float(np.linalg.norm(residual, ord=np.inf))
        if norm < tolerance:
            return z, norm
        jacobian = np.column_stack(
            [(function(z + h * unit) - function(z - h * unit)) / (2.0 * h) for unit in eye]
        )
        step = np.linalg.solve(jacobian, -residual)
        accepted = False
        damping = 1.0
        for _ in range(12):
            candidate = z + damping * step
            try:
                candidate_norm = float(np.linalg.norm(function(candidate), ord=np.inf))
            except (RuntimeError, OverflowError, ValueError):
                candidate_norm = math.inf
            if candidate_norm < norm:
                z = candidate
                accepted = True
                break
            damping *= 0.5
        if not accepted:
            raise RuntimeError(f"Newton line search failed: z={z}, residual={residual}")
    raise RuntimeError(f"Newton failed to converge: z={z}, residual={function(z)}")


def solve_fold(theta: float, xi: float, guess, branch: str, kmax=K_DEEP):
    tolerance = SMALL_B_NEWTON_TOL if guess[0] < 5.0e-4 else NEWTON_TOL
    z, residual = damped_newton(
        lambda value: fold_residual(value, theta, xi, kmax),
        np.log(guess),
        tolerance=tolerance,
    )
    b, tau = np.exp(z)
    return Fold(theta, xi, branch, float(b), float(tau), residual, kmax)


def solve_cusp(theta: float, guess, kmax=K_DEEP):
    initial = np.asarray([math.log(guess[0]), math.log(guess[1]), guess[2]])
    z, residual = damped_newton(lambda value: cusp_residual(value, theta, kmax), initial)
    return {
        "theta": theta,
        "xi": float(z[2]),
        "b": float(math.exp(z[0])),
        "tau": float(math.exp(z[1])),
        "residual": residual,
        "kmax": kmax,
    }


def continue_antipodal_upper(targets):
    targets = sorted(set(float(value) for value in targets), reverse=True)
    current_xi = 0.5
    current = solve_fold(0.5, current_xi, (3.07643236, 0.038363052), "upper")
    found = {0.5: current}
    for target in targets:
        while current_xi > target + 5.0e-13:
            next_xi = max(target, current_xi - 0.002)
            current = solve_fold(0.5, next_xi, (current.b, current.tau), "upper")
            current_xi = next_xi
        found[round(target, 10)] = current
    return [found[round(value, 10)] for value in targets]


def continue_antipodal_lower(targets):
    targets = sorted(set(float(value) for value in targets))
    current_xi = min(targets)
    current = solve_fold(0.5, current_xi, (5.5704, 0.014635), "lower")
    found = {round(current_xi, 10): current}
    target_set = {round(value, 10) for value in targets}
    while current_xi < max(targets) - 5.0e-13:
        next_xi = min(max(targets), current_xi + 0.001)
        current = solve_fold(0.5, next_xi, (current.b, current.tau), "lower")
        current_xi = next_xi
        if round(current_xi, 10) in target_set:
            found[round(current_xi, 10)] = current
    return [found[round(value, 10)] for value in targets]


def critical_count(theta: float, xi: float, b: float, kmax=K_DEEP):
    mu, G = modes(theta, b, xi, kmax)
    tau_lo = max(2.0e-3, 50.0 / (kmax * kmax))
    tau = np.exp(np.linspace(math.log(tau_lo), math.log(0.8), 18000))
    exponential = np.exp(-np.outer(tau, mu))
    with np.errstate(all="ignore"):   # some BLAS builds raise spurious flags on underflow
        density = exponential @ G
        s1 = exponential @ (G * mu)
    derivative = -s1
    roots = []
    kinds = []
    changes = np.flatnonzero(np.signbit(derivative[:-1]) != np.signbit(derivative[1:]))
    for index in changes:
        root_tau = bisect_scalar(
            lambda value: -float(np.sum(G * mu * np.exp(-mu * value))),
            float(tau[index]),
            float(tau[index + 1]),
            xtol=2.0e-14,
        )
        second = float(np.sum(G * mu * mu * np.exp(-mu * root_tau)))
        roots.append(root_tau)
        kinds.append("max" if second < 0.0 else "min")

    scale = float(np.max(density))
    sampled_peaks, sampled_valleys = sampled_extrema(
        density, minimum_prominence=max(1.0e-12, 1.0e-7 * scale)
    )
    minimum_ratio = float(np.min(density) / scale)
    return {
        "theta": theta,
        "xi": xi,
        "b": b,
        "tau_interval": [tau_lo, 0.8],
        "moment_roots": roots,
        "moment_kinds": kinds,
        "n_max_moments": kinds.count("max"),
        "n_min_moments": kinds.count("min"),
        "n_max_sampled": len(sampled_peaks),
        "n_min_sampled": len(sampled_valleys),
        "minimum_density_over_maximum": minimum_ratio,
    }


def cusp_nondegeneracy(cusp):
    theta, xi, b, tau = cusp["theta"], cusp["xi"], cusp["b"], cusp["tau"]
    m = moments(tau, theta, xi, b, K_DEEP, order=4)
    s4 = float(tau**4 * m[4] / m[0])

    def unfolding(logb, start):
        bb = math.exp(logb)
        values = moments(tau, theta, start, bb, K_DEEP, order=2)
        return np.asarray(
            [tau * values[1] / values[0], tau**2 * values[2] / values[0]]
        )

    h = 1.0e-5
    column_b = (unfolding(math.log(b) + h, xi) - unfolding(math.log(b) - h, xi)) / (2 * h)
    column_xi = (unfolding(math.log(b), xi + h) - unfolding(math.log(b), xi - h)) / (2 * h)
    jacobian = np.column_stack([column_b, column_xi])
    return {
        "tau4_S4_over_Phi": s4,
        "unfolding_jacobian_logb_xi": jacobian.tolist(),
        "unfolding_determinant": float(np.linalg.det(jacobian)),
        "unfolding_singular_values": np.linalg.svd(jacobian)[1].tolist(),
    }


def lower_sheet_asymptotics(lower_folds):
    """Short-time essential asymptotic as xi approaches an antipodal gate."""
    by_xi = {round(fold.xi, 10): fold for fold in lower_folds}
    rows = []
    for delta in (0.08, 0.07, 0.06):
        xi = 0.5 - delta
        fold = by_xi[round(xi, 10)]
        a = xi
        A = a * a - delta * delta
        tau_asymptotic = delta * delta * (1.0 + 2.0 * delta * delta / A)
        b_asymptotic = (
            math.e * a * A * A / (2.0 * delta**6)
            * math.exp(-A / (2.0 * delta * delta))
        )
        rows.append(
            {
                "delta": delta,
                "xi": xi,
                "b_direct": fold.b,
                "b_asymptotic": b_asymptotic,
                "b_direct_over_asymptotic": fold.b / b_asymptotic,
                "tau_direct": fold.tau,
                "tau_asymptotic": tau_asymptotic,
                "tau_direct_over_asymptotic": fold.tau / tau_asymptotic,
            }
        )
    ratios = [row["b_direct_over_asymptotic"] for row in rows]
    assert 0.65 < ratios[0] < ratios[1] < ratios[2] < 1.0
    assert max(abs(row["tau_direct_over_asymptotic"] - 1.0) for row in rows) < 0.02
    return rows


def write_outputs(folds, cusp_left, cusp_right, nondegeneracy, lower_asymptotics, checks, convergence):
    csv_path = TAB / "dpma_offgate_phase.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["theta", "xi", "branch", "b", "tau", "residual_inf", "kmax"])
        for fold in folds:
            writer.writerow(
                [
                    f"{fold.theta:.10f}",
                    f"{fold.xi:.10f}",
                    fold.branch,
                    f"{fold.b:.12g}",
                    f"{fold.tau:.12g}",
                    f"{fold.residual:.3e}",
                    f"{fold.kmax:.1f}",
                ]
            )
        for cusp in (cusp_left, cusp_right):
            writer.writerow(
                [
                    f"{cusp['theta']:.10f}",
                    f"{cusp['xi']:.10f}",
                    "cusp",
                    f"{cusp['b']:.12g}",
                    f"{cusp['tau']:.12g}",
                    f"{cusp['residual']:.3e}",
                    f"{cusp['kmax']:.1f}",
                ]
            )

    payload = {
        "settings": {
            "numerical_dependencies": ["numpy"],
            "k_deep": K_DEEP,
            "k_convergence_check": K_CHECK,
            "root_grid_step_general_theta": ROOT_STEP,
            "newton_tolerance": NEWTON_TOL,
            "small_b_newton_tolerance": SMALL_B_NEWTON_TOL,
            "scope": "continuum rank-one delta sink; folds listed in CSV; critical-count scans use the printed tau interval",
        },
        "cusp_left": cusp_left,
        "cusp_right": cusp_right,
        "cusp_nondegeneracy": nondegeneracy,
        "lower_sheet_asymptotics": lower_asymptotics,
        "folds": [asdict(fold) for fold in folds],
        "critical_count_checks": checks,
        "kmax_convergence": convergence,
    }
    (TAB / "dpma_offgate_phase.json").write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "Two-peak region for release away from the sink: signed-moment fold continuation",
        f"deep spectrum k <= {K_DEEP:g}; general-theta root grid dk={ROOT_STEP:g}",
        "",
        "On the fully continued theta=1/2 section, two genuine interior maxima occur only for",
        "    b_-(xi) < b < b_+(xi).",
        "Below b_- only the late maximum remains; above b_+ only the early maximum remains.",
        "",
        "theta=1/2 cusp (left; right is its exact reflection):",
        f"  xi*={cusp_left['xi']:.12f}, b*={cusp_left['b']:.12f}, tau*={cusp_left['tau']:.12f}",
        f"  reflected xi*={cusp_right['xi']:.12f}",
        f"  tau^4 S4/Phi={nondegeneracy['tau4_S4_over_Phi']:.8f}",
        "  unfolding coordinates=(tau S1/Phi, tau^2 S2/Phi); controls=(log b, xi), fixed tau",
        f"  unfolding det={nondegeneracy['unfolding_determinant']:.8f}",
        "",
        "Lower-sheet short-time asymptotic at theta=1/2, xi=1/2-delta:",
        "  b_- ~ e a A^2/(2 delta^6) exp[-A/(2 delta^2)], a=xi, A=a^2-delta^2",
        f"  {'delta':>7} {'b_direct':>14} {'b_asymptotic':>14} {'ratio':>9} {'tau ratio':>10}",
    ]
    for row in lower_asymptotics:
        lines.append(
            f"  {row['delta']:7.3f} {row['b_direct']:14.8g} {row['b_asymptotic']:14.8g} "
            f"{row['b_direct_over_asymptotic']:9.4f} {row['tau_direct_over_asymptotic']:10.4f}"
        )
    lines.extend(["", "Representative folds:",
                  f"  {'theta':>6} {'xi':>7} {'branch':>7} {'b':>14} {'tau':>14} {'|res|inf':>10}"])
    for fold in folds:
        lines.append(
            f"  {fold.theta:6.3f} {fold.xi:7.3f} {fold.branch:>7} "
            f"{fold.b:14.9g} {fold.tau:14.9g} {fold.residual:10.2e}"
        )
    lines.extend(["", "Critical-point cross-checks (moment roots = sampled prominence):"])
    for check in checks:
        lines.append(
            f"  theta={check['theta']:.2f} xi={check['xi']:.2f} b={check['b']:.4g}: "
            f"max={check['n_max_moments']}, min={check['n_min_moments']}, "
            f"sampled=({check['n_max_sampled']},{check['n_min_sampled']}), "
            f"min(Phi)/max(Phi)={check['minimum_density_over_maximum']:.2e}"
        )
    lines.extend(
        [
            "",
            "Summary:",
            "  Near xi=theta the upper fold line continues the release-at-sink threshold; the lower",
            "  fold line is positive at every sampled xi!=theta (although exponentially small near",
            "  the sink), and the two lines terminate at a cusp.",
            "",
        ]
    )
    (TAB / "dpma_offgate_phase.txt").write_text("\n".join(lines) + "\n")


def main():
    cusp_left = solve_cusp(0.5, (5.81630219, 0.01607957, 0.3788069561))
    cusp_right = solve_cusp(0.5, (5.81630219, 0.01607957, 0.6211930439))
    nondegeneracy = cusp_nondegeneracy(cusp_left)

    requested = [0.38, 0.382, 0.385, 0.39, 0.40, 0.41, 0.42, 0.43, 0.44]
    lower = continue_antipodal_lower(requested)
    upper = continue_antipodal_upper(requested)
    lower_asymptotics = lower_sheet_asymptotics(lower)

    # General-theta representative cross-sections, each with both fold sheets.
    representative_specs = [
        (0.38, 0.30, (3.52, 0.00645), (5.23, 0.01280)),
        (0.38, 0.50, (1.66, 0.01438), (4.62, 0.03604)),
        (0.30, 0.24, (3.07, 0.00374), (6.03, 0.00841)),
        (0.30, 0.42, (3.23, 0.01462), (3.71, 0.02582)),
    ]
    general = []
    for theta, xi, lower_guess, upper_guess in representative_specs:
        general.append(solve_fold(theta, xi, lower_guess, "lower"))
        general.append(solve_fold(theta, xi, upper_guess, "upper"))

    by_key = {(round(fold.theta, 3), round(fold.xi, 3), fold.branch): fold for fold in lower + upper + general}

    # Regression checks against previously computed values.
    assert abs(by_key[(0.5, 0.4, "lower")].b - 1.8390558587) < 8.0e-7
    assert abs(by_key[(0.5, 0.4, "upper")].b - 3.8620786232) < 8.0e-7
    assert abs(by_key[(0.5, 0.44, "upper")].b - 3.1366036780) < 8.0e-7
    assert abs(cusp_left["xi"] - 0.3788069561) < 4.0e-8
    assert abs(cusp_left["b"] - 5.81630219) < 4.0e-7
    assert abs(cusp_left["tau"] - 0.01607957) < 4.0e-8
    assert abs(cusp_right["xi"] - (1.0 - cusp_left["xi"])) < 3.0e-10
    assert abs(cusp_right["b"] - cusp_left["b"]) < 3.0e-9
    assert abs(cusp_right["tau"] - cusp_left["tau"]) < 3.0e-10
    assert abs(nondegeneracy["tau4_S4_over_Phi"]) > 1.0
    assert abs(nondegeneracy["unfolding_determinant"]) > 1.0

    # k-cutoff convergence on the hardest small-tau representative and the main upper fold.
    convergence = []
    for theta, xi, branch, guess in [
        (0.30, 0.24, "lower", (3.07, 0.00374)),
        (0.50, 0.44, "upper", (3.14, 0.03457)),
    ]:
        shallow = solve_fold(theta, xi, guess, branch, kmax=K_CHECK)
        deep = by_key[(round(theta, 3), round(xi, 3), branch)]
        row = {
            "theta": theta,
            "xi": xi,
            "branch": branch,
            "delta_b": abs(shallow.b - deep.b),
            "delta_tau": abs(shallow.tau - deep.tau),
            "k_shallow": K_CHECK,
            "k_deep": K_DEEP,
        }
        convergence.append(row)
        assert row["delta_b"] < 3.0e-6
        assert row["delta_tau"] < 3.0e-8

    checks = []
    expected = [
        (0.50, 0.40, 1.0, 1, 0),
        (0.50, 0.40, 2.5, 2, 1),
        (0.50, 0.40, 4.5, 1, 0),
        (0.38, 0.30, 2.0, 1, 0),
        (0.38, 0.30, 4.2, 2, 1),
        (0.38, 0.30, 6.0, 1, 0),
        (0.30, 0.24, 2.0, 1, 0),
        (0.30, 0.24, 4.5, 2, 1),
        (0.30, 0.24, 7.0, 1, 0),
        (0.50, 0.37, 5.8, 1, 0),  # outside the cusp wedge
    ]
    for theta, xi, b, maxima, minima in expected:
        check = critical_count(theta, xi, b)
        checks.append(check)
        assert check["n_max_moments"] == maxima
        assert check["n_min_moments"] == minima
        assert check["n_max_sampled"] == maxima
        assert check["n_min_sampled"] == minima
        assert check["minimum_density_over_maximum"] > -2.0e-10

    folds = sorted(
        lower + upper + general,
        key=lambda fold: (round(fold.theta, 10), round(fold.xi, 10), fold.branch),
    )
    write_outputs(folds, cusp_left, cusp_right, nondegeneracy, lower_asymptotics, checks, convergence)
    print((TAB / "dpma_offgate_phase.txt").read_text(), end="")
    print("wrote outputs/tables/dpma_offgate_phase.csv")
    print("wrote outputs/tables/dpma_offgate_phase.json")


if __name__ == "__main__":
    main()
