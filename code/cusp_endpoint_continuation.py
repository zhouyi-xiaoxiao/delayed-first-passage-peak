#!/usr/bin/env python3
"""Cusps of the two-peak region near an end: half-line cusps and interval cusp loci (App. C).

This script constructs the half-line arrival transform, locates its two nondegenerate A3 cusps
(release on the wall side, X < 1, and on the far side, X > 1, of the sink), and continues the
corresponding finite-interval cusp branches from theta=1/2 toward the left end
(theta = 0.50, 0.49, ..., 0.20, k <= 160; repeated with k <= 120 and k <= 220 at
theta = 0.2, 0.3, 0.4, 0.5).

The half-line gate is at y=1, the absorbing boundary is y=0, and the scaled
start is X>0.  The total absorption-time transform is

  fhat(s;B,X) = exp(-k X)
      + B (1-exp(-k)) [exp(-k|1-X|)-exp(-k(1+X))]
        / [k + B(1-exp(-2k))],       k=sqrt(2s).

The density and its time derivatives are inverted from the negative-real-axis
branch cut with deterministic composite Gauss-Legendre quadrature (orders 112 and 80, two
cutoffs).  No SciPy or stored constant is used in the solve.

Outputs: outputs/tables/cusp_endpoint_continuation.{txt,json}, outputs/tables/dpma_cusp_endpoint_loci.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from typing import Callable, Iterable

import numpy as np
from numpy.polynomial.legendre import leggauss

from paths import TABLES as ART


def composite_gauss(order: int = 112, vmax: float = 60.0, width: float = math.pi / 2):
    """Deterministic nodes on equal subintervals; endpoints are never sampled."""
    local_x, local_w = leggauss(order)
    edges = np.arange(0.0, vmax + width, width)
    nodes: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        nodes.append(0.5 * (lo + hi) + 0.5 * (hi - lo) * local_x)
        weights.append(0.5 * (hi - lo) * local_w)
    return np.concatenate(nodes), np.concatenate(weights)


class HalfLineCusp:
    def __init__(self, order: int = 112, vmax: float = 60.0):
        self.order = int(order)
        self.vmax = float(vmax)
        self.v, self.qw = composite_gauss(self.order, self.vmax)

    def transform_on_cut(self, B: float, X: float) -> np.ndarray:
        """Boundary value fhat(-v^2/2+i0); k=i v."""
        kappa = 1j * self.v
        r11 = (1.0 - np.exp(-2.0 * kappa)) / kappa
        r1x = (
            np.exp(-kappa * abs(1.0 - X))
            - np.exp(-kappa * (1.0 + X))
        ) / kappa
        return (
            np.exp(-kappa * X)
            + B * r1x * (1.0 - np.exp(-kappa)) / (1.0 + B * r11)
        )

    def moments(self, B: float, T: float, X: float, order: int = 4) -> np.ndarray:
        if not (B > 0.0 and T > 0.0 and X > 0.0):
            raise ValueError("B, T, and X must be positive")
        transform = self.transform_on_cut(B, X)
        # Bromwich branch-cut inversion: A(v)=-v Im fhat(-v^2/2+i0)/pi.
        spectral_weight = -self.v * np.imag(transform) / math.pi
        base = self.qw * spectral_weight * np.exp(-0.5 * T * self.v**2)
        mu = 0.5 * self.v**2
        return np.asarray([np.sum(base * mu**n) for n in range(order + 1)])

    def normalized_cusp_residual(self, z: np.ndarray) -> np.ndarray:
        B, T = np.exp(z[:2])
        X = float(z[2])
        m = self.moments(float(B), float(T), X, order=3)
        if not np.all(np.isfinite(m)) or m[0] <= 0.0:
            raise RuntimeError("invalid half-line density during cusp solve")
        return np.asarray([T * m[1] / m[0], T**2 * m[2] / m[0], T**3 * m[3] / m[0]])

    def solve_fold_at_start(self, X: float, guess: Iterable[float]) -> dict[str, float]:
        """Two-equation fold solve, used to recover the release-at-sink constant (X=1)."""
        B0, T0 = [float(value) for value in guess]

        def residual(z: np.ndarray) -> np.ndarray:
            B, T = np.exp(z)
            m = self.moments(float(B), float(T), X, order=3)
            return np.asarray([T * m[1] / m[0], T**2 * m[2] / m[0]])

        z, residual_inf = damped_newton(
            residual,
            np.asarray([math.log(B0), math.log(T0)]),
            tolerance=2.0e-13,
            difference_step=1.0e-5,
            iterations=20,
        )
        B, T = np.exp(z)
        m = self.moments(float(B), float(T), X, order=3)
        return {
            "X": float(X),
            "B": float(B),
            "T": float(T),
            "residual_inf": float(residual_inf),
            "T3_S3_over_Phi": float(T**3 * m[3] / m[0]),
        }

    def solve(self, guess: Iterable[float]) -> dict[str, float]:
        B, T, X = [float(value) for value in guess]
        z, residual = damped_newton(
            self.normalized_cusp_residual,
            np.asarray([math.log(B), math.log(T), X]),
            tolerance=2.0e-13,
            difference_step=1.0e-5,
            iterations=20,
        )
        B, T = np.exp(z[:2])
        X = float(z[2])
        m = self.moments(float(B), float(T), X, order=4)

        def unfolding(logB: float, start: float) -> np.ndarray:
            values = self.moments(math.exp(logB), float(T), start, order=2)
            return np.asarray([T * values[1] / values[0], T**2 * values[2] / values[0]])

        h = 1.0e-5
        column_B = (unfolding(math.log(B) + h, X) - unfolding(math.log(B) - h, X)) / (2.0 * h)
        column_X = (unfolding(math.log(B), X + h) - unfolding(math.log(B), X - h)) / (2.0 * h)
        unfolding_jacobian = np.column_stack([column_B, column_X])

        eye = np.eye(3)
        full_jacobian = np.column_stack(
            [
                (
                    self.normalized_cusp_residual(z + h * direction)
                    - self.normalized_cusp_residual(z - h * direction)
                )
                / (2.0 * h)
                for direction in eye
            ]
        )
        return {
            "X": X,
            "B": float(B),
            "T": float(T),
            "density": float(m[0]),
            "residual_inf": float(residual),
            "T4_S4_over_Phi": float(T**4 * m[4] / m[0]),
            "unfolding_determinant_logB_X": float(np.linalg.det(unfolding_jacobian)),
            "unfolding_singular_values": np.linalg.svd(unfolding_jacobian)[1].tolist(),
            "full_cusp_jacobian_determinant": float(np.linalg.det(full_jacobian)),
            "quadrature_order": self.order,
            "quadrature_vmax": self.vmax,
        }


def damped_newton(
    function: Callable[[np.ndarray], np.ndarray],
    initial: np.ndarray,
    *,
    tolerance: float,
    difference_step: float,
    iterations: int,
):
    z = np.asarray(initial, dtype=float)
    eye = np.eye(len(z))
    for _ in range(iterations):
        residual = function(z)
        norm = float(np.linalg.norm(residual, ord=np.inf))
        if norm < tolerance:
            return z, norm
        jacobian = np.column_stack(
            [
                (
                    function(z + difference_step * direction)
                    - function(z - difference_step * direction)
                )
                / (2.0 * difference_step)
                for direction in eye
            ]
        )
        step = np.linalg.solve(jacobian, -residual)
        damping = 1.0
        for _ in range(16):
            candidate = z + damping * step
            try:
                candidate_norm = float(np.linalg.norm(function(candidate), ord=np.inf))
            except (FloatingPointError, OverflowError, RuntimeError, ValueError):
                candidate_norm = math.inf
            if candidate_norm < norm:
                z = candidate
                break
            damping *= 0.5
        else:
            raise RuntimeError(f"Newton line search failed at {z}, residual={residual}")
    raise RuntimeError(f"Newton did not converge: {function(z)}")


def load_interval_solver():
    import dpma_offgate_phase
    return dpma_offgate_phase


def continue_interval_cusps(module, theta_values: list[float], kmax: float = 160.0):
    branches = {
        "boundary_side": (5.8163021864, 0.01607956718, 0.3788069561),
        "domain_side": (5.8163021864, 0.01607956718, 0.6211930439),
    }
    rows: list[dict[str, float | str]] = []
    for branch, initial in branches.items():
        guess = initial
        for theta in theta_values:
            cusp = module.solve_cusp(theta, guess, kmax=kmax)
            nondegeneracy = module.cusp_nondegeneracy(cusp)
            row: dict[str, float | str] = {
                "branch": branch,
                "theta": float(theta),
                "xi": cusp["xi"],
                "b": cusp["b"],
                "tau": cusp["tau"],
                "X_scaled": cusp["xi"] / theta,
                "B_scaled": cusp["b"] * theta,
                "T_scaled": cusp["tau"] / theta**2,
                "residual_inf": cusp["residual"],
                "T4_S4_over_Phi": nondegeneracy["tau4_S4_over_Phi"],
                "unfolding_determinant_logb_xi": nondegeneracy["unfolding_determinant"],
                "kmax": kmax,
            }
            rows.append(row)
            guess = (cusp["b"], cusp["tau"], cusp["xi"])
    return rows


def interval_cutoff_checks(module, rows: list[dict[str, float | str]]):
    checks = []
    for branch in ("boundary_side", "domain_side"):
        branch_rows = [row for row in rows if row["branch"] == branch]
        for theta in (0.20, 0.30, 0.40, 0.50):
            reference = min(branch_rows, key=lambda row: abs(float(row["theta"]) - theta))
            guess = (float(reference["b"]), float(reference["tau"]), float(reference["xi"]))
            shallow = module.solve_cusp(theta, guess, kmax=120.0)
            deep = module.solve_cusp(theta, guess, kmax=220.0)
            checks.append(
                {
                    "branch": branch,
                    "theta": theta,
                    "delta_b_k120_k220": abs(shallow["b"] - deep["b"]),
                    "delta_tau_k120_k220": abs(shallow["tau"] - deep["tau"]),
                    "delta_xi_k120_k220": abs(shallow["xi"] - deep["xi"]),
                    "residual_k220": deep["residual"],
                }
            )
    return checks


def write_outputs(
    halfline: dict[str, dict[str, float]],
    source_start_fold: dict[str, float],
    quadrature_check: dict[str, dict[str, float]],
    rows: list[dict[str, float | str]],
    cutoff_checks: list[dict[str, float | str]],
):
    with (ART / "dpma_cusp_endpoint_loci.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "claim_scope": {
            "analytic_structure": "conditional local finite-interval persistence by IFT around any exact nondegenerate half-line cusp root in either isolated numerical neighborhood",
            "numerical_structure": "two distinguished nondegenerate half-line roots and their finite-interval loci continued on theta in [0.20,0.50]",
            "not_claimed": "global exclusion of every additional fold sheet or cusp outside the computed parameter range",
        },
        "halfline_cusps": halfline,
        "source_start_fold_transform_crosscheck": source_start_fold,
        "quadrature_crosscheck": quadrature_check,
        "finite_interval_loci": rows,
        "finite_interval_cutoff_checks": cutoff_checks,
    }
    (ART / "cusp_endpoint_continuation.json").write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "Half-line cusps and interval cusp loci near an end",
        "===================================================",
        "",
        "Half-line transform:",
        "  fhat = exp(-k X) + B(1-exp(-k))[exp(-k|1-X|)-exp(-k(1+X))]",
        "         / [k+B(1-exp(-2k))],  k=sqrt(2s).",
        "",
        "Source-start transform cross-check (X=1):",
        f"  B_fold={source_start_fold['B']:.14g}",
        f"  T_fold={source_start_fold['T']:.14g}",
        f"  normalized T^3 S3/Phi={source_start_fold['T3_S3_over_Phi']:.12g}",
        "",
    ]
    for name in ("boundary_side", "domain_side"):
        value = halfline[name]
        lines.extend(
            [
                f"{name}:",
                f"  X*={value['X']:.14g}",
                f"  B*={value['B']:.14g}",
                f"  T*={value['T']:.14g}",
                f"  normalized T^4 S4/Phi={value['T4_S4_over_Phi']:.12g}",
                f"  det d(T S1/Phi,T^2 S2/Phi)/d(log B,X)={value['unfolding_determinant_logB_X']:.12g}",
                f"  full cusp Jacobian determinant={value['full_cusp_jacobian_determinant']:.12g}",
                "",
            ]
        )
    lines.extend(
        [
            "Locally certified endpoint-persistence proposition:",
            "  xi_c/theta -> X*,  b_c theta -> B*,  tau_c/theta^2 -> T* as theta -> 0.",
            "  Reflection gives the corresponding theta -> 1 laws.",
            "  The far-boundary heat-kernel images and their required derivatives are",
            "  exponentially small on a compact neighborhood of each half-line cusp;",
            "  nonzero S4 and the full-rank unfolding permit the implicit-function theorem.",
            "",
            "Scope guard:",
            "  This script supplies the floating-point locations and continuation diagnostics.",
            "  Exact local existence, uniqueness, and nondegeneracy in the two displayed boxes",
            "  are proved independently by halfline_cusp_interval_newton.py (interval Newton)",
            "  and halfline_cusp_krawczyk.py (Krawczyk).  The combined results do",
            "  not prove a global no-other-root or no-other-sheet theorem.",
        ]
    )
    (ART / "cusp_endpoint_continuation.txt").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--theta-min", type=float, default=0.20)
    parser.add_argument("--theta-step", type=float, default=0.01)
    arguments = parser.parse_args()

    primary = HalfLineCusp(order=112, vmax=60.0)
    secondary = HalfLineCusp(order=80, vmax=75.0)
    seeds = {
        "boundary_side": (2.87, 0.0636, 0.758),
        "domain_side": (1.086, 0.207, 1.425),
    }
    halfline = {name: primary.solve(seed) for name, seed in seeds.items()}
    source_start_fold = primary.solve_fold_at_start(1.0, (0.789, 0.158))
    assert abs(source_start_fold["B"] - 0.7890262) < 5.0e-7
    assert abs(source_start_fold["T"] - 0.1579221) < 5.0e-7
    assert abs(source_start_fold["T3_S3_over_Phi"]) > 1.0
    secondary_values = {name: secondary.solve(seed) for name, seed in seeds.items()}
    quadrature_check: dict[str, dict[str, float]] = {}
    for name in seeds:
        quadrature_check[name] = {
            "delta_X": abs(halfline[name]["X"] - secondary_values[name]["X"]),
            "delta_B": abs(halfline[name]["B"] - secondary_values[name]["B"]),
            "delta_T": abs(halfline[name]["T"] - secondary_values[name]["T"]),
        }
        assert max(quadrature_check[name].values()) < 2.0e-11
        assert abs(halfline[name]["T4_S4_over_Phi"]) > 1.0
        assert abs(halfline[name]["unfolding_determinant_logB_X"]) > 1.0
        assert abs(halfline[name]["full_cusp_jacobian_determinant"]) > 1.0e-2

    count = int(round((0.50 - arguments.theta_min) / arguments.theta_step))
    theta_values = [round(0.50 - index * arguments.theta_step, 12) for index in range(count + 1)]
    if abs(theta_values[-1] - arguments.theta_min) > 2.0e-10:
        raise ValueError("theta range is not divisible by theta-step")
    interval = load_interval_solver()
    rows = continue_interval_cusps(interval, theta_values, kmax=160.0)
    cutoff_checks = interval_cutoff_checks(interval, rows)
    for check in cutoff_checks:
        # The shortest-time theta=0.20 boundary-side cusp is the limiting
        # cutoff case; its absolute b shift is 2.36e-5 (1.6e-6 relative).
        assert float(check["delta_b_k120_k220"]) < 3.0e-5
        assert float(check["delta_tau_k120_k220"]) < 2.0e-7
        assert float(check["delta_xi_k120_k220"]) < 2.0e-6

    write_outputs(halfline, source_start_fold, quadrature_check, rows, cutoff_checks)
    print((ART / "cusp_endpoint_continuation.txt").read_text(), end="")
    print("wrote outputs/tables/cusp_endpoint_continuation.json")
    print("wrote outputs/tables/dpma_cusp_endpoint_loci.csv")


if __name__ == "__main__":
    main()
