#!/usr/bin/env python3
"""Deterministic finite-width-gate calculation of the upper fold line at xi = 0.44.

This script is deliberately independent of the delta-sink formulas used elsewhere in the
project.  It discretizes

    H = -(1/2) d^2/dx^2 + (b/ell) 1_{|x-theta| <= ell/2}

on the interior of [0, 1], with Dirichlet endpoints.  The rectangular potential is integrated
over each nodal dual cell, so its discrete mass is exactly one even when a gate edge cuts a cell.
The resulting real symmetric tridiagonal operator is diagonalized with NumPy only (no SciPy).

For an orthonormal eigenpair (mu_n, v_n), a point start at grid node j gives

    S_h(t)   = sum_n A_n exp(-mu_n t),       A_n = v_n[j] sum_i v_n[i],
    Phi_h(t) = sum_n mu_n A_n exp(-mu_n t).

Thus a fold obeys M_2 = M_3 = 0, where
M_k(t) = sum_n mu_n^k A_n exp(-mu_n t).  We solve these equations on three nested meshes,
check second-order convergence, and Richardson-extrapolate the fold.  The reported observable
R uses the same fixed +/-20% peak and valley windows as the measurement protocol; each window is
integrated analytically in time from the finite-gate spectrum, rather than sampled at its centre.

Outputs
-------
outputs/tables/dpma_finite_gate_check.txt
outputs/tables/dpma_finite_gate_check.csv
outputs/tables/dpma_finite_gate_check.json
"""

from __future__ import annotations

import csv
import io
import json
import math
from dataclasses import dataclass

import numpy as np

from paths import TABLES as TABLE_DIR


D = 0.5
THETA = 0.5
XI = 0.44
ELLS = (0.04, 0.02, 0.01)
MESHES = (200, 400, 800)  # number of equal subintervals; XI is a node on every mesh

# Fixed measurement windows from the below-fold delta-sink reference curve.  They are held fixed
# while ell and b change; R=1 is therefore not the fold condition.
PEAK_CENTRE = 0.0462
VALLEY_CENTRE = 0.0253
WINDOW_FRACTION = 0.20

# Reference values from the mesh-free solution of the same finite-gate problem (piecewise-analytic
# eigenfunctions; measurement_window_ratio.py, part [7]).  They are comparisons, not inputs to the
# fold solver: the finite-difference Richardson values below must reproduce them.
BC_REFERENCE = {0.04: 3.0233509, 0.02: 3.0755334, 0.01: 3.1049443}
RC_ELL02_REFERENCE = 0.9700000
BC_ZERO_REFERENCE = 3.1365683

# The Richardson-extrapolated finite-difference values agree with the mesh-free ones to a few 1e-7;
# the tolerances leave room for LAPACK-level roundoff differences across machines.
BC_REGRESSION_ATOL = 2.0e-6
RC_REGRESSION_ATOL = 2.0e-6
BC_ZERO_REGRESSION_ATOL = 2.0e-6

ROOT_TAU_HINT = 0.035
TAU_SCAN = np.linspace(0.020, 0.055, 141)
EXPONENT_CUTOFF = 700.0

TXT_PATH = TABLE_DIR / "dpma_finite_gate_check.txt"
CSV_PATH = TABLE_DIR / "dpma_finite_gate_check.csv"
JSON_PATH = TABLE_DIR / "dpma_finite_gate_check.json"


def require(condition: bool, message: str) -> None:
    """An assertion that remains active under ``python -O``."""
    if not condition:
        raise AssertionError(message)


@dataclass(frozen=True)
class Spectrum:
    mu: np.ndarray
    amplitude: np.ndarray


@dataclass(frozen=True)
class MeshFold:
    intervals: int
    b: float
    tau: float
    ratio: float
    phi: float
    scaled_m2_residual: float
    scaled_m3_residual: float


def gate_profile(ell: float, intervals: int) -> tuple[np.ndarray, float]:
    """Return the cell-averaged unit-mass rectangular gate on the interior nodes."""
    h = 1.0 / intervals
    x = np.arange(1, intervals, dtype=float) * h
    left = THETA - ell / 2.0
    right = THETA + ell / 2.0
    overlap = np.maximum(
        0.0,
        np.minimum(x + h / 2.0, right) - np.maximum(x - h / 2.0, left),
    )
    profile = overlap / (h * ell)
    require(abs(h * float(profile.sum()) - 1.0) < 5e-14,
            f"gate quadrature is not unit mass for ell={ell}, M={intervals}")
    return profile, h


def finite_difference_spectrum(b: float, ell: float, intervals: int) -> Spectrum:
    """Diagonalize the symmetric tridiagonal finite-difference Hamiltonian."""
    gate, h = gate_profile(ell, intervals)
    n = intervals - 1
    diagonal = np.full(n, 2.0 * D / h**2) + b * gate
    off_diagonal = np.full(n - 1, -D / h**2)

    # NumPy does not expose a tridiagonal eigensolver.  Materializing the three nonzero diagonals
    # keeps the public script dependency-free; np.linalg.eigh then uses the platform LAPACK.
    hamiltonian = np.diag(diagonal)
    index = np.arange(n - 1)
    hamiltonian[index, index + 1] = off_diagonal
    hamiltonian[index + 1, index] = off_diagonal
    mu, vectors = np.linalg.eigh(hamiltonian)
    require(bool(np.all(mu > 0.0)), "Dirichlet finite-gate spectrum must be strictly positive")

    start_coordinate = XI * intervals
    start_index = int(round(start_coordinate)) - 1
    require(abs(start_coordinate - round(start_coordinate)) < 1e-13,
            f"xi={XI} is not a node for M={intervals}")
    amplitude = vectors[start_index, :] * vectors.sum(axis=0)
    require(np.all(np.isfinite(amplitude)), "non-finite spectral amplitude")
    return Spectrum(mu=mu, amplitude=amplitude)


def moment(spectrum: Spectrum, tau: float, power: int) -> float:
    """Evaluate M_power(tau), dropping terms below the double-precision exponential floor."""
    active = spectrum.mu * tau < EXPONENT_CUTOFF
    mu = spectrum.mu[active]
    amplitude = spectrum.amplitude[active]
    return float(np.dot(mu**power * amplitude, np.exp(-mu * tau)))


def inflection_time(spectrum: Spectrum) -> float:
    """Find the M_3=0 branch that becomes the late peak/valley fold."""
    values = np.array([moment(spectrum, tau, 3) for tau in TAU_SCAN])
    changes = np.flatnonzero(values[:-1] * values[1:] < 0.0)
    require(changes.size > 0, "no M3 sign change in the fold-time scan")
    candidates: list[tuple[float, float, float]] = []
    for idx in changes:
        lo = float(TAU_SCAN[idx])
        hi = float(TAU_SCAN[idx + 1])
        candidates.append((abs((lo + hi) / 2.0 - ROOT_TAU_HINT), lo, hi))
    _, lo, hi = min(candidates)
    f_lo = moment(spectrum, lo, 3)
    for _ in range(55):
        mid = (lo + hi) / 2.0
        f_mid = moment(spectrum, mid, 3)
        if f_lo * f_mid <= 0.0:
            hi = mid
        else:
            lo, f_lo = mid, f_mid
    tau = (lo + hi) / 2.0
    require(abs(moment(spectrum, tau, 3)) < 1e-6,
            "M3 bisection failed to converge")
    return tau


def fold_residual(b: float, ell: float, intervals: int) -> tuple[float, float, Spectrum]:
    spectrum = finite_difference_spectrum(b, ell, intervals)
    tau = inflection_time(spectrum)
    return moment(spectrum, tau, 2), tau, spectrum


def window_average(spectrum: Spectrum, centre: float) -> float:
    """Average Phi over a fixed relative window, integrated exactly mode by mode."""
    lo = centre * (1.0 - WINDOW_FRACTION)
    hi = centre * (1.0 + WINDOW_FRACTION)
    integral_terms = spectrum.amplitude * (
        np.exp(-spectrum.mu * lo) - np.exp(-spectrum.mu * hi)
    )
    return float(integral_terms.sum() / (hi - lo))


def solve_mesh_fold(ell: float, intervals: int) -> MeshFold:
    """Solve M2=M3=0 with a safeguarded secant iteration in b."""
    b_prev, b_now = 2.70, 3.40
    f_prev, _, _ = fold_residual(b_prev, ell, intervals)
    f_now, _, _ = fold_residual(b_now, ell, intervals)
    require(f_prev * f_now < 0.0,
            f"fold is not bracketed in b for ell={ell}, M={intervals}")

    spectrum: Spectrum | None = None
    tau = math.nan
    for _ in range(12):
        denominator = f_now - f_prev
        require(abs(denominator) > 1e-14, "degenerate secant denominator")
        b_next = b_now - f_now * (b_now - b_prev) / denominator
        if not (2.30 < b_next < 4.00):
            b_next = (b_prev + b_now) / 2.0
        f_next, tau, spectrum = fold_residual(b_next, ell, intervals)
        if abs(f_next) < 1e-7:
            b_now, f_now = b_next, f_next
            break
        b_prev, f_prev, b_now, f_now = b_now, f_now, b_next, f_next
    else:
        raise RuntimeError(f"fold secant did not converge for ell={ell}, M={intervals}")

    require(spectrum is not None, "internal fold-solver error")
    phi = moment(spectrum, tau, 1)
    m3 = moment(spectrum, tau, 3)
    scaled_m2 = abs(f_now) * tau**2 / abs(phi)
    scaled_m3 = abs(m3) * tau**3 / abs(phi)
    require(scaled_m2 < 1e-9 and scaled_m3 < 1e-9,
            f"scaled fold residual too large for ell={ell}, M={intervals}")

    peak_average = window_average(spectrum, PEAK_CENTRE)
    valley_average = window_average(spectrum, VALLEY_CENTRE)
    require(peak_average > 0.0 and valley_average > 0.0,
            "window-averaged FPT density must be positive")
    ratio = peak_average / valley_average
    return MeshFold(
        intervals=intervals,
        b=b_now,
        tau=tau,
        ratio=ratio,
        phi=phi,
        scaled_m2_residual=scaled_m2,
        scaled_m3_residual=scaled_m3,
    )


def richardson(coarser: float, finer: float, order: float = 2.0) -> float:
    return finer + (finer - coarser) / (2.0**order - 1.0)


def observed_order(coarse: float, medium: float, fine: float) -> float:
    numerator = abs(medium - coarse)
    denominator = abs(fine - medium)
    require(numerator > 0.0 and denominator > 0.0, "zero mesh difference")
    return math.log(numerator / denominator, 2.0)


def build_outputs(results: list[dict[str, float]], b_zero: float) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    lines = [
        "Deterministic finite-width gate calculation",
        "model: D=1/2, Dirichlet [0,1], theta=0.5, xi=0.44, "
        "killing rate (b/ell) on |x-theta|<=ell/2",
        "numerics: cell-integrated rectangular potential; symmetric tridiagonal second-order FD; "
        "all modes from numpy.linalg.eigh; M=200,400,800",
        "fold equations: M2(tau,b)=M3(tau,b)=0",
        "observable protocol: fixed +/-20% windows centred at tau_p=0.0462 and tau_v=0.0253; "
        "mode-by-mode time integrals (not centre-point values)",
        "",
        " ell       b(M=200)    b(M=400)    b(M=800)    p_obs   b_Richardson  "
        "|d_b|       tau_fold      R_fold       mesh-free      abs.diff",
    ]
    for row in results:
        lines.append(
            f" {row['ell']:.3f}   {row['b_M200']:.9f}  {row['b_M400']:.9f}  "
            f"{row['b_M800']:.9f}  {row['observed_order_b']:.4f}   "
            f"{row['b_richardson']:.9f}  {row['b_last_correction']:.2e}  "
            f"{row['tau_richardson']:.9f}  {row['R_richardson']:.9f}  "
            f"{row['b_reference']:.7f}  {row['b_abs_error']:.2e}"
        )
    lines.extend([
        "",
        f"quadratic ell->0 extrapolation from Richardson values: b_c(0)={b_zero:.9f} "
        f"(mesh-free gate {BC_ZERO_REFERENCE:.7f}, abs.diff={abs(b_zero-BC_ZERO_REFERENCE):.2e}; point sink 3.13660)",
        f"ell=0.02 same-window fold ratio: R_c={next(r['R_richardson'] for r in results if r['ell'] == 0.02):.9f} "
        f"(mesh-free gate {RC_ELL02_REFERENCE:.7f})",
        "",
        "Discretization note: |d_b|=|b_Richardson-b(M=800)| is the conservative last mesh "
        "correction.  The observed orders are approximately two.  The ell->0 quadratic fit is "
        "a three-width consistency extrapolation, not a rigorous error bound.",
        "Protocol note: R is formed from the finite-width density at that width's own fold, while "
        "the two time windows are held fixed across widths.  A centre-point ratio is a different "
        "observable.",
        "",
        "regression assertions: PASS",
    ])
    TXT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    fieldnames = list(results[0].keys())
    csv_buffer = io.StringIO()
    writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(results)
    CSV_PATH.write_text(csv_buffer.getvalue(), encoding="utf-8")

    payload = {
        "model": {
            "D": D,
            "domain": [0.0, 1.0],
            "boundary": "Dirichlet",
            "theta": THETA,
            "xi": XI,
            "gate": "rectangular killing rate b/ell on |x-theta| <= ell/2",
        },
        "observable": {
            "definition": "ratio of window-averaged FPT densities, peak / valley",
            "window_fraction": WINDOW_FRACTION,
            "peak_centre": PEAK_CENTRE,
            "valley_centre": VALLEY_CENTRE,
            "windows_fixed_across_ell": True,
            "evaluated_at_each_widths_own_fold": True,
        },
        "numerics": {
            "method": "cell-integrated rectangular potential plus symmetric tridiagonal second-order finite differences",
            "eigensolver": "numpy.linalg.eigh",
            "scipy_used": False,
            "meshes": list(MESHES),
            "richardson_order": 2,
            "all_discrete_modes_used": True,
        },
        "regression": {
            "b_reference": {str(key): value for key, value in BC_REFERENCE.items()},
            "b_absolute_tolerance": BC_REGRESSION_ATOL,
            "R_ell_0.02_reference": RC_ELL02_REFERENCE,
            "R_absolute_tolerance": RC_REGRESSION_ATOL,
            "b_ell_to_zero_reference": BC_ZERO_REFERENCE,
            "b_ell_to_zero_absolute_tolerance": BC_ZERO_REGRESSION_ATOL,
            "status": "PASS",
        },
        "results": results,
        "b_ell_to_zero_quadratic": b_zero,
    }
    JSON_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    results: list[dict[str, float]] = []
    for ell in ELLS:
        mesh_results = [solve_mesh_fold(ell, intervals) for intervals in MESHES]
        coarse, medium, fine = mesh_results
        p_b = observed_order(coarse.b, medium.b, fine.b)
        require(1.8 < p_b < 2.2,
                f"fold threshold is not second-order convergent for ell={ell}: p={p_b}")

        b_rich = richardson(medium.b, fine.b)
        tau_rich = richardson(medium.tau, fine.tau)
        ratio_rich = richardson(medium.ratio, fine.ratio)
        reference = BC_REFERENCE[ell]
        error = abs(b_rich - reference)
        require(error < BC_REGRESSION_ATOL,
                f"b_c regression failed for ell={ell}: {b_rich} vs {reference}")
        results.append({
            "ell": ell,
            "b_M200": coarse.b,
            "b_M400": medium.b,
            "b_M800": fine.b,
            "observed_order_b": p_b,
            "b_richardson": b_rich,
            "b_last_correction": abs(b_rich - fine.b),
            "tau_M200": coarse.tau,
            "tau_M400": medium.tau,
            "tau_M800": fine.tau,
            "tau_richardson": tau_rich,
            "R_M200": coarse.ratio,
            "R_M400": medium.ratio,
            "R_M800": fine.ratio,
            "R_richardson": ratio_rich,
            "b_reference": reference,
            "b_abs_error": error,
            "scaled_M2_residual_M800": fine.scaled_m2_residual,
            "scaled_M3_residual_M800": fine.scaled_m3_residual,
        })

    ratio_ell02 = next(row["R_richardson"] for row in results if row["ell"] == 0.02)
    require(abs(ratio_ell02 - RC_ELL02_REFERENCE) < RC_REGRESSION_ATOL,
            f"ell=0.02 window-ratio regression failed: {ratio_ell02}")

    ell_array = np.array([row["ell"] for row in results])
    b_array = np.array([row["b_richardson"] for row in results])
    b_zero = float(np.polyfit(ell_array, b_array, 2)[2])
    require(abs(b_zero - BC_ZERO_REFERENCE) < BC_ZERO_REGRESSION_ATOL,
            f"ell->0 regression failed: {b_zero} vs {BC_ZERO_REFERENCE}")

    build_outputs(results, b_zero)
    print(TXT_PATH.read_text(encoding="utf-8"), end="")
    print("wrote outputs/tables/dpma_finite_gate_check.csv")
    print("wrote outputs/tables/dpma_finite_gate_check.json")


if __name__ == "__main__":
    main()
