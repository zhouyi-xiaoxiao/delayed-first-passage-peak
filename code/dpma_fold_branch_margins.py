#!/usr/bin/env python3
"""Nondegeneracy margins along the threshold curve and spectral-cutoff sensitivity.

A nondegenerate fold requires ``S3 != 0`` and ``partial_b S1 != 0``.  This script evaluates
dimensionless margins at theta = 0.20, 0.25, ..., 0.50 plus the refined minimum, and repeats the
moments with two spectral cutoffs (w <= 40 and w <= 80, i.e. k <= 80 and k <= 160).
"""
from __future__ import annotations

import csv
import numpy as np

from dpma_saddle_node_bc_theta import G_affected, continue_fold, locate_boundary_minimum, roots
from paths import TABLES


def moments(theta: float, b: float, tau: float, wmax: float) -> np.ndarray:
    """Return Phi,S1,S2,S3 for a source-started gate using a specified root cutoff."""
    amplitudes: list[float] = []
    rates: list[float] = []
    for w in roots(theta, b, wmax=wmax):
        # Sink-node modes have zero FPT amplitude when xi=theta.  At a rational
        # theta they may be roots of the unperturbed factor rather than sign-changing roots.
        if abs(np.sin(2.0 * theta * w)) < 1e-9:
            continue
        amplitudes.append(G_affected(w, theta, theta, b))
        rates.append(2.0 * w * w)
    G = np.asarray(amplitudes)
    mu = np.asarray(rates)
    weighted = G * np.exp(-mu * tau)
    return np.asarray([np.sum(weighted * mu**order) for order in range(4)])


def main() -> None:
    continuation = continue_fold(np.linspace(0.5, 0.2, 61))
    minimum = locate_boundary_minimum(continuation)
    by_theta = {round(th, 10): (b, tau) for th, b, tau in continuation}
    selected = [(th, *by_theta[round(th, 10)]) for th in (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)]
    selected.append(minimum)
    selected.sort(key=lambda row: row[0])

    lines = [
        "Branch-wide nondegeneracy of the source-started fold",
        "  moments use w<=80; cutoff sensitivity compares w<=40 and w<=80",
        "  theta       b_c       tau_c   tau^3 S3/Phi  tau d_bS1/Phi  b tau d_bS1/Phi  |res1|    |res2|  rel S3(40,80)",
    ]
    rows = []
    for theta, b, tau in selected:
        m80 = moments(theta, b, tau, 80.0)
        m40 = moments(theta, b, tau, 40.0)
        h = 1e-5 * max(1.0, b)
        s1_b = (moments(theta, b + h, tau, 80.0)[1] - moments(theta, b - h, tau, 80.0)[1]) / (2.0 * h)
        phi, s1, s2, s3 = m80
        n3 = tau**3 * s3 / phi
        n1b = tau * s1_b / phi
        res1 = tau * s1 / phi
        res2 = tau**2 * s2 / phi
        rel_s3 = abs(m40[3] / m80[3] - 1.0)
        is_min = abs(theta - minimum[0]) < 1e-8
        rows.append((theta, b, tau, n3, n1b, res1, res2, rel_s3, is_min))
        lines.append(
            f"  {theta:6.4f}  {b:9.6f}  {tau:10.7f}  {n3:13.6f}  {n1b:14.6f}  {b * n1b:16.6f}  "
            f"{abs(res1):.2e}  {abs(res2):.2e}  {rel_s3:.2e}" + ("  [minimum]" if is_min else "")
        )

    n3_values = np.abs([row[3] for row in rows])
    n1b_values = np.abs([row[4] for row in rows])
    bn1b_values = np.abs([row[1] * row[4] for row in rows])
    max_residual = max(max(abs(row[5]), abs(row[6])) for row in rows)
    max_cutoff = max(row[7] for row in rows)
    lines.extend([
        "",
        f"  margin ranges: |tau^3 S3/Phi|=[{min(n3_values):.5f},{max(n3_values):.5f}], "
        f"|tau d_bS1/Phi|=[{min(n1b_values):.5f},{max(n1b_values):.5f}], "
        f"|b tau d_bS1/Phi|=[{min(bn1b_values):.5f},{max(bn1b_values):.5f}]",
        f"  maximum normalized fold residual={max_residual:.2e}; maximum S3 cutoff shift={max_cutoff:.2e}",
        "  => no degeneracy is approached on the reported continuation branch.",
    ])
    if min(n3_values) < 1.0 or min(n1b_values) < 0.08 or max_residual > 5e-4 or max_cutoff > 1e-4:
        raise RuntimeError("fold-branch nondegeneracy/cutoff check failed")

    print("\n".join(lines), flush=True)
    (TABLES / "dpma_fold_branch_margins.txt").write_text("\n".join(lines) + "\n")
    with (TABLES / "dpma_fold_branch_margins.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow([
            "theta", "b_c", "tau_c", "tau3_S3_over_Phi", "tau_dS1db_over_Phi",
            "tau_S1_over_Phi", "tau2_S2_over_Phi", "relative_S3_cutoff_shift", "is_minimum",
        ])
        writer.writerows(rows)


if __name__ == "__main__":
    main()
