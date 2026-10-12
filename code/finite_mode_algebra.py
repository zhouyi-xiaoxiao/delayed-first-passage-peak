#!/usr/bin/env python3
"""Two- and three-mode fold algebra on random rate sets (Table II; App. C, minimum number of modes).

With A_j = G_j mu_j exp(-mu_j tau), a fold of a two-term sum would require A_1 + A_2 = 0 and
mu_1 A_1 + mu_2 A_2 = 0, i.e. a singular 2x2 matrix unless mu_1 = mu_2.  For three terms the fold
equations are solved by A_1:A_2:A_3 = (mu_3 - mu_2):(mu_1 - mu_3):(mu_2 - mu_1), whose signs
alternate for ordered rates.  This script checks both statements on random, well-separated rate
sets (500 draws with seed 20260716; draws with a rate gap < 0.1 are skipped).
Writes outputs/tables/finite_mode_algebra.txt and .json.
"""
from __future__ import annotations

import json

import numpy as np

from paths import TABLES


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def finite_mode_checks() -> dict[str, object]:
    rng = np.random.default_rng(20260716)
    minimum_two_mode_determinant = np.inf
    maximum_three_mode_residual = 0.0
    minimum_three_mode_sign_changes = 3
    trials = 500
    accepted_trials = 0
    for _ in range(trials):
        rates = np.sort(rng.uniform(0.2, 8.0, size=3))
        if min(np.diff(rates)) < 0.1:
            continue
        accepted_trials += 1
        mu1, mu2, mu3 = rates
        two_mode_matrix = np.array(((1.0, 1.0), (mu1, mu2)))
        minimum_two_mode_determinant = min(
            minimum_two_mode_determinant, abs(float(np.linalg.det(two_mode_matrix)))
        )
        contributions = np.array((mu3 - mu2, mu1 - mu3, mu2 - mu1))
        residual = np.array((np.sum(contributions), rates @ contributions), dtype=float)
        maximum_three_mode_residual = max(maximum_three_mode_residual, float(np.max(np.abs(residual))))
        signs = np.sign(contributions)
        sign_changes = int(np.sum(signs[:-1] * signs[1:] < 0))
        minimum_three_mode_sign_changes = min(minimum_three_mode_sign_changes, sign_changes)

    require(minimum_two_mode_determinant > 0.09, "two-mode fold matrix became singular")
    require(maximum_three_mode_residual < 2.0e-14, "three-mode fold ratio residual failed")
    require(minimum_three_mode_sign_changes == 2, "three-mode fold ratio lost alternation")
    require(accepted_trials >= 450, "too few separated-rate constructive checks were accepted")
    return {
        "requested_trials": trials,
        "accepted_constructive_checks": accepted_trials,
        "minimum_two_mode_determinant": float(minimum_two_mode_determinant),
        "maximum_three_mode_double_zero_residual": float(maximum_three_mode_residual),
        "minimum_three_mode_sign_changes": minimum_three_mode_sign_changes,
    }


def main() -> None:
    payload = finite_mode_checks()
    lines = [
        "Two- and three-mode fold algebra on random separated rate sets: PASS",
        f"Rate sets accepted (rate gaps >= 0.1): {payload['accepted_constructive_checks']}/"
        f"{payload['requested_trials']}",
        f"Minimum |det| of the two-mode fold matrix: {payload['minimum_two_mode_determinant']:.3e}",
        f"Three-mode double-zero max residual: {payload['maximum_three_mode_double_zero_residual']:.3e}",
        f"Minimum number of sign changes of the three-mode ratio: "
        f"{payload['minimum_three_mode_sign_changes']}",
    ]
    print("\n".join(lines))
    (TABLES / "finite_mode_algebra.txt").write_text("\n".join(lines) + "\n")
    (TABLES / "finite_mode_algebra.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
