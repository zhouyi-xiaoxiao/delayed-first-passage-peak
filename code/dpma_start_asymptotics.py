#!/usr/bin/env python3
"""Quartic robustness of the antipodal fold to an off-gate release.

For ``theta=1/2`` and ``delta=|xi-1/2|``, the sink-coupled amplitude obeys

    G_j(delta)/G_j(0) = cos(2 w_j delta) + b sin(2 w_j delta)/(2 w_j).

Its logarithm is a common amplitude/time shift through cubic order; the first
shape-changing term is ``-mu_j^2 delta^4/3``.  Consequently the morphology
upper threshold has no linear, quadratic, or cubic displacement:

    b_+(delta) = b_c + [S3/(3 partial_b S1)] delta^4 + O(delta^5).

This script evaluates the coefficient at high precision and checks it against
direct off-gate fold solves.
"""
from __future__ import annotations

import csv

from mpmath import mp

from dpma_saddle_node_bc_theta import solve_fold
from paths import TABLES


mp.dps = 40
BC = mp.mpf("3.0764323603659522")
TAUC = mp.mpf("0.03836305186128459")
NMODES = 60


def roots_antipodal(b: mp.mpf) -> list[mp.mpf]:
    values = []
    for j in range(1, NMODES + 1):
        lo = (2 * j - 1) * mp.pi / 2 + mp.mpf("1e-30")
        hi = j * mp.pi - mp.mpf("1e-30")
        secular = lambda w: 2 * w * mp.cos(w) + b * mp.sin(w)
        f_lo = secular(lo)
        for _ in range(130):
            mid = (lo + hi) / 2
            if f_lo * secular(mid) <= 0:
                hi = mid
            else:
                lo, f_lo = mid, secular(mid)
        values.append((lo + hi) / 2)
    return values


def moment(order: int, tau: mp.mpf, b: mp.mpf) -> mp.mpf:
    total = mp.mpf("0")
    for w in roots_antipodal(b):
        mu = 2 * w * w
        amplitude = 4 * w * (1 - mp.cos(w)) / (
            mp.sin(w) * (1 + b * (b + 2) / (4 * w * w))
        )
        total += amplitude * mu**order * mp.exp(-mu * tau)
    return total


def main() -> None:
    h = mp.mpf("1e-10")
    s3 = moment(3, TAUC, BC)
    s1b = (moment(1, TAUC, BC + h) - moment(1, TAUC, BC - h)) / (2 * h)
    quartic = s3 / (3 * s1b)
    cubic_time = 2 * BC / 3
    if abs(quartic - mp.mpf("4352.6561855")) > mp.mpf("0.01"):
        raise RuntimeError(f"quartic coefficient drifted: {quartic}")

    lines = [
        "Antipodal start-offset asymptotics",
        f"  b_c={mp.nstr(BC, 16)}, tau_c={mp.nstr(TAUC, 16)}",
        f"  S3/(3 partial_b S1) = {mp.nstr(quartic, 14)}",
        f"  2 b_c/3 = {mp.nstr(cubic_time, 14)}",
        "  b_+(delta)=b_c+4352.6561855 delta^4+O(delta^5)",
        "  tau_c(delta)=tau_c-delta^2+(2b_c/3)delta^3+O(delta^4)",
        "",
        "  delta      b_+(direct)   (b_+(delta)-b_c)/delta^4   tau-shift/delta^3",
    ]
    rows = []
    for delta in (0.04, 0.03, 0.02, 0.01, 0.005, 0.002, 0.001):
        b_direct, tau_direct = solve_fold(
            0.5, (float(BC), float(TAUC)), xi=0.5 - delta, tol=1e-13, max_iter=20
        )
        b_ratio = (b_direct - float(BC)) / delta**4
        tau_ratio = (tau_direct - float(TAUC) + delta**2) / delta**3
        rows.append((delta, b_direct, tau_direct, b_ratio, tau_ratio))
        lines.append(f"  {delta:7.4f}  {b_direct:13.10f}  {b_ratio:27.6f}  {tau_ratio:19.6f}")
    # The finite-delta corrections are O(delta); the smallest offset must already approach
    # the analytic coefficient and cubic time shift within 5%.
    if abs(rows[-1][3] / float(quartic) - 1.0) > 0.05 or abs(rows[-1][4] / float(cubic_time) - 1.0) > 0.05:
        raise RuntimeError("direct start-offset folds do not approach the analytic asymptotics")
    lines.append("  => the displacement of the upper fold line begins at quartic order in xi - 1/2.")

    print("\n".join(lines), flush=True)
    (TABLES / "dpma_start_asymptotics.txt").write_text("\n".join(lines) + "\n")
    with (TABLES / "dpma_start_asymptotics.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["delta", "b_c_direct", "tau_c_direct", "quartic_ratio", "cubic_time_ratio"])
        writer.writerows(rows)


if __name__ == "__main__":
    main()
