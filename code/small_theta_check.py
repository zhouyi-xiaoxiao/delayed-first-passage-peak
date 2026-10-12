#!/usr/bin/env python3
"""Direct check of the hatched (near-end) part of Fig. 2(a) and App. D, "Threshold curve".

For theta = 0.10 and 0.15 (outside the computed continuation theta in [0.2, 0.8]) the density
Phi(tau) for release at the sink (xi = theta) is evaluated from the spectral solution of
dpma_saddle_node_bc_theta.py (roots, G_affected, G_node) with a deeper modal cutoff (wmax = 150
and 250, convergence check), and its interior extrema are counted at b = f * B*/theta for f below
and above 1.  Expected: a valley and a delayed peak for f < 1 and none for f > 1.
Writes outputs/tables/small_theta_check.txt.
"""
from __future__ import annotations

import math

import numpy as np

from dpma_saddle_node_bc_theta import roots, G_affected, G_node
from paths import TABLES

B_STAR, C_STAR = 0.7890262, 0.1579221


def phi(taus, xi, th, b, wmax):
    ws = roots(th, b, wmax=wmax, step=0.0012)
    amps, rates = [], []
    for w in ws:
        rates.append(2 * w * w)
        if abs(math.sin(2 * th * w)) < 1e-3:
            amps.append(G_node(int(round(2 * w / math.pi)), xi))
        else:
            amps.append(G_affected(w, xi, th, b))
    amps, rates = np.array(amps), np.array(rates)
    with np.errstate(all="ignore"):
        return np.exp(-np.outer(taus, rates)) @ amps, len(ws)


def extrema(th, b, wmax):
    tau_c = C_STAR * th * th
    taus = np.exp(np.linspace(math.log(0.2 * tau_c), math.log(2.0), 12000))
    values, n_modes = phi(taus, th, th, b, wmax)
    d = np.diff(values)
    maxima = np.where((d[:-1] > 0) & (d[1:] <= 0))[0] + 1
    minima = np.where((d[:-1] < 0) & (d[1:] >= 0))[0] + 1
    return taus[minima], taus[maxima], n_modes


def main():
    lines = ["theta  f=b*theta/B*  b  wmax  modes  tau_valley  tau_peak  pair?"]
    ok = True
    for th in (0.10, 0.15):
        for f in (0.3, 0.9, 0.98, 1.02, 1.1, 1.5):
            b = f * B_STAR / th
            for wmax in (150, 250):
                mn, mx, nm = extrema(th, b, wmax)
                pair = len(mn) > 0 and len(mx) > 0 and mn[0] < mx[-1]
                ok &= pair == (f < 1)
                lines.append(
                    f"{th:.2f}  {f:.2f}  {b:.4f}  {wmax}  {nm}  "
                    f"{(f'{mn[0]:.6f}' if len(mn) else '-')}  "
                    f"{(f'{mx[-1]:.6f}' if len(mx) else '-')}  {pair}"
                )
    lines.append(f"result: {'PASS' if ok else 'FAIL'} (pair exists iff b < B*/theta at all tested points)")
    (TABLES / "small_theta_check.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
