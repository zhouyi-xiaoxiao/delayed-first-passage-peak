#!/usr/bin/env python3
"""Brownian-dynamics data of Fig. 6: finite gate, release away from the sink.

A particle diffuses on [0,1] with D = 1/2, is absorbed at the ends, and is captured at rate b/ell
while inside a gate of width ell = 0.02 centred at theta = 1/2; release at xi = 0.44.  This tends
to the point-sink problem as ell -> 0.  Euler-Maruyama histograms (6x10^4 trajectories per value
of b, time step 1.5e-5, seeds 7 and 8, normalized by all trajectories) are compared with the exact
point-sink density at b = 2.5 (two interior peaks) and b = 3.8 (above the upper fold line).

Modes
  default      recompute the exact point-sink curves and b_+(0.44) and compare them with the
               stored data file (data/dpma_brownian_fold_data.npz).  Light.
  --simulate   rerun the Brownian dynamics with the recorded seeds and write new data files to
               outputs/data/ (a few minutes).

Logs: outputs/tables/dpma_brownian_fold_check.txt (default) or outputs/tables/dpma_brownian_fold.txt.
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np

from dpma_saddle_node_bc_theta import bc, phi_vals
from paths import DATA, GENERATED, TABLES

THETA, XI, ELL, DT, NWALK, TAU_MAX = 0.5, 0.44, 0.02, 1.5e-5, 60_000, 1.4
CASES = ((2.5, 7), (3.8, 8))              # (b, seed)
OUT: list[str] = []


def say(s=""):
    print(s, flush=True)
    OUT.append(s)


def brownian_fpt(theta, xi, b, ell=ELL, dt=DT, nwalk=NWALK, tau_max=TAU_MAX, seed=7):
    """Euler-Maruyama; return times of absorption at an end or capture in the gate."""
    rng = np.random.default_rng(seed)
    x = np.full(nwalk, xi)
    alive = np.ones(nwalk, dtype=bool)
    tabs = np.zeros(nwalk)
    glo, ghi = theta - ell / 2, theta + ell / 2
    kill_p = (b / ell) * dt
    sig = math.sqrt(dt)  # sqrt(2 D dt), D = 1/2
    nsteps = int(tau_max / dt)
    for n in range(1, nsteps + 1):
        idx = np.where(alive)[0]
        if idx.size == 0:
            break
        xi_ = x[idx] + sig * rng.standard_normal(idx.size)
        ingate = (xi_ >= glo) & (xi_ <= ghi)
        killed = ingate & (rng.random(idx.size) < kill_p)
        absorbed_end = (xi_ <= 0.0) | (xi_ >= 1.0)
        done = killed | absorbed_end
        d = idx[done]
        if d.size:
            alive[d] = False
            tabs[d] = n * dt
        still = idx[~done]
        x[still] = xi_[~done]
    return tabs[~alive]


def exact_part():
    taus_exact = np.exp(np.linspace(math.log(1.3e-3), math.log(0.6), 1200))
    bplus = bc(XI, THETA, lo=0.3, hi=6.0)
    curves = {b: phi_vals(taus_exact, XI, THETA, b) for b, _ in CASES}
    return taus_exact, bplus, curves


def check_stored():
    say("Exact point-sink curves vs the stored data of Fig. 6 (no Brownian-dynamics rerun)")
    taus, bplus, curves = exact_part()
    z = np.load(DATA / "dpma_brownian_fold_data.npz")
    say(f"  upper fold line at xi=0.44 (bisection on the valley/peak count): b_+ = {bplus:.10f}; "
        f"stored {float(z['b_c'][0]):.10f}")
    worst = float(np.max(np.abs(taus - z["exact_tau"]) / z["exact_tau"]))
    for b, _ in CASES:
        key = f"b_{str(b).replace('.', 'p')}"
        stored = z[f"{key}_exact_density"]
        rel = float(np.max(np.abs(curves[b] - stored) / np.abs(stored)))
        worst = max(worst, rel)
        nmax = int(np.sum((curves[b][1:-1] > curves[b][:-2]) & (curves[b][1:-1] > curves[b][2:])))
        cnt = z[f"{key}_histogram_counts"]
        say(f"  b={b}: exact density max relative difference recomputed/stored = {rel:.1e}; "
            f"interior maxima = {nmax}; stored histogram: {int(cnt.sum())} of {NWALK} trajectories "
            f"in the displayed window")
    if worst > 1e-9 or abs(bplus - float(z["b_c"][0])) > 1e-9:
        raise RuntimeError("exact Brownian-figure curves differ from the stored data")
    (TABLES / "dpma_brownian_fold_check.txt").write_text("\n".join(OUT) + "\n")


def simulate():
    taus, bplus, curves = exact_part()
    say(f"point-sink b_+(theta=0.5; xi=0.44) = {bplus:.4f}; gate width ell={ELL}")
    saved = {}
    stored = np.load(DATA / "dpma_brownian_fold_data.npz")
    for b, seed in CASES:
        say(f"\nBrownian dynamics at b={b} (seed {seed}):")
        fpt = brownian_fpt(THETA, XI, b, seed=seed)
        say(f"  absorbed {len(fpt)} trajectories; median tau={np.median(fpt):.4f}")
        bins = np.logspace(math.log10(1.3e-3), math.log10(0.6), 60)
        counts, edges = np.histogram(fpt, bins=bins)
        density = counts / (NWALK * np.diff(edges))      # normalized by all trajectories
        say(f"  probability inside displayed bins={counts.sum() / NWALK:.6f}")
        key = str(b).replace(".", "p")
        same = np.array_equal(counts, stored[f"b_{key}_histogram_counts"])
        say(f"  histogram counts identical to the stored run: {same}")
        saved[f"b_{key}_histogram_edges"] = edges
        saved[f"b_{key}_histogram_counts"] = counts
        saved[f"b_{key}_histogram_density"] = density
        saved[f"b_{key}_exact_density"] = curves[b]
    np.savez_compressed(
        GENERATED / "dpma_brownian_fold_data.npz",
        exact_tau=taus, D=np.array([0.5]), dt=np.array([DT]), tau_max=np.array([TAU_MAX]),
        histogram_denominator_trials=np.array([NWALK]), theta=np.array([THETA]), xi=np.array([XI]),
        ell=np.array([ELL]), b_c=np.array([bplus]), b=np.array([2.5, 3.8]), seeds=np.array([7, 8]),
        **saved,
    )
    (GENERATED / "dpma_brownian_fold_data.json").write_text(json.dumps({
        "description": "Processed Brownian histogram counts/densities and exact plotted curves; no raw trajectories.",
        "D": 0.5, "theta": THETA, "xi": XI, "ell": ELL, "dt": DT, "nwalk_per_condition": NWALK,
        "simulation_tau_max": TAU_MAX, "b": [2.5, 3.8], "seeds": [7, 8], "b_c_delta_sink": bplus,
    }, indent=2, sort_keys=True) + "\n")
    say("\nwrote outputs/data/dpma_brownian_fold_data.npz and .json")
    (TABLES / "dpma_brownian_fold.txt").write_text("\n".join(OUT) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--simulate", action="store_true", help="rerun the Brownian dynamics (slow)")
    args = ap.parse_args()
    simulate() if args.simulate else check_stored()


if __name__ == "__main__":
    main()
