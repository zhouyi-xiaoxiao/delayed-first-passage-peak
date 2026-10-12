#!/usr/bin/env python3
"""Brownian-dynamics sensitivity to the gate width and the time step (Sec. V and App. D).

The Brownian dynamics of Fig. 6 (dpma_brownian_fold.py) uses a gate of width ell=0.02 and the
Euler-Maruyama step 1.5e-5.  This script sweeps ell in {0.04, 0.02, 0.01} at dt=1.5e-5 and
dt in {3e-5, 1.5e-5, 7.5e-6} at ell=0.02 (theta=1/2, xi=0.44, b=2.5, capture rate b/ell inside
the gate; 3 x 6x10^4 trajectories per cell, fixed seeds) and reports the window ratio
R (+/-20% windows around tau_p=0.0462 and tau_v=0.0253) and the late-peak location, against
the exact point-sink values.  About two minutes; run by reproduce_all.py --full.
Writes outputs/tables/dpma_bd_convergence.txt
"""
from __future__ import annotations
import math
import numpy as np
from dpma_saddle_node_bc_theta import phi_vals
from dpma_window_observable import exact_window_ratio, sample_window_ratio
from paths import TABLES as TAB
OUT = []
def say(s=""):
    print(s, flush=True); OUT.append(s)

theta, xi, b = 0.5, 0.44, 2.5
NW = 60_000
TAU_MAX = 0.6

# exact delta-sink references on the same protocol
tex = np.exp(np.linspace(math.log(2e-3), math.log(0.5), 4000))
ye = phi_vals(tex, xi, theta, b)
d1 = np.diff(ye)
mx = np.where((d1[:-1] > 0) & (d1[1:] <= 0))[0] + 1
mn = np.where((d1[:-1] < 0) & (d1[1:] >= 0))[0] + 1
tau_v_exact, tau_p_exact = float(tex[mn[0]]), float(tex[mx[-1]])
# The measurement protocol publishes these rounded reference centres and keeps the windows fixed.
# Use exactly the same observable here; grid-located centres define a slightly different ratio.
tau_v, tau_p = 0.0253, 0.0462
pwin, vwin = (0.8 * tau_p, 1.2 * tau_p), (0.8 * tau_v, 1.2 * tau_v)
R_exact, p_exact, v_exact = exact_window_ratio(
    lambda tau: phi_vals(tau, xi, theta, b), pwin, vwin
)
if not abs(R_exact - 1.03586221) < 2e-7:
    raise RuntimeError(f"fixed-window delta-sink comparator drifted: R={R_exact}")
say(f"Exact delta-sink references (theta=1/2, xi=0.44, b=2.5): "
    f"located tau_v={tau_v_exact:.5f}, tau_p={tau_p_exact:.5f}; "
    f"fixed window centres=({tau_v:.4f},{tau_p:.4f}); "
    f"R_window={R_exact:.6f} (P_peak={p_exact:.6f}, P_valley={v_exact:.6f})")

def bd_fpt(ell, dt, seed):
    rng = np.random.default_rng(seed)
    x = np.full(NW, xi); alive = np.ones(NW, bool); tabs = np.zeros(NW)
    glo, ghi = theta - ell / 2, theta + ell / 2
    kp = (b / ell) * dt; sig = math.sqrt(2 * 0.5 * dt)
    for n in range(1, int(TAU_MAX / dt) + 1):
        idx = np.where(alive)[0]
        if idx.size == 0: break
        xn = x[idx] + sig * rng.standard_normal(idx.size)
        done = ((xn >= glo) & (xn <= ghi) & (rng.random(idx.size) < kp)) | (xn <= 0) | (xn >= 1)
        d = idx[done]
        if d.size: alive[d] = False; tabs[d] = n * dt
        st = idx[~done]; x[st] = xn[~done]
    return tabs[~alive]

def metrics(fpt):
    ratio = sample_window_ratio(fpt, pwin, vwin)
    # late-peak location: smoothed log-binned histogram, window BEYOND the valley (0.03-0.09)
    # so the capture-peak tail cannot masquerade as the late peak.
    bins = np.exp(np.linspace(math.log(0.03), math.log(0.09), 26))
    cnt, edges = np.histogram(fpt, bins=bins)
    dens = cnt / np.diff(edges)
    k = np.ones(5) / 5.0
    sm = np.convolve(dens, k, mode="same")
    j = int(np.argmax(sm[2:-2])) + 2                    # avoid edge artifacts of the smoothing
    tpk = float(np.sqrt(edges[j] * edges[j + 1]))
    return ratio, tpk

NSEED = 3
LATE_BIN_LOG_WIDTH = math.log(0.09 / 0.03) / 25.0
say(f"\nBD convergence sweeps ({NSEED} x {NW} walkers per cell, tau_max={TAU_MAX}):")
say(f"  {'ell':>6} {'dt':>9} {'R_BD (mean+/-sem)':>18} {'R_exact':>8} {'tau_peak':>9} "
    f"{'|R_BD-R_exact|/sem':>18} {'peak offset (bins)':>18}   (exact tau_p {tau_p_exact:.4f})")
sweep_rows = []
for ell, dt in ((0.04, 1.5e-5), (0.02, 1.5e-5), (0.01, 1.5e-5), (0.02, 3.0e-5), (0.02, 7.5e-6)):
    Rs, tps = [], []
    for sd in range(NSEED):
        R, tpk = metrics(bd_fpt(ell, dt, seed=7000 + int(1e6 * ell) + int(1e7 * dt) + 131 * sd))
        Rs.append(R); tps.append(tpk)
    Rs = np.array(Rs); tps = np.array(tps)
    mean_R = float(np.mean(Rs))
    sem_R = float(np.std(Rs, ddof=1) / math.sqrt(NSEED))
    mean_peak = float(np.mean(tps))
    z_score = abs(mean_R - R_exact) / sem_R
    peak_bin_offset = abs(math.log(mean_peak / tau_p_exact)) / LATE_BIN_LOG_WIDTH
    sweep_rows.append((ell, dt, z_score, peak_bin_offset))
    say(f"  {ell:>6.3f} {dt:>9.1e} {mean_R:>10.4f} +/- {sem_R:.4f} "
        f"{R_exact:>8.4f} {mean_peak:>9.4f} {z_score:>18.2f} {peak_bin_offset:>18.2f}")
if max(row[2] for row in sweep_rows) > 1.5:
    raise RuntimeError(f"BD window-ratio discrepancy exceeded 1.5 SEM: {sweep_rows}")
if max(row[3] for row in sweep_rows) > 2.0:
    raise RuntimeError(f"BD late-peak displacement exceeded two histogram bins: {sweep_rows}")
say("")
say(f"largest late-peak offset {max(row[3] for row in sweep_rows):.2f} bins (bin width "
    f"{LATE_BIN_LOG_WIDTH:.4f} in log tau); largest ratio discrepancy {max(row[2] for row in sweep_rows):.2f} SEM")
say("The late-peak location stays within two histogram bins of the exact tau_p in every cell. The")
say("ratio observable is consistent with the same-window delta-sink comparator at the stated")
say("three-seed sampling resolution. Gate-width and time-step changes leave the fold morphology")
say("unchanged; this is a sensitivity check, while the deterministic finite-gate calculation")
say("separately quantifies the ell -> 0 systematic limit.")
say("Regression gates: maximum ratio discrepancy <= 1.5 SEM; every cell's mean peak <= 2 bins.")
(TAB / "dpma_bd_convergence.txt").write_text("\n".join(OUT) + "\n")
