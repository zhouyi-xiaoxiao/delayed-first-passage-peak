#!/usr/bin/env python3
"""Fig. 6: Brownian dynamics with a gate of finite width, release away from the sink (plot only).

Plot-only rendering of the stored data written by dpma_brownian_fold.py --simulate; the
Brownian-dynamics runs and the exact point-sink density are not recomputed here
(dpma_brownian_fold.py without options recomputes the exact curves and compares them with the
stored data).  Use --data DIR to plot data files regenerated in another directory.

Data: data/dpma_brownian_fold_data.npz (+ .json sidecar), keys used
  b_2p5_histogram_edges, b_2p5_histogram_density, b_2p5_exact_density   (panel a, b = 2.5)
  b_3p8_histogram_edges, b_3p8_histogram_density, b_3p8_exact_density   (panel b, b = 3.8)
  exact_tau; parameters theta, xi, ell, dt, histogram_denominator_trials, seeds, b, b_c.

Peak/valley markers: discrete local extrema of the stored exact density (no new evaluation).

Outputs: outputs/figures/fig6_brownian_dynamics.{pdf,png},
         outputs/tables/fig6_brownian_dynamics.txt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import StepPatch, Patch
from matplotlib.ticker import LogLocator, NullFormatter, MultipleLocator

import figstyle as S
from paths import DATA, FIGURES as FIG, TABLES as TAB, rel

NPZ = DATA / "dpma_brownian_fold_data.npz"
SIDECAR = DATA / "dpma_brownian_fold_data.json"
LOG = []


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


def local_extrema(y):
    i = np.arange(1, len(y) - 1)
    mx = i[(y[1:-1] > y[:-2]) & (y[1:-1] > y[2:])]
    mn = i[(y[1:-1] < y[:-2]) & (y[1:-1] < y[2:])]
    return mx, mn


def main():
    global NPZ, SIDECAR
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=None, help="directory with the data files")
    args = ap.parse_args()
    if args.data is not None:
        NPZ, SIDECAR = args.data / NPZ.name, args.data / SIDECAR.name
    S.apply_style()
    z = np.load(NPZ)
    side = json.load(open(SIDECAR))
    say("Fig. 6: Brownian dynamics (plot only; stored data, no simulation or exact solve rerun)")
    say(f"input {rel(NPZ)} sha256 {S.sha256(NPZ)}")
    say(f"input {rel(SIDECAR)} sha256 {S.sha256(SIDECAR)}")

    # ---- stored parameters: b = 2.5, 3.8; xi = 0.44; theta = 1/2; ell = 0.02;
    #      6x10^4 trajectories; point-sink upper fold b_+ = 3.1366
    theta, xi, ell = float(z["theta"][0]), float(z["xi"][0]), float(z["ell"][0])
    nwalk = int(z["histogram_denominator_trials"][0])
    bvals = [float(b) for b in z["b"]]
    assert (theta, xi, ell, nwalk, bvals) == (0.5, 0.44, 0.02, 60_000, [2.5, 3.8])
    assert (side["theta"], side["xi"], side["ell"], side["nwalk_per_condition"], side["b"]) == \
        (theta, xi, ell, nwalk, bvals)
    b_plus = float(z["b_c"][0])
    assert abs(b_plus - 3.1366) < 5e-5
    say(f"theta={theta}, xi={xi}, gate ell={ell}, walkers per b={nwalk}, dt={float(z['dt'][0]):g}, "
        f"seeds={[int(s) for s in z['seeds']]}; point-sink upper fold b_+(0.44) = {b_plus:.5f}")

    tau = z["exact_tau"]
    fig, axes = plt.subplots(1, 2, figsize=(S.FULL_WIDTH, 2.45))
    C_EX = S.OI["vermillion"]
    plotted = []
    for ax, b, letter, title in ((axes[0], 2.5, "a", "two interior peaks"),
                                 (axes[1], 3.8, "b", "one interior peak")):
        key = f"b_{str(b).replace('.', 'p')}"
        edges, dens, ex = z[f"{key}_histogram_edges"], z[f"{key}_histogram_density"], z[f"{key}_exact_density"]
        assert np.allclose(dens * np.diff(edges), z[f"{key}_histogram_counts"] / nwalk, rtol=0, atol=1e-15)
        st = ax.stairs(dens, edges, fill=True, facecolor=S.HIST_FACE, edgecolor=S.HIST_EDGE,
                       lw=0.5, zorder=2)
        ln, = ax.plot(tau, ex, "-", color=C_EX, lw=1.4, zorder=3)
        mx, mn = local_extrema(ex)
        assert len(mx) == (2 if b < b_plus else 1) and len(mn) == (1 if b < b_plus else 0), (b, mx, mn)
        ax.plot(tau[mx], ex[mx], "o", ms=4.2, mfc=S.OI["blue"], mec=S.OI["blue"], zorder=5)
        ax.plot(tau[mn], ex[mn], "o", ms=4.2, mfc="white", mec=S.OI["blue"], mew=1.0, zorder=5)
        say(f"b={b}: exact density maxima at tau = {', '.join(f'{tau[i]:.4f}' for i in mx)}"
            + (f"; minimum at tau = {', '.join(f'{tau[i]:.4f}' for i in mn)}" if len(mn) else "")
            + f"; histogram mass in window = {dens @ np.diff(edges):.6f}")
        plotted.append((st, ln, edges, dens, ex))
        ax.set_xscale("log")
        ax.set_xlim(edges[0], edges[-1])
        ax.set_ylim(0.0, 1.17 * max(dens.max(), ex.max()))
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=6))
        ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=20))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_major_locator(MultipleLocator(2 if b < 3 else 4))
        ax.yaxis.set_minor_locator(MultipleLocator(1))
        ax.tick_params(which="both", top=True, right=True)
        ax.set_xlabel(r"time $\tau$")
        S.panel_title(ax, letter, rf"$b={b:g}$: {title}")
        ax.legend([Patch(facecolor=S.HIST_FACE, edgecolor=S.HIST_EDGE, lw=0.5), ln],
                  [rf"Brownian dynamics (gate $\ell={ell:g}$)", "exact point-sink density"],
                  loc="upper right", frameon=False, handlelength=1.8, borderaxespad=0.4)
        ax.text(0.97, 0.70, rf"$\xi={xi:g}$, $\theta=1/2$", transform=ax.transAxes, ha="right",
                va="top", fontsize=S.SMALL_PT, color=S.MUTED)
    axes[0].set_ylabel("absorption-time density")

    # peak names (terminology: release away from the sink -> early peak, valley, late peak)
    lead = dict(arrowstyle="-", color=S.MUTED, lw=0.6, shrinkA=1, shrinkB=3.5)
    ann = dict(fontsize=S.NOTE_PT, color=S.INK, ha="center")
    _, _, _, _, ex_a = plotted[0]
    mx, mn = local_extrema(ex_a)
    axes[0].annotate("early peak", xy=(tau[mx[0]], ex_a[mx[0]]), xytext=(tau[mx[0]], 10.6),
                     va="center", arrowprops=lead, **ann)
    axes[0].annotate("valley", xy=(tau[mn[0]], ex_a[mn[0]]), xytext=(tau[mn[0]] * 0.8, 1.4),
                     va="center", arrowprops=lead, **ann)
    axes[0].annotate("late peak", xy=(tau[mx[1]], ex_a[mx[1]]), xytext=(tau[mx[1]], 6.75),
                     va="center", arrowprops=lead, **ann)
    _, _, _, _, ex_b = plotted[1]
    mxb, _ = local_extrema(ex_b)
    axes[1].annotate("early peak", xy=(tau[mxb[0]], ex_b[mxb[0]]), xytext=(tau[mxb[0]], 15.95),
                     va="center", arrowprops=lead, **ann)
    i_sh = int(np.argmin(np.abs(tau - 0.035)))
    axes[1].annotate("no late peak\n(shoulder)", xy=(tau[i_sh], ex_b[i_sh]),
                     xytext=(0.115, 8.7), va="center", arrowprops=lead, linespacing=1.1, **ann)
    fig.subplots_adjust(left=0.065, right=0.99, bottom=0.165, top=0.915, wspace=0.14)

    # ---- verification: plotted arrays are the stored arrays (bit-for-bit)
    n = 0
    for st, ln, edges, dens, ex in plotted:
        vals, e, _ = st.get_data()
        assert np.array_equal(vals, dens) and np.array_equal(e, edges)
        assert np.array_equal(ln.get_xdata(), tau) and np.array_equal(ln.get_ydata(), ex)
        n += 4
    say(f"verified: {n} plotted arrays equal the stored npz arrays bit-for-bit")

    # figures from regenerated data are written next to the reference build, not over it
    info = S.save(fig, FIG, "fig6_brownian_dynamics" + ("_regenerated" if args.data is not None else ""))
    say(f"wrote {rel(info['pdf'])} and .png ({S.PNG_DPI} dpi); "
        f"size {info['width_in']:.2f} x {info['height_in']:.2f} in; smallest font {info['min_font_pt']:g} pt")
    for row in info["fonts"]:
        say("  pdffonts: " + " ".join(row.split()))
    (TAB / ("fig6_brownian_dynamics" + ("_regenerated" if args.data is not None else "") + ".txt")).write_text("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
