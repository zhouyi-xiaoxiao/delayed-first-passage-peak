#!/usr/bin/env python3
"""Fig. 5: lattice checks and absorption routes (plot only).

Plot-only rendering of the stored data written by dpma_channel_mc.py --simulate; the Monte Carlo
and the exact lattice densities are not recomputed here (dpma_channel_mc.py without options
recomputes the exact curves and compares them with the stored data).  Use --data DIR to plot
data files regenerated in another directory (e.g. outputs/data).

Data: data/dpma_channel_mc_data.npz (+ .json sidecar), keys used
  (a) histogram_edges, histogram_density            MC, 4x10^5 walkers, seed 12345
      exact_tau, exact_total_density                 exact lattice density
  (b) exact_tau, exact_total_density, exact_shortcut_density, exact_diffusive_density
  (c) nonantipodal_tau, nonantipodal_density[0..2], nonantipodal_b
  parameters: q, N_antipodal, u_antipodal, r_antipodal, b_antipodal, N_nonantipodal,
              u_nonantipodal, r_nonantipodal, histogram_denominator_trials

Lattice: lazy ring of N = 100 sites, target site 0, one-way shortcut u -> 0 with probability
lambda per step; continuum variables tau = q t / N^2, theta = u/N, xi = r/N, b = lambda N/q.
"Capture via shortcut" = absorption through the shortcut; "arrival at the ends" = absorption by
a nearest-neighbour hop onto site 0 (the two ends of the unrolled interval).

Outputs: outputs/figures/fig5_lattice_checks.{pdf,png},
         outputs/tables/fig5_lattice_checks.txt
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
from matplotlib.ticker import LogLocator, NullFormatter

import figstyle as S
from paths import DATA, FIGURES as FIG, TABLES as TAB, rel

NPZ = DATA / "dpma_channel_mc_data.npz"
SIDECAR = DATA / "dpma_channel_mc_data.json"
XLIM = (3e-4, 3.0)
YLIM = (3e-2, 400.0)
LOG = []


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


def style_axes(ax):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(LogLocator(base=10, numticks=8))
        axis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=30))
        axis.set_minor_formatter(NullFormatter())
    ax.tick_params(which="both", top=True, right=True)
    ax.set_xlabel(r"time $\tau=qt/N^{2}$")


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
    say("Fig. 5: lattice checks (plot only; stored data, no Monte Carlo or exact solve rerun)")
    say(f"input {rel(NPZ)} sha256 {S.sha256(NPZ)}")
    say(f"input {rel(SIDECAR)} sha256 {S.sha256(SIDECAR)}")

    N, u, r = int(z["N_antipodal"][0]), int(z["u_antipodal"][0]), int(z["r_antipodal"][0])
    b_a = float(z["b_antipodal"][0])
    ntot = int(z["histogram_denominator_trials"][0])
    Nc, uc, rc = int(z["N_nonantipodal"][0]), int(z["u_nonantipodal"][0]), int(z["r_nonantipodal"][0])
    bs = [float(b) for b in z["nonantipodal_b"]]
    q = float(z["q"][0])
    # parameters agree with the sidecar written by the data script
    a = side["antipodal"]
    assert (a["N"], a["u"], a["r"], a["b"], a["nwalk"], a["seed"]) == (N, u, r, b_a, ntot, 12345)
    assert side["nonantipodal"]["b"] == bs and (Nc, uc, rc) == (100, 38, 38)
    assert abs(q - 2.0 / 3.0) < 1e-15 and ntot == 400_000
    # histogram normalised by ALL trials
    assert np.allclose(z["histogram_density"] * np.diff(z["histogram_edges"]),
                       z["histogram_counts"] / ntot, rtol=0, atol=1e-15)
    # exact channel decomposition: total = shortcut + ends
    res = np.max(np.abs(z["exact_total_density"] - z["exact_shortcut_density"]
                        - z["exact_diffusive_density"]))
    say(f"N={N}, u={u}, r={r} (theta=xi={u / N:g}), b={b_a}, q={q:.6f}; MC trials {ntot}; "
        f"max|total - shortcut - ends| = {res:.1e}")
    say(f"(c): N={Nc}, u=r={uc} (theta=xi={uc / Nc:g}), b = {bs}")

    fig, axes = plt.subplots(1, 3, figsize=(S.FULL_WIDTH, 2.45), sharey=True)
    axA, axB, axC = axes

    # (a) MC vs exact
    st = axA.stairs(z["histogram_density"], z["histogram_edges"], fill=True,
                    facecolor=S.HIST_FACE, edgecolor=S.HIST_EDGE, lw=0.5, zorder=2)
    lA, = axA.plot(z["exact_tau"], z["exact_total_density"], "-", color=S.OI["vermillion"],
                   lw=1.3, zorder=3)
    axA.legend([Patch(facecolor=S.HIST_FACE, edgecolor=S.HIST_EDGE, lw=0.5), lA],
               [r"MC, $4\times10^{5}$ walkers", "exact"], loc="upper right", frameon=False,
               handlelength=1.8, borderaxespad=0.4)
    S.panel_title(axA, "a", "lattice MC vs exact")

    # (b) absorption routes
    lB0, = axB.plot(z["exact_tau"], z["exact_total_density"], "-", color=S.INK, lw=1.4, zorder=3)
    lB1, = axB.plot(z["exact_tau"], z["exact_shortcut_density"], "--", color=S.OI["green"],
                    lw=1.4, zorder=4)
    lB2, = axB.plot(z["exact_tau"], z["exact_diffusive_density"], "-.", color=S.OI["vermillion"],
                    lw=1.4, zorder=4)
    axB.legend([lB0, lB1, lB2], ["total", "capture via shortcut", "arrival at the ends"],
               loc="upper right", frameon=False, handlelength=2.2, borderaxespad=0.4)
    S.panel_title(axB, "b", "absorption routes")

    # (c) sink at theta = 0.38, release at the sink
    stylesC = [(S.OI["blue"], "-", 1.4), (S.INK, "--", 1.4), (S.OI["vermillion"], ":", 1.7)]
    linesC = []
    for i, (b, (col, ls, lw)) in enumerate(zip(bs, stylesC)):
        ln, = axC.plot(z["nonantipodal_tau"], z["nonantipodal_density"][i], ls=ls, color=col,
                       lw=lw, zorder=3)
        linesC.append(ln)
    axC.legend(linesC, [rf"$b={b:g}$" for b in bs], loc="upper right", frameon=False,
               handlelength=2.2, borderaxespad=0.4)
    S.panel_title(axC, "c", rf"sink at $\theta={uc / Nc:.2f}$")

    for ax in axes:
        style_axes(ax)
    axA.set_ylabel("absorption-time density")
    for ax in (axB, axC):
        ax.tick_params(labelleft=False)
    # parameter notes where no curve runs: (a) under the legend; (b), (c) lower left
    for ax, txt, xy, ha, va in (
            (axA, f"$N={N}$\n" + rf"$\theta=\xi=1/2$" + f"\n$b={b_a:g}$", (0.965, 0.735), "right", "top"),
            (axB, f"$N={N}$\n" + rf"$\theta=\xi=1/2$" + f"\n$b={b_a:g}$", (0.04, 0.04), "left", "bottom"),
            (axC, f"$N={Nc}$\nrelease at the sink", (0.04, 0.04), "left", "bottom")):
        ax.text(*xy, txt, transform=ax.transAxes, ha=ha, va=va, fontsize=S.SMALL_PT,
                color=S.MUTED, linespacing=1.25)
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.165, top=0.915, wspace=0.07)

    # ---- verification: plotted arrays are the stored arrays (bit-for-bit)
    sp = [p for p in axA.patches if isinstance(p, StepPatch)][0]
    vals, edges, _ = sp.get_data()
    checks = [
        ("(a) histogram density", vals, z["histogram_density"]),
        ("(a) histogram edges", edges, z["histogram_edges"]),
        ("(a) exact tau", lA.get_xdata(), z["exact_tau"]),
        ("(a) exact density", lA.get_ydata(), z["exact_total_density"]),
        ("(b) total", lB0.get_ydata(), z["exact_total_density"]),
        ("(b) capture via shortcut", lB1.get_ydata(), z["exact_shortcut_density"]),
        ("(b) arrival at the ends", lB2.get_ydata(), z["exact_diffusive_density"]),
    ] + [(f"(c) b={b:g}", linesC[i].get_ydata(), z["nonantipodal_density"][i]) for i, b in enumerate(bs)] \
      + [(f"(c) tau b={b:g}", linesC[i].get_xdata(), z["nonantipodal_tau"]) for i, b in enumerate(bs)]
    for name, plotted, stored in checks:
        assert np.array_equal(np.asarray(plotted), np.asarray(stored)), name
    say(f"verified: {len(checks)} plotted arrays equal the stored npz arrays bit-for-bit")

    # figures from regenerated data are written next to the reference build, not over it
    info = S.save(fig, FIG, "fig5_lattice_checks" + ("_regenerated" if args.data is not None else ""))
    say(f"wrote {rel(info['pdf'])} and .png ({S.PNG_DPI} dpi); "
        f"size {info['width_in']:.2f} x {info['height_in']:.2f} in; smallest font {info['min_font_pt']:g} pt")
    for row in info["fonts"]:
        say("  pdffonts: " + " ".join(row.split()))
    (TAB / ("fig5_lattice_checks" + ("_regenerated" if args.data is not None else "") + ".txt")).write_text("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
