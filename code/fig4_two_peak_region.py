#!/usr/bin/env python3
"""Fig. 4: two-peak region for a midpoint sink and release at xi (plot only).

Plot-only: nothing is solved here.

Data (stored, read-only)
  data/dpma_offgate_dense.csv    branch, xi, b, tau, residual, param
      dense fold lines b_-(xi), b_+(xi) and the two cusps, written by dpma_offgate_dense.py
      (k <= 160, xi step <= 0.0025, tau-parametrised and refined near the cusps; right half =
      exact mirror xi -> 1 - xi)
  data/dpma_offgate_phase.csv    9 fold points per line + cusps (dpma_offgate_phase.py)
  data/dpma_offgate_phase.json   cusp_left / cusp_right (cross-check of the CSV)
  data/dpma_boundary_generated.csv  b_c(theta = 1/2) (release at the sink)

Each line is drawn as one polyline cusp -> dense rows -> xi = 1/2 -> mirrored rows -> cusp.  At
xi = 1/2 the upper line is the solved fold b_+(1/2) = b_c(1/2); the lower line is closed at its
limit b = 0: the dense lower line stops at xi = 0.44 (b_- = 1.4e-6 there and smaller beyond,
where the cancellation in the mode sum limits the solver), i.e. ~3e-5 pt above the drawn value.

The 9 fold points of dpma_offgate_phase.csv are not marked: they lie on the dense lines to
|db| < 3e-10 (asserted below).

Outputs: outputs/figures/fig4_two_peak_region.{pdf,png}, outputs/tables/fig4_two_peak_region.txt
"""
from __future__ import annotations

import csv
import json

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

import figstyle as S
from paths import DATA, FIGURES as FIG, TABLES as TAB, rel

DENSE_PATH = DATA / "dpma_offgate_dense.csv"
CSV_PATH = DATA / "dpma_offgate_phase.csv"
JSON_PATH = DATA / "dpma_offgate_phase.json"
BOUNDARY = DATA / "dpma_boundary_generated.csv"
LOG = []


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


def load():
    # dense lines: left half (solved) and right half (mirror rows), each sorted in xi
    dense = list(csv.DictReader(open(DENSE_PATH, newline="")))
    lines = {}
    for name in ("lower", "upper"):
        rows = [(float(r["xi"]), float(r["b"]), r["param"]) for r in dense if r["branch"] == name]
        left = sorted((x, b) for x, b, p in rows if p != "mirror")
        right = sorted((x, b) for x, b, p in rows if p == "mirror")
        lines[name] = (np.array(left), np.array(right))
    cusp_rows = sorted((float(r["xi"]), float(r["b"])) for r in dense if r["branch"] == "cusp")
    assert len(cusp_rows) == 2
    # sparse table: cusps (and json) + the 9 stored fold points per line
    rows = list(csv.DictReader(open(CSV_PATH, newline="")))
    sec = [r for r in rows if float(r["theta"]) == 0.5]
    stored = {name: sorted((float(r["xi"]), float(r["b"])) for r in sec if r["branch"] == name)
              for name in ("lower", "upper")}
    cusps = sorted((float(r["xi"]), float(r["b"])) for r in sec if r["branch"] == "cusp")
    assert len(cusps) == 2 and abs(cusps[0][0] + cusps[1][0] - 1.0) < 1e-9
    js = json.load(open(JSON_PATH))
    for key, (x, b) in zip(("cusp_left", "cusp_right"), cusps):
        assert abs(js[key]["xi"] - x) < 1e-9 and abs(js[key]["b"] - b) < 1e-9 * b, key
    for (xd, bd), (xs, bs) in zip(cusp_rows, cusps):
        assert abs(xd - xs) < 1e-9 and abs(bd - bs) < 1e-9, "dense cusp != stored cusp"
    bnd = [r for r in csv.DictReader(open(BOUNDARY, newline="")) if float(r["theta"]) == 0.5]
    assert len(bnd) == 1
    b_c = float(bnd[0]["b_c"])
    assert abs(b_c - 3.0764323604) < 1e-9
    return lines, np.array(cusp_rows), stored, b_c


def main():
    S.apply_style()
    lines, cusps, stored, b_c = load()
    (lo_L, lo_R), (up_L, up_R) = lines["lower"], lines["upper"]
    xi_c, b_cusp = cusps[0]
    say("Fig. 4: two-interior-peak region at theta = 1/2 (plot only, no solves)")
    for path in (DENSE_PATH, CSV_PATH, JSON_PATH, BOUNDARY):
        say(f"input {rel(path)} sha256 {S.sha256(path)}")
    say(f"dense rows per half: lower {len(lo_L)} (xi {lo_L[0, 0]:.10f} .. {lo_L[-1, 0]:g}), "
        f"upper {len(up_L)} (xi {up_L[0, 0]:.10f} .. {up_L[-1, 0]:g}); max xi step "
        f"{max(np.diff(np.r_[xi_c, lo_L[:, 0]]).max(), np.diff(np.r_[xi_c, up_L[:, 0]]).max()):.4f}")
    say(f"cusps: xi = {cusps[0, 0]:.10f}, {cusps[1, 0]:.10f}; b = {b_cusp:.11g}; b_c(1/2) = {b_c:.10f}")

    # ---- checks on the stored data before drawing (no solves)
    for name, (L, R) in lines.items():
        assert np.all(np.diff(L[:, 0]) > 0) and np.all(np.diff(R[:, 0]) > 0)
        assert L[0, 0] > xi_c and L[-1, 0] <= 0.5 and R[-1, 0] < 1.0 - xi_c
        Lm = L[L[:, 0] < 0.5]
        assert np.allclose(1.0 - R[::-1, 0], Lm[:, 0], rtol=0, atol=1e-15)
        assert np.array_equal(R[::-1, 1], Lm[:, 1]), f"{name}: right half is not the mirror"
    assert up_L[-1, 0] == 0.5 and abs(up_L[-1, 1] - b_c) < 1e-9     # b_+(1/2) = b_c(1/2)
    assert lo_L[-1, 0] < 0.5 and lo_L[-1, 1] < 2e-6                 # lower line ends at b ~ 1e-6
    worst = 0.0
    for name, (L, _) in lines.items():
        for xs, bs in stored[name]:
            j = int(np.argmin(np.abs(L[:, 0] - xs)))
            assert abs(L[j, 0] - xs) < 1e-12, f"{name}: stored xi {xs} missing from dense rows"
            worst = max(worst, abs(L[j, 1] - bs))
    assert worst < 1e-6
    say(f"stored fold points (9 per line, not marked) lie on the dense lines: max |db| = {worst:.1e}")
    say(f"lower line: last dense point xi = {lo_L[-1, 0]:g}, b = {lo_L[-1, 1]:.2e}; "
        "closed at b = 0 on xi = 1/2")

    # ---- polylines: cusp -> left rows -> xi = 1/2 -> mirrored rows -> cusp
    x_up = np.r_[xi_c, up_L[:, 0], up_R[:, 0], cusps[1, 0]]
    y_up = np.r_[b_cusp, up_L[:, 1], up_R[:, 1], cusps[1, 1]]
    x_lo = np.r_[xi_c, lo_L[:, 0], 0.5, lo_R[:, 0], cusps[1, 0]]
    y_lo = np.r_[b_cusp, lo_L[:, 1], 0.0, lo_R[:, 1], cusps[1, 1]]

    C_FILL = "#bfe0f5"          # light tint of Okabe-Ito sky blue
    C_UP, C_LO = S.OI["vermillion"], S.OI["blue"]

    fig, ax = plt.subplots(figsize=(S.COL_WIDTH, 2.65))
    region = ax.fill(np.r_[x_up, x_lo[::-1]], np.r_[y_up, y_lo[::-1]], facecolor=C_FILL,
                     edgecolor="none", zorder=1)[0]
    l_up, = ax.plot(x_up, y_up, "-", color=C_UP, lw=1.4, zorder=3, solid_joinstyle="round")
    l_lo, = ax.plot(x_lo, y_lo, "--", color=C_LO, lw=1.4, zorder=3, dash_joinstyle="round")
    ax.axvline(0.5, color=S.MUTED, lw=0.8, ls=":", zorder=2)
    ax.plot(cusps[:, 0], cusps[:, 1], "o", ms=5.0, color=S.INK, zorder=5)
    ax.plot([0.5], [b_c], "o", ms=5.0, mfc="white", mec=S.INK, mew=1.0, zorder=5)

    ann = dict(fontsize=S.NOTE_PT, color=S.INK)
    lead = dict(arrowstyle="-", color=S.MUTED, lw=0.6, shrinkA=1, shrinkB=3.5)
    ax.annotate("cusp", xy=cusps[0], xytext=(xi_c + 0.016, b_cusp - 0.45), ha="left",
                va="center", arrowprops=lead, **ann)
    ax.annotate("cusp", xy=cusps[1], xytext=(1 - xi_c - 0.016, b_cusp - 0.45), ha="right",
                va="center", arrowprops=lead, **ann)
    ax.annotate(rf"$b_c={b_c:.3f}$", xy=(0.5, b_c), xytext=(0.512, 3.62), ha="left",
                va="bottom", arrowprops=lead, **ann)
    ax.text(0.4965, 0.18, "release at the sink", rotation=90, ha="right", va="bottom",
            fontsize=S.SMALL_PT, color=S.MUTED)

    ax.set_xlim(0.37, 0.63)
    ax.set_ylim(0.0, 6.35)
    ax.xaxis.set_major_locator(MultipleLocator(0.05))
    ax.xaxis.set_minor_locator(MultipleLocator(0.01))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.5))
    ax.tick_params(which="both", top=True, right=True)
    ax.set_xlabel(r"release point $\xi$ (sink at $\theta=1/2$)")
    ax.set_ylabel(r"sink strength $b$")
    handles = [Patch(facecolor=C_FILL, edgecolor="none"),
               Line2D([], [], color=C_UP, ls="-", lw=1.4),
               Line2D([], [], color=C_LO, ls="--", lw=1.4)]
    ax.legend(handles, ["two interior peaks", r"upper fold line $b_+(\xi)$",
                        r"lower fold line $b_-(\xi)$"],
              loc="upper center", bbox_to_anchor=(0.5, 0.995), frameon=True, facecolor="white",
              edgecolor="none", framealpha=1.0, borderpad=0.25,
              handlelength=2.2, borderaxespad=0.3, labelspacing=0.3).set_zorder(6)
    fig.subplots_adjust(left=0.125, right=0.975, bottom=0.155, top=0.985)

    # ---- verification: plotted arrays are the stored values (no interpolation, no solves)
    nu, nl = len(up_L), len(lo_L)
    xd, yd = l_up.get_data()
    assert np.array_equal(xd[1:1 + nu], up_L[:, 0]) and np.array_equal(yd[1:1 + nu], up_L[:, 1])
    assert np.array_equal(yd[1 + nu:-1], up_R[:, 1]) and yd[0] == yd[-1] == b_cusp
    xd, yd = l_lo.get_data()
    assert np.array_equal(xd[1:1 + nl], lo_L[:, 0]) and np.array_equal(yd[1:1 + nl], lo_L[:, 1])
    assert np.array_equal(yd[2 + nl:-1], lo_R[:, 1]) and yd[1 + nl] == 0.0 and xd[1 + nl] == 0.5
    for xs_, ys_ in ((x_up, y_up), (x_lo, y_lo)):
        assert np.allclose(1.0 - xs_[::-1], xs_, rtol=0, atol=1e-15) and np.array_equal(ys_[::-1], ys_)
    poly = region.get_xy()
    assert np.array_equal(poly[:len(x_up), 1], y_up)
    say(f"verified: plotted polylines ({len(x_up)} upper, {len(x_lo)} lower vertices) equal the dense "
        "CSV values bit-for-bit; right half is the exact mirror; shading is bounded by the two lines")

    info = S.save(fig, FIG, "fig4_two_peak_region")
    say(f"wrote {rel(info['pdf'])} and .png ({S.PNG_DPI} dpi); "
        f"size {info['width_in']:.2f} x {info['height_in']:.2f} in; smallest font {info['min_font_pt']:g} pt")
    for row in info["fonts"]:
        say("  pdffonts: " + " ".join(row.split()))
    (TAB / "fig4_two_peak_region.txt").write_text("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
