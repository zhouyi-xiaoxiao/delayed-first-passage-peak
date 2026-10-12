#!/usr/bin/env python3
"""Lattice data of Fig. 5: direct Monte Carlo of the ring, exact densities, absorption routes.

Model: lazy ring of N sites (stay 1-q, hop q/2 to each neighbour); absorbing target at site 0;
a one-way shortcut from site u to the target, taken with probability lambda = beta(1-q) per step
(subtracted from the probability of staying).  Continuum map: b = lambda N/q, theta = u/N,
tau = q t/N^2.  The one-step absorption vector splits by route: the shortcut entry (lambda at u)
and the two end entries (q/2 at the neighbours of the target).

  (a) direct Monte Carlo (4x10^5 walkers, seed 12345) vs the exact lattice density
      (N = 100, u = r = 50, b = 2.6);
  (b) exact split of the density into capture through the shortcut and arrival at the ends;
  (c) shortcut site u = 38 (theta = 0.38), release at the shortcut site, b = 1.5, 2.16, 2.8.

Modes
  default      recompute the exact densities and compare them with the stored data file
               (data/dpma_channel_mc_data.npz); recompute pi_sc and its binomial standard error
               from the stored per-route counts of the Monte Carlo run and compare them with the
               closed form.  Light (seconds).
  --simulate   rerun the Monte Carlo with the recorded seed and write new data files to
               outputs/data/ (about two minutes for 4x10^5 walkers; --nwalk sets the number).

Logs: outputs/tables/dpma_channel_mc_check.txt (default) or outputs/tables/dpma_channel_mc.txt.
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np
from scipy.stats import chi2 as chi2_distribution

from lattice_mixture import evaluate_geometric_mixture
from paths import DATA, GENERATED, TABLES

q = 2.0 / 3.0
SEED = 12345
OUT: list[str] = []


def say(s=""):
    print(s, flush=True)
    OUT.append(s)


# ------------------------------------------------------------------ exact model
def build_ring(N, u, beta):
    """Transient block M (sites 1..N-1) + absorption vectors split by route."""
    n = N - 1
    M = np.zeros((n, n))
    a_sc = np.zeros(n)    # capture through the shortcut (at u)
    a_end = np.zeros(n)   # hop onto the target (the two ends of the unrolled interval)
    for i in range(1, N):
        k = i - 1
        be = beta if i == u else 0.0
        M[k, k] = (1 - q) * (1 - be)
        if be:
            a_sc[k] += be * (1 - q)
        for j in (i - 1, i + 1):
            if j % N == 0:            # neighbour is the absorbing target
                a_end[k] += q / 2.0
            elif 1 <= j <= N - 1:
                M[k, (j % N) - 1] += q / 2.0
    return M, a_sc, a_end


def fpt_channels(M, a_sc, a_end, r, ts):
    """Exact first-passage pmf and its two route components at integer times ts (start r)."""
    s, V = np.linalg.eigh(M)
    er = np.zeros(M.shape[0])
    er[r - 1] = 1.0
    with np.errstate(all="ignore"):
        proj = er @ V
        B_tot = proj * (V.T @ (a_sc + a_end))
        B_sc = proj * (V.T @ a_sc)
        B_end = proj * (V.T @ a_end)
    return (evaluate_geometric_mixture(s, B_tot, ts), evaluate_geometric_mixture(s, B_sc, ts),
            evaluate_geometric_mixture(s, B_end, ts))


# ------------------------------------------------------------------ direct Monte Carlo
def montecarlo(N, u, beta, r, nwalk, tmax, seed_offset):
    """Vectorised direct simulation of the discrete chain.

    Returns absorption times (in tau) and route labels (0 = ends, 1 = shortcut).  A local
    generator with a fixed seed makes the histogram exactly reproducible.
    """
    rng = np.random.default_rng(SEED + seed_offset)
    pos = np.full(nwalk, r, dtype=np.int64)
    alive = np.ones(nwalk, dtype=bool)
    tabs = np.zeros(nwalk, dtype=np.int64)
    chan = np.full(nwalk, -1, dtype=np.int8)
    stay_u = (1 - q) * (1 - beta)          # stay probability at u
    for t in range(1, tmax + 1):
        idx = np.where(alive)[0]
        if idx.size == 0:
            break
        p = pos[idx]
        rnd = rng.random(idx.size)
        atu = (p == u)
        # order of the outcomes: shortcut (only at u), stay, hop left, hop right
        thr_stay = np.where(atu, stay_u, (1 - q))
        c_sc = np.where(atu, beta * (1 - q), 0.0)
        c_stay = c_sc + thr_stay
        c_left = c_stay + q / 2.0
        move = np.where(rnd < c_sc, 3,
                        np.where(rnd < c_stay, 0,
                                 np.where(rnd < c_left, 1, 2))).astype(np.int8)
        newp = p.copy()
        newp = np.where(move == 1, p - 1, newp)
        newp = np.where(move == 2, p + 1, newp)
        sc_abs = (move == 3)
        end_abs = (newp <= 0) | (newp >= N)
        done_sc = idx[sc_abs]
        done_end = idx[end_abs & ~sc_abs]
        if done_sc.size:
            alive[done_sc] = False
            tabs[done_sc] = t
            chan[done_sc] = 1
        if done_end.size:
            alive[done_end] = False
            tabs[done_end] = t
            chan[done_end] = 0
        still = idx[~sc_abs & ~end_abs]
        pos[still] = newp[~sc_abs & ~end_abs]
    absorbed = ~alive
    tau = q * tabs[absorbed] / (N * N)
    return tau, chan[absorbed], int(absorbed.sum()), nwalk


def tgrid(N, tau_max, npts=1400):
    tmax = int(tau_max * N * N / q)
    return np.unique(np.round(np.exp(np.linspace(math.log(2), math.log(tmax), npts))).astype(np.int64))


def exact_curves():
    """Exact lattice curves of panels (a)-(c), as stored in the data file."""
    Na, ua = 100, 50
    ba = 2.6
    beta_a = ba * q / ((1 - q) * Na)
    tsA = tgrid(Na, 3.0)
    Ftot, Fsc, Fend = fpt_channels(*build_ring(Na, ua, beta_a), ua, tsA)
    scale = (Na * Na) / q
    Nc, uc = 100, 38
    tsC = tgrid(Nc, 2.5)
    curves_c = []
    for b in (1.5, 2.16, 2.8):
        beta_c = b * q / ((1 - q) * Nc)
        Ft, _, _ = fpt_channels(*build_ring(Nc, uc, beta_c), uc, tsC)
        curves_c.append(Ft * (Nc * Nc) / q)
    a_sm = q / (beta_a * (1 - q))               # a = q/lambda = N/b
    pi_sc_exact = 2 * min(ua, ua) * (Na - max(ua, ua)) / (a_sm * Na + 2 * ua * (Na - ua))
    return dict(Na=Na, ua=ua, ba=ba, beta_a=beta_a, tauA=q * tsA / (Na * Na), Ftot=Ftot * scale,
                Fsc=Fsc * scale, Fend=Fend * scale, Nc=Nc, uc=uc, tauC=q * tsC / (Nc * Nc),
                curves_c=np.vstack(curves_c), pi_sc_exact=pi_sc_exact,
                channel_residual=float(np.max(np.abs(Ftot - Fsc - Fend))))


def interior_maxima(y):
    return int(np.sum((y[1:-1] > y[:-2]) & (y[1:-1] > y[2:])))


def check_stored():
    say("Exact lattice densities vs the stored data of Fig. 5 (no Monte Carlo rerun)")
    ex = exact_curves()
    z = np.load(DATA / "dpma_channel_mc_data.npz")
    summary = json.loads((DATA / "dpma_channel_mc_summary.json").read_text())
    pairs = [("exact_tau", ex["tauA"]), ("exact_total_density", ex["Ftot"]),
             ("exact_shortcut_density", ex["Fsc"]), ("exact_diffusive_density", ex["Fend"]),
             ("nonantipodal_tau", ex["tauC"]), ("nonantipodal_density", ex["curves_c"])]
    worst = 0.0
    for key, new in pairs:
        old = z[key]
        # relative difference where the value exceeds 1e-8 of its maximum, and absolute
        # difference relative to the maximum elsewhere (round-off of the mode sum at times
        # before the ends can be reached, where the exact value is zero)
        scale = float(np.max(np.abs(old)))
        big = np.abs(old) > 1e-8 * scale
        rel = float(np.max(np.abs(new[big] - old[big]) / np.abs(old[big])))
        absolute = float(np.max(np.abs(new - old))) / scale
        worst = max(worst, rel / 1e-6, absolute / 1e-12)
        say(f"  {key:26s}: max relative difference recomputed/stored = {rel:.1e} "
            f"(absolute, relative to the maximum: {absolute:.1e})")
    if worst > 1.0:
        raise RuntimeError("exact lattice curves differ from the stored data "
                           "(tolerance 1e-6 relative, 1e-12 absolute)")
    say(f"  exact route decomposition: max |total - shortcut - ends| = {ex['channel_residual']:.1e}")
    F, F_sc, tauA = ex["Ftot"], ex["Fsc"], ex["tauA"]
    peaks = np.flatnonzero((F[1:-1] > F[:-2]) & (F[1:-1] > F[2:])) + 1
    valleys = np.flatnonzero((F[1:-1] < F[:-2]) & (F[1:-1] < F[2:])) + 1
    jp, jv = peaks[-1], valleys[-1]
    say(f"  panel (b): valley at tau = {tauA[jv]:.4f}, delayed peak at tau = {tauA[jp]:.4f}; capture share "
        f"of the lattice density at the delayed peak = {F_sc[jp] / F[jp]:.3f}; capture current monotone "
        f"decreasing for tau >= 1e-3: {bool(np.all(np.diff(F_sc[tauA >= 1e-3]) < 0.0))}")
    if ex["channel_residual"] > 2e-15:
        raise RuntimeError("route decomposition failed")
    counts = [interior_maxima(c) for c in ex["curves_c"]]
    say(f"  theta = 0.38: local maxima of the lattice density at b = 1.5, 2.16, 2.8: {counts} "
        "(the first maximum of each count is the discrete parity maximum at the start)")
    if counts != [2, 2, 1]:
        raise RuntimeError(f"theta=0.38 regression failed: {counts}")
    ntot = int(z["histogram_denominator_trials"][0])
    cnts = z["histogram_counts"]
    if not np.allclose(z["histogram_density"] * np.diff(z["histogram_edges"]), cnts / ntot,
                       rtol=0.0, atol=1e-15):
        raise RuntimeError("stored histogram is not normalized by all trials")
    n_sc, n_end = (int(x) for x in z["route_counts_shortcut_end"])
    if n_sc + n_end != ntot or not np.array_equal(z["histogram_counts_shortcut"] + z["histogram_counts_end"], cnts):
        raise RuntimeError("stored per-route counts are inconsistent with the histogram")
    pi_mc = n_sc / ntot                       # recomputed from the stored per-route counts
    if round(pi_mc, 5) != float(summary["pi_sc_monte_carlo"]):
        raise RuntimeError("per-route counts disagree with the stored summary")
    pi_ex = ex["pi_sc_exact"]
    se = math.sqrt(pi_mc * (1 - pi_mc) / ntot)
    say(f"  Monte Carlo ({ntot} walkers, seed {SEED}): {len(cnts)} logarithmic bins, median occupancy "
        f"of non-empty bins {int(np.median(cnts[cnts > 0]))}")
    say(f"  absorption routes of the stored run: {n_sc} through the shortcut, {n_end} at the ends")
    say(f"  pi_sc: Monte Carlo {pi_mc:.5f} +- {se:.5f} (binomial s.e.), exact closed form {pi_ex:.5f}, "
        f"|difference| = {abs(pi_mc - pi_ex):.1e} ({abs(pi_mc - pi_ex) / se:.2f} s.e.)")
    if abs(pi_mc - pi_ex) > 4 * se:
        raise RuntimeError("Monte Carlo pi_sc disagrees with the closed form")
    # Monte Carlo histogram vs the exact lattice bin probabilities (Fig. 5(a)): the exact pmf is
    # summed over the integer times whose tau = q t/N^2 falls in each logarithmic bin.
    edges = z["histogram_edges"]
    Na, ua = ex["Na"], ex["ua"]
    ts = np.arange(1, int(math.ceil(edges[-1] * Na * Na / q)) + 2, dtype=np.int64)
    Ft, _, _ = fpt_channels(*build_ring(Na, ua, ex["beta_a"]), ua, ts)
    p_bin, _ = np.histogram(q * ts / (Na * Na), bins=edges, weights=Ft)
    expected = ntot * p_bin
    use = expected >= 5.0
    chi2 = float(np.sum((cnts[use] - expected[use]) ** 2 / expected[use]))
    dof = int(use.sum())
    p_value = float(chi2_distribution.sf(chi2, dof))
    nonempty = np.flatnonzero(cnts > 0)
    lo, hi = float(edges[nonempty[0]]), float(edges[nonempty[-1] + 1])
    big = expected >= 1e3
    say(f"  histogram vs exact lattice bin probabilities: chi^2/dof = {chi2:.1f}/{dof} (p = {p_value:.2f}; "
        f"bins with >= 5 expected events), largest |count/expected - 1| in bins with >= 10^3 expected "
        f"events {float(np.max(np.abs(cnts[big] / expected[big] - 1.0))):.3f}; non-empty bins span "
        f"tau = {lo:.1e}..{hi:.2f} ({math.log10(hi / lo):.1f} decades)")
    if p_value < 1e-3:
        raise RuntimeError("Monte Carlo histogram disagrees with the exact lattice density")
    (TABLES / "dpma_channel_mc_check.txt").write_text("\n".join(OUT) + "\n")


def simulate(nwalk):
    ex = exact_curves()
    Na, ua, ba, beta_a = ex["Na"], ex["ua"], ex["ba"], ex["beta_a"]
    say("Panel (a): direct Monte Carlo vs exact density (theta = xi = 1/2)")
    say(f"  N={Na}, u={ua}, b={ba} (beta={beta_a:.4f}), start r={ua}")
    tmc = int(6.0 * Na * Na / q)
    tau_mc, chan_mc, nabs, ntot = montecarlo(Na, ua, beta_a, ua, nwalk, tmc, 0)
    n_sc = int(np.sum(chan_mc == 1))
    n_end = int(np.sum(chan_mc == 0))
    pi_sc_mc = n_sc / ntot
    pi_ex = ex["pi_sc_exact"]
    pi_se = math.sqrt(pi_sc_mc * (1.0 - pi_sc_mc) / ntot)
    say(f"  MC walkers={ntot}, absorbed={nabs} ({100 * nabs / ntot:.2f}%), through the shortcut {n_sc}, "
        f"at the ends {n_end}, pi_sc(MC)={pi_sc_mc:.5f}")
    say(f"  pi_sc(exact, closed form)={pi_ex:.5f}; binomial s.e. {pi_se:.5f}")
    if abs(pi_sc_mc - pi_ex) > 4.0 * pi_se:
        raise RuntimeError("Monte Carlo splitting probability disagrees with the closed form")
    logbins = np.logspace(math.log10(3e-4), math.log10(3.0), 60)
    cnts, edges = np.histogram(tau_mc, bins=logbins)
    cnts_sc, _ = np.histogram(tau_mc[chan_mc == 1], bins=logbins)
    cnts_end, _ = np.histogram(tau_mc[chan_mc == 0], bins=logbins)
    # normalized by all trials (density=True would condition on the displayed bins)
    mc_density = cnts / (ntot * np.diff(edges))
    say(f"  MC histogram bins: {len(logbins) - 1}, nonempty {int((cnts > 0).sum())}, "
        f"min nonempty count {int(cnts[cnts > 0].min())}, median {int(np.median(cnts[cnts > 0]))}")
    say(f"  MC probability inside displayed bins={cnts.sum() / ntot:.6f}")
    stored = np.load(DATA / "dpma_channel_mc_data.npz")
    if ntot == int(stored["histogram_denominator_trials"][0]):
        same = np.array_equal(cnts, stored["histogram_counts"])
        if "route_counts_shortcut_end" in stored.files:
            same = (same and np.array_equal(cnts_sc, stored["histogram_counts_shortcut"])
                    and [n_sc, n_end] == [int(x) for x in stored["route_counts_shortcut_end"]])
        say(f"  histogram and route counts identical to the stored run: {same}")
    Nc, uc = ex["Nc"], ex["uc"]
    np.savez_compressed(
        GENERATED / "dpma_channel_mc_data.npz",
        q=np.array([q]), N_antipodal=np.array([Na]), u_antipodal=np.array([ua]),
        r_antipodal=np.array([ua]), b_antipodal=np.array([ba]), beta_antipodal=np.array([beta_a]),
        monte_carlo_seed=np.array([SEED]), monte_carlo_tmax_steps=np.array([tmc]),
        monte_carlo_tau_max=np.array([6.0]), displayed_tau_max_antipodal=np.array([3.0]),
        histogram_edges=edges, histogram_counts=cnts, histogram_density=mc_density,
        histogram_denominator_trials=np.array([ntot]),
        histogram_counts_shortcut=cnts_sc, histogram_counts_end=cnts_end,
        route_counts_shortcut_end=np.array([n_sc, n_end]),
        exact_tau=ex["tauA"], exact_total_density=ex["Ftot"], exact_shortcut_density=ex["Fsc"],
        exact_diffusive_density=ex["Fend"],
        N_nonantipodal=np.array([Nc]), u_nonantipodal=np.array([uc]), r_nonantipodal=np.array([uc]),
        displayed_tau_max_nonantipodal=np.array([2.5]), nonantipodal_tau=ex["tauC"],
        nonantipodal_b=np.array([1.5, 2.16, 2.8]),
        nonantipodal_beta=np.array([b * q / ((1 - q) * Nc) for b in (1.5, 2.16, 2.8)]),
        nonantipodal_density=ex["curves_c"],
    )
    (GENERATED / "dpma_channel_mc_data.json").write_text(json.dumps({
        "description": "Processed histogram counts/densities and exact plotted curves; no raw trajectories.",
        "antipodal": {"q": q, "N": Na, "u": ua, "r": ua, "b": ba, "beta": beta_a, "nwalk": ntot,
                      "seed": SEED, "simulation_tau_max": 6.0, "displayed_tau_max": 3.0},
        "nonantipodal": {"q": q, "N": Nc, "u": uc, "r": uc, "b": [1.5, 2.16, 2.8],
                         "beta": [b * q / ((1 - q) * Nc) for b in (1.5, 2.16, 2.8)],
                         "displayed_tau_max": 2.5},
    }, indent=2, sort_keys=True) + "\n")
    (GENERATED / "dpma_channel_mc_summary.json").write_text(json.dumps({
        "walkers": ntot, "seed": SEED, "absorbed": nabs, "absorbed_through_shortcut": n_sc,
        "absorbed_at_ends": n_end, "pi_sc_monte_carlo": round(pi_sc_mc, 5),
        "pi_sc_exact": round(pi_ex, 5), "histogram_bins": len(logbins) - 1,
        "median_nonempty_bin_count": int(np.median(cnts[cnts > 0])),
    }, indent=2) + "\n")
    say("wrote outputs/data/dpma_channel_mc_data.npz, .json and dpma_channel_mc_summary.json")
    (TABLES / "dpma_channel_mc.txt").write_text("\n".join(OUT) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--simulate", action="store_true", help="rerun the Monte Carlo (slow)")
    ap.add_argument("--nwalk", type=int, default=400_000, help="number of walkers for --simulate")
    args = ap.parse_args()
    if args.simulate:
        simulate(args.nwalk)
    else:
        check_stored()


if __name__ == "__main__":
    main()
