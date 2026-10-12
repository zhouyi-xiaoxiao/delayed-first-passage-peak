#!/usr/bin/env python3
"""Reproduce the figures and the quoted numbers of the paper

    "Geometry-controlled disappearance of a delayed first-passage peak under localized absorption"
    (Xiaoxiao Zhouyi).

Usage (from any directory):

    python reproduce_all.py              # default: all light computations, the six figures from
                                         # stored data, and a PASS/FAIL comparison with the paper
    python reproduce_all.py --full       # additionally rerun the simulations with the recorded seeds
                                         # (lattice Monte Carlo, Brownian dynamics, gate-width/time-step
                                         # sweep), if python-flint is installed the Arb certificates,
                                         # and if lean is installed the two core-Lean modules of formal/
    python reproduce_all.py --list       # list the steps
    python reproduce_all.py --only fig1_model_morphology dpma_halfline_bstar   # run a subset
    python reproduce_all.py --checks-only                                     # only compare numbers

With --only or --checks-only, comparisons whose inputs have not been produced yet are reported as
NOT RUN (not as failures).

Every script writes below outputs/ (figures, tables/logs, generated data); the console output of
each step is kept in outputs/logs/<step>.log and the comparison table in outputs/summary.txt.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CODE = ROOT / "code"
DATA = ROOT / "data"
OUT = ROOT / "outputs"
TAB = OUT / "tables"
FIG = OUT / "figures"
GEN = OUT / "data"
LOGS = OUT / "logs"
REF_FIG = ROOT / "figures"

FIGURES = ("fig1_model_morphology", "fig2_threshold", "fig3_near_end", "fig4_two_peak_region",
           "fig5_lattice_checks", "fig6_brownian_dynamics")

# (step name, command line after the interpreter, what it reproduces)
STEPS = [
    ("test_dpma_identities", ["-m", "pytest", "code/test_dpma_identities.py", "-q", "-p", "no:cacheprovider"],
     "exact identities (App. A, B) and headline numbers"),
    ("dpma_verification_table", ["code/dpma_verification_table.py"], "Table II, rows 1-3"),
    ("finite_mode_algebra", ["code/finite_mode_algebra.py"], "Table II, two/three-mode fold algebra"),
    ("dpma_saddle_node_bc_theta", ["code/dpma_saddle_node_bc_theta.py"], "threshold curve b_c(theta)"),
    ("dpma_saddle_node_certification", ["code/dpma_saddle_node_certification.py"], "fold scaling, Fig. 2(c)"),
    ("dpma_fold_branch_margins", ["code/dpma_fold_branch_margins.py"], "nondegeneracy margins, cutoff"),
    ("dpma_normal_form", ["code/dpma_normal_form.py"], "prefactors A_tau, A_Phi; curvature at 1/2"),
    ("mode_truncation_check", ["code/mode_truncation_check.py"], "K-mode truncation"),
    ("dpma_window_scan", ["code/dpma_window_scan.py"], "dense scans in b (one interval)"),
    ("small_theta_check", ["code/small_theta_check.py"], "near-end law at theta = 0.10, 0.15"),
    ("dpma_halfline_bstar", ["code/dpma_halfline_bstar.py"], "B*, c_* and half-line margins"),
    ("dpma_endpoint_law", ["code/dpma_endpoint_law.py"], "b_c(theta) theta -> B* by bisection"),
    ("endpoint_physics_checks", ["code/endpoint_physics_checks.py"], "physical reading of the near-end law"),
    ("dpma_mfpt_smoothness", ["code/dpma_mfpt_smoothness.py"], "mean, splitting probability, d mu_1/db"),
    ("dpma_bcN_convergence", ["code/dpma_bcN_convergence.py"], "lattice threshold b_{c,N}"),
    ("lattice_continuum_convergence", ["code/lattice_continuum_convergence.py"], "lattice vs continuum density"),
    ("lattice_closed_form_residues", ["code/lattice_closed_form_residues.py"], "closed-form lattice poles and residues"),
    ("dpma_offgate_phase", ["code/dpma_offgate_phase.py"], "two-peak region: fold lines, cusp"),
    ("dpma_offgate_dense", ["code/dpma_offgate_dense.py"], "dense fold lines (data of Fig. 4)"),
    ("dpma_start_asymptotics", ["code/dpma_start_asymptotics.py"], "quartic law of b_+ near xi = 1/2"),
    ("scan_beyond_cusp", ["code/scan_beyond_cusp.py"], "no two peaks beyond the cusp"),
    ("cusp_endpoint_continuation", ["code/cusp_endpoint_continuation.py"], "half-line cusps, cusp loci"),
    ("halfline_transform_checks", ["code/halfline_transform_checks.py"], "half-line transform, arbitrary release"),
    ("certificate_rational_checks", ["code/certificate_rational_checks.py"], "exact checks of the cusp certificates"),
    ("dpma_finite_gate_check", ["code/dpma_finite_gate_check.py"], "finite gate b_+^(ell)"),
    ("measurement_window_ratio", ["code/measurement_window_ratio.py"], "Sec. VI window ratio, trials"),
    ("measurement_shape_test", ["code/measurement_shape_test.py"], "Sec. VI shape test, calibration, gate of the example"),
    ("dpma_channel_mc", ["code/dpma_channel_mc.py"], "lattice data of Fig. 5 (exact part)"),
    ("dpma_brownian_fold", ["code/dpma_brownian_fold.py"], "data of Fig. 6 (exact part)"),
    ("supplementary_checks", ["code/supplementary_checks.py"], "supplementary physical checks"),
    ("fig1_model_morphology", ["code/fig1_model_morphology.py"], "Fig. 1"),
    ("fig2_threshold", ["code/fig2_threshold.py"], "Fig. 2"),
    ("fig3_near_end", ["code/fig3_near_end.py"], "Fig. 3 (stored Brownian-dynamics samples)"),
    ("fig4_two_peak_region", ["code/fig4_two_peak_region.py"], "Fig. 4"),
    ("fig5_lattice_checks", ["code/fig5_lattice_checks.py"], "Fig. 5"),
    ("fig6_brownian_dynamics", ["code/fig6_brownian_dynamics.py"], "Fig. 6"),
]

FULL_STEPS = [
    ("dpma_channel_mc_simulate", ["code/dpma_channel_mc.py", "--simulate"],
     "lattice Monte Carlo, 4x10^5 walkers, seed 12345"),
    ("dpma_brownian_fold_simulate", ["code/dpma_brownian_fold.py", "--simulate"],
     "Brownian dynamics of Fig. 6, seeds 7, 8"),
    ("fig3_near_end_rerun_bd", ["code/fig3_near_end.py", "--rerun-bd"],
     "Fig. 3 with the half-line Brownian dynamics rerun (seeds 20261012, 20261013)"),
    ("fig5_lattice_checks_regenerated", ["code/fig5_lattice_checks.py", "--data", "outputs/data"],
     "Fig. 5 from the regenerated data"),
    ("fig6_brownian_dynamics_regenerated", ["code/fig6_brownian_dynamics.py", "--data", "outputs/data"],
     "Fig. 6 from the regenerated data"),
    ("dpma_bd_convergence", ["code/dpma_bd_convergence.py"], "gate-width and time-step sweep"),
    ("halfline_cusp_interval_newton", ["code/halfline_cusp_interval_newton.py"], "Arb interval Newton"),
    ("halfline_cusp_krawczyk", ["code/halfline_cusp_krawczyk.py"], "Arb Krawczyk"),
    ("formal_cusp_check", ["code/formal_cusp_check.py"], "core-Lean modules of formal/ (lean)"),
]
NEEDS_FLINT = {"halfline_cusp_interval_newton", "halfline_cusp_krawczyk"}
NEEDS_LEAN = {"formal_cusp_check"}


# ------------------------------------------------------------------------------------ running
def run_steps(steps):
    LOGS.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.setdefault("MPLCONFIGDIR", str(OUT / ".mplconfig"))
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    results = {}
    have_flint = importlib.util.find_spec("flint") is not None
    have_lean = shutil.which("lean") is not None
    for name, args, what in steps:
        if (name in NEEDS_FLINT and not have_flint) or (name in NEEDS_LEAN and not have_lean):
            missing = "python-flint" if name in NEEDS_FLINT else "lean"
            print(f"  SKIP  {name:38s} ({missing} not installed)", flush=True)
            results[name] = ("SKIP", 0.0)
            continue
        print(f"  run   {name:38s} {what}", flush=True)
        t0 = time.time()
        proc = subprocess.run([sys.executable, *args], cwd=ROOT, env=env, capture_output=True, text=True)
        dt = time.time() - t0
        (LOGS / f"{name}.log").write_text(proc.stdout + ("\n" + proc.stderr if proc.stderr else ""))
        status = "ok" if proc.returncode == 0 else f"exit {proc.returncode}"
        results[name] = (status, dt)
        print(f"        {status} ({dt:.0f} s)", flush=True)
    return results


# ------------------------------------------------------------------------------------ helpers
def text(path):
    return Path(path).read_text()


def num(path, pattern, group=1):
    match = re.search(pattern, text(path), flags=re.M)
    if match is None:
        raise ValueError(f"pattern {pattern!r} not found in {Path(path).name}")
    return float(match.group(group))


def nums(path, pattern):
    return [float(x) for x in re.findall(pattern, text(path), flags=re.M)]


def jload(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def close(value, target, tol):
    return abs(value - target) <= tol


def pdf_equal_except_date(a, b):
    strip = lambda raw: re.sub(rb"/CreationDate \(D:[^)]*\)", b"", raw)
    return strip(Path(a).read_bytes()) == strip(Path(b).read_bytes())


# ------------------------------------------------------------------------------------ checks
def boundary_row(theta):
    for r in rows(TAB / "dpma_boundary_generated.csv"):
        if abs(float(r["theta"]) - theta) < 1e-9 and r["is_min"] == "false":
            return r
    raise ValueError(theta)


def boundary_min():
    return next(r for r in rows(TAB / "dpma_boundary_generated.csv") if r["is_min"] == "true")


def offgate_fold(xi, branch):
    for r in rows(TAB / "dpma_offgate_phase.csv"):
        if abs(float(r["theta"]) - 0.5) < 1e-9 and abs(float(r["xi"]) - xi) < 1e-9 and r["branch"] == branch:
            return float(r["b"])
    raise ValueError((xi, branch))


WINDOW_ROW = r"^\s+0\.\d\d\s+\[0\.05,[\d.]+\]\s+1\s+below=\[2\] above=\[0\]"


def check_list():
    """(quantity, value quoted in the paper, callable returning (computed text, pass?))."""
    C = []

    def add(label, quoted, fn):
        C.append((label, quoted, fn))

    # --- Sec. III: release at the sink
    add("b_c(1/2) [Eq. (antip)]", "3.0764",
        lambda: (lambda b: (f"{b:.6f}", close(b, 3.0764, 5e-5)))(float(boundary_row(0.5)["b_c"])))
    add("tau_c at theta=1/2", "0.03836",
        lambda: (lambda t: (f"{t:.6f}", close(t, 0.03836, 5e-6)))(float(boundary_row(0.5)["tau_c"])))
    add("min b_c(theta), at theta", "2.165 at 0.381",
        lambda: (lambda r: (f"{float(r['b_c']):.5f} at {float(r['theta']):.5f}",
                            close(float(r["b_c"]), 2.165, 5e-4) and close(float(r["theta"]), 0.381, 5e-4)))(boundary_min()))
    add("curvature: b_c(1/2+eps) = b_c - c2 eps^2", "c2 = 133 (133.1)",
        lambda: (lambda c: (f"{-c:.2f}", close(-c, 133.1, 0.05)))(num(TAB / "dpma_normal_form.txt", r"c2=(-?[\d.]+), c4")))
    add("near-end law joins the curve at theta=0.2", "six digits",
        lambda: (lambda r: (f"b_c*0.2 = {float(r['b_c']) * 0.2:.7f}", close(float(r["b_c"]) * 0.2, 0.7890262, 1e-6)))(boundary_row(0.2)))
    add("|b_c d - B*| for d <= 0.30", "<= 2e-5",
        lambda: (lambda m: (f"{m:.1e}", m <= 2e-5))(max(abs(float(r["b_c"]) * float(r["theta"]) - 0.7890261736)
                                                       for r in rows(TAB / "dpma_boundary_generated.csv")
                                                       if float(r["theta"]) <= 0.30 + 1e-12)))
    add("b_c(theta) theta for theta in [0.20,0.27]", "0.789026",
        lambda: (lambda v: (", ".join(f"{x:.6f}" for x in v), all(close(x, 0.789026, 5e-7) for x in v)))(
            nums(TAB / "dpma_endpoint_law.txt", r"^\s+0\.2[0-7]0?\s+[\d.]+\s+([\d.]+)\s*$")))
    add("direct fold d=0.05-0.175: |b_c d - B*|", "1e-10",
        lambda: (lambda v: (f"max {max(abs(x) for x in v):.1e}", max(abs(x) for x in v) <= 1e-10))(
            nums(TAB / "fig3_near_end.txt", r"^\s+0\.(?:050|075|100|125|150|175)\s+[\d.]+\s+[\d.]+\s+([+-][\d.e+-]+)")))
    add("theta=0.10,0.15: pair at 0.98 B*/theta, none at 1.02", "pair iff b < B*/theta",
        lambda: ("PASS" if "result: PASS" in text(TAB / "small_theta_check.txt") else "FAIL",
                 "result: PASS" in text(TAB / "small_theta_check.txt")))
    add("one interval in b, seven theta (step 0.02, k<=240)", "exactly one transition",
        lambda: (lambda n, t: (f"{n} of 7 rows with one transition", n == 7 and "no re-entrance" in t))(
            len(re.findall(WINDOW_ROW, text(TAB / "dpma_window_scan.txt"), re.M)), text(TAB / "dpma_window_scan.txt")))
    add("at most one interior maximum (release at sink)", "at most one",
        lambda: (lambda v: (f"max over 7 theta = {int(max(v))}", max(v) == 1 and len(v) == 7))(
            nums(TAB / "supplementary_checks.txt", r"max interior maxima=(\d+)")))
    add("lattice threshold: b_{c,N}-b_c at N=100, 1200", "4.742e-3, 3.281e-5",
        lambda: (lambda a, b: (f"{a:.3e}, {b:.3e}", close(a, 4.742e-3, 5e-7) and close(b, 3.281e-5, 5e-9)))(
            num(TAB / "dpma_bcN_convergence.txt", r"^\s+100\s+[\d.]+\s+\+([\d.e-]+)"),
            num(TAB / "dpma_bcN_convergence.txt", r"^\s+1200\s+[\d.]+\s+\+([\d.e-]+)")))
    add("lattice threshold: log-log slope; N^2(b_cN-b_c) at N=1200, limit", "-2.001; 47.247, 47.25",
        lambda: (lambda p, n2, lim: (f"{p:.4f}; {n2:.3f}, {lim:.3f}", close(p, -2.001, 6e-4) and close(n2, 47.247, 1.5e-3)
                                     and close(lim, 47.25, 5e-3)))(
            num(TAB / "dpma_bcN_convergence.txt", r"N\^\((-[\d.]+)\)"),
            num(TAB / "dpma_bcN_convergence.txt", r"^\s+1200\s+[\d.]+\s+\S+\s+([\d.]+)"),
            num(TAB / "dpma_bcN_convergence.txt", r"extrapolation A \+ C/N\^2: A = ([\d.]+)")))
    add("nondegeneracy margins |tau^3 Phi'''/Phi|, |b tau dPhi'/db/Phi|", ">= 1.82, >= 0.49",
        lambda: (lambda a, b: (f"{a:.5f}, {b:.5f}", a >= 1.82 and b >= 0.49))(
            num(TAB / "dpma_fold_branch_margins.txt", r"\|tau\^3 S3/Phi\|=\[([\d.]+),"),
            num(TAB / "dpma_fold_branch_margins.txt", r"\|b tau d_bS1/Phi\|=\[([\d.]+),")))
    add("cutoff doubling (k<=80 -> 160): relative change of S3", "<= 4e-6",
        lambda: (lambda v: (f"{v:.2e}", v <= 4e-6))(num(TAB / "dpma_fold_branch_margins.txt", r"maximum S3 cutoff shift=([\d.e-]+)")))
    add("K-mode truncation: b_c^(3), b_c^(4), b_c^(5)", "3.046, 3.0766, 3.0764",
        lambda: (lambda v: (", ".join(f"{x:.5f}" for x in v), len(v) == 3 and close(v[0], 3.046, 5e-4)
                            and close(v[1], 3.0766, 5e-5) and close(v[2], 3.0764, 5e-5)))(
            nums(TAB / "mode_truncation_check.txt", r"fold near the midpoint threshold: tau=[\d.]+ b=([\d.]+)")))
    add("K=2 truncation: no fold", "no fold",
        lambda: ("no fold" if "K=2: no fold" in text(TAB / "mode_truncation_check.txt") else "fold",
                 "K=2: no fold" in text(TAB / "mode_truncation_check.txt")))
    add("prefactors A_tau, A_Phi [Eq. (prefac)]", "0.0247518, 0.357444",
        lambda: (lambda a, b: (f"{a:.7f}, {b:.6f}", close(a, 0.0247518, 5e-8) and close(b, 0.357444, 5e-7)))(
            num(TAB / "dpma_normal_form.txt", r"tau_\+ - tau_-\s+= ([\d.]+)"),
            num(TAB / "dpma_normal_form.txt", r"Phi\(tau_-\) = ([\d.]+)")))
    add("prefactors vs measured extrema, b_c-b <= 1e-3", "< 2e-4 relative",
        lambda: (lambda v: (f"max {max(v):.1e}", max(v) <= 2e-4))(
            [float(x) for line in text(TAB / "dpma_normal_form.txt").splitlines() if re.match(r"\s+(0\.001|0\.0001|1\.0e-5)\s", line)
             for x in line.split()[3:4] + line.split()[6:7]]))
    add("d S1/db: central difference vs dk/db = -D_b/D_k", "agree",
        lambda: (lambda v: (f"{v:.1e}", v < 1e-15))(num(TAB / "dpma_normal_form.txt", r"relative difference to the central difference ([\d.e-]+)")))
    add("fitted slopes (seven points)", "0.5010, 1.5020",
        lambda: (lambda a, b: (f"{a:.4f}, {b:.4f}", close(a, 0.5010, 6e-5) and close(b, 1.5020, 6e-5)))(
            num(TAB / "dpma_saddle_node_certification.txt", r"gap = ([\d.]+) \+/-"),
            num(TAB / "dpma_saddle_node_certification.txt", r"prominence = ([\d.]+) \+/-")))
    add("fit-window ranges of the slopes", "0.5005-0.5022, 1.5009-1.5043",
        lambda: (lambda t: (t, t == "0.5005--0.5022 / 1.5009--1.5043"))(
            "{} / {}".format(re.search(r"gap slopes ([\d.]+--[\d.]+)", text(TAB / "dpma_saddle_node_certification.txt")).group(1),
                             re.search(r"prominence slopes ([\d.]+--[\d.]+)", text(TAB / "dpma_saddle_node_certification.txt")).group(1))))
    add("B*, c_* [Eq. (Bstar)]", "0.78903, 0.15792",
        lambda: (lambda b, c: (f"{b:.8f}, {c:.8f}", close(b, 0.78903, 5e-6) and close(c, 0.15792, 5e-6)))(
            num(TAB / "dpma_halfline_bstar.txt", r"B\* = ([\d.]+),"), num(TAB / "dpma_halfline_bstar.txt", r"c_\*\) = ([\d.]+)")))
    add("half-line fold margins and Jacobian", "-2.69248, -0.50033, -1.69e3",
        lambda: (lambda a, b, c: (f"{a:.5f}, {b:.5f}, {c:.4g}", close(a, -2.69248, 5e-6) and close(b, -0.50033, 5e-6)
                                  and close(c, -1.69e3, 6e0)))(
            num(TAB / "dpma_halfline_bstar.txt", r"sigma\^3 Phi'''/Phi = ([+-][\d.]+)"),
            num(TAB / "dpma_halfline_bstar.txt", r"B sigma \(partial_B Phi'\)/Phi = ([+-][\d.]+)"),
            num(TAB / "dpma_halfline_bstar.txt", r"det d\(Phi',Phi''\)/d\(sigma,B\) = ([+-][\d.e+]+)")))
    add("crude estimate: merger at sigma=1/6, B=18e^-3", "0.90",
        lambda: (lambda v: (f"{v:.5f}", close(v, 0.90, 5e-3)))(num(TAB / "endpoint_physics_checks.txt", r"value ([\d.]+) = 18 e\^-3")))
    add("at the merger: still diffusing / reached the wall", "78%, 1%",
        lambda: (lambda s, e: (f"{100 * s:.1f}%, {100 * e:.2f}%", close(s, 0.78, 5e-3) and close(e, 0.01, 5e-3)))(
            num(TAB / "endpoint_physics_checks.txt", r"survival S\(sigma\*\) = ([\d.]+)"),
            num(TAB / "endpoint_physics_checks.txt", r"escape ([\d.]+); fraction")))
    add("capture odds 2B (lattice, every M)", "2B exactly",
        lambda: (lambda v: (f"{len(v)} lattice sizes", len(v) == 3 and all(close(x, 2 * 0.7890261736 / (1 + 2 * 0.7890261736), 1e-10) for x in v)))(
            nums(TAB / "endpoint_physics_checks.txt", r"u\(M\) = ([\d.]+)")))
    add("mean at threshold <tau>/d = 1/(1+2B*)", "0.39",
        lambda: (lambda v: (f"{v:.4f}", close(v, 0.39, 5e-3)))(num(TAB / "dpma_mfpt_smoothness.txt", r"<tau>_c/d -> 1/\(1\+2B\*\) = ([\d.]+)")))
    add("d mu_1/db at b_c (theta=1/2)", "1.10",
        lambda: (lambda v: (f"{v:.4f}", close(v, 1.10, 5e-3)))(num(TAB / "dpma_mfpt_smoothness.txt", r"b=3\.0764323604: mu_1=[\d.]+, dmu_1/db=([\d.]+)")))
    add("lattice mean and pi_sc vs closed forms", "< 6.2e-14",
        lambda: (lambda v: (f"{v:.2e}", v < 6.2e-14))(jload(TAB / "dpma_mfpt_smoothness.json")["finite_identity_max_abs_error"]))
    add("capture share at the delayed peak, theta=xi=1/2, b=2.6", "about one half",
        lambda: (lambda v: (f"{v:.3f}", 0.4 <= v <= 0.6))(num(TAB / "supplementary_checks.txt", r"delayed peak\s+tau=[\d.]+: Phi=[\d.]+, capture share=([\d.]+)")))
    # --- Sec. IV: release away from the sink
    add("two-peak range at xi=0.40", "1.839 < b < 3.862",
        lambda: (lambda a, b: (f"{a:.4f} < b < {b:.4f}", close(a, 1.839, 5e-4) and close(b, 3.862, 5e-4)))(
            offgate_fold(0.40, "lower"), offgate_fold(0.40, "upper")))
    add("cusp (xi*, b*, tau*) [Eq. (offgatecusp)]", "(0.3788069561, 5.8163021864, 0.01607956718)",
        lambda: (lambda c: (f"({c['xi']:.10f}, {c['b']:.10f}, {c['tau']:.11f})",
                            close(c["xi"], 0.3788069561, 2e-10) and close(c["b"], 5.8163021864, 2e-9) and close(c["tau"], 0.01607956718, 2e-11)))(
            jload(TAB / "dpma_offgate_phase.json")["cusp_left"]))
    add("cusp: tau^4 S4/Phi, unfolding determinant", "-5.07254, 7.57313",
        lambda: (lambda n: (f"{n['tau4_S4_over_Phi']:.5f}, {n['unfolding_determinant']:.5f}",
                            close(n["tau4_S4_over_Phi"], -5.07254, 5e-6) and close(n["unfolding_determinant"], 7.57313, 5e-6)))(
            jload(TAB / "dpma_offgate_phase.json")["cusp_nondegeneracy"]))
    add("midpoint-sink cusp residual", "< 1.4e-11",
        lambda: (lambda r: (f"{r:.1e}", r < 1.4e-11))(jload(TAB / "dpma_offgate_phase.json")["cusp_left"]["residual"]))
    add("b_- / asymptote at delta = 0.08, 0.07, 0.06", "0.72, 0.83, 0.89",
        lambda: (lambda v: (", ".join(f"{x:.3f}" for x in v), all(close(a, b, 5e-3) for a, b in zip(v, (0.72, 0.83, 0.89)))))(
            [r["b_direct_over_asymptotic"] for r in jload(TAB / "dpma_offgate_phase.json")["lower_sheet_asymptotics"]]))
    add("b_+(xi=0.44)", "3.13660 (3.137)",
        lambda: (lambda b: (f"{b:.5f}", close(b, 3.13660, 5e-6)))(offgate_fold(0.44, "upper")))
    add("b_-(xi=0.44)", "1.4e-6",
        lambda: (lambda b: (f"{b:.2e}", close(b, 1.4e-6, 5e-8)))(
            next(float(r["b"]) for r in rows(TAB / "dpma_offgate_dense.csv")
                 if r["branch"] == "lower" and r["param"] != "mirror" and abs(float(r["xi"]) - 0.44) < 1e-12)))
    add("C_b = S3/(3 d_b S1) [Eq. (startquartic)]", "4352.656",
        lambda: (lambda v: (f"{v:.4f}", close(v, 4352.656, 5e-4)))(num(TAB / "dpma_start_asymptotics.txt", r"S3/\(3 partial_b S1\) = ([\d.]+)")))
    add("direct solve at delta=1e-3: quartic and cubic ratios", "4331.7, 1.998",
        lambda: (lambda r: (f"{float(r['quartic_ratio']):.1f}, {float(r['cubic_time_ratio']):.3f}",
                            close(float(r["quartic_ratio"]), 4331.7, 0.05) and close(float(r["cubic_time_ratio"]), 1.998, 5e-4)))(
            next(r for r in rows(TAB / "dpma_start_asymptotics.csv") if float(r["delta"]) == 0.001)))
    add("scan beyond the cusp (39 xi x 400 b)", "no two-peak density",
        lambda: (lambda n: (f"{int(n)} cases", n == 0))(num(TAB / "scan_beyond_cusp.txt", r"cases with >= 2 interior maxima = (\d+)")))
    add("half-line cusps (X, B, sigma), wall side", "(0.758314, 2.868772, 0.0636296)",
        lambda: (lambda c: (f"({c['X']:.6f}, {c['B']:.6f}, {c['T']:.7f})", close(c["X"], 0.758314, 5e-7)
                            and close(c["B"], 2.868772, 5e-7) and close(c["T"], 0.0636296, 5e-8)))(
            jload(TAB / "cusp_endpoint_continuation.json")["halfline_cusps"]["boundary_side"]))
    add("half-line cusps (X, B, sigma), far side", "(1.424897, 1.085683, 0.207230)",
        lambda: (lambda c: (f"({c['X']:.6f}, {c['B']:.6f}, {c['T']:.6f})", close(c["X"], 1.424897, 5e-7)
                            and close(c["B"], 1.085683, 5e-7) and close(c["T"], 0.207230, 5e-7)))(
            jload(TAB / "cusp_endpoint_continuation.json")["halfline_cusps"]["domain_side"]))
    add("interval cusp loci: k<=120 vs k<=220", "|db| < 3e-5",
        lambda: (lambda v: (f"max {v:.1e}", v < 3e-5))(max(c["delta_b_k120_k220"] for c in
                                                        jload(TAB / "cusp_endpoint_continuation.json")["finite_interval_cutoff_checks"])))
    add("half-line transform, arbitrary X: fhat(0)=1, B=0, X=1, branch cut", "Eqs. (fhl), (phihl)",
        lambda: (lambda t: (re.search(r"worst relative difference of the density: (\S+)", t).group(0), "result: PASS" in t))(
            text(TAB / "halfline_transform_checks.txt")))
    add("exact rational checks of the cusp certificates (Q-, Q+)", "every inclusion and sign exclusion",
        lambda: (lambda d: (f"{d['exact_comparisons']} comparisons in {len(d['groups'])} groups; "
                            f"{sum(c['holds'] for c in d['interval_consistency'])}/{len(d['interval_consistency'])} consistency checks",
                            d["pass"]))(jload(TAB / "certificate_rational_checks.json")))
    add("Lean statements of these checks == formal/FormalLean/CuspCertificateDecimals.lean", "identical",
        lambda: (lambda same: ("identical" if same else "differs", same))(
            (TAB / "CuspCertificateDecimals.lean").read_bytes() == (ROOT / "formal/FormalLean/CuspCertificateDecimals.lean").read_bytes()))
    add("Arb boxes Q-, Q+ (stored certificates)", "unique; nondegenerate",
        lambda: (lambda a, b: ("interval Newton: " + a["status"] + "; Krawczyk: " + b["status"],
                               a["status"] == "PASS" and b["status"] == "VERIFIED"
                               and all(c["status"] == "VERIFIED" for c in b["certificates"])))(
            jload(DATA / "dpma_interval_certificates.json"), jload(DATA / "dpma_halfline_cusp_interval_certificate.json")))
    add("Q-: enclosures of sigma^4 Phi''''/Phi and the unfolding det", "[-5.53353583,-5.53353573], [3.72087719,3.72087726]",
        lambda: (lambda c: (f"[{float(c['normalized_fourth_margin']['lower']):.8f},{float(c['normalized_fourth_margin']['upper']):.8f}], "
                            f"[{float(c['normalized_unfolding_determinant_logB_X']['lower']):.8f},"
                            f"{float(c['normalized_unfolding_determinant_logB_X']['upper']):.8f}]",
                            float(c["normalized_fourth_margin"]["lower"]) >= -5.53353583 and float(c["normalized_fourth_margin"]["upper"]) <= -5.53353573
                            and float(c["normalized_unfolding_determinant_logB_X"]["lower"]) >= 3.72087719
                            and float(c["normalized_unfolding_determinant_logB_X"]["upper"]) <= 3.72087726))(
            jload(DATA / "dpma_interval_certificates.json")["boundary_side_cusp"]))
    add("Q+: centres of the two enclosures", "-6.052596, -1.379343",
        lambda: (lambda c: (f"{c['normalized_fourth_margin']['ball']}, {c['normalized_unfolding_determinant_logB_X']['ball']}",
                            close(0.5 * (float(c["normalized_fourth_margin"]["lower"]) + float(c["normalized_fourth_margin"]["upper"])), -6.052596, 5e-7)
                            and close(0.5 * (float(c["normalized_unfolding_determinant_logB_X"]["lower"])
                                             + float(c["normalized_unfolding_determinant_logB_X"]["upper"])), -1.379343, 5e-7)))(
            jload(DATA / "dpma_interval_certificates.json")["domain_side_cusp"]))
    # --- Sec. V: numerical checks
    add("lattice vs continuum density, N=1200", "<~8e-5; exponent 2.00",
        lambda: (lambda e, p: (f"{e:.1e}; {', '.join(f'{x:.2f}' for x in p)}", e < 8e-5 and all(close(x, 2.00, 0.005) for x in p)))(
            num(TAB / "lattice_continuum_convergence.txt", r"N=1200 over the three cases: ([\d.e-]+)"),
            nums(TAB / "lattice_continuum_convergence.txt", r"fitted exponent ([\d.]+)")))
    add("scaled residues (N^2/q) W_j vs G_j, ten slowest modes, N=1200", "about 2e-4 per mode",
        lambda: (lambda med, mx: (f"median {max(med):.1e} (largest single mode {max(mx):.1e})", max(med) <= 3e-4 and max(mx) <= 1e-3))(
            nums(TAB / "lattice_continuum_convergence.txt", r"\|\(N\^2/q\)W_j/G_j - 1\| median ([\d.e-]+)"),
            nums(TAB / "lattice_continuum_convergence.txt", r"\|\(N\^2/q\)W_j/G_j - 1\| median [\d.e-]+, max ([\d.e-]+)")))
    add("lattice poles = roots of D_u; residues: closed form vs eigh", "closed forms of App. A",
        lambda: (lambda d: (f"root shift {d['worst']['newton_shift']:.0e}, max|dW|/max|W| {d['worst']['residue_abs_dev_over_max']:.1e}, "
                            f"|sum W/(1-Lambda)-1| {d['worst']['total_probability_minus_one']:.0e}", d["pass"]))(
            jload(TAB / "lattice_closed_form_residues.json")))
    add("lattice (closed-form residues) vs continuum, N=1200", "<~8e-5",
        lambda: (lambda e: (f"{e:.1e}", e < 8e-5))(jload(TAB / "lattice_closed_form_residues.json")["worst"]["continuum_max_rel_error_N1200"]))
    add("MC histogram vs exact lattice density (Fig. 5a)", "agrees over about 4 decades",
        lambda: (lambda m: (f"chi^2/dof = {m.group(1)}/{m.group(2)} (p = {m.group(3)}), {m.group(4)} decades",
                            float(m.group(3)) >= 0.01 and float(m.group(4)) >= 3.5))(
            re.search(r"chi\^2/dof = ([\d.]+)/(\d+) \(p = ([\d.]+);.*\(([\d.]+) decades\)", text(TAB / "dpma_channel_mc_check.txt"))))
    add("lattice capture share at the delayed peak (Fig. 5b)", "about one half",
        lambda: (lambda v: (f"{v:.3f}", 0.4 <= v <= 0.6))(num(TAB / "dpma_channel_mc_check.txt", r"capture share of the lattice density at the delayed peak = ([\d.]+)")))
    add("MC pi_sc vs exact (4x10^5 walkers)", "0.5647 +- 0.0008 vs 0.5652",
        lambda: (lambda t: (t.group(0), close(float(t.group(1)), 0.5647, 5e-5) and close(float(t.group(2)), 0.0008, 5e-5)
                            and close(float(t.group(3)), 0.5652, 5e-5)))(
            re.search(r"Monte Carlo ([\d.]+) \+- ([\d.]+) \(binomial s\.e\.\), exact closed form ([\d.]+)", text(TAB / "dpma_channel_mc_check.txt"))))
    add("theta=0.38 lattice: delayed peak at b=1.5, 2.16, not 2.8", "present, present, absent",
        lambda: ("[2, 2, 1] (incl. parity maximum)" if "[2, 2, 1]" in text(TAB / "dpma_channel_mc_check.txt") else "other",
                 "[2, 2, 1]" in text(TAB / "dpma_channel_mc_check.txt")))
    add("finite gate b_+^(ell), ell = 0.04, 0.02, 0.01", "3.02335, 3.07553, 3.10494",
        lambda: (lambda v: (", ".join(f"{x:.5f}" for x in v), all(close(a, b, 5e-6) for a, b in zip(v, (3.02335, 3.07553, 3.10494)))))(
            [r["b_richardson"] for r in jload(TAB / "dpma_finite_gate_check.json")["results"]]))
    add("quadratic extrapolation ell -> 0", "3.13657",
        lambda: (lambda v: (f"{v:.5f}", close(v, 3.13657, 5e-6)))(jload(TAB / "dpma_finite_gate_check.json")["b_ell_to_zero_quadratic"]))
    add("finite gate: window ratio at its fold vs point sink", "4.5e-5",
        lambda: (lambda a, b: (f"{abs(a - b):.1e}", close(abs(a - b), 4.5e-5, 5e-6)))(
            next(r["R_richardson"] for r in jload(TAB / "dpma_finite_gate_check.json")["results"] if r["ell"] == 0.02),
            jload(TAB / "measurement_window_ratio.json")["R_c"]))
    add("exact finite gate (no mesh): b_+^(0.02)", "3.0755",
        lambda: (lambda v: (f"{v:.5f}", close(v, 3.0755, 5e-5)))(num(TAB / "measurement_window_ratio.txt", r"ell=0\.02\s*: b_\+\^\(ell\)=([\d.]+)")))
    add("Brownian dynamics of Fig. 6: fixed time step (stored run)", "1.5e-5",
        lambda: (lambda v: (f"{v:.1e}", close(v, 1.5e-5, 1e-12)))(jload(DATA / "dpma_brownian_fold_data.json")["dt"]))
    add("half-line Brownian dynamics chi^2/dof", "42.0/42, 45.7/42",
        lambda: (lambda v: (", ".join(f"{x:.1f}/42" for x in v), len(v) == 2 and close(v[0], 42.0, 0.05) and close(v[1], 45.7, 0.05)))(
            nums(TAB / "fig3_near_end.txt", r"chi2/dof vs delta-sink density: ([\d.]+)/42")))
    # --- Sec. VI: proposed measurement
    add("valley and late-peak times at b=2.5 (xi=0.44)", "0.0253296, 0.0462268",
        lambda: (lambda d: (f"{d['tau_v_exact']:.7f}, {d['tau_p_exact']:.7f}", close(d["tau_v_exact"], 0.0253296, 5e-8)
                            and close(d["tau_p_exact"], 0.0462268, 5e-8)))(jload(TAB / "measurement_window_ratio.json")))
    add("R = 1 at b", "2.821 (2.82)",
        lambda: (lambda v: (f"{v:.4f}", close(v, 2.821, 5e-4)))(num(TAB / "measurement_shape_test.txt", r"R = 1 at b = ([\d.]+)")))
    add("two-s.d. test of R>1: trials at b = 1.5, 2, 2.5", "5.0e3, 1.4e4, 9.7e4",
        lambda: (lambda v: (", ".join(f"{x:.3g}" for x in v), len(v) == 3
                            and all(close(a / b, 1.0, 0.02) for a, b in zip(v, (5.0e3, 1.4e4, 9.7e4)))))(
            nums(TAB / "measurement_shape_test.txt", r"N_t\(2 s\.d\., R > 1\) = ([\d.e+]+)")))
    add("two-s.d. test: probability of R>1 at the one-sided 5% level; trials factor for 0.95", "0.64, 2.7",
        lambda: (lambda a, b: (f"{a:.3f}, {b:.2f}", close(a, 0.64, 5e-3) and close(b, 2.7, 0.05)))(
            num(TAB / "measurement_shape_test.txt", r"one-sided 5% level = ([\d.]+)"),
            num(TAB / "measurement_shape_test.txt", r"trials factor for probability 0\.95 = ([\d.]+)")))
    add("at b_+: R_c, f_p, f_v, dR/db", "0.969955, 0.104529, 0.059015, -0.088385",
        lambda: (lambda d: (f"{d['R_c']:.6f}, {d['p']:.6f}, {d['v']:.6f}, {d['dR_db']:.6f}",
                            close(d["R_c"], 0.969955, 5e-7) and close(d["p"], 0.104529, 5e-7) and close(d["v"], 0.059015, 5e-7)
                            and close(d["dR_db"], -0.088385, 5e-7)))(jload(TAB / "measurement_window_ratio.json")))
    add("trials N_t for +-0.25 and +-0.1", "51086, 319287",
        lambda: (lambda d: (f"{d['N_025']:.0f}, {d['N_010']:.0f}", round(d["N_025"]) == 51086 and round(d["N_010"]) == 319287))(
            jload(TAB / "measurement_window_ratio.json")))
    add("other window designs: N_t for +-0.25", "2e4 - 5e5",
        lambda: (lambda v: (f"{min(v):.2g} - {max(v):.2g}", 1.5e4 <= min(v) and max(v) <= 6e5))(
            [row[8] for row in jload(TAB / "measurement_window_ratio.json")["design_sensitivity"]]))
    add("mean at b_+: <tau>, s.d., d<tau>/db; trials for +-0.25", "0.1121, 0.107, -0.0167; 6.5e2",
        lambda: (lambda t: (f"{t[0]:.4f}, {t[1]:.5f}, {t[2]:.5f}; {t[3]:.0f}", close(t[0], 0.1121, 5e-5) and close(t[1], 0.107, 6e-4)
                            and close(t[2], -0.0167, 5e-5) and close(t[3], 650, 10)))(
            (num(TAB / "measurement_shape_test.txt", r"mean = ([\d.]+) \(closed"),
             num(TAB / "measurement_shape_test.txt", r"s\.d\. = ([\d.]+)"),
             num(TAB / "measurement_shape_test.txt", r"d<tau>/db = (-[\d.]+)"),
             num(TAB / "measurement_shape_test.txt", r"to \+-0\.25: ([\d.e+]+) trials"))))
    add("late peak above valley at b = b_+ - 0.25", "about 1%",
        lambda: (lambda v: (f"{v:.2f}%", close(v, 1.0, 0.1)))(num(TAB / "measurement_shape_test.txt", r"b = 2\.8866:.*\n.*exceeds the valley by ([\d.]+)%")))
    add("early peak time (xi=0.44, b=2.5)", "0.003",
        lambda: (lambda v: (f"{v:.4f}", close(v, 0.003, 5e-4)))(num(TAB / "measurement_shape_test.txt", r"early peak at tau = ([\d.]+)")))
    add("kappa ell L/D at b_c(0.381) and b_+", "4.3, 6.3",
        lambda: (lambda a, b: (f"{a:.2f}, {b:.2f}", close(a, 4.3, 0.05) and close(b, 6.3, 0.05)))(
            num(TAB / "measurement_shape_test.txt", r"2 b: ([\d.]+) for b_c"), num(TAB / "measurement_shape_test.txt", r"; ([\d.]+) for b_\+")))
    add("gate of the example (ell = 0.05, exact gate, no mesh): upper fold b_+", "3.0005 (3.00)",
        lambda: (lambda v: (f"{v:.5f}", close(v, 3.0005, 5e-5)))(
            num(TAB / "measurement_shape_test.txt", r"upper fold b_\+\^\(ell\) = ([\d.]+)")))
    add("gate of the example (ell = 0.05): R = 1 at b", "2.714",
        lambda: (lambda v: (f"{v:.4f}", close(v, 2.714, 5e-4)))(
            num(TAB / "measurement_shape_test.txt", r"R\^\(ell\) = 1 at b = ([\d.]+)")))
    add("gate of the example (ell = 0.05), b = 2: R, trials (two-s.d. test), mean", "1.098, 1.6e4, 0.1346",
        lambda: (lambda r, n, t: (f"{r:.4f}, {n:.3g}, {t:.5f}", close(r, 1.098, 5e-4) and close(n / 1.6e4, 1.0, 0.02)
                                  and close(t, 0.1346, 5e-5)))(
            num(TAB / "measurement_shape_test.txt", r"b = 2\.0: R\^\(ell\) = ([\d.]+)"),
            num(TAB / "measurement_shape_test.txt", r"N_t\^\(ell\)\(2 s\.d\., R > 1\) = ([\d.e+]+)"),
            num(TAB / "measurement_shape_test.txt", r"mean absorption time at b = 2\.0: ([\d.]+)")))
    add("colloid example: L^2/(2D), ell^2/(2D), kappa at the finite-gate fold, p per frame", "28 s, 0.07 s, 2.1/s, 0.01",
        lambda: (lambda a, b, c, d: (f"{a:.1f} s, {b:.3f} s, {c:.2f}/s, {d:.4f}", close(a, 28, 0.5) and close(b, 0.07, 0.005)
                                     and close(c, 2.1, 0.05) and close(d, 0.01, 0.002)))(
            num(TAB / "measurement_shape_test.txt", r"L\^2/\(2D\) = ([\d.]+) s"),
            num(TAB / "measurement_shape_test.txt", r"ell\^2/\(2D\) = ([\d.]+) s"),
            num(TAB / "measurement_shape_test.txt", r"needs kappa = ([\d.]+) 1/s"),
            num(TAB / "measurement_shape_test.txt", r"1 - exp\(-kappa dt\) = ([\d.]+)")))
    add("colloid example (finite gate): mean trial, accumulated time at b=2", "3.8 s, 17 h",
        lambda: (lambda a, b: (f"{a:.2f} s, {b:.1f} h", close(a, 3.8, 0.05) and close(b, 17, 0.5)))(
            num(TAB / "measurement_shape_test.txt", r"mean trial duration at b = 2: ([\d.]+) s"),
            num(TAB / "measurement_shape_test.txt", r"take ([\d.]+) h")))
    # --- Table II rows not listed above
    add("Table II: jump, I/J quadrature, J identity", "< 6e-14, < 3e-16, < 6e-15",
        lambda: (lambda a, b, c: (f"{a:.1e}, {b:.1e}, {c:.1e}", a < 6e-14 and b < 3e-16 and c < 6e-15))(
            num(TAB / "dpma_verification_table.txt", r"secular rhs\| = ([\d.e+-]+)"),
            num(TAB / "dpma_verification_table.txt", r"quadrature\| = ([\d.e+-]+)"),
            num(TAB / "dpma_verification_table.txt", r"D_k/\(4b\)\| = ([\d.e+-]+)")))
    add("Table II: fold algebra on random rate sets", "466 sets, residual < 7.2e-15",
        lambda: (lambda d: (f"{d['accepted_constructive_checks']} sets, {d['maximum_three_mode_double_zero_residual']:.2e}",
                            d["accepted_constructive_checks"] == 466 and d["maximum_three_mode_double_zero_residual"] < 7.2e-15))(
            jload(TAB / "finite_mode_algebra.json")))
    add("pytest: exact identities", "all pass",
        lambda: (lambda t: (t.strip().splitlines()[-1] if t.strip() else "no output", bool(re.search(r"\b6 passed\b", t)) and "failed" not in t))(
            text(LOGS / "test_dpma_identities.log")))
    # --- regenerated data and figures vs the stored versions
    for name in ("dpma_boundary_generated.csv", "dpma_fold_scaling_generated.csv", "dpma_offgate_phase.csv",
                 "dpma_offgate_dense.csv", "dpma_offgate_phase.json", "dpma_endpoint_law.txt"):
        add(f"regenerated {name} == data/{name}", "identical",
            lambda name=name: (lambda same: ("identical" if same else "differs", same))(
                (TAB / name).read_bytes() == (DATA / name).read_bytes()))
    add("regenerated dpma_cusp_endpoint_loci.csv vs stored", "agree to 1e-9",
        lambda: (lambda d: (f"max relative difference {d:.1e}", d < 1e-9))(
            max(abs(float(a[k]) - float(b[k])) / max(abs(float(b[k])), 1e-300)
                for a, b in zip(rows(TAB / "dpma_cusp_endpoint_loci.csv"), rows(DATA / "dpma_cusp_endpoint_loci.csv"))
                for k in ("xi", "b", "tau", "T4_S4_over_Phi", "unfolding_determinant_logb_xi"))))
    for fig in FIGURES:
        add(f"{fig}.pdf == figures/{fig}.pdf", "identical",
            lambda fig=fig: (lambda same: ("identical" if same else "differs", same))(
                pdf_equal_except_date(FIG / f"{fig}.pdf", REF_FIG / f"{fig}.pdf")))
    return C


def full_checks():
    C = []
    add = lambda label, quoted, fn: C.append((label, quoted, fn))
    add("rerun lattice MC: histogram identical to stored", "identical",
        lambda: (lambda t: ("identical" if "identical to the stored run: True" in t else "differs",
                            "identical to the stored run: True" in t))(text(TAB / "dpma_channel_mc.txt")))
    add("rerun Brownian dynamics (Fig. 6): histograms identical", "identical",
        lambda: (lambda t: (f"{t.count('identical to the stored run: True')} of 2", t.count("identical to the stored run: True") == 2))(
            text(TAB / "dpma_brownian_fold.txt")))
    add("rerun half-line Brownian dynamics (Fig. 3c): samples identical", "identical",
        lambda: (lambda same: ("identical" if same else "differs", same))(
            __import__("numpy").load(GEN / "dpma_endpoint_bd.npz")["T0"].tobytes()
            == __import__("numpy").load(DATA / "dpma_endpoint_bd.npz")["T0"].tobytes()
            and __import__("numpy").load(GEN / "dpma_endpoint_bd.npz")["T1"].tobytes()
            == __import__("numpy").load(DATA / "dpma_endpoint_bd.npz")["T1"].tobytes()))
    add("gate width / time step: late peak within two bins", "within two bins",
        lambda: (lambda v: (f"largest offset {v:.2f} bins", v <= 2.0))(
            num(TAB / "dpma_bd_convergence.txt", r"largest late-peak offset ([\d.]+) bins")))
    for fig in ("fig5_lattice_checks", "fig6_brownian_dynamics"):
        add(f"{fig} from the regenerated data == figures/{fig}.pdf", "identical",
            lambda fig=fig: (lambda same: ("identical" if same else "differs", same))(
                pdf_equal_except_date(FIG / f"{fig}_regenerated.pdf", REF_FIG / f"{fig}.pdf")))
    if importlib.util.find_spec("flint") is not None:
        newton_fields = ("box", "interval_newton_image", "normalized_fourth_margin",
                         "normalized_unfolding_determinant_logB_X", "full_jacobian_determinant", "density", "m4")
        add("rerun Arb interval Newton: enclosures == stored certificate", "identical",
            lambda: (lambda d, s: ("identical" if d["status"] == "PASS" and all(
                d[k][f] == s[k][f] for k in ("boundary_side_cusp", "domain_side_cusp") for f in newton_fields) else "differs",
                d["status"] == "PASS" and all(d[k][f] == s[k][f] for k in ("boundary_side_cusp", "domain_side_cusp")
                                              for f in newton_fields)))(
                jload(TAB / "halfline_cusp_interval_newton.json"), jload(DATA / "dpma_interval_certificates.json")))
        kraw_fields = ("box", "krawczyk_image", "box_enclosures", "sign_checks", "status")
        add("rerun Arb Krawczyk: enclosures == stored certificate", "identical",
            lambda: (lambda d, s: (lambda same: ("identical" if same else "differs", same))(
                d["status"] == "VERIFIED" and len(d["certificates"]) == len(s["certificates"]) and all(
                    a[f] == b[f] for a, b in zip(d["certificates"], s["certificates"]) for f in kraw_fields)))(
                jload(TAB / "halfline_cusp_krawczyk.json"), jload(DATA / "dpma_halfline_cusp_interval_certificate.json")))
    if shutil.which("lean") is not None:
        add("core-Lean modules of formal/ (cusp Jacobian, certificate decimals)", "proved",
            lambda: (lambda t: ("PASS" if "result: PASS" in t else "FAIL", "result: PASS" in t))(text(TAB / "formal_cusp_check.txt")))
    return C


def evaluate(checks, partial=False):
    """Run the comparisons.  In a partial run (--only, --checks-only) a comparison whose input file
    has not been produced yet is reported as NOT RUN instead of FAIL."""
    lines, n_pass, n_fail, n_not_run = [], 0, 0, 0
    width = max(len(c[0]) for c in checks)
    for label, quoted, fn in checks:
        try:
            shown, ok = fn()
            status = "PASS" if ok else "FAIL"
        except FileNotFoundError as exc:
            shown = f"input not produced: {Path(exc.filename or '').name}"
            status = "NOT RUN" if partial else "FAIL"
        except Exception as exc:  # parse failure
            shown, status = f"error: {type(exc).__name__}: {exc}"[:90], "FAIL"
        n_pass += status == "PASS"
        n_fail += status == "FAIL"
        n_not_run += status == "NOT RUN"
        lines.append(f"{status:<7} {label:<{width}}  paper: {quoted}  |  computed: {shown}")
    return lines, n_pass, n_fail, n_not_run


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", action="store_true", help="also rerun the simulations and certificates")
    ap.add_argument("--only", nargs="+", metavar="STEP", help="run only these steps")
    ap.add_argument("--list", action="store_true", help="list the steps and exit")
    ap.add_argument("--checks-only", action="store_true", help="skip the computations; compare outputs")
    args = ap.parse_args()
    steps = STEPS + (FULL_STEPS if args.full else [])
    if args.list:
        for name, cmd, what in STEPS:
            print(f"{name:38s} {what}")
        print("--full adds:")
        for name, cmd, what in FULL_STEPS:
            print(f"{name:38s} {what}")
        return 0
    if args.only:
        known = {s[0] for s in STEPS + FULL_STEPS}
        unknown = [n for n in args.only if n not in known]
        if unknown:
            ap.error(f"unknown step(s): {unknown}")
        steps = [s for s in STEPS + FULL_STEPS if s[0] in args.only]
    t0 = time.time()
    results = {}
    if not args.checks_only:
        print(f"Running {len(steps)} steps (python {sys.version.split()[0]}); logs in outputs/logs/", flush=True)
        results = run_steps(steps)
    failed_steps = [n for n, (status, _) in results.items() if status not in ("ok", "SKIP")]
    checks = check_list() + (full_checks() if args.full else [])
    lines, n_pass, n_fail, n_not_run = evaluate(checks, partial=bool(args.only or args.checks_only))
    header = [
        "=" * 100,
        "Comparison with the values quoted in the paper",
        "=" * 100,
    ]
    footer = [
        "-" * 100,
        f"steps run: {len(results)}; failed steps: {failed_steps if failed_steps else 'none'}; "
        f"skipped: {[n for n, (s, _) in results.items() if s == 'SKIP'] or 'none'}",
        f"checks: {n_pass} PASS, {n_fail} FAIL"
        + (f", {n_not_run} NOT RUN (inputs of steps not run)" if n_not_run else "")
        + f"; total time {time.time() - t0:.0f} s",
    ]
    report = "\n".join(header + lines + footer)
    print(report)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.txt").write_text(report + "\n")
    return 1 if (n_fail or failed_steps) else 0


if __name__ == "__main__":
    sys.exit(main())
