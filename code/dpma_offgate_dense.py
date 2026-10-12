#!/usr/bin/env python3
"""Dense continuation of the theta = 1/2 fold lines b_-(xi), b_+(xi) (data of Fig. 4).

dpma_offgate_phase.py stores 9 fold points per line (dpma_offgate_phase.csv).  This script
computes and stores a dense version of both fold lines with the solver functions of
dpma_offgate_phase.py:

    fold system   S_1 = S_2 = 0 at theta = 1/2  (P.fold_residual, normalised: tau S1/Phi, tau^2 S2/Phi)
    spectrum      P.modes -> P.antipodal_modes, k <= 160 (P.K_DEEP)
    solver        P.damped_newton / P.solve_fold / P.solve_cusp, tolerances P.NEWTON_TOL = 2e-10,
                  P.SMALL_B_NEWTON_TOL = 5e-8 (used by solve_fold when b < 5e-4)

Continuation
  1. cusp           S_1 = S_2 = S_3 = 0 (P.solve_cusp), left cusp; the right one is its mirror.
  2. cusp segment   tau fixed, unknowns (log b, xi).  At the cusp the fold curve is smooth in tau
                    while xi - xi_* ~ (tau - tau_*)^2, so a uniform tau grid (dtau ~ 5e-5) is
                    automatically refined in xi where the two lines turn; six extra points at
                    |tau - tau_*| = dtau/2^k (k = 1..6) bring the innermost points to
                    xi - xi_* ~ 3e-10, |b - b_*| ~ 6e-8.  The segment runs from the stored
                    lower fold at xi = 0.385 through the cusp to the stored upper fold at
                    xi = 0.385 (their taus are read from the stored CSV).
  3. xi segments    xi fixed, unknowns (log b, log tau) (P.solve_fold), step 0.0025 from 0.385:
                    upper line to xi = 1/2 (where b_+ = b_c(1/2)), lower line to xi = 0.44 and
                    then step 0.001 while rows with b < 5e-4 meet the small-b
                    tolerance 5e-8 with a measured round-off floor <= 2 x 5e-8 (stops at 0.44).
                    The stored xi = 0.38, 0.382 are re-solved with xi fixed from guesses
                    interpolated on the cusp segment and inserted.
  4. mirror         xi -> 1 - xi (exact symmetry of the theta = 1/2 problem); residuals are
                    re-evaluated at the mirrored xi.

Checks (log section 4): the 9 sparse points per line, residuals, cusp meeting, where the lower
line stops, quartic law of the upper line at xi -> 1/2, grid, and spectral cutoff.

Outputs
  outputs/tables/dpma_offgate_dense.csv   branch, xi, b, tau, residual, param
  outputs/tables/dpma_offgate_dense.txt   run log and verification numbers

Run from anywhere:  python code/dpma_offgate_dense.py   (single process)
"""
from __future__ import annotations

import csv
import math
import time

import numpy as np

import dpma_offgate_phase as P          # fold solver of the two-peak region
from paths import DATA, TABLES, rel

STORED_CSV = DATA / "dpma_offgate_phase.csv"
OUT_CSV = TABLES / "dpma_offgate_dense.csv"
OUT_TXT = TABLES / "dpma_offgate_dense.txt"

THETA = 0.5
K = P.K_DEEP                            # 160
K_CHECK = P.K_CHECK                     # 120
K_HIGH = 200.0                          # cutoff convergence check
XI_STEP = 0.0025                        # xi-fixed continuation step (<= 0.003)
XI_STEP_TAIL = 0.001                    # lower line beyond xi = 0.44
DTAU = 5.0e-5                           # tau spacing on the cusp segment
FLOOR_FACTOR = 2.0                      # accept b < 5e-4 rows only if round-off floor <= 2 x 5e-8
N_REFINE = 6                            # extra points at |tau - tau*| = DTAU/2^k, k = 1..N_REFINE
XI_JOIN = 0.385                         # where the cusp segment hands over to xi stepping
B_C_REF = 3.0764323604                  # b_c(1/2), release at the sink (dpma_boundary_generated.csv)
C_B_REF = 4352.66                       # b_+ - b_c ~ C_b (xi - 1/2)^4 (dpma_start_asymptotics.py)

LOG: list[str] = []


def say(s: str = "") -> None:
    print(s, flush=True)
    LOG.append(s)


# ------------------------------------------------------------------ helpers around the solver
def residual_at(xi: float, b: float, tau: float, kmax: float = K) -> float:
    return float(np.linalg.norm(P.fold_residual(np.log([b, tau]), THETA, xi, kmax), ord=np.inf))


def residual_floor(xi: float, b: float, tau: float, rel: float = 1.0e-7, n: int = 41) -> float:
    """Empirical round-off floor of the fold residual at a solution.

    log b is shifted by k*rel (k = -20..20); the exact residual is smooth (quadratic) on this
    scale, so the max deviation from a quadratic fit measures the arithmetic noise (mainly the
    root-bracketing tolerance of antipodal_modes amplified by the cancellation in Phi).
    """
    z0 = np.log([b, tau])
    ks = np.arange(-(n // 2), n // 2 + 1)
    rs = np.array([P.fold_residual(z0 + np.array([k * rel, 0.0]), THETA, xi, K) for k in ks])
    return max(float(np.max(np.abs(rs[:, i] - np.polyval(np.polyfit(ks, rs[:, i], 2), ks))))
               for i in range(2))


def b_uncertainty(xi: float, b: float, tau: float, floor: float) -> float:
    """Relative uncertainty of b implied by a residual floor: floor * |row 0 of J^-1|_1."""
    z0, h = np.log([b, tau]), 1.0e-4
    J = np.column_stack([(P.fold_residual(z0 + h * u, THETA, xi, K)
                          - P.fold_residual(z0 - h * u, THETA, xi, K)) / (2 * h) for u in np.eye(2)])
    return float(floor * np.sum(np.abs(np.linalg.inv(J)[0])))


def solve_tau_fixed(tau: float, guess_logb: float, guess_xi: float):
    """Fold point at fixed tau: unknowns (log b, xi); regular through the cusp."""
    logtau = math.log(tau)
    z, res = P.damped_newton(
        lambda v: P.fold_residual(np.array([v[0], logtau]), THETA, float(v[1]), K),
        [guess_logb, guess_xi], tolerance=P.NEWTON_TOL)
    return float(z[1]), float(math.exp(z[0])), tau, float(res)


def secant(points, x_new):
    """Linear predictor in (log b, log tau) from the last two (xi, b, tau) points."""
    (x1, b1, t1), (x2, b2, t2) = points[-2], points[-1]
    f = (x_new - x2) / (x2 - x1)
    return (math.exp(math.log(b2) + f * (math.log(b2) - math.log(b1))),
            math.exp(math.log(t2) + f * (math.log(t2) - math.log(t1))))


def load_stored():
    rows = [r for r in csv.DictReader(open(STORED_CSV, newline="")) if float(r["theta"]) == 0.5]
    out = {"lower": {}, "upper": {}, "cusp": []}
    for r in rows:
        rec = (float(r["xi"]), float(r["b"]), float(r["tau"]), float(r["residual_inf"]))
        if r["branch"] == "cusp":
            out["cusp"].append(rec)
        else:
            out[r["branch"]][round(rec[0], 10)] = rec
    out["cusp"].sort()
    return out


# ------------------------------------------------------------------------------------- main
def main() -> None:
    t_start = time.perf_counter()
    stored = load_stored()
    xi_s_ref, b_s_ref, tau_s_ref, _ = stored["cusp"][0]
    say("Dense theta = 1/2 fold lines of the two-peak region (S1 = S2 = 0)")
    say(f"solver module {rel(P.__file__)}; k <= {K:g}; "
        f"Newton tol {P.NEWTON_TOL:g} (small b: {P.SMALL_B_NEWTON_TOL:g})")

    # 1. cusp -------------------------------------------------------------------------------
    cusp = P.solve_cusp(THETA, (5.81630219, 0.01607957, 0.3788069561))
    say("")
    say("1. cusp (S1 = S2 = S3 = 0)")
    say(f"   xi* = {cusp['xi']:.12f}  b* = {cusp['b']:.12f}  tau* = {cusp['tau']:.12f}  "
        f"|res| = {cusp['residual']:.1e}")
    say(f"   vs stored cusp: |dxi| = {abs(cusp['xi'] - xi_s_ref):.1e}, "
        f"|db| = {abs(cusp['b'] - b_s_ref):.1e}, |dtau| = {abs(cusp['tau'] - tau_s_ref):.1e}")

    # 2. cusp segment, tau fixed -------------------------------------------------------------
    t0 = time.perf_counter()
    tau_lo_end = stored["lower"][XI_JOIN][2]
    tau_up_end = stored["upper"][XI_JOIN][2]
    n_lo = int(math.ceil((cusp["tau"] - tau_lo_end) / DTAU))
    n_up = int(math.ceil((tau_up_end - cusp["tau"]) / DTAU))
    # extra geometric refinement next to the cusp: |tau - tau*| = DTAU/64, ..., DTAU/2
    refine = DTAU * 2.0 ** -np.arange(N_REFINE, 0, -1)
    taus_lo = np.r_[cusp["tau"] - refine, np.linspace(cusp["tau"], tau_lo_end, n_lo + 1)[1:]]
    taus_up = np.r_[cusp["tau"] + refine, np.linspace(cusp["tau"], tau_up_end, n_up + 1)[1:]]
    # check point exactly at the stored tau*: must return the stored cusp
    xi_c0, b_c0, _, res_c0 = solve_tau_fixed(tau_s_ref, math.log(cusp["b"]), cusp["xi"])
    seg = {}
    for name, taus in (("lower", taus_lo), ("upper", taus_up)):
        hist = [(cusp["tau"], math.log(cusp["b"]), cusp["xi"])]
        pts = []
        for tau in taus:
            if len(hist) == 1:
                guess = hist[-1][1:]
            else:              # secant predictor in tau (steps are non-uniform)
                (t1, l1, x1), (t2, l2, x2) = hist[-2], hist[-1]
                f = (tau - t2) / (t2 - t1)
                guess = (l2 + f * (l2 - l1), x2 + f * (x2 - x1))
            xi, b, _, res = solve_tau_fixed(tau, *guess)
            hist.append((tau, math.log(b), xi))
            pts.append((xi, b, tau, res))
        seg[name] = pts
    t_seg = time.perf_counter() - t0
    say("")
    say(f"2. cusp segment (tau fixed, unknowns log b and xi): dtau = {DTAU:g}, plus |tau - tau*| = "
        f"dtau/2^k (k = 1..{N_REFINE}); "
        f"{len(seg['lower'])} lower + {len(seg['upper'])} upper points; {t_seg:.1f} s")
    say(f"   solve at the stored tau* = {tau_s_ref:.12f}: xi = {xi_c0:.12f}, b = {b_c0:.12f}; "
        f"vs stored cusp |dxi| = {abs(xi_c0 - xi_s_ref):.1e}, |db| = {abs(b_c0 - b_s_ref):.1e}")
    for name in ("lower", "upper"):
        first, last = seg[name][0], seg[name][-1]
        xs, bs, ts, _ = stored[name][XI_JOIN]
        say(f"   {name}: first point xi - xi* = {first[0] - cusp['xi']:.2e}, "
            f"b - b* = {first[1] - cusp['b']:+.2e}; end at stored tau({XI_JOIN}) -> "
            f"xi = {last[0]:.12f} (|dxi| = {abs(last[0] - xs):.1e}), |db| = {abs(last[1] - bs):.1e}")
        assert abs(last[0] - xs) < 1e-8 and abs(last[1] - bs) < 1e-6
    # hand over to xi stepping: the end point is replaced by the xi-fixed solve at XI_JOIN
    for name in ("lower", "upper"):
        seg[name] = seg[name][:-1]
        xs = [p[0] for p in seg[name]]
        assert all(np.diff(xs) > 0), f"{name} cusp segment not monotone in xi"

    # 3. xi segments ------------------------------------------------------------------------
    t0 = time.perf_counter()
    grid = [round(XI_JOIN + XI_STEP * i, 10) for i in range(int(round((0.5 - XI_JOIN) / XI_STEP)) + 1)]
    assert grid[-1] == 0.5 and all(round(x, 10) in grid for x in stored["upper"] if x >= XI_JOIN)
    xi_rows = {"lower": [], "upper": []}
    tail = {}
    for name in ("upper", "lower"):
        hist = [(p[0], p[1], p[2]) for p in seg[name][-2:]]
        targets = grid if name == "upper" else [x for x in grid if x <= 0.44 + 1e-12]
        for xi in targets:
            guess = secant(hist, xi)
            f = P.solve_fold(THETA, xi, guess, name)
            xi_rows[name].append((xi, f.b, f.tau, f.residual))
            hist.append((xi, f.b, f.tau))
        if name == "lower":
            # tail beyond 0.44: continue at step 0.001 while the residual is not noise-dominated
            xi = 0.44
            while True:
                xi = round(xi + XI_STEP_TAIL, 10)
                guess = secant(hist, xi)
                try:
                    f = P.solve_fold(THETA, xi, guess, name)
                except RuntimeError as err:
                    tail["failed_xi"] = xi
                    tail["error"] = str(err).split(":")[0]
                    m = str(err)
                    try:            # last Newton iterate, printed by P.damped_newton
                        z = np.array([float(v) for v in m[m.index("z=[") + 3:m.index("]")].split()])
                        tail["stuck_res"] = residual_at(xi, *np.exp(z))
                        tail["stuck_floor"] = residual_floor(xi, *np.exp(z))
                    except ValueError:
                        tail["stuck_res"] = tail["stuck_floor"] = float("nan")
                    break
                fl = residual_floor(xi, f.b, f.tau)
                if fl > FLOOR_FACTOR * P.SMALL_B_NEWTON_TOL:
                    tail["failed_xi"] = xi
                    tail["error"] = (f"Newton reached |res| = {f.residual:.1e} < {P.SMALL_B_NEWTON_TOL:g}, but the "
                                     f"round-off floor exceeds {FLOOR_FACTOR:g} x tolerance")
                    tail["stuck_res"], tail["stuck_floor"] = f.residual, fl
                    tail["stuck_unc"] = b_uncertainty(xi, f.b, f.tau, fl)
                    break
                xi_rows[name].append((xi, f.b, f.tau, f.residual))
                hist.append((xi, f.b, f.tau))
    # stored xi inside the cusp segment (0.38, 0.382): xi-fixed re-solves from interpolated guesses
    extra = {"lower": [], "upper": []}
    for name in ("lower", "upper"):
        sx = np.array([p[0] for p in seg[name]])
        slb = np.log([p[1] for p in seg[name]])
        slt = np.log([p[2] for p in seg[name]])
        for xi in sorted(x for x in stored[name] if x < XI_JOIN):
            guess = (math.exp(np.interp(xi, sx, slb)), math.exp(np.interp(xi, sx, slt)))
            f = P.solve_fold(THETA, xi, guess, name)
            extra[name].append((xi, f.b, f.tau, f.residual))
    t_xi = time.perf_counter() - t0
    last_lower = xi_rows["lower"][-1]
    say("")
    say(f"3. xi segments (xi fixed, unknowns log b and log tau): step {XI_STEP:g} from {XI_JOIN}; "
        f"{t_xi:.1f} s")
    say(f"   upper: {len(xi_rows['upper'])} points, xi = {XI_JOIN} ... 0.5")
    say(f"   lower: {len(xi_rows['lower'])} points, xi = {XI_JOIN} ... {last_lower[0]:g} "
        f"(then step {XI_STEP_TAIL:g} attempted)")
    say(f"   re-solved stored xi inside the cusp segment: "
        f"{', '.join(f'{x:g}' for x, *_ in extra['lower'])}")

    # assemble left half (+ cusp at the start of each line), then mirror ---------------------
    left = {}
    for name in ("lower", "upper"):
        pts = [(p[0], p[1], p[2], p[3], "tau") for p in seg[name]]
        pts += [(p[0], p[1], p[2], p[3], "xi") for p in xi_rows[name] + extra[name]]
        pts.sort(key=lambda p: p[0])
        left[name] = pts
    rows = [("cusp", cusp["xi"], cusp["b"], cusp["tau"], cusp["residual"], "cusp")]
    xr = 1.0 - cusp["xi"]
    rows.append(("cusp", xr, cusp["b"], cusp["tau"], residual_at(xr, cusp["b"], cusp["tau"]), "mirror"))
    mirror_res = []
    for name in ("lower", "upper"):
        rows += [(name, x, b, t, r, p) for x, b, t, r, p in left[name]]
        for x, b, t, r, p in reversed(left[name]):
            if x == 0.5:
                continue
            xm = 1.0 - x
            rm = residual_at(xm, b, t)
            mirror_res.append(abs(rm - r))
            rows.append((name, xm, b, t, rm, "mirror"))
    with OUT_CSV.open("w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["branch", "xi", "b", "tau", "residual", "param"])
        for name, x, b, t, r, p in rows:
            w.writerow([name, f"{x:.15g}", f"{b:.15g}", f"{t:.15g}", f"{r:.3e}", p])

    # 4. verification -----------------------------------------------------------------------
    say("")
    say("4. verification")
    say("   (a) the 9 stored fold points per line vs the dense rows at the same xi")
    say(f"       {'xi':>6} {'branch':>6} {'b stored':>16} {'b dense':>16} {'|db|':>8} {'|db|/b':>8} {'|dtau|':>8}")
    worst = {"lower": 0.0, "upper": 0.0}
    worst_rel = {"lower": 0.0, "upper": 0.0}
    for name in ("lower", "upper"):
        dense = {round(p[0], 10): p for p in left[name] if p[4] == "xi"}
        for key in sorted(stored[name]):
            xs, bs, ts, _ = stored[name][key]
            xd, bd, td, rd, _ = dense[key]
            db, dt = abs(bd - bs), abs(td - ts)
            worst[name] = max(worst[name], db)
            worst_rel[name] = max(worst_rel[name], db / bs)
            say(f"       {xs:6.3f} {name:>6} {bs:16.12g} {bd:16.12g} {db:8.1e} {db / bs:8.1e} {dt:8.1e}")
            assert db < 1e-6, (name, xs, db)
    say(f"       max |db|: lower {worst['lower']:.1e}, upper {worst['upper']:.1e} "
        f"(max relative {max(worst_rel.values()):.1e}); criterion |db| < 1e-6: PASS")

    say("   (b) residuals |(tau S1/Phi, tau^2 S2/Phi)|_inf of the stored dense rows")
    for name in ("lower", "upper"):
        big = [p for p in left[name] if p[1] >= 5e-4]
        small = [p for p in left[name] if p[1] < 5e-4]
        say(f"       {name}: max {max(p[3] for p in big):.1e} over {len(big)} rows with b >= 5e-4"
            + (f"; max {max(p[3] for p in small):.1e} over {len(small)} rows with b < 5e-4 "
               f"(tolerance {P.SMALL_B_NEWTON_TOL:g})" if small else ""))
        assert max(p[3] for p in big) < P.NEWTON_TOL
        assert all(p[3] < P.SMALL_B_NEWTON_TOL for p in small)
    say(f"       mirror rows: |res(1 - xi) - res(xi)| <= {max(mirror_res):.1e}")

    say("   (c) the two lines meet at the stored cusp")
    for name in ("lower", "upper"):
        p = left[name][0]
        say(f"       innermost {name} point: xi - xi* = {p[0] - xi_s_ref:.2e}, "
            f"|b - b*| = {abs(p[1] - b_s_ref):.2e}")
        assert abs(p[1] - b_s_ref) < 1e-4
    say(f"       tau-fixed solve at tau* hits the stored cusp to |db| = {abs(b_c0 - b_s_ref):.1e} "
        f"(criterion 1e-4: PASS)")

    say("   (d) lower line near xi -> 1/2 (where the dense line stops)")
    say(f"       acceptance for b < 5e-4: Newton |res| < {P.SMALL_B_NEWTON_TOL:g} (small-b tolerance of the solver)"
        f" and measured round-off floor <= {FLOOR_FACTOR:g} x that")
    small = [p for p in left["lower"] if p[1] < 5e-4]
    for x, b, t, r, _ in small:
        fl = residual_floor(x, b, t)
        say(f"       xi = {x:g}: b = {b:.6e}, |res| = {r:.1e}, floor {fl:.1e}, "
            f"round-off uncertainty of b {b_uncertainty(x, b, t, fl):.0e} (relative)")
        assert fl <= FLOOR_FACTOR * P.SMALL_B_NEWTON_TOL
    say(f"       next step xi = {tail['failed_xi']:g}: {tail['error']} (floor {tail['stuck_floor']:.1e}"
        + (f", uncertainty {tail['stuck_unc']:.0e})" if "stuck_unc" in tail else ")"))
    say(f"       -> dense lower line stops at xi = {last_lower[0]:g} (b_- = {last_lower[1]:.2e}); "
        "the sparse table dpma_offgate_phase.csv also ends there.")
    say("       (floor = round-off of the spectral sum: Phi ~ 6e-6 at xi = 0.44 is a cancellation of")
    say("        O(10^2) mode terms, so the 5e-15 root-bracketing tolerance of antipodal_modes is")
    say("        amplified ~10^7; the floor grows ~2-3x per 0.001 in xi.)")
    floors = [(x, residual_floor(x, b, t)) for x, b, t, *_ in left["lower"]
              if round(x, 10) in (0.4, 0.42, 0.43)]
    say("       floor elsewhere: " + ", ".join(f"{f:.0e} at {x:g}" for x, f in floors))
    say("       ratio b_-/asymptotic [e a A^2/(2 delta^6) exp(-A/(2 delta^2)), a = xi, A = a^2 - delta^2]:")
    ratios = []
    for x, b, t, *_ in left["lower"]:
        if x >= 0.42 - 1e-12 and x <= 0.44 + 1e-12:
            d = 0.5 - x
            A = x * x - d * d
            basym = math.e * x * A * A / (2 * d ** 6) * math.exp(-A / (2 * d * d))
            ratios.append((x, b / basym))
    say("       " + ", ".join(f"{r:.4f} ({x:g})" for x, r in ratios))
    assert all(np.diff([r for _, r in ratios]) > 0) and ratios[-1][1] < 1.0

    say("   (e) upper line near xi -> 1/2: quartic flatness b_+ - b_c ~ C_b (xi - 1/2)^4")
    up = {round(p[0], 10): p for p in left["upper"]}
    b_half, tau_half = up[0.5][1], up[0.5][2]
    say(f"       b_+(1/2) = {b_half:.10f} (b_c = {B_C_REF:.10f}, |diff| = {abs(b_half - B_C_REF):.1e}); "
        f"tau = {tau_half:.6f}")
    assert abs(b_half - B_C_REF) < 1e-9
    for x in (0.47, 0.48, 0.49, 0.495, 0.4975):
        d = abs(x - 0.5)
        db = up[x][1] - b_half
        say(f"       xi = {x:.4f}: b_+ - b_c = {db:.6e}; C_b (xi - 1/2)^4 = {C_B_REF * d ** 4:.6e}; "
            f"ratio {db / (C_B_REF * d ** 4):.4f}")
    # Q(d) = (b_+ - b_c)/d^4 on the grid points with d = |xi - 1/2| <= 0.01
    D = np.array([abs(x - 0.5) for x in up if 0.0 < abs(x - 0.5) <= 0.01 + 1e-12])
    Q = np.array([(up[round(0.5 - d, 10)][1] - b_half) / d ** 4 for d in D])
    A = np.column_stack([np.ones_like(D), D, D * D])
    c, *_ = np.linalg.lstsq(A, Q, rcond=None)
    misfit = float(np.max(np.abs(A @ c - Q)))
    A2 = np.column_stack([np.ones_like(D), D * D])
    c2, *_ = np.linalg.lstsq(A2, Q, rcond=None)
    misfit2 = float(np.max(np.abs(A2 @ c2 - Q)))
    say(f"       fit Q = (b_+ - b_c)/d^4 = C0 + C1 d + C2 d^2, d = |xi - 1/2| <= 0.01 ({len(D)} points):")
    say(f"         C0 = {c[0]:.2f} (reference C_b = {C_B_REF}; rel. diff {abs(c[0] - C_B_REF) / C_B_REF:.1e}), "
        f"C1/C0 = {c[1] / c[0]:.2f}, max misfit {misfit:.2f}")
    say(f"         (an even-only fit C0 + C2 d^2 misses by up to {misfit2:.1f}, C0 = {c2[0]:.1f}: the first")
    say(f"          correction is odd, ~ |xi - 1/2|^5, so b_+ - b_c falls {abs(c[1] / c[0]) * 0.005 * 100:.1f}% "
        f"below C_b d^4 at d = 0.005)")
    assert abs(c[0] - C_B_REF) / C_B_REF < 1e-3 and misfit < 1.0

    say("   (f) grid")
    for name in ("lower", "upper"):
        xs = np.array([p[0] for p in left[name]])
        dx = np.diff(np.r_[cusp["xi"], xs])
        assert np.all(dx > 0) and dx.max() <= 0.003 + 1e-12
        bs = np.array([p[1] for p in left[name]])
        # leave-one-out chord deviation (linear interpolation from the two neighbours)
        dev = np.abs(bs[1:-1] - (bs[:-2] + (bs[2:] - bs[:-2]) * (xs[1:-1] - xs[:-2]) / (xs[2:] - xs[:-2])))
        say(f"       {name}: {len(xs)} points on 0 < xi - xi* <= {xs[-1] - cusp['xi']:.4f} "
            f"(left half), xi steps {dx.min():.1e} .. {dx.max():.4f} (<= 0.003)")
        say(f"         max leave-one-out deviation {dev.max():.1e} in b at xi = {xs[1 + int(np.argmax(dev))]:.4f}"
            f" -> chord error of the drawn polyline ~ {dev.max() / 4:.0e} (Fig. 4: 1 pt ~ 0.04 in b)")

    say(f"   (g) spectral cutoff (dense rows use k <= {K:g})")
    r200 = {name: max(residual_at(x, b, t, K_HIGH) for x, b, t, *_ in left[name]) for name in left}
    say(f"       dense rows re-evaluated with k <= {K_HIGH:g}: max residual {r200['lower']:.1e} (lower), "
        f"{r200['upper']:.1e} (upper)")
    say(f"       re-solves from a guess offset by +1% in b and tau (|db|/b):")
    for name, xi in (("lower", 0.38), ("upper", 0.38), ("lower", 0.42), ("lower", 0.44),
                     ("upper", 0.45), ("upper", 0.5)):
        p = {round(q[0], 10): q for q in left[name]}[xi]
        g = (1.01 * p[1], 1.01 * p[2])
        f = {k: P.solve_fold(THETA, xi, g, name, kmax=k) for k in (K_CHECK, K, K_HIGH)}
        say(f"       {name} xi = {xi:g}: k160 vs dense {abs(f[K].b - p[1]) / p[1]:.1e}; "
            f"k200 vs k160 {abs(f[K_HIGH].b - f[K].b) / p[1]:.1e}; "
            f"k120 vs k160 {abs(f[K_CHECK].b - f[K].b) / p[1]:.1e}")
    say("       (k <= 120 is not enough only at the shortest fold times, xi -> 0.44 on the lower line;")
    say("        k <= 160 is converged there.)")

    n_rows = len(rows)
    say("")
    say(f"wrote {rel(OUT_CSV)}: {n_rows} rows "
        f"(param: cusp = S1=S2=S3=0 solve, tau = tau-fixed solve, xi = xi-fixed solve, "
        f"mirror = xi -> 1 - xi copy with residual re-evaluated)")
    say(f"total runtime {time.perf_counter() - t_start:.1f} s (single process)")
    OUT_TXT.write_text("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
