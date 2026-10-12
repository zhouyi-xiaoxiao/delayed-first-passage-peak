#!/usr/bin/env python3
"""Threshold curve b_c(theta) for release at the sink (xi = theta).

With signed spectral moments S_m(b,tau)=sum_j G_j mu_j^m exp(-mu_j tau), Phi'=-S_1 and
Phi''=S_2, the finite-time peak is born or destroyed at S_1(b_c,tau_c)=S_2(b_c,tau_c)=0,
subject to the nondegeneracy conditions S3 != 0 and d_b S1 != 0. This script (1) solves the two
fold equations and continues their solution in theta, (2) checks selected values against an
independently computed reference table (bisection on the valley/peak count), and (3) supports the
near-end law b_c ~ 0.7890261736/min(theta,1-theta).

Continuum: Phi_{xi,theta}(tau;b)=sum_j G_j e^{-2 w_j^2 tau}, w_j>0 roots of
  M_theta(w;b)=w sin2w + b sin(2 theta w) sin(2(1-theta) w)=0; start xi=theta.
Writes outputs/tables/dpma_boundary_generated.csv and
outputs/tables/dpma_saddle_node_bc_theta.txt.
"""
from __future__ import annotations
import argparse
import csv
import math
from pathlib import Path
import numpy as np

from paths import TABLES

OUT = []
def say(s=""):
    print(s, flush=True); OUT.append(s)

def M_theta(w, th, b):
    return w*math.sin(2*w) + b*math.sin(2*th*w)*math.sin(2*(1-th)*w)

def roots(th, b, wmax=40.0, step=0.0012):
    ws = []; w0 = 1e-4; f0 = M_theta(w0, th, b); w = w0 + step
    while w <= wmax:
        f1 = M_theta(w, th, b)
        if f0*f1 < 0:
            a, fa, bb = w0, f0, w
            for _ in range(60):
                m = 0.5*(a+bb); fm = M_theta(m, th, b)
                if fa*fm <= 0: bb = m
                else: a, fa = m, fm
            ws.append(0.5*(a+bb))
        w0, f0 = w, f1; w += step
    out = []
    for r in ws:
        if not out or abs(r-out[-1]) > 1e-3: out.append(r)
    return out

def G_affected(w, xi, th, b):
    k = 2*w
    phi = math.sin(k*(1-th))*math.sin(k*xi) if xi <= th else math.sin(k*th)*math.sin(k*(1-xi))
    I = (math.sin(k*(1-th))*(1-math.cos(k*th)) + math.sin(k*th)*(1-math.cos(k*(1-th))))/k
    J = (math.sin(k*(1-th))**2)*(th/2 - math.sin(2*k*th)/(4*k)) + \
        (math.sin(k*th)**2)*((1-th)/2 - math.sin(2*k*(1-th))/(4*k))
    return 2*w*w*phi*I/J

def G_node(n, xi):
    return n*math.pi*(1-(-1)**n)*math.sin(n*math.pi*xi)

def phi_vals(taus, xi, th, b, max_modes=None):
    ws = roots(th, b)
    if max_modes is not None:  # keep first K AFFECTED modes (node modes carry ~0 weight at xi=theta=1/2)
        ws = [w for w in ws if abs(math.sin(2*th*w)) >= 1e-3][:max_modes]
    G = []; mu = []
    for w in ws:
        mu.append(2*w*w)
        if abs(math.sin(2*th*w)) < 1e-3:
            n = int(round(2*w/math.pi)); G.append(G_node(n, xi))
        else:
            G.append(G_affected(w, xi, th, b))
    G = np.array(G); mu = np.array(mu)
    return np.array([float(np.sum(G*np.exp(-mu*t))) for t in taus])

def has_pair(xi, th, b, xlo=1e-5, xhi=2.0, n=14000, max_modes=None):
    xs = np.exp(np.linspace(math.log(xlo), math.log(xhi), n))
    v = phi_vals(xs, xi, th, b, max_modes)
    d1 = np.diff(v)
    mx = np.where((d1[:-1] > 0) & (d1[1:] <= 0))[0] + 1
    mn = np.where((d1[:-1] < 0) & (d1[1:] >= 0))[0] + 1
    return len(mx) > 0 and len(mn) > 0 and mn[0] < mx[-1]

def bc(xi, th, lo=0.5, hi=20.0, max_modes=None):
    if not has_pair(xi, th, lo, max_modes=max_modes):
        return None
    for _ in range(46):
        mid = 0.5*(lo+hi)
        if has_pair(xi, th, mid, max_modes=max_modes): lo = mid
        else: hi = mid
    return 0.5*(lo+hi)


def spectral_moments(tau, xi, th, b):
    """Return (Phi, S1, S2, S3) from the same converged spectrum used by ``phi_vals``."""
    G = []
    mu = []
    for w in roots(th, b):
        mu.append(2*w*w)
        if abs(math.sin(2*th*w)) < 1e-3:
            n = int(round(2*w/math.pi))
            G.append(G_node(n, xi))
        else:
            G.append(G_affected(w, xi, th, b))
    G = np.asarray(G, dtype=float)
    mu = np.asarray(mu, dtype=float)
    weighted = G*np.exp(-mu*tau)
    return np.asarray([
        np.sum(weighted),
        np.sum(weighted*mu),
        np.sum(weighted*mu**2),
        np.sum(weighted*mu**3),
    ], dtype=float)


def fold_residual(log_parameters, th, xi=None):
    """Dimensionless residual for the direct equations S1=S2=0.

    Positive ``b`` and ``tau`` are represented in logarithmic coordinates.  Dividing
    by Phi and the appropriate powers of tau keeps the Newton system well scaled
    across the continued theta interval.
    """
    if xi is None:
        xi = th
    b, tau = np.exp(np.asarray(log_parameters, dtype=float))
    phi, s1, s2, _ = spectral_moments(tau, xi, th, b)
    if not np.isfinite(phi) or abs(phi) < 1e-14:
        raise RuntimeError(f"invalid fold normalization at theta={th}: Phi={phi}")
    return np.asarray([tau*s1/phi, tau*tau*s2/phi], dtype=float)


def solve_fold(th, guess, xi=None, tol=5e-11, max_iter=12):
    """Solve S1=S2=0 at fixed theta by damped Newton iteration."""
    x = np.log(np.asarray(guess, dtype=float))
    unit = np.eye(2)
    jac_step = 2e-4
    for _ in range(max_iter):
        residual = fold_residual(x, th, xi)
        residual_norm = float(np.linalg.norm(residual, ord=np.inf))
        if residual_norm < tol:
            b, tau = np.exp(x)
            return float(b), float(tau)
        jacobian = np.column_stack([
            (fold_residual(x + jac_step*unit[j], th, xi)
             - fold_residual(x - jac_step*unit[j], th, xi))/(2*jac_step)
            for j in range(2)
        ])
        try:
            step = np.linalg.solve(jacobian, -residual)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError(f"singular fold Jacobian at theta={th}") from exc

        accepted = False
        damping = 1.0
        for _ in range(9):
            candidate = x + damping*step
            candidate_residual = fold_residual(candidate, th, xi)
            if np.linalg.norm(candidate_residual, ord=np.inf) < residual_norm:
                x = candidate
                accepted = True
                break
            damping *= 0.5
        if not accepted:
            raise RuntimeError(
                f"fold Newton line search failed at theta={th}; "
                f"b,tau={tuple(np.exp(x))}, residual={tuple(residual)}"
            )
    raise RuntimeError(
        f"fold Newton did not converge at theta={th}; "
        f"b,tau={tuple(np.exp(x))}, residual={tuple(fold_residual(x, th, xi))}"
    )


def continue_fold(thetas, initial=(3.0764323604, 0.03836305186)):
    """Continue the source-started fold through an ordered sequence of theta values."""
    rows = []
    guess = np.asarray(initial, dtype=float)
    for th in map(float, thetas):
        if len(rows) >= 2:
            th0, b0, tau0 = rows[-1]
            th1, b1, tau1 = rows[-2]
            log_slope = (
                np.log([b0, tau0]) - np.log([b1, tau1])
            )/(th0 - th1)
            guess = np.exp(np.log([b0, tau0]) + log_slope*(th - th0))
        elif rows:
            guess = np.asarray(rows[-1][1:], dtype=float)
        try:
            b, tau = solve_fold(th, guess)
        except RuntimeError:
            if not rows:
                raise
            b, tau = solve_fold(th, rows[-1][1:])
        rows.append((th, b, tau))
        guess = np.asarray([b, tau])
    return rows


def locate_boundary_minimum(rows, bracket=(0.36, 0.40), tolerance=2e-9):
    """Minimize the directly solved fold branch with a golden-section search."""
    cache = {round(th, 14): (b, tau) for th, b, tau in rows}

    def at(th):
        key = round(float(th), 14)
        if key not in cache:
            nearest = min(cache, key=lambda candidate: abs(candidate - th))
            cache[key] = solve_fold(float(th), cache[nearest])
        return cache[key]

    lo, hi = map(float, bracket)
    golden = (math.sqrt(5.0) - 1.0)/2.0
    left = hi - golden*(hi - lo)
    right = lo + golden*(hi - lo)
    b_left = at(left)[0]
    b_right = at(right)[0]
    while hi - lo > tolerance:
        if b_left > b_right:
            lo, left, b_left = left, right, b_right
            right = lo + golden*(hi - lo)
            b_right = at(right)[0]
        else:
            hi, right, b_right = right, left, b_left
            left = hi - golden*(hi - lo)
            b_left = at(left)[0]
    theta_min = 0.5*(lo + hi)
    b_min, tau_min = at(theta_min)
    return theta_min, b_min, tau_min


def generated_boundary_rows():
    """Generate the half-domain continuation plus its explicitly refined minimum."""
    continuation = continue_fold(np.linspace(0.5, 0.2, 61))
    minimum = locate_boundary_minimum(continuation)
    rows = [(th, b, tau, False) for th, b, tau in continuation]
    rows.append((*minimum, True))
    return sorted(rows, key=lambda row: row[0])


def boundary_csv_path():
    return TABLES / "dpma_boundary_generated.csv"


def write_boundary_csv(rows, path=None):
    """Write the boundary table (theta, b_c, tau_c, is_min) read by the threshold figure."""
    if path is None:
        path = boundary_csv_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["theta", "b_c", "tau_c", "is_min"])
        for th, b, tau, is_min in rows:
            writer.writerow([f"{th:.10f}", f"{b:.10f}", f"{tau:.10f}", str(is_min).lower()])
    return path

# Reference values (xi=theta) from bisection on the valley/peak count, kept as a regression check.
REFERENCE = {0.5: 3.07643, 0.45: 2.77162, 0.40: 2.22447, 0.35: 2.26550, 0.30: 2.63014,
       0.25: 3.15610, 0.20: 3.94513, 0.15: 5.26017, 0.10: 7.89026, 0.05: 15.78052}

def main(csv_only=False):
    OUT.clear()
    say("="*74)
    say("b_c(theta) from direct S1=S2=0 fold continuation (start xi=theta)")
    say("="*74)
    rows = generated_boundary_rows()
    write_boundary_csv(rows)
    minimum = next(row for row in rows if row[3])
    theta_min, b_min, tau_min, _ = minimum
    expected = np.asarray([0.3809084, 2.16451853, 0.0241544])
    observed = np.asarray([theta_min, b_min, tau_min])
    tolerances = np.asarray([5e-6, 2e-6, 2e-6])
    if np.any(np.abs(observed - expected) > tolerances):
        raise RuntimeError(f"boundary minimum regression failed: observed={tuple(observed)}")
    say(
        f"  minimum: theta={theta_min:.10f}, b_c={b_min:.10f}, "
        f"tau_c={tau_min:.10f}"
    )
    say("  wrote outputs/tables/dpma_boundary_generated.csv")

    continuation = {round(th, 10): (b, tau) for th, b, tau, is_min in rows if not is_min}
    say(f"\n  {'theta':>6} {'b_c(fold)':>11} {'tau_c':>11} {'b_c(ref)':>10} {'rel-diff':>9}")
    worst = 0.0
    for th in (0.5, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20):
        val, tau = continuation[round(th, 10)]
        ref = REFERENCE.get(th)
        rd = abs(val-ref)/ref
        worst = max(worst, rd)
        say(f"  {th:>6.3f} {val:>11.6f} {tau:>11.7f} {ref:>10.5f} {rd:>9.2e}")
    say(f"  => direct fold continuation matches the reference table: worst rel-diff = {worst:.2e}")
    say("  => endpoint b_c*d should approach 0.7890261736 as theta->0")

    if not csv_only:
        p = TABLES / "dpma_saddle_node_bc_theta.txt"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(OUT) + "\n")
        say("\nwrote outputs/tables/dpma_saddle_node_bc_theta.txt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv-only", action="store_true",
        help="write/validate the boundary CSV without writing the text log",
    )
    main(csv_only=parser.parse_args().csv_only)
