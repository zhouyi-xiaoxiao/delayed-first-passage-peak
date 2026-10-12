#!/usr/bin/env python3
"""Closed-form lattice poles and residues of App. A, checked against the spectral decomposition.

For the lazy ring with a one-way shortcut (q = 2/3, lambda = b q/N, u = theta N, r = xi N) the
first-passage probability is F_r(t) = sum_j W_j(r) Lambda_j^(t-1) [Eq. (Ft)].  App. A gives the
poles as the roots y_j of the Chebyshev determinant [Eq. (Du)]

    D_u(y) = eta U_{N-1}(y) + 2 U_{u-1}(y) U_{N-u-1}(y),     eta = q/lambda,  Lambda_j = 1-q+q y_j,

and the residues as W_j(r) = q N_{r,u}(y_j)/D_u'(y_j) with [Eq. (num)]

    N_{r,u} = eta [A_r + A_{N-r}] + 2 A_{N-u} [A_r + A_{u-r}]   (r <= u),   A_m = U_{m-1},

and its mirror image (r -> N-r, u -> N-u) for r >= u.  This script

  (i)   computes the eigenvalues of the symmetric matrix P_lambda (numpy eigvalsh), maps them to
        y_j and refines each one by Newton's method on D_u (the poles are the roots of D_u);
  (ii)  evaluates the closed-form residues and compares them with the residues (a.v_j)(v_j)_r
        obtained from the eigenvectors v_j (numpy eigh);
  (iii) checks with the closed-form residues that absorption is certain,
        sum_j W_j/(1-Lambda_j) = 1, and that the mean sum_j W_j/(1-Lambda_j)^2 agrees with a
        direct linear solve <a|(I-P_lambda)^-2|r>;
  (iv)  repeats the lattice-continuum comparison of Table II (b = 1.5, N = 1200,
        0.01 <= tau <= 0.2, three (theta, xi)) with the closed-form residues.

Cases: the three (theta, xi) of lattice_continuum_convergence.py at N = 150, 300, 600, 1200; the
midpoint shortcut of Fig. 5 (N = 100, u = r = 50, b = 2.6); theta = 0.38 (N = 100, u = r = 38,
b = 1.5, 2.16, 2.8); and a release beyond the shortcut (N = 300, u = 90, r = 210, b = 1.5) for
the mirror form of N_{r,u}.
Writes outputs/tables/lattice_closed_form_residues.{txt,json}.
"""
from __future__ import annotations

import json
import math

import numpy as np

from dpma_offgate_phase import general_modes
from lattice_mixture import evaluate_geometric_mixture
from paths import TABLES

Q = 2.0 / 3.0
LINES: list[str] = []


def say(s=""):
    print(s, flush=True)
    LINES.append(s)


def chebyshev_u(nmax, y):
    """U_0..U_nmax and their derivatives at the points y (arrays of shape (nmax+1, len(y)))."""
    y = np.asarray(y, dtype=float)
    U = np.zeros((nmax + 1, y.size))
    dU = np.zeros((nmax + 1, y.size))
    U[0] = 1.0
    if nmax >= 1:
        U[1] = 2.0 * y
        dU[1] = 2.0
    for n in range(1, nmax):
        U[n + 1] = 2.0 * y * U[n] - U[n - 1]
        dU[n + 1] = 2.0 * U[n] + 2.0 * y * dU[n] - dU[n - 1]
    return U, dU


def u_index(U, m):
    """U_m with the convention U_{-1} = 0 (rows of U are U_0, U_1, ...)."""
    return U[m] if m >= 0 else np.zeros(U.shape[1])


def determinant(N, u, eta, y):
    """D_u(y) and D_u'(y) [Eq. (Du)]."""
    U, dU = chebyshev_u(N, y)
    D = eta * U[N - 1] + 2.0 * u_index(U, u - 1) * u_index(U, N - u - 1)
    dD = eta * dU[N - 1] + 2.0 * (u_index(dU, u - 1) * u_index(U, N - u - 1)
                                  + u_index(U, u - 1) * u_index(dU, N - u - 1))
    return D, dD, U


def numerator(N, u, r, eta, U):
    """N_{r,u}(y) [Eq. (num)] from the Chebyshev table U (A_m = U_{m-1}, A_0 = 0)."""
    if r > u:                                       # mirror image r -> N-r, u -> N-u
        return numerator(N, N - u, N - r, eta, U)
    A = lambda m: u_index(U, m - 1)
    return eta * (A(r) + A(N - r)) + 2.0 * A(N - u) * (A(r) + A(u - r))


def path_matrix(N, u, lam):
    n = N - 1
    P = np.diag(np.full(n, 1.0 - Q)) + np.diag(np.full(n - 1, Q / 2), 1) + np.diag(np.full(n - 1, Q / 2), -1)
    P[u - 1, u - 1] -= lam
    a = np.zeros(n)
    a[u - 1] += lam
    a[0] += Q / 2
    a[-1] += Q / 2
    return P, a


def closed_form(N, u, r, b):
    """Poles (Newton-refined roots of D_u) and closed-form residues; eigh residues for comparison."""
    lam = b * Q / N
    eta = Q / lam
    P, a = path_matrix(N, u, lam)
    s_eig, V = np.linalg.eigh(P)
    y0 = (s_eig - (1.0 - Q)) / Q
    y = y0.copy()
    for _ in range(3):                               # Newton on D_u from the eigvalsh values
        D, dD, _ = determinant(N, u, eta, y)
        y = y - D / dD
    D, dD, U = determinant(N, u, eta, y)
    W_cf = Q * numerator(N, u, r, eta, U) / dD
    with np.errstate(all="ignore"):   # some BLAS builds raise spurious flags on underflow
        W_eig = (a @ V) * V[r - 1, :]
    Lam = 1.0 - Q + Q * y
    order = np.argsort(Lam)[::-1]
    shift = float(np.max(np.abs(y - y0)))
    distinct = bool(np.all(np.diff(np.sort(y)) > 0.0)) and np.array_equal(np.argsort(y), np.argsort(y0))
    return dict(Lam=Lam[order], W_cf=W_cf[order], W_eig=W_eig[order], shift=shift, distinct=distinct,
                P=P, a=a, lam=lam)


def case_report(N, u, r, b):
    c = closed_form(N, u, r, b)
    W_cf, W_eig, Lam = c["W_cf"], c["W_eig"], c["Lam"]
    scale = float(np.max(np.abs(W_eig)))
    abs_dev = float(np.max(np.abs(W_cf - W_eig))) / scale
    weighted = np.flatnonzero(np.abs(W_eig) > 1e-6 * scale)[:10]
    rel_slow = float(np.max(np.abs(W_cf[weighted] / W_eig[weighted] - 1.0)))
    total = float(np.sum(W_cf / (1.0 - Lam)))
    mean_cf = float(np.sum(W_cf / (1.0 - Lam) ** 2))
    n = N - 1
    er = np.zeros(n)
    er[r - 1] = 1.0
    IminusP = np.eye(n) - c["P"]
    with np.errstate(all="ignore"):   # some BLAS builds raise spurious flags on underflow
        x = np.linalg.solve(IminusP, np.linalg.solve(IminusP, er))
        mean_direct = float(np.dot(c["a"], x))
    rel_mean = abs(mean_cf / mean_direct - 1.0)
    say(f"  N={N:5d} u={u:4d} r={r:4d} b={b:<5}: poles = roots of D_u (Newton shift from eigvalsh "
        f"{c['shift']:.1e}, {len(Lam)} distinct roots); residues closed form vs eigh: "
        f"max|dW|/max|W| = {abs_dev:.1e}, ten slowest weighted modes max rel. {rel_slow:.1e}; "
        f"sum W/(1-Lambda) - 1 = {total - 1.0:+.1e}; mean vs direct solve {rel_mean:.1e}")
    return dict(N=N, u=u, r=r, b=b, newton_shift=c["shift"], distinct_roots=c["distinct"],
                residue_abs_dev_over_max=abs_dev, residue_rel_dev_slowest10=rel_slow,
                total_probability_minus_one=total - 1.0, mean_rel_dev=rel_mean), c


def main():
    say("Closed-form lattice poles and residues (App. A) vs the spectral decomposition, q = 2/3")
    records = []
    cases = []
    for theta, xi in ((1.0 / 3.0, 0.2), (0.4, 0.3), (0.5, 0.3)):
        for N in (150, 300, 600, 1200):
            cases.append((N, int(round(theta * N)), int(round(xi * N)), 1.5))
    cases += [(100, 50, 50, 2.6), (100, 38, 38, 1.5), (100, 38, 38, 2.16), (100, 38, 38, 2.8), (300, 90, 210, 1.5)]
    worst_continuum = 0.0
    for N, u, r, b in cases:
        rec, c = case_report(N, u, r, b)
        if N == 1200:
            theta, xi = u / N, r / N
            mu, G = general_modes(theta, b, xi, 400.0)
            t = np.arange(math.ceil(0.01 * N * N / Q), math.floor(0.2 * N * N / Q) + 1, dtype=np.int64)
            lat = evaluate_geometric_mixture(c["Lam"], c["W_cf"], t) * N * N / Q
            tau = Q * t / (N * N)
            con = np.array([float(np.sum(G * np.exp(-mu * x))) for x in tau])
            err = float(np.max(np.abs(lat / con - 1.0)))
            worst_continuum = max(worst_continuum, err)
            rec["continuum_max_rel_error"] = err
            say(f"      lattice density from the closed-form residues vs continuum, 0.01 <= tau <= 0.2: "
                f"max relative error {err:.2e}")
        records.append(rec)
    worst = {
        "newton_shift": max(r["newton_shift"] for r in records),
        "residue_abs_dev_over_max": max(r["residue_abs_dev_over_max"] for r in records),
        "residue_rel_dev_slowest10": max(r["residue_rel_dev_slowest10"] for r in records),
        "total_probability_minus_one": max(abs(r["total_probability_minus_one"]) for r in records),
        "mean_rel_dev": max(r["mean_rel_dev"] for r in records),
        "continuum_max_rel_error_N1200": worst_continuum,
    }
    ok = (all(r["distinct_roots"] for r in records) and worst["newton_shift"] < 1e-12
          and worst["residue_abs_dev_over_max"] < 1e-10 and worst["residue_rel_dev_slowest10"] < 1e-9
          and worst["total_probability_minus_one"] < 1e-11 and worst["mean_rel_dev"] < 1e-10
          and worst_continuum < 8e-5)
    say("")
    say(f"worst over {len(records)} cases: Newton shift {worst['newton_shift']:.1e}; residues "
        f"max|dW|/max|W| {worst['residue_abs_dev_over_max']:.1e}, slowest modes {worst['residue_rel_dev_slowest10']:.1e}; "
        f"|sum W/(1-Lambda) - 1| {worst['total_probability_minus_one']:.1e}; mean {worst['mean_rel_dev']:.1e}; "
        f"lattice (closed-form residues) vs continuum at N=1200: {worst_continuum:.1e}")
    say(f"result: {'PASS' if ok else 'FAIL'}")
    (TABLES / "lattice_closed_form_residues.txt").write_text("\n".join(LINES) + "\n")
    (TABLES / "lattice_closed_form_residues.json").write_text(
        json.dumps({"cases": records, "worst": worst, "pass": ok}, indent=2) + "\n")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
