#!/usr/bin/env python3
"""Lattice-to-continuum convergence of the density (Sec. II C, App. B and Table II).

For b = 1.5 and (theta, xi) = (1/3, 0.2), (0.4, 0.3), (0.5, 0.3) the exact lattice density
(N^2/q) F_r(t), q = 2/3, u = theta N, r = xi N, lambda = b q/N, is compared with the continuum
density Phi(tau) at the same times tau = q t/N^2, on every integer t with 0.01 <= tau <= 0.2,
for N = 150, 300, 600, 1200.  Also compared are the residues, (N^2/q) W_j against G_j, and the
lattice wavenumbers k_{N,j} = N arccos(1 - (1 - Lambda_j)/q) against k_j, for the ten slowest
modes with nonzero weight at N = 1200.  The convergence exponent is fitted on log(max error) versus log N.
Writes outputs/tables/lattice_continuum_convergence.txt.
"""
from __future__ import annotations

import math

import numpy as np

from dpma_offgate_phase import general_modes
from lattice_mixture import evaluate_geometric_mixture
from paths import TABLES

Q = 2.0 / 3.0
B = 1.5
CASES = ((1.0 / 3.0, 0.2), (0.4, 0.3), (0.5, 0.3))
NS = (150, 300, 600, 1200)
LINES: list[str] = []


def say(s=""):
    print(s, flush=True)
    LINES.append(s)


def lattice_modes(N, u, r, b):
    lam = b * Q / N
    n = N - 1
    P = np.diag(np.full(n, 1 - Q)) + np.diag(np.full(n - 1, Q / 2), 1) + np.diag(np.full(n - 1, Q / 2), -1)
    P[u - 1, u - 1] -= lam
    s, V = np.linalg.eigh(P)
    a = np.zeros(n)
    a[u - 1] += lam
    a[0] += Q / 2
    a[-1] += Q / 2
    with np.errstate(all="ignore"):   # some BLAS builds raise spurious flags on underflow
        W = (a @ V) * V[r - 1, :]
    order = np.argsort(s)[::-1]
    return s[order], W[order]


def continuum(theta, xi, b, kmax=400.0):
    mu, G = general_modes(theta, b, xi, kmax)
    order = np.argsort(mu)
    return mu[order], G[order]


def main():
    say(f"Lattice vs continuum density, b = {B}, q = 2/3, window 0.01 <= tau <= 0.2")
    exponents = []
    worst_1200 = 0.0
    worst_mode = 0.0
    for theta, xi in CASES:
        mu, G = continuum(theta, xi, B)
        errors = []
        for N in NS:
            u, r = int(round(theta * N)), int(round(xi * N))
            assert abs(u - theta * N) < 1e-9 and abs(r - xi * N) < 1e-9
            s, W = lattice_modes(N, u, r, B)
            t = np.arange(math.ceil(0.01 * N * N / Q), math.floor(0.2 * N * N / Q) + 1, dtype=np.int64)
            lat = evaluate_geometric_mixture(s, W, t) * N * N / Q
            tau = Q * t / (N * N)
            con = np.array([float(np.sum(G * np.exp(-mu * x))) for x in tau])
            err = float(np.max(np.abs(lat / con - 1.0)))
            errors.append(err)
            if N == 1200:
                worst_1200 = max(worst_1200, err)
                idx = np.flatnonzero(np.abs(G) > 1e-10)[:10]     # modes with nonzero weight
                kN = N * np.arccos(1.0 - (1.0 - s[idx]) / Q)
                kc = np.sqrt(2.0 * mu[idx])
                Wn = W[idx] * N * N / Q
                rel_amp = np.abs(Wn / G[idx] - 1.0)
                rel_k = np.abs(kN / kc - 1.0)
                worst_mode = max(worst_mode, float(np.median(rel_amp)))
                say(f"  theta={theta:.4f}, xi={xi}: N=1200, ten slowest weighted modes: |(N^2/q)W_j/G_j - 1| "
                    f"median {np.median(rel_amp):.1e}, max {np.max(rel_amp):.1e}; "
                    f"max |k_Nj/k_j - 1| = {np.max(rel_k):.1e}")
        p = float(np.polyfit(np.log(NS), np.log(errors), 1)[0])
        exponents.append(p)
        say(f"  theta={theta:.4f}, xi={xi}: max relative error " +
            ", ".join(f"N={N}: {e:.2e}" for N, e in zip(NS, errors)) + f"; fitted exponent {-p:.3f}")
    say(f"max relative error at N=1200 over the three cases: {worst_1200:.1e}; "
        f"largest median per-mode amplitude error: {worst_mode:.1e}; fitted exponents "
        + ", ".join(f"{-p:.2f}" for p in exponents))
    (TABLES / "lattice_continuum_convergence.txt").write_text("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
