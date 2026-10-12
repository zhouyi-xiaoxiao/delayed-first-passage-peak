#!/usr/bin/env python3
"""Fast checks of exact identities and headline numbers of the paper (pytest).

Run from the archive root:  python -m pytest code/test_dpma_identities.py -q
Each test checks an algebraic identity or a headline number at float precision.
"""
from __future__ import annotations

import math

import numpy as np

q = 2.0 / 3.0


def _roots_D(th, b, wmax=40.0, n=80001):
    k = np.linspace(1e-6, 2 * wmax, n)
    f = k * np.sin(k) + 2 * b * np.sin(k * th) * np.sin(k * (1 - th))
    s = np.sign(f)
    i = np.where(s[:-1] * s[1:] < 0)[0]
    lo, hi, flo = k[i], k[i + 1], f[i]
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        fm = mid * np.sin(mid) + 2 * b * np.sin(mid * th) * np.sin(mid * (1 - th))
        left = flo * fm <= 0
        hi = np.where(left, mid, hi)
        lo = np.where(left, lo, mid)
        flo = np.where(left, flo, fm)
    return 0.5 * (lo + hi)


def test_delta_sink_jump_equals_secular_condition():
    """phi'(th+)-phi'(th-) = 2b phi(th)  <=>  D_theta(k;b)=0 (App. B)."""
    for th, b in ((0.5, 3.0), (0.38, 2.0), (0.27, 4.0)):
        for k in _roots_D(th, b)[:6]:
            jump = -k * math.sin(k)                       # exact: -k[sin(k th)cos(..)+cos(..)sin(..)]
            rhs = 2 * b * math.sin(k * th) * math.sin(k * (1 - th))
            assert abs(jump - rhs) < 1e-8 * max(1.0, abs(rhs))


def test_normalization_identity_J():
    """J = sin(k) D_k / (4b) at every root (App. B, normalization identity)."""
    for th, b in ((0.5, 3.0), (0.38, 2.0), (0.31, 1.3)):
        for k in _roots_D(th, b)[:6]:
            A, Bs = math.sin(k * th), math.sin(k * (1 - th))
            J = 0.5 * (th * Bs**2 + (1 - th) * A**2) - A * Bs * math.sin(k) / (2 * k)
            Dk = math.sin(k) + k * math.cos(k) + 2 * b * (
                th * math.cos(k * th) * Bs + (1 - th) * A * math.cos(k * (1 - th)))
            assert abs(J - math.sin(k) * Dk / (4 * b)) < 1e-10 * max(1.0, abs(J))


def test_chebyshev_product_identity():
    """A_{N-r} A_u - A_{N-u} A_r = A_N A_{u-r} with A_m = sin(m phi)/sin(phi) (App. A)."""
    rng = np.random.default_rng(3)
    for _ in range(200):
        N = int(rng.integers(6, 60))
        u = int(rng.integers(2, N - 1))
        r = int(rng.integers(1, u + 1))
        ph = float(rng.uniform(0.05, 3.09))

        def A(m: int) -> float:
            return math.sin(m * ph) / math.sin(ph)

        lhs = A(N - r) * A(u) - A(N - u) * A(r)
        rhs = A(N) * A(u - r)
        assert abs(lhs - rhs) < 1e-7 * max(1.0, abs(rhs))


def test_pisc_closed_form_vs_linear_algebra():
    """pi_sc = 2 min(r,u)(N-max(r,u)) / (aN + 2u(N-u)) vs direct resolvent (App. A)."""
    N, b = 40, 2.6
    lam = b * q / N
    a = q / lam
    for u, r in ((20, 20), (13, 7), (28, 33)):
        M = np.zeros((N - 1, N - 1))
        for i in range(N - 1):
            M[i, i] = 1 - q
            if i > 0:
                M[i, i - 1] = q / 2
            if i < N - 2:
                M[i, i + 1] = q / 2
        M[u - 1, u - 1] -= lam
        er = np.zeros(N - 1)
        er[r - 1] = 1.0
        eu = np.zeros(N - 1)
        eu[u - 1] = 1.0
        pisc = lam * (eu @ np.linalg.solve(np.eye(N - 1) - M, er))
        closed = 2 * min(r, u) * (N - max(r, u)) / (a * N + 2 * u * (N - u))
        assert abs(pisc - closed) < 1e-12


def test_antipodal_fold_values():
    """b_c(1/2)=3.076432, tau_c=0.038363 from S1=S2=0 (float Newton on tan w = -2w/b)."""
    def roots_antip(b, nmax=40):
        ws = []
        for j in range(1, nmax + 1):
            lo, hi = (2 * j - 1) * math.pi / 2 + 1e-12, j * math.pi - 1e-12

            def f(w: float) -> float:
                return 2 * w * math.cos(w) + b * math.sin(w)

            flo = f(lo)
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                if flo * f(mid) <= 0:
                    hi = mid
                else:
                    lo, flo = mid, f(mid)
            ws.append(0.5 * (lo + hi))
        return ws

    def S(n, tau, b):
        tot = 0.0
        for w in roots_antip(b):
            mu = 2 * w * w
            G = 4 * w * (1 - math.cos(w)) / (math.sin(w) * (1 + b * (b + 2) / (4 * w * w)))
            tot += G * mu**n * math.exp(-mu * tau)
        return tot

    tau, b = 0.0384, 3.076
    for _ in range(25):
        s1, s2, s3 = S(1, tau, b), S(2, tau, b), S(3, tau, b)
        db = 1e-7
        s1b = (S(1, tau, b + db) - S(1, tau, b - db)) / (2 * db)
        s2b = (S(2, tau, b + db) - S(2, tau, b - db)) / (2 * db)
        det = (-s2) * s2b - s1b * (-s3)
        tau += (-s1 * s2b + s2 * s1b) / det
        b += (s2 * s2 - s1 * s3) / det
    assert abs(b - 3.0764323604) < 1e-6
    assert abs(tau - 0.0383630519) < 1e-7


def test_master_function_uses_continuum_density_normalization():
    """The continuum density obeys F(t) ~ (q/N^2) Phi(qt/N^2), not F ~ Phi/N^2."""
    from dpma_antipodal_spectrum import Q_DEFAULT, density
    from lattice_mixture import evaluate_geometric_mixture

    N, b, d = 200, 1.5, 3
    q0 = Q_DEFAULT
    beta = q0 * b / ((1.0 - q0) * N)
    u, r = N // 2, N // 2 + d
    lam = beta * (1.0 - q0)
    M = np.zeros((N - 1, N - 1)); b_abs = np.zeros(N - 1)
    for i in range(N - 1):
        M[i, i] = 1.0 - q0
        if i > 0:
            M[i, i - 1] = q0 / 2.0
        if i < N - 2:
            M[i, i + 1] = q0 / 2.0
    M[u - 1, u - 1] -= lam
    b_abs[0] += q0 / 2.0; b_abs[-1] += q0 / 2.0; b_abs[u - 1] += lam
    s, vectors = np.linalg.eigh(M)
    start = np.zeros(N - 1); start[r - 1] = 1.0
    with np.errstate(all="ignore"):   # some BLAS builds raise spurious flags on underflow
        amplitudes = (start @ vectors) * (vectors.T @ b_abs)
    tau = 0.1
    t = int(round(tau * N * N / q0))
    f_t = float(evaluate_geometric_mixture(s, amplitudes, [t])[0])
    continuum = float(density(np.array([tau]), b, n_modes=120)[0])
    finite_n_density = N * N * f_t / q0

    # The O(1/N) residue correction is about 2.1% here.  The alternative
    # N^2 F scaling differs by the order-one factor q=2/3 and must not pass.
    assert abs(continuum / finite_n_density - 1.0) < 0.03
    assert abs(continuum / (N * N * f_t) - 1.0) > 0.2
