#!/usr/bin/env python3
"""Arb/Krawczyk certificates for the two half-line A3 cusp roots (App. C and App. D).

The branch-cut integral is split into [0,delta], [delta,V], and [V,infinity].
The middle part is enclosed by Arb's validated complex integrator.  The two
omitted tails are enclosed by explicit analytic inequalities, including the
parameter derivatives used by the Krawczyk operator.  No trapezoidal or
floating-point quadrature value enters the proof.

Install the only extra dependency with

    python -m pip install python-flint==0.9.0

Writes outputs/tables/halfline_cusp_krawczyk.json (the stored reference output of this script is
data/dpma_halfline_cusp_interval_certificate.json).
"""
from __future__ import annotations

import json
import platform

try:
    import flint
    from flint import acb, arb, arb_mat, ctx
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "python-flint is required: python -m pip install python-flint==0.9.0"
    ) from exc


from paths import TABLES

OUT = TABLES / "halfline_cusp_krawczyk.json"
PREC_BITS = 192
ctx.prec = PREC_BITS

DELTA = arb("0.001")
VCUT = arb("70")
SPLITS = (("0.001", "1"), ("1", "5"), ("5", "15"), ("15", "40"), ("40", "70"))
BOX_RADIUS = "1e-7"  # in (log B, log T, X)
IUNIT = acb(0, 1)

CANDIDATES = (
    {
        "label": "start_between_boundary_and_gate",
        "branch": "X_lt_1",
        "B": "2.868771925558732",
        "T": "0.06362964429621742",
        "X": "0.7583136631287805",
    },
    {
        "label": "start_beyond_gate",
        "branch": "X_gt_1",
        "B": "1.0856828303187454",
        "T": "0.20723008335295337",
        "X": "1.4248971567811466",
    },
)


def ball_record(value: arb) -> dict[str, object]:
    mid, rad, exponent = value.mid_rad_10exp()
    return {
        "arb": value.str(40, more=True),
        "decimal_outer_ball": {
            "mid_integer": str(mid),
            "radius_integer": str(rad),
            "exponent": int(exponent),
            "meaning": "(mid_integer +/- radius_integer) * 10^exponent",
        },
        "contains_zero": bool(value.contains(0)),
    }


def symmetric_error(radius: arb) -> arb:
    return arb(0, radius.abs_upper())


def tail_bounds(
    n: int, B: arb, T: arb, X: arb, branch: str
) -> tuple[arb, arb, arb, dict[str, arb]]:
    """Bounds for I_n, d_logB I_n, d_X I_n outside [delta,V]."""
    b_min = B.lower()
    b_max = B.upper()
    t_min = T.lower()
    if branch == "X_lt_1":
        c_max = 2 * X.upper()
    else:
        c_max = arb(2)

    # On 0 <= v <= delta:
    # |D| >= v [1+2 B_min-(4/3) B_max delta^2].
    d0 = 1 + 2 * b_min - arb(4) * b_max * DELTA**2 / 3
    if d0.contains(0) or d0.lower() <= 0:
        raise RuntimeError("small-v denominator lower bound is not positive")
    h0 = 1 + b_max * c_max * DELTA / d0
    hu0 = b_max * c_max / d0**2
    hx0 = 1 + 2 * b_max / d0
    small_i = (
        h0 * DELTA ** (2 * n + 2)
        / (arb.pi() * (2**n) * (2 * n + 2))
    )
    small_u = (
        hu0 * DELTA ** (2 * n + 3)
        / (arb.pi() * (2**n) * (2 * n + 3))
    )
    small_x = (
        hx0 * DELTA ** (2 * n + 3)
        / (arb.pi() * (2**n) * (2 * n + 3))
    )

    # On v >= V, |D| >= v-B_max.  The resulting Gaussian tails are
    # integrated exactly with the upper incomplete gamma function.
    gap = VCUT - b_max
    if gap.contains(0) or gap.lower() <= 0:
        raise RuntimeError("large-v denominator lower bound is not positive")
    h_inf = 1 + 4 * b_max / gap
    hu_inf = 4 * b_max * VCUT / gap**2
    hx_inf = 1 + 4 * b_max / gap
    z = t_min * VCUT**2 / 2
    gamma_integer = z.gamma_upper(n + 1)
    large_i = h_inf * gamma_integer / (arb.pi() * t_min ** (n + 1))
    large_u = hu_inf * gamma_integer / (arb.pi() * t_min ** (n + 1))
    gamma_half = z.gamma_upper(arb(n) + arb("1.5"))
    large_x = (
        hx_inf * arb(2).sqrt() * gamma_half
        / (arb.pi() * t_min ** (arb(n) + arb("1.5")))
    )
    return (
        (small_i + large_i).abs_upper(),
        (small_u + large_u).abs_upper(),
        (small_x + large_x).abs_upper(),
        {
            "small_I": small_i.abs_upper(),
            "small_logB": small_u.abs_upper(),
            "small_X": small_x.abs_upper(),
            "large_I": large_i.abs_upper(),
            "large_logB": large_u.abs_upper(),
            "large_X": large_x.abs_upper(),
        },
    )


def transform_and_controls(v: acb, B: arb, X: arb, branch: str) -> tuple[acb, acb, acb]:
    kappa = IUNIT * v
    one_minus = 1 - (-kappa).exp()
    if branch == "X_lt_1":
        near = (-kappa * (1 - X)).exp()
        far = (-kappa * (1 + X)).exp()
        difference = near - far
        difference_x = kappa * (near + far)
    elif branch == "X_gt_1":
        near = (-kappa * (X - 1)).exp()
        far = (-kappa * (1 + X)).exp()
        difference = near - far
        difference_x = -kappa * near + kappa * far
    else:  # pragma: no cover
        raise ValueError(branch)
    denominator = kappa + B * (1 - (-2 * kappa).exp())
    direct = (-kappa * X).exp()
    transform = direct + B * one_minus * difference / denominator
    # B*d/dB of the rank-one term; D-B(1-exp(-2k))=k.
    derivative_log_b = B * one_minus * difference * kappa / denominator**2
    derivative_x = -kappa * direct + B * one_minus * difference_x / denominator
    return transform, derivative_log_b, derivative_x


def enclosed_integrals(
    B: arb, T: arb, X: arb, branch: str, max_order: int = 4
) -> tuple[list[arb], list[arb], list[arb], dict[str, dict[str, arb]]]:
    integrals: list[arb] = []
    log_b_derivatives: list[arb] = []
    x_derivatives: list[arb] = []
    tails: dict[str, dict[str, arb]] = {}
    for n in range(max_order + 1):
        totals = [acb(0), acb(0), acb(0)]

        def values(v: acb) -> tuple[acb, acb, acb]:
            transform, derivative_log_b, derivative_x = transform_and_controls(v, B, X, branch)
            weight = v * (v * v / 2) ** n * (-T * v * v / 2).exp()
            return weight * transform, weight * derivative_log_b, weight * derivative_x

        for lo, hi in SPLITS:
            for component in range(3):
                totals[component] += acb.integral(
                    lambda value, analytic, component=component: values(value)[component],
                    arb(lo),
                    arb(hi),
                    rel_tol=arb("1e-32"),
                    abs_tol=arb("1e-32"),
                    eval_limit=200000,
                )
        middle = [-total.imag / arb.pi() for total in totals]
        error_i, error_u, error_x, detail = tail_bounds(n, B, T, X, branch)
        integrals.append(middle[0] + symmetric_error(error_i))
        log_b_derivatives.append(middle[1] + symmetric_error(error_u))
        x_derivatives.append(middle[2] + symmetric_error(error_x))
        tails[str(n)] = detail
    return integrals, log_b_derivatives, x_derivatives, tails


def cusp_map_and_jacobian(
    log_b: arb, log_t: arb, X: arb, branch: str
) -> tuple[arb_mat, arb_mat, dict[str, object]]:
    B = log_b.exp()
    T = log_t.exp()
    values, derivative_b, derivative_x, tails = enclosed_integrals(B, T, X, branch)
    function = arb_mat([[values[1]], [values[2]], [values[3]]])
    jacobian = arb_mat(
        [
            [derivative_b[n], -T * values[n + 1], derivative_x[n]]
            for n in range(1, 4)
        ]
    )
    diagnostics = {
        "B": B,
        "T": T,
        "density": values[0],
        "fourth_derivative": values[4],
        "unfolding_determinant": (
            derivative_b[1] * derivative_x[2]
            - derivative_x[1] * derivative_b[2]
        ),
        "full_jacobian_determinant": jacobian.det(),
        "tails": tails,
    }
    return function, jacobian, diagnostics


def certify(candidate: dict[str, str]) -> dict[str, object]:
    log_b_center = arb(candidate["B"]).log().mid()
    log_t_center = arb(candidate["T"]).log().mid()
    x_center = arb(candidate["X"]).mid()
    radius = arb(BOX_RADIUS)
    log_b_box = log_b_center + arb(0, radius)
    log_t_box = log_t_center + arb(0, radius)
    x_box = x_center + arb(0, radius)

    if candidate["branch"] == "X_lt_1" and not x_box.upper() < 1:
        raise RuntimeError("X<1 branch box crosses the gate")
    if candidate["branch"] == "X_gt_1" and not x_box.lower() > 1:
        raise RuntimeError("X>1 branch box crosses the gate")

    function_center, jacobian_center, center_diag = cusp_map_and_jacobian(
        log_b_center, log_t_center, x_center, candidate["branch"]
    )
    _, jacobian_box, box_diag = cusp_map_and_jacobian(
        log_b_box, log_t_box, x_box, candidate["branch"]
    )
    inverse_ball = jacobian_center.inv()
    preconditioner = arb_mat(
        [[inverse_ball[row, column].mid() for column in range(3)] for row in range(3)]
    )
    identity = arb_mat(3, 3)
    for index in range(3):
        identity[index, index] = 1
    delta_box = arb_mat([[arb(0, radius)] for _ in range(3)])
    center = arb_mat([[log_b_center], [log_t_center], [x_center]])
    contraction_matrix = identity - preconditioner * jacobian_box
    krawczyk = (
        center
        - preconditioner * function_center
        + contraction_matrix * delta_box
    )
    contraction_row_sums = [
        sum(contraction_matrix[row, column].abs_upper() for column in range(3))
        for row in range(3)
    ]
    preconditioner_determinant = preconditioner.det()
    coordinate_boxes = (log_b_box, log_t_box, x_box)
    inclusion = [
        coordinate_boxes[index].contains_interior(krawczyk[index, 0])
        for index in range(3)
    ]
    density = box_diag["density"]
    fourth = box_diag["fourth_derivative"]
    unfolding = box_diag["unfolding_determinant"]
    full_det = box_diag["full_jacobian_determinant"]
    expected_unfolding_sign = 1 if candidate["branch"] == "X_lt_1" else -1
    sign_checks = {
        "density_positive": not density.contains(0) and density.lower() > 0,
        "fourth_derivative_negative": not fourth.contains(0) and fourth.upper() < 0,
        "unfolding_determinant_expected_sign": (
            not unfolding.contains(0)
            and ((unfolding.lower() > 0) if expected_unfolding_sign > 0 else (unfolding.upper() < 0))
        ),
        "full_cusp_jacobian_invertible": not full_det.contains(0),
        "preconditioner_invertible": not preconditioner_determinant.contains(0),
        "weighted_infinity_contraction": all(value.upper() < 1 for value in contraction_row_sums),
    }
    verified = all(inclusion) and all(sign_checks.values())

    return {
        "label": candidate["label"],
        "status": "VERIFIED" if verified else "FAILED",
        "branch": candidate["branch"],
        "coordinate_order": ["log_B", "log_T", "X"],
        "box_radius_each_coordinate": BOX_RADIUS,
        "box": {
            "log_B": ball_record(log_b_box),
            "log_T": ball_record(log_t_box),
            "X": ball_record(x_box),
            "B": ball_record(log_b_box.exp()),
            "T": ball_record(log_t_box.exp()),
        },
        "center_function": {
            "I1": ball_record(function_center[0, 0]),
            "I2": ball_record(function_center[1, 0]),
            "I3": ball_record(function_center[2, 0]),
        },
        "krawczyk_image": {
            "log_B": ball_record(krawczyk[0, 0]),
            "log_T": ball_record(krawczyk[1, 0]),
            "X": ball_record(krawczyk[2, 0]),
            "strictly_inside_box": inclusion,
            "contraction_row_sum_bounds": [ball_record(value) for value in contraction_row_sums],
            "preconditioner_determinant": ball_record(preconditioner_determinant),
        },
        "box_enclosures": {
            "density_I0": ball_record(density),
            "fourth_time_derivative_I4": ball_record(fourth),
            "control_unfolding_determinant": ball_record(unfolding),
            "full_cusp_jacobian_determinant": ball_record(full_det),
        },
        "sign_checks": sign_checks,
        "tail_bounds_at_box": {
            order: {name: ball_record(value) for name, value in bounds.items()}
            for order, bounds in box_diag["tails"].items()
        },
        "claim": (
            "Exactly one zero of (I1,I2,I3), equivalently of the first three "
            "time derivatives, lies in this box; I4 and the two-control unfolding "
            "determinant are nonzero throughout the box."
        ),
    }


def main() -> None:
    ctx.prec = PREC_BITS
    certificates = [certify(candidate) for candidate in CANDIDATES]
    verified = all(item["status"] == "VERIFIED" for item in certificates)
    payload = {
        "status": "VERIFIED" if verified else "FAILED",
        "method": (
            "Krawczyk inclusion using Arb validated complex integration on [delta,V], "
            "plus explicit small-v and Gaussian-tail enclosures"
        ),
        "integral_definition": (
            "I_n=-pi^{-1} integral_0^infinity v (v^2/2)^n exp(-T v^2/2) "
            "Im fhat(-v^2/2+i0;B,X) dv; f^(n)=(-1)^n I_n"
        ),
        "precision_bits": PREC_BITS,
        "delta": DELTA.str(20, more=True),
        "V": VCUT.str(20, more=True),
        "software": {
            "python": platform.python_version(),
            "python_flint": flint.__version__,
            "arb_integrator": "acb_calc_integrate through acb.integral",
        },
        "certificates": certificates,
        "scope": (
            "local existence and uniqueness in the two displayed boxes only; "
            "no global assertion that these are the only half-line cusps"
        ),
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    if not verified:
        raise SystemExit("one or more half-line cusp certificates failed")


if __name__ == "__main__":
    main()
