#!/usr/bin/env python3
"""Interval-Newton (Arb) certificates for the two half-line A3 cusps (App. C and App. D).

The two half-line cusps on the X<1 and X>1 branches, located in floating point by
cusp_endpoint_continuation.py, are turned into interval-Newton statements: python-flint/Arb
integrates the analytic complex branch-cut integrand on [0, 60] at 60 decimal digits, and the
omitted [60, infinity) contribution is enclosed by an explicit Gaussian/incomplete-gamma bound.
No sampled quadrature rule is used in the certificate.

Install the only extra dependency with

    python -m pip install python-flint==0.9.0

Writes outputs/tables/halfline_cusp_interval_newton.{json,txt} (the stored reference output is
data/dpma_interval_certificates.json).
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

from paths import TABLES as ART

CTX_DPS = 60
ctx.dps = CTX_DPS

VMAX = arb(60)
REL_TOL = arb("1e-40")
ABS_TOL = arb("1e-40")
IUNIT = acb(0, 1)
PI = arb.pi()


def ball(center: str, radius: str) -> arb:
    """Return an Arb ball enclosing center +/- radius."""
    return arb(center, radius)


def interval_record(value: arb, digits: int = 30) -> dict[str, str]:
    return {
        "ball": value.str(digits),
        "lower": value.lower().str(digits, radius=False),
        "upper": value.upper().str(digits, radius=False),
    }


def box_record(center: str, radius: str) -> dict[str, str]:
    value = ball(center, radius)
    return {
        "center": center,
        "radius": radius,
        **interval_record(value),
    }


def excludes_zero(value: arb) -> bool:
    return not value.contains(0)


def strictly_positive(value: arb) -> bool:
    return bool(value.lower() > 0)


def gaussian_power_tail(power: int, decay: arb) -> arb:
    """Rigorous integral of v**power exp(-decay*v**2) from VMAX to inf."""
    shape = arb(power + 1) / 2
    lower_argument = decay * VMAX * VMAX
    return (
        arb(1)
        / 2
        * decay ** (-shape)
        * lower_argument.gamma_upper(shape)
    )


def cut_component(
    strength: arb,
    time: arb,
    start: arb,
    moment_order: int,
    component: str,
    side: str,
) -> tuple[arb, arb]:
    """Enclose a cusp moment or its B/X derivative, including the tail."""
    if component not in {"f", "B", "X"}:
        raise ValueError(component)
    if side not in {"boundary", "domain"}:
        raise ValueError(side)

    def analytic_integrand(v: acb, _analytic: bool) -> acb:
        iv = IUNIT * v
        exp_start = (-iv * start).exp()
        r11 = 2 * (-iv).exp() * v.sinc()
        one_minus_exp = -(-iv).expm1()
        if side == "boundary":  # X<1
            r1x = 2 * start * (-iv).exp() * (v * start).sinc()
            r1x_start = 2 * (-iv).exp() * (v * start).cos()
        else:  # X>1
            r1x = 2 * exp_start * v.sinc()
            r1x_start = -2 * iv * exp_start * v.sinc()
        denominator = 1 + strength * r11
        if component == "f":
            transfer = (
                exp_start
                + strength * r1x * one_minus_exp / denominator
            )
        elif component == "B":
            transfer = r1x * one_minus_exp / denominator**2
        else:
            transfer = (
                -iv * exp_start
                + strength * r1x_start * one_minus_exp / denominator
            )
        mu = v * v / 2
        return (
            v
            * transfer
            * mu**moment_order
            * (-time * mu).exp()
        )

    finite_integral = acb.integral(
        analytic_integrand,
        arb(0),
        VMAX,
        rel_tol=REL_TOL,
        abs_tol=ABS_TOL,
        eval_limit=1_000_000,
    )
    finite_value = -finite_integral.imag / PI

    strength_max = strength.upper()
    time_min = time.lower()
    denominator_margin = 1 - 2 * strength_max / VMAX
    if not strictly_positive(denominator_margin):
        raise RuntimeError("tail denominator bound is not positive")
    decay = time_min / 2
    scale = PI * (2**moment_order)
    if component == "f":
        tail = (
            gaussian_power_tail(2 * moment_order + 1, decay)
            + 4
            * strength_max
            / denominator_margin
            * gaussian_power_tail(2 * moment_order, decay)
        ) / scale
    elif component == "B":
        tail = (
            4
            / denominator_margin**2
            * gaussian_power_tail(2 * moment_order, decay)
        ) / scale
    else:
        tail = (
            gaussian_power_tail(2 * moment_order + 2, decay)
            + 4
            * strength_max
            / denominator_margin
            * gaussian_power_tail(2 * moment_order + 1, decay)
        ) / scale
    enclosed = finite_value + arb(0, tail.abs_upper())
    return enclosed, tail


def certify_cusp(
    *,
    name: str,
    side: str,
    strength_center: str,
    time_center: str,
    start_center: str,
    strength_radius: str,
    time_radius: str,
    start_radius: str,
) -> dict[str, object]:
    strength_box = ball(strength_center, strength_radius)
    time_box = ball(time_center, time_radius)
    start_box = ball(start_center, start_radius)
    center = [arb(strength_center), arb(time_center), arb(start_center)]

    center_residual = [
        cut_component(
            center[0], center[1], center[2], order, "f", side
        )[0]
        for order in (1, 2, 3)
    ]
    box_moments = {
        order: cut_component(
            strength_box, time_box, start_box, order, "f", side
        )[0]
        for order in (0, 2, 3, 4)
    }
    strength_derivatives = {
        order: cut_component(
            strength_box, time_box, start_box, order, "B", side
        )[0]
        for order in (1, 2, 3)
    }
    start_derivatives = {
        order: cut_component(
            strength_box, time_box, start_box, order, "X", side
        )[0]
        for order in (1, 2, 3)
    }

    # Variable order (B,T,X); F=(m1,m2,m3), where Phi^(n)=(-1)^n m_n.
    jacobian = arb_mat(
        [
            [
                strength_derivatives[order],
                -box_moments[order + 1],
                start_derivatives[order],
            ]
            for order in (1, 2, 3)
        ]
    )
    residual = arb_mat([[value] for value in center_residual])
    center_vector = arb_mat([[value] for value in center])
    image = center_vector - jacobian.inv() * residual
    boxes = [strength_box, time_box, start_box]
    inclusion = [
        boxes[index].contains_interior(image[index, 0])
        for index in range(3)
    ]

    unfolding = arb_mat(
        [
            [strength_derivatives[1], start_derivatives[1]],
            [strength_derivatives[2], start_derivatives[2]],
        ]
    )
    density = box_moments[0]
    normalized_unfolding = (
        strength_box
        * time_box**3
        / density**2
        * unfolding.det()
    )
    normalized_fourth = (
        time_box**4 * box_moments[4] / density
    )
    determinant = jacobian.det()
    passed = bool(
        all(inclusion)
        and excludes_zero(determinant)
        and excludes_zero(box_moments[4])
        and excludes_zero(normalized_unfolding)
        and strictly_positive(density)
    )
    if not passed:
        raise RuntimeError(f"{name} cusp interval-Newton certificate failed")

    return {
        "status": "PASS",
        "name": name,
        "side": "X<1" if side == "boundary" else "X>1",
        "method": (
            "Arb analytic integration on [0,60], explicit Gaussian tail "
            "bound, and interval Newton"
        ),
        "variables": ["B", "T", "X"],
        "box": {
            "B": box_record(strength_center, strength_radius),
            "T": box_record(time_center, time_radius),
            "X": box_record(start_center, start_radius),
        },
        "center_residual": {
            f"m{order}": interval_record(center_residual[order - 1])
            for order in (1, 2, 3)
        },
        "interval_newton_image": {
            "B": interval_record(image[0, 0]),
            "T": interval_record(image[1, 0]),
            "X": interval_record(image[2, 0]),
            "strictly_inside": inclusion,
        },
        "full_jacobian_determinant": interval_record(determinant),
        "density": interval_record(density),
        "m4": interval_record(box_moments[4]),
        "normalized_fourth_margin": interval_record(normalized_fourth),
        "normalized_unfolding_determinant_logB_X": interval_record(
            normalized_unfolding
        ),
        "tail_bound": {
            "cutoff": "60",
            "proof": (
                "|1+B r11| >= 1-2 Bmax/v on v>=60; polynomial-Gaussian "
                "remainders are integrated with the upper incomplete gamma function"
            ),
        },
        "claim": "exactly one cusp root exists in the displayed box",
    }


def write_text(payload: dict[str, object]) -> str:
    lines = [
        "INTERVAL-NEWTON CERTIFICATES FOR THE TWO HALF-LINE CUSPS",
        "=========================================================",
        f"python-flint={payload['environment']['python_flint']}, Arb dps={CTX_DPS}",
        "",
    ]
    for key in ("boundary_side_cusp", "domain_side_cusp"):
        cusp = payload[key]
        lines.extend(
            [
                f"{key}: PASS",
                f"  B box: {cusp['box']['B']['ball']}",
                f"  T box: {cusp['box']['T']['ball']}",
                f"  X box: {cusp['box']['X']['ball']}",
                f"  Newton B: {cusp['interval_newton_image']['B']['ball']}",
                f"  Newton T: {cusp['interval_newton_image']['T']['ball']}",
                f"  Newton X: {cusp['interval_newton_image']['X']['ball']}",
                f"  det J: {cusp['full_jacobian_determinant']['ball']}",
                f"  T^4 Phi_TTTT/Phi: {cusp['normalized_fourth_margin']['ball']}",
                "  normalized unfolding det(log B,X): "
                f"{cusp['normalized_unfolding_determinant_logB_X']['ball']}",
                "",
            ]
        )
    lines.extend(
        [
            "Scope: each uniqueness statement is local to its displayed box.",
            "The certificates do not exclude other roots elsewhere in parameter space.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    boundary = certify_cusp(
        name="boundary-side half-line cusp",
        side="boundary",
        strength_center="2.868771925558728",
        time_center="0.06362964429621742",
        start_center="0.7583136631287806",
        strength_radius="1e-10",
        time_radius="1e-11",
        start_radius="1e-11",
    )
    domain = certify_cusp(
        name="domain-side half-line cusp",
        side="domain",
        strength_center="1.0856828303187425",
        time_center="0.2072300833529535",
        start_center="1.424897156781147",
        strength_radius="1e-10",
        time_radius="1e-11",
        start_radius="1e-10",
    )
    payload: dict[str, object] = {
        "status": "PASS",
        "scope": (
            "local existence and uniqueness in the displayed boxes; no global "
            "exclusion of other folds or cusps"
        ),
        "environment": {
            "python": platform.python_version(),
            "python_flint": flint.__version__,
            "arb_decimal_digits": CTX_DPS,
        },
        "boundary_side_cusp": boundary,
        "domain_side_cusp": domain,
    }
    (ART / "halfline_cusp_interval_newton.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    text = write_text(payload)
    (ART / "halfline_cusp_interval_newton.txt").write_text(
        text, encoding="utf-8"
    )
    print(text, end="")


if __name__ == "__main__":
    main()
