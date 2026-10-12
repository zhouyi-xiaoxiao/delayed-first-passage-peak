#!/usr/bin/env python3
"""Exact rational checks of the interval certificates of the two half-line cusps (App. C, App. D).

The Arb certificates (python-flint; halfline_cusp_interval_newton.py and halfline_cusp_krawczyk.py)
store every enclosure as decimal numbers: lower/upper bounds in data/dpma_interval_certificates.json
(interval Newton) and integer midpoints and radii with a power of ten in
data/dpma_halfline_cusp_interval_certificate.json (Krawczyk).  This script reads these decimals as
exact rationals (fractions.Fraction; no floating point) and verifies every inclusion and sign
exclusion quoted in the paper:

  interval Newton, for each box Q- (wall side, X < 1) and Q+ (far side, X > 1)
    N1  the box of Eq. (cusp-certificate-boxes) lies inside the certified box (uniqueness in Q)
    N2  the interval-Newton image lies strictly inside Q (existence in Q)
    N3  density > 0, Phi'''' < 0, det of the full cusp Jacobian excludes zero
    N4  Q-: the enclosures of sigma^4 Phi''''/Phi and of the normalized unfolding determinant lie in
        the quoted intervals; Q+: both enclosures exclude zero and are centred at the quoted values
    N5  the enclosure of det J is consistent with the factorization
        det J = Phi^3/(B sigma^7) (sigma^4 Phi''''/Phi) (normalized unfolding determinant)
  Krawczyk, for the same two cusps
    K1  the Krawczyk image lies strictly inside the box (coordinates log B, log T, X)
    K2  contraction: every row-sum bound is < 1; the preconditioner determinant excludes zero
    K3  density > 0, I4 < 0, unfolding and full Jacobian determinants exclude zero
    K4  the X-components of the Krawczyk and interval-Newton images overlap
    K5  the two Jacobian enclosures agree: det_(log B, log T, X) = B T det_(B, T, X)
  independent double-precision solve (cusp_endpoint_continuation.py; run that step first)
    F1  the cusp found by Newton's method in double precision lies in Q- and Q+, and its margins
        lie inside the certified enclosures.

The comparisons N1-N4 and K1-K4 are also written as Lean 4 statements on integers (closed by
`decide`, no mathlib needed) to outputs/tables/CuspCertificateDecimals.lean; reproduce_all.py checks
that this file is identical to formal/FormalLean/CuspCertificateDecimals.lean.
Writes outputs/tables/certificate_rational_checks.{txt,json} and the Lean file.
"""
from __future__ import annotations

import json
from fractions import Fraction as Fr

from paths import DATA, TABLES

LINES: list[str] = []


def say(s=""):
    print(s, flush=True)
    LINES.append(s)


# Eq. (cusp-certificate-boxes): centre and radius of each coordinate, as printed in the paper.
PAPER_BOXES = {
    "boundary": {"B": ("2.868771925558728", "1e-10"), "T": ("0.06362964429621742", "1e-11"),
                 "X": ("0.7583136631287806", "1e-11")},
    "domain": {"B": ("1.0856828303187425", "1e-10"), "T": ("0.2072300833529535", "1e-11"),
               "X": ("1.424897156781147", "1e-10")},
}
# App. C: Q- enclosures; Q+ centres (six decimals, so |centre - quoted| <= 5e-7).
PAPER_QMINUS = {"normalized_fourth_margin": ("-5.53353583", "-5.53353573"),
                "normalized_unfolding_determinant_logB_X": ("3.72087719", "3.72087726")}
PAPER_QPLUS_CENTRES = {"normalized_fourth_margin": "-6.052596",
                       "normalized_unfolding_determinant_logB_X": "-1.379343"}
SIDE_NAME = {"boundary": "Q- (wall side)", "domain": "Q+ (far side)"}
LEAN_SIDE = {"boundary": "boundaryCusp", "domain": "domainCusp"}


# ------------------------------------------------------------------ exact interval helpers
def iv(record):
    """[lower, upper] of an interval-Newton record (decimal strings)."""
    return Fr(record["lower"]), Fr(record["upper"])


def ball(record):
    """[mid - rad, mid + rad] * 10^exponent of a Krawczyk record (integers)."""
    d = record["decimal_outer_ball"]
    scale = Fr(10) ** int(d["exponent"])
    mid, rad = int(d["mid_integer"]), int(d["radius_integer"])
    return (mid - rad) * scale, (mid + rad) * scale


def mul(a, b):
    p = [a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1]]
    return min(p), max(p)


def power(a, n):
    out = (Fr(1), Fr(1))
    for _ in range(n):
        out = mul(out, a)
    return out


def inverse(a):
    if a[0] <= 0 <= a[1]:
        raise ZeroDivisionError("interval contains zero")
    return 1 / a[1], 1 / a[0]


def overlap(a, b):
    return a[0] <= b[1] and b[0] <= a[1]


def excludes_zero(a):
    return a[1] < 0 or a[0] > 0


def fmt(x, digits=12):
    return f"{float(x):.{digits}g}"


# ------------------------------------------------------------------ comparison bookkeeping
class Group:
    """A named list of exact comparisons a < b or a <= b (all rationals with power-of-ten denominators)."""

    def __init__(self, name, doc):
        self.name, self.doc, self.items = name, doc, []

    def lt(self, a, b, label):
        self.items.append((a, "<", b, label))

    def le(self, a, b, label):
        self.items.append((a, "<=", b, label))

    def ok(self):
        return all((a < b) if op == "<" else (a <= b) for a, op, b, _ in self.items)


def decimal_scale(*values):
    k = 0
    while any((v * 10 ** k).denominator != 1 for v in values):
        k += 1
        if k > 400:
            raise ValueError("value is not a terminating decimal")
    return k


def lean_group(group):
    lines = [f"/-- {group.doc} -/", f"theorem {group.name} :"]
    for i, (a, op, b, label) in enumerate(group.items):
        k = decimal_scale(a, b)
        A, B = int(a * 10 ** k), int(b * 10 ** k)
        sym = "<" if op == "<" else "≤"
        joint = " ∧" if i < len(group.items) - 1 else " := by"
        lines.append(f"    -- {label} (both sides × 10^{k})")
        lines.append(f"    ({A} : Int) {sym} {B}{joint}")
    lines.append("  decide")
    return "\n".join(lines)


def main():
    newton = json.loads((DATA / "dpma_interval_certificates.json").read_text())
    kraw = json.loads((DATA / "dpma_halfline_cusp_interval_certificate.json").read_text())
    kraw_by_side = {"boundary": None, "domain": None}
    for cert in kraw["certificates"]:
        kraw_by_side["boundary" if cert["branch"] == "X_lt_1" else "domain"] = cert
    groups: list[Group] = []
    python_only: list[tuple[str, bool, str]] = []
    say("Exact rational checks of the stored Arb certificates (fractions.Fraction, no floating point)")

    for side in ("boundary", "domain"):
        c = newton[f"{side}_side_cusp"]
        name = LEAN_SIDE[side]
        paper = {v: (Fr(cen) - Fr(rad), Fr(cen) + Fr(rad)) for v, (cen, rad) in PAPER_BOXES[side].items()}
        cert_box = {v: iv(c["box"][v]) for v in ("B", "T", "X")}
        image = {v: iv(c["interval_newton_image"][v]) for v in ("B", "T", "X")}
        # N1, N2
        g1 = Group(f"{name}_paperBox_inside_certifiedBox",
                   f"{SIDE_NAME[side]}: the box of Eq. (cusp-certificate-boxes) lies inside the box "
                   f"certified by interval Newton (coordinates B, σ, X).")
        g2 = Group(f"{name}_newtonImage_inside_paperBox",
                   f"{SIDE_NAME[side]}: the interval-Newton image lies strictly inside the box of "
                   f"Eq. (cusp-certificate-boxes) (coordinates B, σ, X).")
        for v in ("B", "T", "X"):
            sym = "σ" if v == "T" else v
            g1.le(cert_box[v][0], paper[v][0], f"{sym}: certified lower bound ≤ quoted lower bound")
            g1.le(paper[v][1], cert_box[v][1], f"{sym}: quoted upper bound ≤ certified upper bound")
            g2.lt(paper[v][0], image[v][0], f"{sym}: quoted lower bound < image lower bound")
            g2.lt(image[v][1], paper[v][1], f"{sym}: image upper bound < quoted upper bound")
        # N3
        dens, m4, det = iv(c["density"]), iv(c["m4"]), iv(c["full_jacobian_determinant"])
        g3 = Group(f"{name}_newton_signs",
                   f"{SIDE_NAME[side]}: on the certified box the density is positive, Φ'''' is negative "
                   f"and the determinant of the full cusp Jacobian excludes zero.")
        g3.lt(Fr(0), dens[0], "density: 0 < lower bound")
        g3.lt(m4[1], Fr(0), "Φ'''': upper bound < 0")
        if det[1] < 0:
            g3.lt(det[1], Fr(0), "det J: upper bound < 0")
        else:
            g3.lt(Fr(0), det[0], "det J: 0 < lower bound")
        # N4
        fourth, unfold = iv(c["normalized_fourth_margin"]), iv(c["normalized_unfolding_determinant_logB_X"])
        if side == "boundary":
            g4 = Group(f"{name}_margins_in_quoted_intervals",
                       "Q- (wall side): the enclosures of σ⁴Φ''''/Φ and of the normalized unfolding "
                       "determinant lie in the intervals quoted in App. C, so both exclude zero.")
            lo, hi = (Fr(x) for x in PAPER_QMINUS["normalized_fourth_margin"])
            g4.le(lo, fourth[0], "σ⁴Φ''''/Φ: quoted lower bound ≤ certified lower bound")
            g4.le(fourth[1], hi, "σ⁴Φ''''/Φ: certified upper bound ≤ quoted upper bound")
            g4.lt(hi, Fr(0), "σ⁴Φ''''/Φ: quoted upper bound < 0")
            lo, hi = (Fr(x) for x in PAPER_QMINUS["normalized_unfolding_determinant_logB_X"])
            g4.le(lo, unfold[0], "unfolding determinant: quoted lower bound ≤ certified lower bound")
            g4.le(unfold[1], hi, "unfolding determinant: certified upper bound ≤ quoted upper bound")
            g4.lt(Fr(0), lo, "unfolding determinant: 0 < quoted lower bound")
        else:
            g4 = Group(f"{name}_margins_exclude_zero",
                       "Q+ (far side): the enclosures of σ⁴Φ''''/Φ and of the normalized unfolding "
                       "determinant exclude zero and are centred at the values quoted in App. C "
                       "(|centre − quoted| ≤ 5×10⁻⁷).")
            for key, label in (("normalized_fourth_margin", "σ⁴Φ''''/Φ"),
                               ("normalized_unfolding_determinant_logB_X", "unfolding determinant")):
                enc = fourth if key == "normalized_fourth_margin" else unfold
                centre2 = 2 * Fr(PAPER_QPLUS_CENTRES[key])
                g4.lt(enc[1], Fr(0), f"{label}: upper bound < 0")
                g4.le(centre2 - Fr(1, 10 ** 6), enc[0] + enc[1], f"{label}: 2 × centre ≥ 2 × quoted − 10⁻⁶")
                g4.le(enc[0] + enc[1], centre2 + Fr(1, 10 ** 6), f"{label}: 2 × centre ≤ 2 × quoted + 10⁻⁶")
        groups += [g1, g2, g3, g4]
        # N5 (factorization of the cusp Jacobian; interval arithmetic, Python only)
        sigma, B = cert_box["T"], cert_box["B"]
        product = mul(mul(mul(power(dens, 3), inverse(mul(B, power(sigma, 7)))), fourth), unfold)
        consistent = overlap(product, det)
        python_only.append((f"{SIDE_NAME[side]}: det J enclosure [{fmt(det[0])}, {fmt(det[1])}] overlaps "
                            f"Phi^3/(B sigma^7) (sigma^4 Phi''''/Phi)(normalized unfolding det) "
                            f"[{fmt(product[0])}, {fmt(product[1])}]", consistent, f"{name}_factorization"))

        # Krawczyk
        k = kraw_by_side[side]
        kb = {v: ball(k["box"][v]) for v in ("log_B", "log_T", "X", "B", "T")}
        ki = {v: ball(k["krawczyk_image"][v]) for v in ("log_B", "log_T", "X")}
        h1 = Group(f"{name}_krawczykImage_inside_box",
                   f"{SIDE_NAME[side]}, Krawczyk: the image lies strictly inside the box "
                   f"(coordinates log B, log σ, X).")
        for v, sym in (("log_B", "log B"), ("log_T", "log σ"), ("X", "X")):
            h1.lt(kb[v][0], ki[v][0], f"{sym}: box lower bound < image lower bound")
            h1.lt(ki[v][1], kb[v][1], f"{sym}: image upper bound < box upper bound")
        h2 = Group(f"{name}_krawczyk_contraction",
                   f"{SIDE_NAME[side]}, Krawczyk: every row-sum bound of the contraction is below 1 and "
                   f"the preconditioner determinant excludes zero.")
        for i, row in enumerate(k["krawczyk_image"]["contraction_row_sum_bounds"]):
            h2.lt(ball(row)[1], Fr(1), f"row {i + 1}: upper bound < 1")
        pdet = ball(k["krawczyk_image"]["preconditioner_determinant"])
        if pdet[0] > 0:
            h2.lt(Fr(0), pdet[0], "preconditioner determinant: 0 < lower bound")
        else:
            h2.lt(pdet[1], Fr(0), "preconditioner determinant: upper bound < 0")
        h3 = Group(f"{name}_krawczyk_signs",
                   f"{SIDE_NAME[side]}, Krawczyk: on the box the density is positive, I₄ = Φ'''' is "
                   f"negative, and the unfolding and full Jacobian determinants exclude zero.")
        enc = {key: ball(val) for key, val in k["box_enclosures"].items()}
        h3.lt(Fr(0), enc["density_I0"][0], "density: 0 < lower bound")
        h3.lt(enc["fourth_time_derivative_I4"][1], Fr(0), "I₄: upper bound < 0")
        for key, label in (("control_unfolding_determinant", "unfolding determinant"),
                           ("full_cusp_jacobian_determinant", "full Jacobian determinant")):
            if enc[key][0] > 0:
                h3.lt(Fr(0), enc[key][0], f"{label}: 0 < lower bound")
            else:
                h3.lt(enc[key][1], Fr(0), f"{label}: upper bound < 0")
        h4 = Group(f"{name}_krawczyk_newton_X_overlap",
                   f"{SIDE_NAME[side]}: the X-components of the Krawczyk and interval-Newton images overlap.")
        h4.le(ki["X"][0], image["X"][1], "Krawczyk lower bound ≤ Newton upper bound")
        h4.le(image["X"][0], ki["X"][1], "Newton lower bound ≤ Krawczyk upper bound")
        groups += [h1, h2, h3, h4]
        # K5: det in (log B, log T, X) = B T det in (B, T, X)  (Python only)
        scaled = mul(mul(kb["B"], kb["T"]), det)
        python_only.append((f"{SIDE_NAME[side]}: Krawczyk det [{fmt(enc['full_cusp_jacobian_determinant'][0])}, "
                            f"{fmt(enc['full_cusp_jacobian_determinant'][1])}] overlaps B T det_Newton "
                            f"[{fmt(scaled[0])}, {fmt(scaled[1])}]",
                            overlap(scaled, enc["full_cusp_jacobian_determinant"]), f"{name}_two_jacobians"))

    # F1: independent double-precision cusp solve vs the certified boxes and enclosures
    solve_path = TABLES / "cusp_endpoint_continuation.json"
    if solve_path.exists():
        solved = json.loads(solve_path.read_text())["halfline_cusps"]
        for side, key in (("boundary", "boundary_side"), ("domain", "domain_side")):
            s = solved[key]
            c = newton[f"{side}_side_cusp"]
            inside = all(Fr(cen) - Fr(rad) < Fr(s[v]) < Fr(cen) + Fr(rad)
                         for v, (cen, rad) in PAPER_BOXES[side].items())
            m_in = (iv(c["normalized_fourth_margin"])[0] <= Fr(s["T4_S4_over_Phi"]) <= iv(c["normalized_fourth_margin"])[1]
                    and iv(c["normalized_unfolding_determinant_logB_X"])[0] <= Fr(s["unfolding_determinant_logB_X"])
                    <= iv(c["normalized_unfolding_determinant_logB_X"])[1])
            python_only.append((f"{SIDE_NAME[side]}: double-precision cusp (X, B, sigma) = "
                                f"({s['X']:.13f}, {s['B']:.12f}, {s['T']:.14f}) lies in the box; margins "
                                f"{s['T4_S4_over_Phi']:.9f}, {s['unfolding_determinant_logB_X']:.9f} lie in the "
                                f"certified enclosures", inside and m_in, f"{LEAN_SIDE[side]}_double_precision"))
    else:
        python_only.append(("double-precision cusp solve not found (run cusp_endpoint_continuation.py first)",
                            False, "double_precision"))

    n_cmp = sum(len(g.items) for g in groups)
    for g in groups:
        say(f"  {'ok  ' if g.ok() else 'FAIL'} {g.name}: {len(g.items)} exact comparisons")
    for text, ok, _ in python_only:
        say(f"  {'ok  ' if ok else 'FAIL'} {text}")
    all_ok = all(g.ok() for g in groups) and all(ok for _, ok, _ in python_only)
    say(f"exact comparisons: {n_cmp} in {len(groups)} groups, all hold: {all(g.ok() for g in groups)}; "
        f"interval-arithmetic consistency checks: {sum(ok for _, ok, _ in python_only)} of {len(python_only)} hold")
    say(f"result: {'PASS' if all_ok else 'FAIL'}")

    header = [
        "/-",
        "Exact decimal checks of the interval certificates of the two half-line cusps (App. C, App. D).",
        "",
        "Generated by code/certificate_rational_checks.py from data/dpma_interval_certificates.json",
        "(interval Newton) and data/dpma_halfline_cusp_interval_certificate.json (Krawczyk); do not edit.",
        "Every certified bound is a terminating decimal; each comparison below is the corresponding",
        "comparison of integers after multiplying both sides by the same power of ten, decided by the",
        "kernel.  Only core Lean is used (no mathlib).",
        "-/",
        "",
        "namespace DPMA.CuspCertificates",
        "",
        "",
    ]
    lean = "\n".join(header) + "\n\n".join(lean_group(g) for g in groups) + "\n\nend DPMA.CuspCertificates\n"
    (TABLES / "CuspCertificateDecimals.lean").write_text(lean)
    (TABLES / "certificate_rational_checks.txt").write_text("\n".join(LINES) + "\n")
    (TABLES / "certificate_rational_checks.json").write_text(json.dumps({
        "exact_comparisons": n_cmp, "groups": [{"name": g.name, "comparisons": len(g.items), "hold": g.ok()} for g in groups],
        "interval_consistency": [{"name": n, "holds": ok} for _, ok, n in python_only], "pass": all_ok,
    }, indent=2) + "\n")
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
