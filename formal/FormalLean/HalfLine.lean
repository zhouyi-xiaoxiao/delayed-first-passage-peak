/-
Module 7: the half-line transform of the near-end limit (Sec. III D and App. C of the paper).

  * the absorption-time transform for release at the sink
      f̂(s;B) = [e^{−ϰ} + (B/ϰ)(1−e^{−2ϰ})] / [1 + (B/ϰ)(1−e^{−2ϰ})],  ϰ = √(2s),
    assembled from the half-line Dirichlet resolvent r₀(y,y') = 2 sinh(ϰ y_<)e^{−ϰ y_>}/ϰ
    by Sherman–Morrison:
      — `r0_11_closed`: B·r₀(1,1) = (B/ϰ)(1−e^{−2ϰ})  (the sinh→exp reduction);
      — `sm_scalar`: r_b(y,1) = r₀(y,1)/(1 + B r₀(1,1))  (rank-one defect at y = 1);
      — `flux_at_zero`: the boundary flux ½∂_y r₀(y,1)|_{y=0} = e^{−ϰ};
      — `fhat_assembly`: flux + capture at the sink = f̂;
      — `fhat_levy`: B = 0 reduces f̂ to the Lévy transform e^{−ϰ}.
  * the branch-cut integrand — on s = −v²/2 (ϰ = iv), with g = 2B sin v/v:
      — `cut_g_transform`: (B/(iv))(1−e^{−2iv}) = g·e^{−iv};
      — `cut_normSq`: |1 + g e^{−iv}|² = 1 + 2g cos v + g²;
      — `cut_denominator_form`: 1 + 2g cos v + g² = (g + cos v)² + sin²v, and
        `cut_denominator_pos`: it is positive for v ≠ 0;
      — `cut_im`: −Im[(1+g)e^{−iv}/(1+g e^{−iv})] = (1+g) sin v/(1+2g cos v+g²).
  The numerical values B* = 0.7890262 and c_* = 0.1579221 are the fold of the resulting
  integral and are not formalized (they are computed by the Python scripts).
-/
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Deriv
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Analysis.SpecialFunctions.Complex.Circle
import Mathlib.Tactic

namespace DPMA

open Real

set_option linter.unusedVariables false in
/-- sinh→exp reduction: `B·[2 sinh(ϰ)e^{−ϰ}/ϰ] = (B/ϰ)(1 − e^{−2ϰ})`. -/
theorem r0_11_closed (ϰ B : ℝ) (hϰ : ϰ ≠ 0) :
    B * (2 * Real.sinh ϰ * exp (-ϰ) / ϰ) = (B / ϰ) * (1 - exp (-(2 * ϰ))) := by
  have h1 : Real.exp ϰ * Real.exp (-ϰ) = 1 := by
    rw [← Real.exp_add]
    simp
  have h2 : Real.exp (-(2 * ϰ)) = Real.exp (-ϰ) * Real.exp (-ϰ) := by
    rw [← Real.exp_add]
    congr 1
    ring
  have key : 2 * Real.sinh ϰ * Real.exp (-ϰ) = 1 - Real.exp (-(2 * ϰ)) := by
    rw [Real.sinh_eq, h2]
    linear_combination h1
  rw [key]
  ring

/-- Scalar Sherman–Morrison for the defect kernel:
with `p = r₀(y,1)`, `c = r₀(1,1)` and `1 + B c ≠ 0`,
`p − B·p·c/(1 + B·c) = p/(1 + B·c)`. -/
theorem sm_scalar (p c B : ℝ) (hden : 1 + B * c ≠ 0) :
    p - B * p * c / (1 + B * c) = p / (1 + B * c) := by
  field_simp [hden]
  ring

/-- Boundary flux: `y ↦ 2 sinh(ϰy)e^{−ϰ}/ϰ` has derivative `2·e^{−ϰ}·cosh(ϰy)`
at any `y`; at `y = 0` the half flux is `½·2e^{−ϰ} = e^{−ϰ}` (the Lévy weight). -/
theorem flux_at_zero (ϰ : ℝ) (hϰ : ϰ ≠ 0) :
    HasDerivAt (fun y => 2 * Real.sinh (ϰ * y) * exp (-ϰ) / ϰ)
      (2 * exp (-ϰ) * Real.cosh (ϰ * 0)) 0
    ∧ (1 / 2) * (2 * exp (-ϰ) * Real.cosh (ϰ * 0)) = exp (-ϰ) := by
  constructor
  · have h1 : HasDerivAt (fun y : ℝ => ϰ * y) ϰ 0 := by
      simpa using (hasDerivAt_id (0 : ℝ)).const_mul ϰ
    have h2 : HasDerivAt (fun y : ℝ => Real.sinh (ϰ * y)) (Real.cosh (ϰ * 0) * ϰ) 0 :=
      h1.sinh
    have h4 : HasDerivAt (fun y : ℝ => 2 * Real.sinh (ϰ * y) * Real.exp (-ϰ) / ϰ)
        (2 * (Real.cosh (ϰ * 0) * ϰ) * Real.exp (-ϰ) / ϰ) 0 :=
      ((h2.const_mul 2).mul_const (Real.exp (-ϰ))).div_const ϰ
    have heq : 2 * (Real.cosh (ϰ * 0) * ϰ) * Real.exp (-ϰ) / ϰ
        = 2 * Real.exp (-ϰ) * Real.cosh (ϰ * 0) := by
      rw [div_eq_iff hϰ]
      ring
    rw [← heq]
    exact h4
  · rw [mul_zero, Real.cosh_zero]
    ring

set_option linter.unusedVariables false in
/-- Assembly of the half-line transform: flux term + sink-capture term over the common Sherman–Morrison
denominator `Δ = 1 + (B/ϰ)(1−e^{−2ϰ})`:
`e^{−ϰ}/Δ + [(B/ϰ)(1−e^{−2ϰ})]/Δ = [e^{−ϰ} + (B/ϰ)(1−e^{−2ϰ})]/Δ`. -/
theorem fhat_assembly (ϰ B : ℝ)
    (hden : 1 + (B / ϰ) * (1 - exp (-(2 * ϰ))) ≠ 0) :
    exp (-ϰ) / (1 + (B / ϰ) * (1 - exp (-(2 * ϰ))))
        + (B / ϰ) * (1 - exp (-(2 * ϰ))) / (1 + (B / ϰ) * (1 - exp (-(2 * ϰ))))
      = (exp (-ϰ) + (B / ϰ) * (1 - exp (-(2 * ϰ))))
        / (1 + (B / ϰ) * (1 - exp (-(2 * ϰ)))) := by
  rw [← add_div]

/-- Lévy reduction: at `B = 0`, f̂ = e^{−ϰ}. -/
theorem fhat_levy (ϰ : ℝ) :
    (exp (-ϰ) + (0 / ϰ) * (1 - exp (-(2 * ϰ))))
        / (1 + (0 / ϰ) * (1 - exp (-(2 * ϰ)))) = exp (-ϰ) := by
  simp

/-- Euler split of the branch-cut phase factor: `e^{−iv} = cos v − i sin v` (real `v`). -/
private lemma exp_neg_ofReal_mul_I (v : ℝ) :
    Complex.exp (-((v : ℂ) * Complex.I))
      = (Real.cos v : ℂ) - (Real.sin v : ℂ) * Complex.I := by
  rw [show -((v : ℂ) * Complex.I) = (-(v : ℂ)) * Complex.I by ring, Complex.exp_mul_I,
    Complex.cos_neg, Complex.sin_neg, ← Complex.ofReal_cos, ← Complex.ofReal_sin]
  ring

/-- Branch-cut substitution: on `ϰ = iv` (`v ≠ 0`), with `g = 2B sin v/v`,
`(B/(iv))·(1 − e^{−2iv}) = g·e^{−iv}` as complex numbers. -/
theorem cut_g_transform (B v : ℝ) (hv : v ≠ 0) :
    (B / (Complex.I * v)) * (1 - Complex.exp (-(2 * v * Complex.I)))
      = ((2 * B * Real.sin v / v : ℝ) : ℂ) * Complex.exp (-(v * Complex.I)) := by
  have hw : (v : ℂ) ≠ 0 := Complex.ofReal_ne_zero.mpr hv
  have hexp := exp_neg_ofReal_mul_I v
  have h2 : Complex.exp (-(2 * (v : ℂ) * Complex.I))
      = ((Real.cos v : ℂ) - (Real.sin v : ℂ) * Complex.I) ^ 2 := by
    rw [show -(2 * (v : ℂ) * Complex.I)
          = -((v : ℂ) * Complex.I) + -((v : ℂ) * Complex.I) by ring,
      Complex.exp_add, hexp]
    ring
  have hpyth : (Real.sin v : ℂ) ^ 2 + (Real.cos v : ℂ) ^ 2 = 1 := by
    exact_mod_cast Real.sin_sq_add_cos_sq v
  have hkey : (1 : ℂ) - ((Real.cos v : ℂ) - (Real.sin v : ℂ) * Complex.I) ^ 2
      = 2 * (Real.sin v : ℂ) * Complex.I
          * ((Real.cos v : ℂ) - (Real.sin v : ℂ) * Complex.I) := by
    linear_combination (Real.sin v : ℂ) ^ 2 * Complex.I_sq - hpyth
  have hcast : ((2 * B * Real.sin v / v : ℝ) : ℂ)
      = 2 * (B : ℂ) * (Real.sin v : ℂ) / (v : ℂ) := by
    push_cast
    try ring
  have hIv : Complex.I * (v : ℂ) ≠ 0 := mul_ne_zero Complex.I_ne_zero hw
  rw [h2, hexp, hcast, hkey, div_mul_eq_mul_div, div_mul_eq_mul_div,
    div_eq_div_iff hIv hw]
  ring

/-- Branch-cut denominator modulus: for real `g`, `v`,
`|1 + g e^{−iv}|² = 1 + 2g cos v + g²`. -/
theorem cut_normSq (g v : ℝ) :
    Complex.normSq (1 + (g : ℂ) * Complex.exp (-(v * Complex.I)))
      = 1 + 2 * g * Real.cos v + g ^ 2 := by
  have h1 : (1 : ℂ) + (g : ℂ) * Complex.exp (-((v : ℂ) * Complex.I))
      = ((1 + g * Real.cos v : ℝ) : ℂ) + ((-(g * Real.sin v) : ℝ) : ℂ) * Complex.I := by
    rw [exp_neg_ofReal_mul_I]
    push_cast
    ring
  rw [h1, Complex.normSq_add_mul_I]
  linear_combination g ^ 2 * Real.sin_sq_add_cos_sq v

/-- Positivity form of the branch-cut denominator:
`1 + 2g cos v + g² = (g + cos v)² + sin²v`. -/
theorem cut_denominator_form (g v : ℝ) :
    1 + 2 * g * Real.cos v + g ^ 2 = (g + Real.cos v) ^ 2 + Real.sin v ^ 2 := by
  linear_combination -Real.sin_sq_add_cos_sq v

/-- Strict positivity of the branch-cut denominator on the support:
with the actual `g(v) = 2B sin v/v`, `1 + 2g cos v + g² > 0` for every `v ≠ 0`
(where `sin v = 0` forces `g = 0` and the denominator is `1`; elsewhere `sin²v > 0`). -/
theorem cut_denominator_pos (B v : ℝ) (hv : v ≠ 0) :
    0 < 1 + 2 * (2 * B * Real.sin v / v) * Real.cos v + (2 * B * Real.sin v / v) ^ 2 := by
  rw [cut_denominator_form]
  rcases eq_or_ne (Real.sin v) 0 with hs | hs
  · have hg : 2 * B * Real.sin v / v = 0 := by rw [hs]; simp
    rw [hg, hs]
    nlinarith [Real.sin_sq_add_cos_sq v, hs]
  · have h1 : 0 < Real.sin v ^ 2 := by positivity
    nlinarith [sq_nonneg (2 * B * Real.sin v / v + Real.cos v)]

set_option linter.unusedVariables false in
/-- Branch-cut integrand: for real `g`, `v` with `1 + 2g cos v + g² ≠ 0`,
`−Im[(1+g)·e^{−iv}/(1 + g·e^{−iv})] = (1+g)·sin v/(1 + 2g cos v + g²)`. -/
theorem cut_im (g v : ℝ) (hden : 1 + 2 * g * Real.cos v + g ^ 2 ≠ 0) :
    -((((1 + g : ℝ) : ℂ) * Complex.exp (-(v * Complex.I))
        / (1 + (g : ℂ) * Complex.exp (-(v * Complex.I)))).im)
      = (1 + g) * Real.sin v / (1 + 2 * g * Real.cos v + g ^ 2) := by
  rw [Complex.div_im, cut_normSq g v, exp_neg_ofReal_mul_I v]
  simp only [Complex.mul_re, Complex.mul_im, Complex.add_re, Complex.add_im, Complex.sub_re,
    Complex.sub_im, Complex.one_re, Complex.one_im, Complex.ofReal_re, Complex.ofReal_im,
    Complex.I_re, Complex.I_im, mul_zero, mul_one, zero_mul, sub_zero, zero_sub, add_zero,
    zero_add, neg_zero, mul_neg]
  field_simp [hden]
  ring

end DPMA
