/-
Module 4: derivative structure of a signed sum of exponentials and the minimum number of modes
of a fold (Sec. III B and App. C of the paper).

  * Φ = Σ G_j e^{−μ_j τ} has Φ' = −S₁ and Φ'' = S₂ with S_n = Σ G_j μ_j^n e^{−μ_j τ} —
    `Phi_hasDerivAt`, `S1_hasDerivAt` (finite signed sums, differentiated term by term);
  * a two-term fold is impossible, even with signs: with A_j = G_j μ_j e^{−μ_j τ}, S₁ = 0
    forces A₂ = −A₁, and then S₂ = (μ₁−μ₂)A₁ = 0 gives A₁ = A₂ = 0 — `no_two_mode_fold`
    (abstract) and `no_two_mode_fold_exp` (with the exponential weights and μ > 0, concluding
    G₁ = G₂ = 0);
  * with three terms the fold ratio A₁:A₂:A₃ = (μ₃−μ₂):(μ₁−μ₃):(μ₂−μ₁) forces alternating
    signs — `three_mode_ratio` and `three_mode_alternating`.
-/
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Tactic

namespace DPMA

open Real Finset

/-- Term-by-term derivative: Φ(τ) = Σ_j G_j e^{−μ_j τ} has Φ'(τ) = −S₁(τ)
with S₁(τ) = Σ_j G_j μ_j e^{−μ_j τ}. -/
theorem Phi_hasDerivAt (n : ℕ) (G μ : Fin n → ℝ) (τ : ℝ) :
    HasDerivAt (fun t => ∑ j, G j * exp (-(μ j) * t))
      (-(∑ j, G j * μ j * exp (-(μ j) * τ))) τ := by
  have h : HasDerivAt (fun t => ∑ j, G j * exp (-(μ j) * t))
      (∑ j, -(G j * μ j * exp (-(μ j) * τ))) τ := by
    refine HasDerivAt.fun_sum fun j _ => ?_
    have hlin : HasDerivAt (fun t : ℝ => -(μ j) * t) (-(μ j)) τ :=
      hasDerivAt_const_mul (-(μ j))
    have hterm := hlin.exp.const_mul (G j)
    have hval : -(G j * μ j * exp (-(μ j) * τ)) = G j * (exp (-(μ j) * τ) * -(μ j)) := by
      ring
    rw [hval]
    exact hterm
  simpa using h

/-- Second derivative: S₁'(τ) = −S₂(τ), i.e. Φ'' = S₂. -/
theorem S1_hasDerivAt (n : ℕ) (G μ : Fin n → ℝ) (τ : ℝ) :
    HasDerivAt (fun t => ∑ j, G j * μ j * exp (-(μ j) * t))
      (-(∑ j, G j * μ j ^ 2 * exp (-(μ j) * τ))) τ := by
  have h : HasDerivAt (fun t => ∑ j, G j * μ j * exp (-(μ j) * t))
      (∑ j, -(G j * μ j ^ 2 * exp (-(μ j) * τ))) τ := by
    refine HasDerivAt.fun_sum fun j _ => ?_
    have hlin : HasDerivAt (fun t : ℝ => -(μ j) * t) (-(μ j)) τ :=
      hasDerivAt_const_mul (-(μ j))
    have hterm := hlin.exp.const_mul (G j * μ j)
    have hval : -(G j * μ j ^ 2 * exp (-(μ j) * τ))
        = G j * μ j * (exp (-(μ j) * τ) * -(μ j)) := by
      ring
    rw [hval]
    exact hterm
  simpa using h

/-- Two-mode fold impossibility, abstract linear-algebra core:
if `μ₁ ≠ μ₂`, `A₁ + A₂ = 0` and `μ₁A₁ + μ₂A₂ = 0`, then `A₁ = A₂ = 0`. -/
theorem no_two_mode_fold (μ₁ μ₂ A₁ A₂ : ℝ) (hμ : μ₁ ≠ μ₂)
    (h1 : A₁ + A₂ = 0) (h2 : μ₁ * A₁ + μ₂ * A₂ = 0) : A₁ = 0 ∧ A₂ = 0 := by
  have hd : μ₁ - μ₂ ≠ 0 := sub_ne_zero.mpr hμ
  have key : (μ₁ - μ₂) * A₁ = 0 := by linear_combination h2 - μ₂ * h1
  have hA1 : A₁ = 0 := by
    rcases mul_eq_zero.mp key with h | h
    · exact absurd h hd
    · exact h
  exact ⟨hA1, by linarith⟩

/-- Two-mode fold impossibility with the exponential weights:
for `μ₁ ≠ μ₂`, `μ₁, μ₂ > 0`, if S₁(τ) = S₂(τ) = 0 for the two-mode mixture then
`G₁ = G₂ = 0` (no nontrivial two-mode fold, even with signs). -/
theorem no_two_mode_fold_exp (μ₁ μ₂ G₁ G₂ τ : ℝ) (hμ : μ₁ ≠ μ₂)
    (hμ₁ : 0 < μ₁) (hμ₂ : 0 < μ₂)
    (hS1 : G₁ * μ₁ * exp (-μ₁ * τ) + G₂ * μ₂ * exp (-μ₂ * τ) = 0)
    (hS2 : G₁ * μ₁ ^ 2 * exp (-μ₁ * τ) + G₂ * μ₂ ^ 2 * exp (-μ₂ * τ) = 0) :
    G₁ = 0 ∧ G₂ = 0 := by
  have h := no_two_mode_fold μ₁ μ₂ (G₁ * μ₁ * exp (-μ₁ * τ)) (G₂ * μ₂ * exp (-μ₂ * τ)) hμ
    hS1 (by linear_combination hS2)
  constructor
  · simpa [mul_eq_zero, hμ₁.ne', exp_ne_zero] using h.1
  · simpa [mul_eq_zero, hμ₂.ne', exp_ne_zero] using h.2

/-- Three-mode fold ratio: if `A₁+A₂+A₃ = 0` and
`μ₁A₁+μ₂A₂+μ₃A₃ = 0` with `μ₂ ≠ μ₃`, then
`A₁ : A₂ : A₃ = (μ₃−μ₂) : (μ₁−μ₃) : (μ₂−μ₁)`, i.e.
`∃ t, A₁ = t(μ₃−μ₂) ∧ A₂ = t(μ₁−μ₃) ∧ A₃ = t(μ₂−μ₁)`. -/
theorem three_mode_ratio (μ₁ μ₂ μ₃ A₁ A₂ A₃ : ℝ) (h23 : μ₂ ≠ μ₃)
    (h1 : A₁ + A₂ + A₃ = 0) (h2 : μ₁ * A₁ + μ₂ * A₂ + μ₃ * A₃ = 0) :
    ∃ t : ℝ, A₁ = t * (μ₃ - μ₂) ∧ A₂ = t * (μ₁ - μ₃) ∧ A₃ = t * (μ₂ - μ₁) := by
  have hd : μ₃ - μ₂ ≠ 0 := sub_ne_zero.mpr (Ne.symm h23)
  refine ⟨A₁ / (μ₃ - μ₂), ?_, ?_, ?_⟩
  · rw [div_mul_eq_mul_div, eq_div_iff hd]
  · have key : (μ₁ - μ₃) * A₁ + (μ₂ - μ₃) * A₂ = 0 := by linear_combination h2 - μ₃ * h1
    rw [div_mul_eq_mul_div, eq_div_iff hd]
    linear_combination -key
  · have key : (μ₁ - μ₂) * A₁ + (μ₃ - μ₂) * A₃ = 0 := by linear_combination h2 - μ₂ * h1
    rw [div_mul_eq_mul_div, eq_div_iff hd]
    linear_combination key

/-- Alternating signs: under the fold ratio with ordered rates
`μ₁ < μ₂ < μ₃` and a nontrivial triple (`A₁ ≠ 0`), the signs alternate:
`A₁·A₂ < 0`, `A₂·A₃ < 0` and `A₁·A₃ > 0`. -/
theorem three_mode_alternating (μ₁ μ₂ μ₃ A₁ A₂ A₃ : ℝ)
    (h12 : μ₁ < μ₂) (h23 : μ₂ < μ₃) (hA : A₁ ≠ 0)
    (h1 : A₁ + A₂ + A₃ = 0) (h2 : μ₁ * A₁ + μ₂ * A₂ + μ₃ * A₃ = 0) :
    A₁ * A₂ < 0 ∧ A₂ * A₃ < 0 ∧ 0 < A₁ * A₃ := by
  obtain ⟨t, hA1, hA2, hA3⟩ := three_mode_ratio μ₁ μ₂ μ₃ A₁ A₂ A₃ h23.ne h1 h2
  have ht : t ≠ 0 := by
    intro h0
    exact hA (by rw [hA1, h0, zero_mul])
  have ht2 : 0 < t * t := mul_self_pos.mpr ht
  have h32 : 0 < μ₃ - μ₂ := sub_pos.mpr h23
  have h31 : 0 < μ₃ - μ₁ := sub_pos.mpr (h12.trans h23)
  have h21 : 0 < μ₂ - μ₁ := sub_pos.mpr h12
  refine ⟨?_, ?_, ?_⟩
  · rw [hA1, hA2]
    nlinarith [mul_pos ht2 (mul_pos h32 h31)]
  · rw [hA2, hA3]
    nlinarith [mul_pos ht2 (mul_pos h31 h21)]
  · rw [hA1, hA3]
    nlinarith [mul_pos ht2 (mul_pos h32 h21)]

end DPMA
