/-
Module 5: normal form of the fold — gap and height-difference prefactors (App. C of the paper).

  * near the fold S₁(τ,b) ≃ ½S₃(τ−τ_c)² − S_{1,b}(b_c−b) with S₃ > 0, S_{1,b} > 0; the two
    extrema sit at τ_± = τ_c ± δ with δ = [2S_{1,b}(b_c−b)/S₃]^{1/2}, and the height
    difference is ΔΦ = −∫_{τ₋}^{τ₊} S₁ dτ = (2/3)S₃δ³;
  * τ₊ − τ₋ = 2[2S_{1,b}/S₃]^{1/2}(b_c−b)^{1/2} and
    ΔΦ = (4√2/3)·S_{1,b}^{3/2}/S₃^{1/2}·(b_c−b)^{3/2}.
  Both constants are proved exactly for the truncated normal form below: `nf_roots` (the
  extrema), `nf_gap` (the gap constant 2√(2S_{1,b}/S₃)) and `nf_prominence` (the constant 4√2/3).
  Notation: δb := b_c − b > 0; Φnf is the exact antiderivative of −S₁nf.
-/
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Analysis.Calculus.Deriv.Add
import Mathlib.Analysis.Calculus.Deriv.Mul
import Mathlib.Analysis.Calculus.Deriv.Pow
import Mathlib.Analysis.Calculus.Deriv.Inv
import Mathlib.Tactic

namespace DPMA

open Real

/-- Truncated normal form of the first moment: `S₁nf(τ) = ½S₃(τ−τ_c)² − S1b·δb`. -/
noncomputable def S1nf (S₃ S1b δb τc τ : ℝ) : ℝ :=
  (1 / 2) * S₃ * (τ - τc) ^ 2 - S1b * δb

/-- Exact antiderivative of `−S₁nf` (the density along the normal form):
`Φnf(τ) = S1b·δb·(τ−τ_c) − S₃(τ−τ_c)³/6`. -/
noncomputable def Phinf (S₃ S1b δb τc τ : ℝ) : ℝ :=
  S1b * δb * (τ - τc) - S₃ * (τ - τc) ^ 3 / 6

/-- Φnf is an antiderivative of −S₁nf. -/
theorem Phinf_hasDerivAt (S₃ S1b δb τc τ : ℝ) :
    HasDerivAt (Phinf S₃ S1b δb τc) (-(S1nf S₃ S1b δb τc τ)) τ := by
  have h1 : HasDerivAt (fun t : ℝ => t - τc) 1 τ := (hasDerivAt_id τ).sub_const τc
  have h3 : HasDerivAt (fun t : ℝ => (t - τc) ^ 3) (3 * (τ - τc) ^ 2) τ := by
    have h := h1.fun_pow 3
    norm_num at h
    exact h
  have h5 : HasDerivAt (fun t : ℝ => S1b * δb * (t - τc) - S₃ * (t - τc) ^ 3 / 6)
      (S1b * δb * 1 - S₃ * (3 * (τ - τc) ^ 2) / 6) τ :=
    (h1.const_mul (S1b * δb)).sub ((h3.const_mul S₃).div_const 6)
  have heq : -(S1nf S₃ S1b δb τc τ) = S1b * δb * 1 - S₃ * (3 * (τ - τc) ^ 2) / 6 := by
    simp only [S1nf]; ring
  rw [heq]
  exact h5

/-- The two extrema (roots of S₁nf): `τ_± = τ_c ± √(2·S1b·δb/S₃)`. -/
theorem nf_roots (S₃ S1b δb τc : ℝ) (hS₃ : 0 < S₃) (hS1b : 0 < S1b) (hδb : 0 < δb) :
    S1nf S₃ S1b δb τc (τc + Real.sqrt (2 * S1b * δb / S₃)) = 0
    ∧ S1nf S₃ S1b δb τc (τc - Real.sqrt (2 * S1b * δb / S₃)) = 0 := by
  have hx : (0 : ℝ) ≤ 2 * S1b * δb / S₃ := by positivity
  have hsq : Real.sqrt (2 * S1b * δb / S₃) ^ 2 = 2 * S1b * δb / S₃ := Real.sq_sqrt hx
  have hS₃' : S₃ ≠ 0 := ne_of_gt hS₃
  constructor
  · simp only [S1nf]
    rw [show τc + Real.sqrt (2 * S1b * δb / S₃) - τc = Real.sqrt (2 * S1b * δb / S₃) by ring,
        hsq]
    field_simp
    ring
  · simp only [S1nf]
    rw [show τc - Real.sqrt (2 * S1b * δb / S₃) - τc = -Real.sqrt (2 * S1b * δb / S₃) by ring,
        neg_sq, hsq]
    field_simp
    ring

/-- Gap prefactor: `τ₊ − τ₋ = 2√(2·S1b/S₃)·√δb`. -/
theorem nf_gap (S₃ S1b δb τc : ℝ) (hS₃ : 0 < S₃) (hS1b : 0 < S1b) (hδb : 0 < δb) :
    (τc + Real.sqrt (2 * S1b * δb / S₃)) - (τc - Real.sqrt (2 * S1b * δb / S₃))
      = 2 * Real.sqrt (2 * S1b / S₃) * Real.sqrt δb := by
  have _hδb := hδb
  rw [show 2 * S1b * δb / S₃ = 2 * S1b / S₃ * δb by ring,
      Real.sqrt_mul (by positivity) δb]
  ring

/-- Prominence, exact form: `ΔΦ = Φnf(τ₊) − Φnf(τ₋) = (2/3)·S₃·δ³` with
`δ = √(2·S1b·δb/S₃)`. -/
theorem nf_prominence_exact (S₃ S1b δb τc : ℝ) (hS₃ : 0 < S₃) (hS1b : 0 < S1b) (hδb : 0 < δb) :
    Phinf S₃ S1b δb τc (τc + Real.sqrt (2 * S1b * δb / S₃))
      - Phinf S₃ S1b δb τc (τc - Real.sqrt (2 * S1b * δb / S₃))
      = (2 / 3) * S₃ * Real.sqrt (2 * S1b * δb / S₃) ^ 3 := by
  have hx : (0 : ℝ) ≤ 2 * S1b * δb / S₃ := by positivity
  have hS₃' : S₃ ≠ 0 := ne_of_gt hS₃
  have hsq : Real.sqrt (2 * S1b * δb / S₃) ^ 2 = 2 * S1b * δb / S₃ := Real.sq_sqrt hx
  have hkey : S₃ * Real.sqrt (2 * S1b * δb / S₃) ^ 2 = 2 * S1b * δb := by
    rw [hsq]; field_simp
  simp only [Phinf]
  rw [show τc + Real.sqrt (2 * S1b * δb / S₃) - τc = Real.sqrt (2 * S1b * δb / S₃) by ring,
      show τc - Real.sqrt (2 * S1b * δb / S₃) - τc = -Real.sqrt (2 * S1b * δb / S₃) by ring]
  linear_combination (-(Real.sqrt (2 * S1b * δb / S₃))) * hkey

/-- Prominence prefactor: `ΔΦ = (4√2/3)·(S1b·δb)·√(S1b·δb)/√S₃`,
i.e. the constant is `4√2/3 = 2^{5/2}/3`. -/
theorem nf_prominence (S₃ S1b δb τc : ℝ) (hS₃ : 0 < S₃) (hS1b : 0 < S1b) (hδb : 0 < δb) :
    Phinf S₃ S1b δb τc (τc + Real.sqrt (2 * S1b * δb / S₃))
      - Phinf S₃ S1b δb τc (τc - Real.sqrt (2 * S1b * δb / S₃))
      = (4 * Real.sqrt 2 / 3) * ((S1b * δb) * Real.sqrt (S1b * δb)) / Real.sqrt S₃ := by
  have hS₃' : S₃ ≠ 0 := ne_of_gt hS₃
  have hs : Real.sqrt S₃ ≠ 0 := ne_of_gt (Real.sqrt_pos.mpr hS₃)
  have hx : (0 : ℝ) ≤ 2 * S1b * δb / S₃ := by positivity
  have hsq : Real.sqrt (2 * S1b * δb / S₃) ^ 2 = 2 * S1b * δb / S₃ := Real.sq_sqrt hx
  have hcube : Real.sqrt (2 * S1b * δb / S₃) ^ 3
      = Real.sqrt (2 * S1b * δb / S₃) ^ 2 * Real.sqrt (2 * S1b * δb / S₃) := by ring
  have hdval : Real.sqrt (2 * S1b * δb / S₃)
      = Real.sqrt 2 * Real.sqrt (S1b * δb) / Real.sqrt S₃ := by
    rw [show 2 * S1b * δb / S₃ = 2 * (S1b * δb) / S₃ by ring,
        Real.sqrt_div (by positivity) S₃,
        Real.sqrt_mul (by norm_num : (0 : ℝ) ≤ 2)]
  rw [nf_prominence_exact S₃ S1b δb τc hS₃ hS1b hδb, hcube, hsq, hdval]
  field_simp
  ring

/-- The prominence constant in power form: `4√2/3 = 2^{5/2}/3`. -/
theorem prominence_constant : 4 * Real.sqrt 2 / 3 = 2 ^ ((5 : ℝ) / 2) / 3 := by
  rw [show (5 : ℝ) / 2 = ((2 : ℕ) : ℝ) + 1 / 2 by norm_num,
      Real.rpow_add (by norm_num : (0 : ℝ) < 2), Real.rpow_natCast,
      ← Real.sqrt_eq_rpow]
  norm_num

end DPMA
