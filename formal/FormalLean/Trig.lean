/-
Module 1: Chebyshev / trigonometric product identities behind the exact lattice solution
(App. A of the paper).

  * the product identity A_{N−r}A_u − A_{N−u}A_r = A_N A_{u−r} with A_m = sin(mφ)/sin φ
    (y = cos φ) — `chebyshev_product`, and the underlying three-angle identity
    `sin_product_identity`;
  * the Green-function column of the path, G_rs = 2 A_min A_{N−max} / A_N: the explicit column
    f_i = sin(min(i,j)φ)·sin((N−max(i,j))φ) solves the three-term recurrence with a point
    source at j (`green_column_solves`), with jump value `green_jump`;
  * the determinant D_u(y) = a U_{N−1} + 2 U_{u−1} U_{N−u−1} obtained by clearing denominators
    in 1 + zλ G⁰_uu — `montroll_determinant`;
  * the numerator collapse that gives the residues — `numerator_collapse`;
  * the factorization U_{N−1} = 2 U_{N/2−1} T_{N/2} for a shortcut opposite the target —
    `antipodal_factorization` (trig form sin(2mφ) = 2 sin(mφ)cos(mφ)).

All statements are identities of real trigonometric functions in the parametrization
y = cos φ, valid for all real angle parameters (stronger than the integer-index statements).
-/
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Tactic

namespace DPMA

open Real

/-- Three-angle product identity behind the Chebyshev collapse:
`sin(α−γ)·sin β − sin(α−β)·sin γ = sin α · sin(β−γ)`. -/
theorem sin_product_identity (α β γ : ℝ) :
    sin (α - γ) * sin β - sin (α - β) * sin γ = sin α * sin (β - γ) := by
  rw [sin_sub, sin_sub, sin_sub]; ring

-- `hφ` is part of the statement (the parametrization `y = cos φ` needs
-- `sin φ ≠ 0`); the proof below happens not to need it, so silence the unused-binder lint.
set_option linter.unusedVariables false in
/-- Chebyshev product identity: with `A m = sin (m·φ)/sin φ` (= U_{m−1}(cos φ)),
`A (N−r) * A u − A (N−u) * A r = A N * A (u−r)`, for all real `N u r`. -/
theorem chebyshev_product (φ : ℝ) (hφ : sin φ ≠ 0) (N u r : ℝ) :
    (sin ((N - r) * φ) / sin φ) * (sin (u * φ) / sin φ)
      - (sin ((N - u) * φ) / sin φ) * (sin (r * φ) / sin φ)
      = (sin (N * φ) / sin φ) * (sin ((u - r) * φ) / sin φ) := by
  have h := sin_product_identity (N * φ) (u * φ) (r * φ)
  rw [show N * φ - r * φ = (N - r) * φ by ring,
      show N * φ - u * φ = (N - u) * φ by ring,
      show u * φ - r * φ = (u - r) * φ by ring] at h
  rw [div_mul_div_comm, div_mul_div_comm, div_mul_div_comm, div_sub_div_same, h]

/-- Chebyshev three-term recurrence in trig form: `2 cos φ · sin(mφ) = sin((m+1)φ) + sin((m−1)φ)`. -/
theorem chebyshev_recurrence (φ m : ℝ) :
    2 * cos φ * sin (m * φ) = sin ((m + 1) * φ) + sin ((m - 1) * φ) := by
  rw [show (m + 1) * φ = m * φ + φ by ring,
      show (m - 1) * φ = m * φ - φ by ring,
      sin_add, sin_sub]
  ring

/-- Green/Wronskian jump at the source:
`sin((N−j)φ)·sin((j+1)φ) − sin((N−j−1)φ)·sin(jφ) = sin(Nφ)·sin φ`. -/
theorem green_jump (φ N j : ℝ) :
    sin ((N - j) * φ) * sin ((j + 1) * φ) - sin ((N - j - 1) * φ) * sin (j * φ)
      = sin (N * φ) * sin φ := by
  have h := sin_product_identity (N * φ) ((j + 1) * φ) (j * φ)
  rw [show N * φ - j * φ = (N - j) * φ by ring,
      show N * φ - (j + 1) * φ = (N - j - 1) * φ by ring,
      show (j + 1) * φ - j * φ = φ by ring] at h
  exact h

/-- The explicit column `f i = sin(min(i,j)·φ)·sin((N−max(i,j))·φ)` solves the Dirichlet
three-term recurrence with a point source at `j`: for `0 < i < N`,
`2cos φ·f i − f (i−1) − f (i+1)` is `sin φ·sin(Nφ)` at `i = j` and `0` otherwise, with
Dirichlet boundary values `f 0 = f N = 0`.  This is the tridiagonal-inverse identity behind
`G⁰_uu` at the trig level. -/
theorem green_column_solves (φ : ℝ) (N j : ℤ) (hj : 0 < j) (hjN : j < N) :
    (∀ i : ℤ, 0 < i → i < N →
        2 * cos φ * (sin ((min i j : ℤ) * φ) * sin (((N - max i j : ℤ)) * φ))
          - (sin ((min (i - 1) j : ℤ) * φ) * sin (((N - max (i - 1) j : ℤ)) * φ))
          - (sin ((min (i + 1) j : ℤ) * φ) * sin (((N - max (i + 1) j : ℤ)) * φ))
          = if i = j then sin φ * sin ((N : ℝ) * φ) else 0)
    ∧ sin ((min 0 j : ℤ) * φ) * sin (((N - max 0 j : ℤ)) * φ) = 0
    ∧ sin ((min N j : ℤ) * φ) * sin (((N - max N j : ℤ)) * φ) = 0 := by
  refine ⟨fun i hi hiN => ?_, ?_, ?_⟩
  · rcases lt_trichotomy i j with hij | hij | hij
    · -- off-source row below the source: min/max collapse and the Chebyshev
      -- recurrence at m = i kills the row.
      rw [if_neg (show ¬ i = j by omega), show min i j = i by omega,
          show max i j = j by omega, show min (i - 1) j = i - 1 by omega,
          show max (i - 1) j = j by omega, show min (i + 1) j = i + 1 by omega,
          show max (i + 1) j = j by omega]
      push_cast
      linear_combination sin (((N : ℝ) - (j : ℝ)) * φ) * chebyshev_recurrence φ (i : ℝ)
    · -- source row i = j: recurrence at m = j plus the Wronskian jump.
      rw [if_pos hij, show min i j = j by omega, show max i j = j by omega,
          show min (i - 1) j = j - 1 by omega, show max (i - 1) j = j by omega,
          show min (i + 1) j = j by omega, show max (i + 1) j = j + 1 by omega]
      push_cast
      rw [show (N : ℝ) - ((j : ℝ) + 1) = (N : ℝ) - (j : ℝ) - 1 by ring]
      linear_combination sin (((N : ℝ) - (j : ℝ)) * φ) * chebyshev_recurrence φ (j : ℝ)
        + green_jump φ (N : ℝ) (j : ℝ)
    · -- off-source row above the source: recurrence at m = N - i kills the row.
      rw [if_neg (show ¬ i = j by omega), show min i j = j by omega,
          show max i j = i by omega, show min (i - 1) j = j by omega,
          show max (i - 1) j = i - 1 by omega, show min (i + 1) j = j by omega,
          show max (i + 1) j = i + 1 by omega]
      push_cast
      rw [show (N : ℝ) - ((i : ℝ) - 1) = (N : ℝ) - (i : ℝ) + 1 by ring,
          show (N : ℝ) - ((i : ℝ) + 1) = (N : ℝ) - (i : ℝ) - 1 by ring]
      linear_combination sin ((j : ℝ) * φ) * chebyshev_recurrence φ ((N : ℝ) - (i : ℝ))
  · -- boundary f 0 = 0 : min 0 j = 0 and sin 0 = 0.
    rw [show min 0 j = 0 by omega]
    simp
  · -- boundary f N = 0 : max N j = N and sin ((N - N)φ) = sin 0 = 0.
    rw [show max N j = N by omega]
    simp

/-- Montroll determinant from the Sherman–Morrison denominator:
clearing `U_{N−1}` in `1 + zλG⁰_uu = 1 + (2/a)·A_u·A_{N−u}/A_N` gives
`D_u/(a·A_N)` with `D_u = a·A_N + 2·A_u·A_{N−u}`. -/
theorem montroll_determinant (a AN Au ANu : ℝ) (ha : a ≠ 0) (hAN : AN ≠ 0) :
    1 + 2 * Au * ANu / (a * AN) = (a * AN + 2 * Au * ANu) / (a * AN) := by
  rw [add_div, div_self (mul_ne_zero ha hAN)]

/-- Numerator collapse: given the product-identity value
`A_{N−r}A_u − A_{N−u}A_r = A_N·A_{u−r}`, the bracket of the generating-function numerator collapses to
`a(A_r + A_{N−r}) + 2A_{N−u}(A_r + A_{u−r})`. -/
theorem numerator_collapse (a Ar ANr Au ANu AN Aur : ℝ) (hAN : AN ≠ 0)
    (hprod : ANr * Au - ANu * Ar = AN * Aur) :
    a * (Ar + ANr) + 2 * Ar * ANu + (2 * ANu / AN) * (ANr * Au - Ar * ANu)
      = a * (Ar + ANr) + 2 * ANu * (Ar + Aur) := by
  field_simp
  linear_combination 2 * ANu * hprod

/-- Antipodal factorization: `U_{2m−1} = 2·U_{m−1}·T_m` in trig form,
`sin(2mφ) = 2·sin(mφ)·cos(mφ)`. -/
theorem antipodal_factorization (φ m : ℝ) :
    sin (2 * m * φ) = 2 * sin (m * φ) * cos (m * φ) := by
  rw [show 2 * m * φ = 2 * (m * φ) by ring, sin_two_mul]

end DPMA
