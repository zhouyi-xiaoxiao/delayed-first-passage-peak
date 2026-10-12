/-
Module 6: the splitting probability π_sc and the rank-one (Sherman–Morrison) solve
(App. A of the paper).

  * Sherman–Morrison column of the resolvent with the shortcut — proved constructively (no
    matrix inverses) in `sherman_morrison_solve`: if `B·g = e_r`, `B·h = e_u` and
    `1 + λ h_u ≠ 0`, then `x = g − [λ g_u/(1+λ h_u)]·h` solves `(B + λ e_u e_uᵀ)·x = e_r`, and
    `x_u = g_u/(1+λ h_u)`;
  * π_sc(r) = 2·min(r,u)·[N−max(r,u)]/(aN + 2u(N−u)), a = q/λ, via the first-step
    (backward-equation) linear system of the killed walk: the explicit vector
    `φ_i = K·min(i,u)·(N−max(i,u))`, `K = 2λ/(qN + 2λu(N−u))`, satisfies  φ_0 = φ_N = 0,
               φ_i = (1−q)φ_i + (q/2)(φ_{i−1}+φ_{i+1})            (0 < i < N, i ≠ u),
               φ_u = λ·1 + (1−q−λ)φ_u + (q/2)(φ_{u−1}+φ_{u+1}),
    (`pisc_solves`), and its value at the start is the closed form (`pisc_value`);
  * shortcut opposite the target, u = N/2: π_sc = r*/(a + N/2) with r* = min(r, N−r) —
    `pisc_antipodal`.
This is the same linear system as the resolvent formula π_sc = λ⟨u|(I−P_λ)⁻¹|r⟩ (the backward
equation for the probability of absorption through the shortcut); the closed form is proved
exactly for all admissible N, u, r.
-/
import Mathlib.Data.Matrix.Basic
import Mathlib.Tactic

namespace DPMA

open Matrix

/-- Constructive Sherman–Morrison solve: if `B g = e_r`, `B h = e_u`
and `1 + λ h_u ≠ 0`, then `x = g − [λ g_u/(1+λ h_u)]·h` satisfies
`B x + λ x_u e_u = e_r` (i.e. `(B + λ e_u e_uᵀ) x = e_r`), and `x_u = g_u/(1+λ h_u)`. -/
theorem sherman_morrison_solve {n : ℕ} (B : Matrix (Fin n) (Fin n) ℝ)
    (g h : Fin n → ℝ) (r u : Fin n) (lam : ℝ)
    (hg : B.mulVec g = Pi.single r 1) (hh : B.mulVec h = Pi.single u 1)
    (hden : 1 + lam * h u ≠ 0) :
    B.mulVec (g - (lam * g u / (1 + lam * h u)) • h)
        + (lam * (g - (lam * g u / (1 + lam * h u)) • h) u) • Pi.single u 1
      = Pi.single r 1
    ∧ (g - (lam * g u / (1 + lam * h u)) • h) u = g u / (1 + lam * h u) := by
  have hxu : (g - (lam * g u / (1 + lam * h u)) • h) u = g u / (1 + lam * h u) := by
    simp only [Pi.sub_apply, Pi.smul_apply, smul_eq_mul]
    field_simp
    ring
  refine ⟨?_, hxu⟩
  rw [Matrix.mulVec_sub, Matrix.mulVec_smul, hg, hh, hxu]
  funext i
  simp only [Pi.add_apply, Pi.sub_apply, Pi.smul_apply, smul_eq_mul]
  ring

/-- The explicit splitting-probability vector `φ_i = K·min(i,u)·(N−max(i,u))` with
`K = 2λ/(qN + 2λu(N−u))` satisfies the first-step system of the killed walk
: absorbing boundary, harmonic off the defect, and the
defect equation at `u`. -/
theorem pisc_solves (N u : ℕ) (hu : 0 < u) (huN : u < N) (q lam : ℝ)
    (hq : 0 < q) (hlam : 0 < lam) :
    let K : ℝ := 2 * lam / (q * N + 2 * lam * u * (N - u))
    let φ : ℕ → ℝ := fun i => K * (min i u : ℕ) * ((N : ℝ) - (max i u : ℕ))
    (φ 0 = 0 ∧ φ N = 0)
    ∧ (∀ i : ℕ, 0 < i → i < N → i ≠ u →
        φ i = (1 - q) * φ i + (q / 2) * φ (i - 1) + (q / 2) * φ (i + 1))
    ∧ φ u = lam * 1 + (1 - q - lam) * φ u + (q / 2) * φ (u - 1) + (q / 2) * φ (u + 1) := by
  intro K φ
  have hφ : ∀ i : ℕ, φ i = K * (min i u : ℕ) * ((N : ℝ) - (max i u : ℕ)) := fun _ => rfl
  have hK : K = 2 * lam / (q * N + 2 * lam * u * ((N : ℝ) - u)) := rfl
  refine ⟨⟨?_, ?_⟩, ?_, ?_⟩
  · -- absorbing boundary at 0
    simp only [hφ]
    rw [min_eq_left (Nat.zero_le u)]
    simp
  · -- absorbing boundary at N
    simp only [hφ]
    rw [max_eq_left (le_of_lt huN)]
    simp
  · -- harmonic away from the defect
    intro i hi0 hiN hne
    simp only [hφ]
    rcases lt_or_gt_of_ne hne with hlt | hgt
    · -- i < u
      rw [min_eq_left hlt.le, max_eq_right hlt.le,
          min_eq_left (by omega : i - 1 ≤ u), max_eq_right (by omega : i - 1 ≤ u),
          min_eq_left (by omega : i + 1 ≤ u), max_eq_right (by omega : i + 1 ≤ u),
          Nat.cast_sub (by omega : 1 ≤ i)]
      push_cast
      ring
    · -- u < i
      rw [min_eq_right hgt.le, max_eq_left hgt.le,
          min_eq_right (by omega : u ≤ i - 1), max_eq_left (by omega : u ≤ i - 1),
          min_eq_right (by omega : u ≤ i + 1), max_eq_left (by omega : u ≤ i + 1),
          Nat.cast_sub (by omega : 1 ≤ i)]
      push_cast
      ring
  · -- defect equation at u
    have hNu : (u : ℝ) < N := by exact_mod_cast huN
    have hu' : (0 : ℝ) < u := by exact_mod_cast hu
    have hN' : (0 : ℝ) < N := hu'.trans hNu
    have hD : 0 < q * (N : ℝ) + 2 * lam * u * ((N : ℝ) - u) := by
      have h1 : 0 < q * (N : ℝ) := mul_pos hq hN'
      have h2 : 0 < 2 * lam * (u : ℝ) * ((N : ℝ) - u) :=
        mul_pos (mul_pos (mul_pos (by norm_num : (0 : ℝ) < 2) hlam) hu') (sub_pos.mpr hNu)
      linarith
    simp only [hφ]
    rw [min_self, max_self,
        min_eq_left (by omega : u - 1 ≤ u), max_eq_right (by omega : u - 1 ≤ u),
        min_eq_right (by omega : u ≤ u + 1), max_eq_left (by omega : u ≤ u + 1),
        Nat.cast_sub (by omega : 1 ≤ u), hK]
    push_cast
    field_simp
    ring

/-- Value at the start: with `a = q/λ`,
`φ_r = 2·min(r,u)·(N−max(r,u)) / (a·N + 2·u·(N−u))`. -/
theorem pisc_value (N u r : ℕ) (hu : 0 < u) (huN : u < N) (q lam : ℝ)
    (hq : 0 < q) (hlam : 0 < lam) :
    (2 * lam / (q * N + 2 * lam * u * (N - u))) * (min r u : ℕ) * ((N : ℝ) - (max r u : ℕ))
      = 2 * (min r u : ℕ) * ((N : ℝ) - (max r u : ℕ))
        / ((q / lam) * N + 2 * u * (N - u)) := by
  -- `hu`, `huN`, `hq` are part of the statement but are not needed for this
  -- algebraic identity (it only requires `lam ≠ 0`); referenced to keep the build silent.
  have _hu := hu; have _huN := huN; have _hq := hq
  have hlam' : lam ≠ 0 := ne_of_gt hlam
  have hE : (q / lam) * (N : ℝ) + 2 * u * ((N : ℝ) - u)
      = (q * N + 2 * lam * u * ((N : ℝ) - u)) / lam := by
    field_simp
  rw [hE, div_div_eq_mul_div]
  ring

/-- Antipodal reduction: for `N = 2m`, `u = m`, `r ≤ N`,
`2·min(r,m)·(N−max(r,m))/(aN + 2m·m) = min(r, N−r)/(a + m)`, i.e.
π_sc = r*/(a + N/2) with r* the short-arc distance min(r, N−r). -/
theorem pisc_antipodal (m r : ℕ) (hm : 0 < m) (hr : r ≤ 2 * m) (a : ℝ) (ha : 0 < a) :
    2 * (min r m : ℕ) * ((2 * m : ℝ) - (max r m : ℕ))
        / (a * (2 * m) + 2 * m * ((2 * m : ℝ) - m))
      = (min r (2 * m - r) : ℕ) / (a + m) := by
  have hm' : (0 : ℝ) < m := by exact_mod_cast hm
  have hden : a * (2 * (m : ℝ)) + 2 * m * ((2 * (m : ℝ)) - m) ≠ 0 := by
    have hpos : 0 < a * (2 * (m : ℝ)) + 2 * m * ((2 * (m : ℝ)) - m) := by
      nlinarith [mul_pos ha hm', mul_pos hm' hm']
    exact ne_of_gt hpos
  have hden2 : a + (m : ℝ) ≠ 0 := by
    have hpos : 0 < a + (m : ℝ) := by linarith
    exact ne_of_gt hpos
  rcases le_total r m with h | h
  · -- short arc on the r ≤ m side
    rw [min_eq_left h, max_eq_right h, min_eq_left (by omega : r ≤ 2 * m - r)]
    field_simp
    ring
  · -- short arc on the r ≥ m side
    rw [min_eq_right h, max_eq_left h, min_eq_right (by omega : 2 * m - r ≤ r),
        Nat.cast_sub hr]
    push_cast
    field_simp
    ring

end DPMA
