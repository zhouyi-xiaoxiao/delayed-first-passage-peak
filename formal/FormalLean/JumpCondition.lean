/-
Module 2: point-sink jump condition ⟺ secular equation (Sec. II B and App. B of the paper).

  * D_θ(k;b) = k sin k + 2b sin(kθ) sin(k(1−θ)) and the piecewise eigenfunction φ_{k,θ};
  * φ'(θ⁺) − φ'(θ⁻) = −k sin k, so the jump condition −k sin k = 2b sin(kθ)sin(k(1−θ)) is
    D_θ = 0 — `branch_deriv_left`, `branch_deriv_right`, `jump_value`, `jump_iff_secular`;
  * continuity of φ at x = θ — `phi_continuous_at_theta`; Dirichlet ends — `phi_dirichlet`;
  * midpoint sink θ = 1/2: D_{1/2}(2w;b) = 0 ⟺ 2w·cos w + b·sin w = 0, i.e. tan w = −2w/b —
    `antipodal_collapse`.
-/
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Deriv
import Mathlib.Tactic

namespace DPMA

open Real

/-- The secular determinant D_θ(k;b) = k sin k + 2b sin(kθ) sin(k(1−θ)). -/
noncomputable def Dtheta (k b θ : ℝ) : ℝ :=
  k * sin k + 2 * b * sin (k * θ) * sin (k * (1 - θ))

/-- The eigenfunction φ_{k,θ} (piecewise sine). -/
noncomputable def phiFun (k θ x : ℝ) : ℝ :=
  if x ≤ θ then sin (k * (1 - θ)) * sin (k * x) else sin (k * θ) * sin (k * (1 - x))

/-- Left-branch derivative: `x ↦ sin(k(1−θ))·sin(kx)` has derivative
`k·sin(k(1−θ))·cos(kx)`. -/
theorem branch_deriv_left (k θ x : ℝ) :
    HasDerivAt (fun y => sin (k * (1 - θ)) * sin (k * y))
      (k * (sin (k * (1 - θ)) * cos (k * x))) x := by
  have h : HasDerivAt (fun y : ℝ => k * y) k x := by
    simpa using (hasDerivAt_id x).const_mul k
  have heq : k * (sin (k * (1 - θ)) * cos (k * x))
      = sin (k * (1 - θ)) * (cos (k * x) * k) := by ring
  rw [heq]
  exact h.sin.const_mul _

/-- Right-branch derivative: `x ↦ sin(kθ)·sin(k(1−x))` has derivative
`−k·sin(kθ)·cos(k(1−x))`. -/
theorem branch_deriv_right (k θ x : ℝ) :
    HasDerivAt (fun y => sin (k * θ) * sin (k * (1 - y)))
      (-(k * (sin (k * θ) * cos (k * (1 - x))))) x := by
  have h : HasDerivAt (fun y : ℝ => k * (1 - y)) (-k) x := by
    have h1 : HasDerivAt (fun y : ℝ => (1 : ℝ) - y) (-1) x := by
      simpa using (hasDerivAt_id x).const_sub (1 : ℝ)
    have heq : (-k) = k * (-1 : ℝ) := by ring
    rw [heq]
    exact h1.const_mul k
  have heq2 : (-(k * (sin (k * θ) * cos (k * (1 - x)))))
      = sin (k * θ) * (cos (k * (1 - x)) * (-k)) := by ring
  rw [heq2]
  exact h.sin.const_mul _

/-- The derivative jump at the sink equals `−k sin k`. -/
theorem jump_value (k θ : ℝ) :
    (-(k * (sin (k * θ) * cos (k * (1 - θ))))) - k * (sin (k * (1 - θ)) * cos (k * θ))
      = -(k * sin k) := by
  have h : sin k = sin (k * θ) * cos (k * (1 - θ)) + cos (k * θ) * sin (k * (1 - θ)) := by
    rw [← Real.sin_add]
    congr 1
    ring
  rw [h]
  ring

/-- δ-sink jump condition ⟺ secular determinant:
`−k sin k = 2b·sin(kθ)·sin(k(1−θ))  ↔  D_θ(k;b) = 0`. -/
theorem jump_iff_secular (k b θ : ℝ) :
    -(k * sin k) = 2 * b * sin (k * θ) * sin (k * (1 - θ)) ↔ Dtheta k b θ = 0 := by
  unfold Dtheta
  constructor <;> intro h <;> linarith

/-- φ_{k,θ} is continuous at the sink: both branches take the common value
`sin(kθ)·sin(k(1−θ))` at `x = θ`. -/
theorem phi_continuous_at_theta (k θ : ℝ) :
    sin (k * (1 - θ)) * sin (k * θ) = sin (k * θ) * sin (k * (1 - θ)) := by
  exact mul_comm _ _

/-- Dirichlet ends: `φ(0) = 0` and `φ(1) = 0` (for `θ ∈ [0,1]`). -/
theorem phi_dirichlet (k θ : ℝ) (h0 : 0 ≤ θ) (h1 : θ ≤ 1) :
    phiFun k θ 0 = 0 ∧ phiFun k θ 1 = 0 := by
  refine ⟨?_, ?_⟩
  · simp [phiFun, h0]
  · by_cases h : (1 : ℝ) ≤ θ
    · have hθ : θ = 1 := le_antisymm h1 h
      subst hθ
      simp [phiFun]
    · simp [phiFun, h]

/-- Antipodal collapse: at `θ = 1/2`, `k = 2w`,
`D_{1/2}(2w;b) = 0  ↔  2w·cos w + b·sin w = 0` (i.e. `tan w = −2w/b` where `cos w ≠ 0`);
uses `sin(2w) = 2 sin w cos w` and `sin w · (…) ` factor structure. -/
theorem antipodal_collapse (w b : ℝ) (hs : sin w ≠ 0) :
    Dtheta (2 * w) b (1 / 2) = 0 ↔ 2 * w * cos w + b * sin w = 0 := by
  have h1 : (2 * w) * (1 / 2 : ℝ) = w := by ring
  have h2 : (2 * w) * (1 - 1 / 2 : ℝ) = w := by ring
  have hD : Dtheta (2 * w) b (1 / 2)
      = 2 * sin w * (2 * w * cos w + b * sin w) := by
    unfold Dtheta
    rw [h1, h2, Real.sin_two_mul]
    ring
  rw [hD]
  constructor
  · intro h
    rcases mul_eq_zero.mp h with h' | h'
    · rcases mul_eq_zero.mp h' with h'' | h''
      · norm_num at h''
      · exact absurd h'' hs
    · exact h'
  · intro h
    rw [h, mul_zero]

end DPMA
