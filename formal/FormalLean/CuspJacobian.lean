/-
Module 8: factorization of the cusp Jacobian (App. C and App. D of the paper).

At a half-line cusp the conditions are m₁ = m₂ = m₃ = 0, where Φ⁽ⁿ⁾ = (−1)ⁿ mₙ are the signed
time derivatives of the density Φ(σ; B, X).  The interval-Newton certificate uses the Jacobian of
(m₁, m₂, m₃) with respect to (B, σ, X).  Because ∂_σ mₙ = −m_{n+1} and m₂ = m₃ = 0 at the cusp, its
σ-column is (0, 0, −m₄), and the determinant factorizes into m₄ times the 2 × 2 unfolding
determinant of (m₁, m₂) with respect to (B, X):

  * `cuspJacobian_factorization`:
      det [[p₁, 0, x₁], [p₂, 0, x₂], [p₃, −m₄, x₃]] = m₄ (p₁ x₂ − x₁ p₂),
    with pₙ = ∂_B mₙ and xₙ = ∂_X mₙ (any commutative ring);
  * `normalizedCuspDet_factorization`: in terms of the normalized margins used in the paper,
    σ⁴Φ''''/Φ = σ⁴ m₄/Φ and the unfolding determinant with respect to (ln B, X) normalized by
    σ³/Φ², i.e. B σ³ (p₁ x₂ − x₁ p₂)/Φ²,
      det = Φ³/(B σ⁷) · (σ⁴ m₄/Φ) · (B σ³ (p₁ x₂ − x₁ p₂)/Φ²)   (Φ, σ, B ≠ 0; any field);
  * `cuspJacobian_ne_zero_from_normalized`: hence the Jacobian is invertible whenever both
    normalized margins are nonzero, which is what the certified enclosures show.

The module uses only core Lean (the `grind` decision procedure for commutative rings and fields);
it does not import mathlib and can also be checked on its own with `lean FormalLean/CuspJacobian.lean`.
-/

namespace DPMA

/-- Determinant of the 3 × 3 matrix with rows (a₁₁, a₁₂, a₁₃), (a₂₁, a₂₂, a₂₃), (a₃₁, a₃₂, a₃₃). -/
def det3 {α : Type} [Lean.Grind.CommRing α] (a₁₁ a₁₂ a₁₃ a₂₁ a₂₂ a₂₃ a₃₁ a₃₂ a₃₃ : α) : α :=
  a₁₁ * (a₂₂ * a₃₃ - a₂₃ * a₃₂) - a₁₂ * (a₂₁ * a₃₃ - a₂₃ * a₃₁) + a₁₃ * (a₂₁ * a₃₂ - a₂₂ * a₃₁)

/-- The cusp Jacobian with σ-column (0, 0, −m₄) factorizes into m₄ times the unfolding determinant. -/
theorem cuspJacobian_factorization {α : Type} [Lean.Grind.CommRing α]
    (p₁ p₂ p₃ x₁ x₂ x₃ m₄ : α) :
    det3 p₁ 0 x₁ p₂ 0 x₂ p₃ (-m₄) x₃ = m₄ * (p₁ * x₂ - x₁ * p₂) := by
  unfold det3
  grind

/-- The same factorization in terms of the normalized margins σ⁴Φ''''/Φ and
B σ³ (p₁ x₂ − x₁ p₂)/Φ² (the unfolding determinant with respect to (ln B, X), normalized). -/
theorem normalizedCuspDet_factorization {α : Type} [Lean.Grind.Field α]
    (Φ σ B p₁ p₂ p₃ x₁ x₂ x₃ m₄ : α) (hΦ : Φ ≠ 0) (hσ : σ ≠ 0) (hB : B ≠ 0) :
    det3 p₁ 0 x₁ p₂ 0 x₂ p₃ (-m₄) x₃
      = Φ ^ 3 / (B * σ ^ 7) * (σ ^ 4 * m₄ / Φ) * (B * σ ^ 3 / Φ ^ 2 * (p₁ * x₂ - x₁ * p₂)) := by
  unfold det3
  grind

/-- Nonzero normalized margins imply an invertible cusp Jacobian. -/
theorem cuspJacobian_ne_zero_from_normalized {α : Type} [Lean.Grind.Field α]
    (Φ σ B p₁ p₂ p₃ x₁ x₂ x₃ m₄ : α) (hΦ : Φ ≠ 0) (hσ : σ ≠ 0) (hB : B ≠ 0)
    (h4 : σ ^ 4 * m₄ / Φ ≠ 0) (hU : B * σ ^ 3 / Φ ^ 2 * (p₁ * x₂ - x₁ * p₂) ≠ 0) :
    det3 p₁ 0 x₁ p₂ 0 x₂ p₃ (-m₄) x₃ ≠ 0 := by
  rw [normalizedCuspDet_factorization Φ σ B p₁ p₂ p₃ x₁ x₂ x₃ m₄ hΦ hσ hB]
  intro h
  grind

end DPMA
