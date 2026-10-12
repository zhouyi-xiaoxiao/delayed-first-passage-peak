/-
Module 3: the normalization identity J = sin(k)·D_k/(4b) and the midpoint closed forms
(App. B of the paper).

  * at a root of D_θ, J(k,θ) = sin(k)·D_k(k;b)/(4b) with
    D_k = sin k + k cos k + 2b[θ cos(kθ)sin(k(1−θ)) + (1−θ)sin(kθ)cos(k(1−θ))].
    With A = sin kθ, B = sin k(1−θ), C = cos kθ, E = cos k(1−θ) (so sin k = AE + CB and
    cos k = CE − AB) and the root condition k sin k = −2bAB, the identity is proved over the
    constraint variety A²+C² = 1, B²+E² = 1, k(AE+CB) = −2bAB in `J_identity_abstract` and then
    instantiated with sines and cosines in `J_identity`;
  * midpoint sink (θ = 1/2, k = 2w, root condition tan w = −2w/b):
      sin²w = 4w²/(4w²+b²)                                  — `antipodal_sin_sq`
      J = ½ sin²w[1 − sin(2w)/(2w)] = ½ sin⁴w[1 + b(b+2)/(4w²)] — `antipodal_J`
      G_{1/2,1/2} = 4w(1−cos w)/(sin w·(1 + b(b+2)/(4w²)))   — `antipodal_G`
    (the last is the amplitude formula used by the numerical scripts).
-/
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Tactic
import FormalLean.JumpCondition

namespace DPMA

open Real

/-- Abstract J-identity over the constraint variety:
if `A² + C² = 1`, `B² + E² = 1` and the root condition `k(AE + CB) = −2bAB` holds
(`k ≠ 0`, `b ≠ 0`), then
`(AE+CB)·[(AE+CB) + k(CE−AB) + 2b(θ·CB + (1−θ)·AE)]/(4b)
   = ½[θB² + (1−θ)A²] − AB(AE+CB)/(2k)`.
Here `AE+CB = sin k`, `CE−AB = cos k`, the bracket is `D_k`, and the right side is `J`. -/
theorem J_identity_abstract (θ k b A B C E : ℝ) (hk : k ≠ 0) (hb : b ≠ 0)
    (hAC : A ^ 2 + C ^ 2 = 1) (hBE : B ^ 2 + E ^ 2 = 1)
    (hroot : k * (A * E + C * B) = -(2 * b * (A * B))) :
    (A * E + C * B) * ((A * E + C * B) + k * (C * E - A * B)
        + 2 * b * (θ * (C * B) + (1 - θ) * (A * E))) / (4 * b)
      = (1 / 2) * (θ * B ^ 2 + (1 - θ) * A ^ 2) - A * B * (A * E + C * B) / (2 * k) := by
  have h4b : (4 : ℝ) * b ≠ 0 := mul_ne_zero (by norm_num) hb
  have h2k : (2 : ℝ) * k ≠ 0 := mul_ne_zero (by norm_num) hk
  rw [eq_sub_iff_add_eq, div_add_div _ _ h4b h2k, div_eq_iff (mul_ne_zero h4b h2k)]
  linear_combination (2 * (A * E + C * B) + 2 * k * (C * E - A * B)) * hroot
    + 4 * b * k * θ * B ^ 2 * hAC + 4 * b * k * (1 - θ) * A ^ 2 * hBE

/-- The normalization identity in trig form: at a root of D_θ (`Dtheta k b θ = 0`,
`k ≠ 0`, `b ≠ 0`),
`sin k · D_k/(4b) = ½[θ sin²(k(1−θ)) + (1−θ) sin²(kθ)] − sin(kθ)sin(k(1−θ))·sin k/(2k) = J`. -/
theorem J_identity (θ k b : ℝ) (hk : k ≠ 0) (hb : b ≠ 0)
    (hroot : Dtheta k b θ = 0) :
    sin k * (sin k + k * cos k
        + 2 * b * (θ * cos (k * θ) * sin (k * (1 - θ))
          + (1 - θ) * sin (k * θ) * cos (k * (1 - θ)))) / (4 * b)
      = (1 / 2) * (θ * sin (k * (1 - θ)) ^ 2 + (1 - θ) * sin (k * θ) ^ 2)
        - sin (k * θ) * sin (k * (1 - θ)) * sin k / (2 * k) := by
  have harg : k * θ + k * (1 - θ) = k := by ring
  have hsin : sin k = sin (k * θ) * cos (k * (1 - θ)) + cos (k * θ) * sin (k * (1 - θ)) := by
    rw [← sin_add, harg]
  have hcos : cos k = cos (k * θ) * cos (k * (1 - θ)) - sin (k * θ) * sin (k * (1 - θ)) := by
    rw [← cos_add, harg]
  simp only [Dtheta] at hroot
  have hroot' : k * (sin (k * θ) * cos (k * (1 - θ)) + cos (k * θ) * sin (k * (1 - θ)))
      = -(2 * b * (sin (k * θ) * sin (k * (1 - θ)))) := by
    linear_combination hroot - k * hsin
  have habs := J_identity_abstract θ k b (sin (k * θ)) (sin (k * (1 - θ)))
    (cos (k * θ)) (cos (k * (1 - θ))) hk hb
    (sin_sq_add_cos_sq (k * θ)) (sin_sq_add_cos_sq (k * (1 - θ))) hroot'
  rw [hsin, hcos]
  linear_combination habs

/-- Antipodal spectral weight: at a root `b sin w = −2w cos w`,
`sin²w·(4w² + b²) = 4w²` (cleared-denominator form of `sin²w = 4w²/(4w²+b²)`;
no further hypotheses needed). -/
theorem antipodal_sin_sq (w b : ℝ) (hroot : b * sin w = -(2 * w * cos w)) :
    sin w ^ 2 * (4 * w ^ 2 + b ^ 2) = 4 * w ^ 2 := by
  linear_combination (b * sin w - 2 * w * cos w) * hroot
    + 4 * w ^ 2 * sin_sq_add_cos_sq w

/-- Antipodal J closed form: at a root `b sin w = −2w cos w` (`w ≠ 0`),
`½ sin²w·(1 − sin w·cos w/w) = ½ sin⁴w·(1 + b(b+2)/(4w²))`.
(The left side is J(2w, 1/2) after `sin(2w) = 2 sin w cos w`; the right side is the
form used by the numerical scripts.) -/
theorem antipodal_J (w b : ℝ) (hw : w ≠ 0) (hroot : b * sin w = -(2 * w * cos w)) :
    (1 / 2) * sin w ^ 2 * (1 - sin w * cos w / w)
      = (1 / 2) * sin w ^ 4 * (1 + b * (b + 2) / (4 * w ^ 2)) := by
  have hs2 := antipodal_sin_sq w b hroot
  field_simp [hw]
  linear_combination (-2 * sin w ^ 3) * hroot + (-(sin w ^ 2)) * hs2

/-- Antipodal amplitude closed form: the amplitude `G = μ·φ(ξ)·I/J` at
`ξ = θ = 1/2`, i.e. `μ = 2w²`, `φ(1/2) = sin²w`, `I = sin w(1−cos w)/w`,
`J = ½ sin⁴w(1+b(b+2)/(4w²))`, collapses to
`G = 4w(1−cos w)/(sin w·(1 + b(b+2)/(4w²)))` — the formula implemented in the scripts.
This is the pure algebraic collapse (`w ≠ 0`, `sin w ≠ 0`, nonvanishing normalization);
the root-conditional content (that `J` takes this form at a root) is `antipodal_J`. -/
theorem antipodal_G (w b : ℝ) (hw : w ≠ 0) (hs : sin w ≠ 0)
    (hJ : 1 + b * (b + 2) / (4 * w ^ 2) ≠ 0) :
    2 * w ^ 2 * (sin w ^ 2) * (sin w * (1 - cos w) / w)
        / ((1 / 2) * sin w ^ 4 * (1 + b * (b + 2) / (4 * w ^ 2)))
      = 4 * w * (1 - cos w) / (sin w * (1 + b * (b + 2) / (4 * w ^ 2))) := by
  have h4w2 : (4 : ℝ) * w ^ 2 ≠ 0 := mul_ne_zero (by norm_num) (pow_ne_zero 2 hw)
  have hrw : 1 + b * (b + 2) / (4 * w ^ 2) = (4 * w ^ 2 + b * (b + 2)) / (4 * w ^ 2) := by
    rw [add_div, div_self h4w2]
  have hJ' : 4 * w ^ 2 + b * (b + 2) ≠ 0 := by
    intro h
    apply hJ
    rw [hrw, h, zero_div]
  rw [hrw]
  field_simp
  ring

end DPMA
