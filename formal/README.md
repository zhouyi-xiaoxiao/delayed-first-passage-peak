# Formal checks (Lean 4 + mathlib)

Machine-checked proofs of exact algebraic identities used in the paper, and exact checks of the
decimal bounds of the two half-line cusp certificates. All 65 theorems (nine modules) are proved
without `sorry`; `axioms_report.txt` (output of `AxiomsReport.lean`) shows that each one depends at
most on Lean's standard axioms `propext`, `Classical.choice` and `Quot.sound`.

| Module | Statements |
|---|---|
| `FormalLean/Trig.lean` | Chebyshev product identity `A_{N−r}A_u − A_{N−u}A_r = A_N A_{u−r}`; Green-function column of the path; determinant `D_u = a U_{N−1} + 2U_{u−1}U_{N−u−1}`; numerator collapse giving the residues; factorization `U_{N−1} = 2U_{N/2−1}T_{N/2}` (App. A) |
| `FormalLean/JumpCondition.lean` | point-sink jump `φ'(θ⁺) − φ'(θ⁻) = −k sin k`; jump condition ⟺ `D_θ(k;b) = 0`; continuity and Dirichlet ends; midpoint reduction `tan w = −2w/b` (Sec. II B, App. B) |
| `FormalLean/Normalization.lean` | normalization identity `J = sin(k) D_k/(4b)` at the roots of `D_θ`; midpoint closed forms of `sin²w`, `J` and the amplitude `G` (App. B) |
| `FormalLean/MinimalModes.lean` | `Φ' = −S₁`, `Φ'' = S₂` for finite signed exponential sums; no fold with two terms; three-term fold ratio and alternating signs (Sec. III B, App. C) |
| `FormalLean/NormalForm.lean` | extrema, gap `2[2S_{1,b}/S₃]^{1/2}(b_c−b)^{1/2}` and height difference `(4√2/3)S_{1,b}^{3/2}S₃^{−1/2}(b_c−b)^{3/2}` of the truncated normal form (App. C) |
| `FormalLean/PiSc.lean` | Sherman–Morrison column with the shortcut; splitting probability `π_sc = 2 min(r,u)[N−max(r,u)]/(aN + 2u(N−u))` and its midpoint form (App. A) |
| `FormalLean/HalfLine.lean` | assembly of the half-line transform from the Dirichlet resolvent; Lévy limit `B = 0`; branch-cut integrand and positivity of its denominator (Sec. III D, App. C) |
| `FormalLean/CuspJacobian.lean` | factorization of the cusp Jacobian: with σ-column `(0, 0, −m₄)` the determinant equals `m₄` times the unfolding determinant, and `Φ³/(Bσ⁷)` times the two normalized margins `σ⁴Φ''''/Φ` and `Bσ³(p₁x₂ − x₁p₂)/Φ²`; nonzero margins give an invertible Jacobian (App. C, App. D) |
| `FormalLean/CuspCertificateDecimals.lean` | exact integer comparisons, closed by `decide`, of every interval-Newton inclusion and sign exclusion quoted for the boxes `Q₋`, `Q₊` of Eq. (cusp-certificate-boxes): quoted box inside the certified box, Newton image strictly inside the quoted box, density > 0, `Φ'''' < 0`, `det J ≠ 0`, the margin enclosures in the quoted intervals (`Q₋`) or excluding zero with the quoted centres (`Q₊`); and the Krawczyk image, contraction bounds and sign checks (App. C, App. D). Generated from the stored certificates by `../code/certificate_rational_checks.py`; `reproduce_all.py` checks that the regenerated file is identical |

Not formalized: the transcendental roots and spectral sums (`b_c(θ)`, `τ_c`, `B*`, `c_*`, the
numerical values of the prefactors), the analytic limit statements (Bromwich inversion, the
far-end estimate, the implicit-function continuation) and the Arb enclosures of transcendental
functions themselves (Module 9 checks the decimal bounds that the Arb certificates report, not
how they were computed). These are computed and cross-checked by the Python scripts in `../code/`.

## Building

Toolchain: `leanprover/lean4:v4.32.0-rc1` (`lean-toolchain`), mathlib `v4.32.0-rc1` (exact
dependency commits in `lake-manifest.json`). With [elan](https://github.com/leanprover/elan)
installed, from this directory:

```bash
lake exe cache get        # prebuilt mathlib (several GB in .lake/)
lake build                # must finish without errors or 'sorry' warnings
lake env lean AxiomsReport.lean   # prints the axioms of every theorem (compare axioms_report.txt)
grep -rn "sorry\|admit\|native_decide" FormalLean/   # must print nothing
```

Modules 8 and 9 use only core Lean (no `import`), so they can also be checked on their own,
without mathlib, from this directory (`python reproduce_all.py --full` does this when `lean` is
installed):

```bash
lean FormalLean/CuspJacobian.lean              # no output means every theorem is proved
lean FormalLean/CuspCertificateDecimals.lean
```
