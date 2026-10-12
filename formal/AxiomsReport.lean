/-
Axiom report for the formal checks.

`lake env lean AxiomsReport.lean` prints, for every theorem, the axioms it depends on. The check
passes iff every line shows a subset of
  [propext, Classical.choice, Quot.sound]
(Lean's standard classical axioms). Any `sorryAx` would indicate an unproved theorem; any other
axiom an additional assumption.
-/
import FormalLean

-- Module 1: Trig
#print axioms DPMA.sin_product_identity
#print axioms DPMA.chebyshev_product
#print axioms DPMA.chebyshev_recurrence
#print axioms DPMA.green_jump
#print axioms DPMA.green_column_solves
#print axioms DPMA.montroll_determinant
#print axioms DPMA.numerator_collapse
#print axioms DPMA.antipodal_factorization

-- Module 2: JumpCondition
#print axioms DPMA.branch_deriv_left
#print axioms DPMA.branch_deriv_right
#print axioms DPMA.jump_value
#print axioms DPMA.jump_iff_secular
#print axioms DPMA.phi_continuous_at_theta
#print axioms DPMA.phi_dirichlet
#print axioms DPMA.antipodal_collapse

-- Module 3: Normalization
#print axioms DPMA.J_identity_abstract
#print axioms DPMA.J_identity
#print axioms DPMA.antipodal_sin_sq
#print axioms DPMA.antipodal_J
#print axioms DPMA.antipodal_G

-- Module 4: MinimalModes
#print axioms DPMA.Phi_hasDerivAt
#print axioms DPMA.S1_hasDerivAt
#print axioms DPMA.no_two_mode_fold
#print axioms DPMA.no_two_mode_fold_exp
#print axioms DPMA.three_mode_ratio
#print axioms DPMA.three_mode_alternating

-- Module 5: NormalForm
#print axioms DPMA.Phinf_hasDerivAt
#print axioms DPMA.nf_roots
#print axioms DPMA.nf_gap
#print axioms DPMA.nf_prominence_exact
#print axioms DPMA.nf_prominence
#print axioms DPMA.prominence_constant

-- Module 6: PiSc
#print axioms DPMA.sherman_morrison_solve
#print axioms DPMA.pisc_solves
#print axioms DPMA.pisc_value
#print axioms DPMA.pisc_antipodal

-- Module 7: HalfLine
#print axioms DPMA.r0_11_closed
#print axioms DPMA.sm_scalar
#print axioms DPMA.flux_at_zero
#print axioms DPMA.fhat_assembly
#print axioms DPMA.fhat_levy
#print axioms DPMA.cut_g_transform
#print axioms DPMA.cut_normSq
#print axioms DPMA.cut_denominator_form
#print axioms DPMA.cut_denominator_pos
#print axioms DPMA.cut_im

-- Module 8: CuspJacobian
#print axioms DPMA.cuspJacobian_factorization
#print axioms DPMA.normalizedCuspDet_factorization
#print axioms DPMA.cuspJacobian_ne_zero_from_normalized

-- Module 9: CuspCertificateDecimals
#print axioms DPMA.CuspCertificates.boundaryCusp_paperBox_inside_certifiedBox
#print axioms DPMA.CuspCertificates.boundaryCusp_newtonImage_inside_paperBox
#print axioms DPMA.CuspCertificates.boundaryCusp_newton_signs
#print axioms DPMA.CuspCertificates.boundaryCusp_margins_in_quoted_intervals
#print axioms DPMA.CuspCertificates.boundaryCusp_krawczykImage_inside_box
#print axioms DPMA.CuspCertificates.boundaryCusp_krawczyk_contraction
#print axioms DPMA.CuspCertificates.boundaryCusp_krawczyk_signs
#print axioms DPMA.CuspCertificates.boundaryCusp_krawczyk_newton_X_overlap
#print axioms DPMA.CuspCertificates.domainCusp_paperBox_inside_certifiedBox
#print axioms DPMA.CuspCertificates.domainCusp_newtonImage_inside_paperBox
#print axioms DPMA.CuspCertificates.domainCusp_newton_signs
#print axioms DPMA.CuspCertificates.domainCusp_margins_exclude_zero
#print axioms DPMA.CuspCertificates.domainCusp_krawczykImage_inside_box
#print axioms DPMA.CuspCertificates.domainCusp_krawczyk_contraction
#print axioms DPMA.CuspCertificates.domainCusp_krawczyk_signs
#print axioms DPMA.CuspCertificates.domainCusp_krawczyk_newton_X_overlap
