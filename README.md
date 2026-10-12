# Reproduction package for "Geometry-controlled disappearance of a delayed first-passage peak under localized absorption"

**Author:** Xiaoxiao Zhouyi, School of Engineering Mathematics and Technology, University of Bristol
(xiaoxiao.zhouyi@bristol.ac.uk)

This archive contains the code, the stored numerical data and the reference figures that
reproduce all six figures and the numerical values quoted in the article (main text, Table II and
the appendices). The article itself is not part of this archive.

**The model.** A Brownian particle (diffusion coefficient 1/2) moves on the interval [0, 1] with
absorbing ends and a partially absorbing point sink of strength `b` at position `θ`; it is released
at `ξ`. The density `Φ(τ)` of the absorption time is an exact signed sum of exponentials. For
release at the sink, the density shows a capture spike, a valley and a delayed peak; the valley
and the delayed peak merge (a fold) at a threshold `b_c(θ)`, which is lowest (`b_c ≃ 2.165`) for a
sink at `θ ≃ 0.381` and grows as `B*/d` near an end (`B* = 0.78903`, merger time `c_* d²`,
`c_* = 0.15792`, from an exactly solvable half-line problem). For release near, but not at, the
sink the density has two interior peaks between two fold lines that meet at a cusp. The discrete
realization is a lazy random walk on a ring with a one-way shortcut to an absorbing target. The
archive reproduces the threshold curve and its near-end law, the normal-form prefactors, the
two-peak region and its cusps (including the interval-arithmetic certificates of the half-line
cusps), the exact lattice calculations and their continuum limit, the lattice Monte Carlo and
Brownian-dynamics checks, the finite-gate calculation, and the numbers of the proposed
measurement.

## Quick start

Python 3.11 or 3.12 is required (the pinned NumPy 2.0.2 has no wheels for Python 3.13). From the
archive root:

```bash
python3.12 -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python reproduce_all.py            # light: all computations, six figures, PASS/FAIL table
python reproduce_all.py --full     # also rerun the simulations with the recorded seeds
```

`reproduce_all.py` runs every script listed below, writes all results below `outputs/`
(`outputs/figures/`, `outputs/tables/`, `outputs/data/`, one log per step in `outputs/logs/`), and
ends with a table that compares each computed value with the value quoted in the article
(`outputs/summary.txt`). It also checks that the regenerated data files are identical to the
stored ones in `data/` and that the regenerated figure PDFs are identical to the reference PDFs in
`figures/`. The exit code is 0 only if every step and every comparison passes.
`python reproduce_all.py --list` lists the steps; `--only STEP ...` runs a subset, and comparisons
whose inputs come from steps that have not been run are then reported as NOT RUN rather than as
failures (`--checks-only` repeats the comparisons on existing outputs). Every script can also be
run on its own, from any directory, e.g. `python code/fig2_threshold.py`.

**Runtime.** The default mode takes about 5 minutes on one core of a recent laptop (the longest
steps are the scan beyond the cusp, about 1.5 minutes, and the half-line checks, about 30 s).
`--full` adds about 4 minutes: the lattice Monte Carlo (4×10⁵ walkers, about 10 s), the Brownian
dynamics of Fig. 6 (about 15 s) and of Fig. 3(c) (about one minute), and the gate-width/time-step
sweep (fifteen Brownian-dynamics runs of 6×10⁴ trajectories, about two minutes). All simulations
use the recorded seeds and, with the package versions below, reproduce the stored histograms,
route counts and samples bit for bit; `--full` checks this and redraws Figs. 5 and 6 from the
regenerated data. If `python-flint` is installed, `--full` also reruns the two Arb certificates and
checks that they reproduce the stored enclosures; otherwise the stored certificates in `data/` are used (their decimal bounds are checked in exact
rational arithmetic in every run, see below). If `lean` is installed, `--full` also checks the two
modules of `formal/` that need no mathlib.

## Requirements

Python 3.11 or 3.12. The stored outputs were produced with Python 3.12.13, NumPy 2.0.2, SciPy
1.17.1, mpmath 1.4.1, Matplotlib 3.9.4 and pytest 9.0.3, with the dependencies of these packages
pinned in `requirements.txt`, on macOS on Apple silicon (arm64; NumPy with the Accelerate BLAS).
Every numerical comparison has a tolerance and is platform independent. The comparisons that
require identical bytes (regenerated data files and figure PDFs, rerun simulations) are expected to
hold on this reference platform with the pinned versions; elsewhere, last-digit differences of the
system math library or of the font subsetting can make them report "differs" while all numerical
comparisons pass. Optional: `python-flint==0.9.0` (Arb certificates), `pdffonts` from poppler (font
check of the figure PDFs), and Lean 4 (`formal/`; see `formal/README.md`).

## Contents

| Path | Contents |
|---|---|
| `reproduce_all.py` | driver: runs all steps and compares the results with the article |
| `code/` | all scripts (see the map below); `paths.py` defines the input/output locations |
| `data/` | stored inputs of the plot-only figure scripts and stored results of the simulations and certificates (see below) |
| `data/reference_logs/` | logs of the full simulation runs (lattice Monte Carlo, Brownian dynamics, gate-width/time-step sweep) |
| `figures/` | the six figure PDFs of the article, as reference outputs |
| `formal/` | Lean 4 sources of the formally checked algebraic identities and of the exact checks of the certificate bounds |
| `MANIFEST.csv`, `SHA256SUMS` | file list with sizes, roles and SHA-256 checksums |

## Map: article → script → data

Figures (all written to `outputs/figures/`, PDF and 150-dpi PNG):

| Figure | Script | Stored input |
|---|---|---|
| Fig. 1 model and density shapes | `code/fig1_model_morphology.py` | none (computed; checked against Talbot inversion) |
| Fig. 2 threshold `b_c(θ)`, densities, approach to the threshold | `code/fig2_threshold.py` | `data/dpma_boundary_generated.csv` (from `dpma_saddle_node_bc_theta.py`), `data/dpma_fold_scaling_generated.csv` (from `dpma_saddle_node_certification.py`) |
| Fig. 3 near-end (half-line) limit | `code/fig3_near_end.py` | `data/dpma_boundary_generated.csv`, `data/dpma_endpoint_law.txt` (from `dpma_endpoint_law.py`), `data/dpma_endpoint_bd.npz` (Brownian dynamics, rerun with `--rerun-bd`) |
| Fig. 4 two-peak region | `code/fig4_two_peak_region.py` | `data/dpma_offgate_dense.csv` (from `dpma_offgate_dense.py`), `data/dpma_offgate_phase.csv/.json` (from `dpma_offgate_phase.py`) |
| Fig. 5 lattice checks | `code/fig5_lattice_checks.py` | `data/dpma_channel_mc_data.npz/.json` (from `dpma_channel_mc.py --simulate`) |
| Fig. 6 Brownian dynamics, finite gate | `code/fig6_brownian_dynamics.py` | `data/dpma_brownian_fold_data.npz/.json` (from `dpma_brownian_fold.py --simulate`) |

Quoted numbers (logs and tables in `outputs/tables/`):

| Article | Quantity | Script |
|---|---|---|
| Sec. II C, App. B | lattice vs continuum density at `N = 1200` (≲ 8×10⁻⁵, exponent 2.00), residues per mode | `lattice_continuum_convergence.py` |
| Sec. II C, App. A, App. D | lattice poles as roots of `D_u`; closed-form residues `W_j = q N_{r,u}(y_j)/D_u'(y_j)` against the eigenvector residues; certain absorption and mean from the closed forms; the `N = 1200` comparison repeated with the closed-form residues | `lattice_closed_form_residues.py` |
| Sec. III A, App. C | at most one interior maximum for release at the sink | `supplementary_checks.py` (part E), `dpma_window_scan.py` |
| Sec. III A, V | capture share near the delayed peak (about one half at `b = 2.6`; continuum 0.523, lattice 0.521) | `supplementary_checks.py` (part A), `dpma_channel_mc.py` |
| Sec. III B, App. C | no fold with two exponentials; three-term ratio | `finite_mode_algebra.py`, `formal/` |
| Sec. III C, Eq. (antip) | `b_c(1/2) = 3.0764`, `τ_c = 0.03836`, minimum 2.165 at `θ ≃ 0.381` | `dpma_saddle_node_bc_theta.py` |
| Sec. III C | curvature `b_c(1/2+ε) ≃ 3.076 − 133 ε²` | `dpma_normal_form.py` (part v) |
| Sec. III C, App. D | single interval in `b` (dense scans, seven `θ`) | `dpma_window_scan.py` |
| Sec. III C, App. D | lattice threshold `b_{c,N} − b_c ≃ 47/N²`, slope −2.001; `N²(b_{c,N} − b_c)` = 47.247 at `N = 1200`, extrapolated limit 47.25 | `dpma_bcN_convergence.py` |
| Sec. III C, Table II, App. D | nondegeneracy margins: minima 1.82996 and 0.49574; cutoff sensitivity ≤ 4×10⁻⁶ | `dpma_fold_branch_margins.py` |
| Sec. III D, Eq. (Bstar), App. C | `B* = 0.78903`, `c_* = 0.15792`; margins −2.69248, −0.50033; Jacobian −1.69×10³ | `dpma_halfline_bstar.py` |
| Sec. III D, App. C | `b_c θ = 0.789026` for `θ ∈ [0.20, 0.27]` (bisection) | `dpma_endpoint_law.py` |
| Sec. III D, App. D | `|b_c d − B*| ≤ 2×10⁻⁵` for `d ≤ 0.30`; direct folds for `d = 0.05–0.175`; half-line Brownian dynamics χ²/dof | `fig3_near_end.py` |
| Sec. III D, App. D | hatched region of Fig. 2(a): pair at `0.98 B*/θ`, none at `1.02 B*/θ` (`θ = 0.10, 0.15`) | `small_theta_check.py` |
| Sec. III D | odds `2B`; crude estimate `18e⁻³ ≈ 0.90` at `σ = 1/6`; 78% still diffusing, 1% at the wall at the merger | `endpoint_physics_checks.py`, `supplementary_checks.py` (part B) |
| Sec. III D, F, App. C | mean `≈ 0.39 d` at threshold; `dμ₁/db ≃ 1.10`; lattice mean and `π_sc` vs closed forms | `dpma_mfpt_smoothness.py` |
| Sec. III E, Eq. (prefac), App. C | `A_τ = 0.0247518`, `A_Φ = 0.357444`; agreement < 2×10⁻⁴; fitted slopes 0.5010, 1.5020 | `dpma_normal_form.py`, `dpma_saddle_node_certification.py` |
| Sec. V, App. C | K-mode truncation: 3.046, 3.0766, 3.0764 | `mode_truncation_check.py` |
| Sec. IV, Eq. (offgatecusp), App. C | range 1.839 < b < 3.862 at `ξ = 0.40`; cusp (0.3788, 5.816, 0.01608) and its nondegeneracy; `b_+(0.44) = 3.13660`; lower-line asymptote ratios | `dpma_offgate_phase.py`, `dpma_offgate_dense.py` |
| Sec. IV, Eq. (startquartic) | `C_b = 4352.656`; direct solves at `δ = 10⁻³` | `dpma_start_asymptotics.py` |
| Sec. IV, App. D | no two-peak density beyond the cusp (39 × 400 grid) | `scan_beyond_cusp.py` |
| Sec. IV, Eqs. (cuspendpoint), (cusp-certificate-boxes) | half-line cusps; interval cusp loci (`k ≤ 120, 160, 220`) | `cusp_endpoint_continuation.py` |
| App. C | half-line transform for arbitrary release `X`: `f̂(0) = 1`, the limits `B = 0` and `X = 1`, branch-cut density against Talbot inversion | `halfline_transform_checks.py` |
| App. C, App. D | exact rational checks of every interval-Newton inclusion and sign exclusion of the boxes `Q₋`, `Q₊`, the Krawczyk certificate, the factorization of the cusp Jacobian, and an independent double-precision cusp solve inside the boxes; the same comparisons as Lean statements | `certificate_rational_checks.py` → `formal/FormalLean/CuspCertificateDecimals.lean` |
| App. C, App. D, Table II | Arb certificates of the two half-line cusp boxes | `halfline_cusp_interval_newton.py`, `halfline_cusp_krawczyk.py` (need `python-flint`); stored results `data/dpma_interval_certificates.json`, `data/dpma_halfline_cusp_interval_certificate.json` |
| Sec. V, Table II | MC `π_sc = 0.5647 ± 0.0008` vs 0.5652 (from the stored per-route counts); MC histogram against the exact lattice bin probabilities; `θ = 0.38` lattice densities | `dpma_channel_mc.py` |
| Sec. V, App. D | finite gate `b_+^(ℓ)` = 3.02335, 3.07553, 3.10494; extrapolation 3.13657 | `dpma_finite_gate_check.py`, `measurement_window_ratio.py` (exact gate, no mesh) |
| Sec. V, App. D | gate width and time step: late peak within two bins (largest offset 1.48 bins) | `dpma_bd_convergence.py` (`--full`; log in `data/reference_logs/`) |
| Sec. VI, App. D | window ratio: `τ_v`, `τ_p`, `R_c`, `f_p`, `f_v`, `dR/db`, `N_t = 51 086, 319 287`, other window designs | `measurement_window_ratio.py` |
| Sec. VI, App. D | `R = 1` at `b = 2.821`; trials for `R > 1`, probability 0.64 of finding `R > 1` at the one-sided 5% level, factor 2.7 for probability 0.95; mean-based calibration; `κ`; gate of the example, `ℓ/L = 0.05` (exact gate, no mesh): `b_+^(ℓ)` = 3.0005, `R = 1` at `b = 2.714`, at `b = 2` `R = 1.098`, 1.6×10⁴ trials and mean 0.1346; colloid example with this finite-gate model (`κ ≃ 2.1 s⁻¹`, about 17 h) | `measurement_shape_test.py` |
| Table II, rows 1–3 | jump condition, closed forms of `I` and `J`, normalization identity | `dpma_verification_table.py`, `test_dpma_identities.py` |
| App. A–C, App. D (formal verification) | exact algebraic identities, the cusp Jacobian factorization, and the decimal bounds of the cusp certificates | `formal/` (Lean 4; modules 8 and 9 need no mathlib and are checked by `--full` when `lean` is installed) |

## Stored data

`data/` holds (i) the inputs of the plot-only figure scripts, which the corresponding scripts
regenerate (`reproduce_all.py` checks that the regenerated files are identical), (ii) processed
results of the simulations (histogram counts and densities, exact curves, parameters and seeds; no
raw trajectories, except the 2×10⁵ absorption times per value of `B` of the half-line Brownian
dynamics in `dpma_endpoint_bd.npz`), and (iii) the outputs of the Arb certificates
(`dpma_interval_certificates.json`, `dpma_halfline_cusp_interval_certificate.json`) and of the cusp
continuation (`cusp_endpoint_continuation.json`, `dpma_cusp_endpoint_loci.csv`).
`dpma_channel_mc_data.npz` also holds the per-route counts of the Monte Carlo run (absorption through
the shortcut or at an end), from which `π_sc` and its standard error are recomputed;
`dpma_channel_mc_summary.json` summarizes them. `data/reference_logs/` holds the logs of the
`--full` simulation runs.

## License

The code (`code/`, `reproduce_all.py`, the Lean sources in `formal/`) and the build configuration
(`requirements.txt`, `.gitignore`, `formal/lakefile.toml`, `formal/lean-toolchain`,
`formal/lake-manifest.json`) are released under the MIT License (`LICENSE`). The data (`data/`),
the figures (`figures/`) and the documentation (`README.md`, `formal/README.md`,
`formal/axioms_report.txt`, `CITATION.cff`, `MANIFEST.csv`, `SHA256SUMS`) are released under the
Creative Commons Attribution 4.0 International License (`LICENSE-DATA`).

## How to cite

Please cite the article and this archive: Xiaoxiao Zhouyi, *Reproduction package for
"Geometry-controlled disappearance of a delayed first-passage peak under localized absorption"*,
version v1.0.1, https://github.com/zhouyi-xiaoxiao/delayed-first-passage-peak (see
`CITATION.cff`). Verify the files from the archive root with `shasum -a 256 -c SHA256SUMS` (macOS)
or `sha256sum -c SHA256SUMS` (Linux).
