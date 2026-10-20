# Derived tables

These CSV files are extracted from the result bundles of the notebook runs by `scripts/extract_tables_main.py` and `scripts/extract_tables_sessionA.py`, except `mm_kappa_diag.csv` and `mm_kappa_path.csv`, which are written by `scripts/mm_kappa_diag.py`. They are the input of `scripts/make_tex_tables.py` and `scripts/make_figures.py`. The suffix gives the session (`_main`: M, `_sessionA`, `_sessionB`); files without a suffix come from session A, except `gpu_matvec.csv` and `kernel_per_iter.csv` (session B) and the Maros-Mészáros files (session M).

Wall times are in seconds per optimisation run. Method names are those of the notebook (`SPECS` in section 9); `cvxopt_default` is the unmodified LDL factorisation of CVXOPT and `custom_exact_reduced` the reduced-system CHOLMOD reference.

## Classical solvers

| file | content | paper |
|---|---|---|
| `instances.csv` | synthetic instances: family, dimensions, condition number of P, density, nonzeros | Table 2 |
| `ablation_E2_sessionA.csv` | acceptance-test ablation per method: accepted, fallback-free, fallbacks, Krylov iterations, wall time | Table 5, Fig. 2 |
| `spectrum_E7.csv`, `deflation_E7.csv` | spectrum of the equilibrated KKT matrix along one IPM run; exact-deflation bound | Fig. 3 |
| `scaling_sparse_{banded,lowfill}_main.csv` | median wall time by KKT dimension (rows) and method (columns) | Figs. 1, 4, 6 |
| `krylov_sparse_{banded,lowfill}_main.csv` | median Krylov iterations, same layout | Section 7.3 |
| `scaling_seeds_main.csv` | per family, method and dimension: median and range over seeds, speed-ups against CVXOPT and the reduced solver | Tables 6, 20, Fig. 4 |
| `scaling_accuracy_main.csv`, `scaling_exponents_main.csv`, `scaling_same_iterations_main.csv` | accuracy of the final iterates, fitted scaling exponents, share of runs with CVXOPT's iteration count | Section 7.3 |
| `memory_main.csv` | peak resident memory and CuPy pool per method and dimension | Fig. 4 |
| `profiles_main.csv`, `profiles_summary_main.csv` | Dolan-Moré performance profiles | Fig. 5 |
| `gpu_matvec_main.csv`, `kernel_per_iter_main.csv` | sparse matrix-vector product on CPU and GPU; time per MINRES iteration of the two kernels | Fig. 6 |
| `gpu_matvec.csv`, `kernel_per_iter.csv` | the same for session B | not used in the paper |
| `mm_instances.csv`, `mm_runs.csv`, `mm_summary.csv`, `mm_fastest.csv` | Maros-Mészáros QPs: instances, runs, per-method summary, number of instances on which each method is fastest | Table 7, Fig. 7 |
| `mm_kappa_diag.csv` | condition numbers before and after equilibration, and fallbacks, on eight Maros-Mészáros QPs | Table 15, Fig. 7(b) |
| `mm_kappa_path.csv` | the same condition numbers at every IPM iteration | not used in the paper |
| `noise_E4.csv` | runs with injected direction error | Section 7.6 |
| `tolerance_sweep_main.csv` | accepted runs by direction-error model, tolerance and IPM tolerance level | Table 8, Fig. 8 |
| `outer_E5.csv`, `outer_E5_main.csv` | optimisation-level refinement, error by round and inner solver (sessions A and M) | Fig. 8 (session A) |
| `oss_classical_main.csv` | CG on the null-space system against augmented MINRES | Table 9, Fig. 14 |
| `equil_main.csv` | maximum KKT condition number along the IPM run before and after equilibration | Tables 15, 16, Fig. 15(a) |
| `sdplib.csv`, `sdplib_instances.csv` | SDPLIB with `conelp`: per method and per instance | Appendix B, Table 18 |

## Quantum solvers

| file | content | paper |
|---|---|---|
| `hhl_q1_ideal_main.csv`, `hhl_q1_shots_main.csv`, `hhl_q1_systems_main.csv` | HHL forward error by clock qubits (ideal and finite-shot readout); the captured systems | Fig. 9(a) |
| `q5a.csv` | forward error, walk steps and success probability of the polynomial solvers on captured systems | Table 16, Fig. 9 |
| `poly_table.csv` | degree and LCU 1-norm of the CKS and the optimal polynomial | Table 10, Fig. 10 |
| `q5b.csv` | refinement to a residual of 1e-10: rounds, samples, queries per system and configuration | Tables 11, 16, Figs. 11, 15(b) |
| `integrated_runs_main.csv`, `integrated_summary_main.csv` | quantum solvers inside `coneqp`: per run and per method | Tables 12, 16, Figs. 1, 12 |
| `q5c_q3_runs.csv`, `q5c_cost.csv` | the same for session A; resources per method | not used in the paper |
| `shots_main.csv`, `shots_per_instance_main.csv`, `shots_accepted_accuracy_main.csv` | finite-shot runs over sampling seeds; accuracy of the accepted directions | Table 13, Fig. 13 |
| `oss_q5a_main.csv`, `oss_q5d_main.csv` | null-space against augmented target on captured systems and inside the IPM | Table 14, Fig. 14 |
| `rounds_vs_minres_main.csv` | refinement rounds per solve against MINRES iterations on the same instance | Fig. 15(c) |

## Replication

| file | content | paper |
|---|---|---|
| `replication_main.csv` | wall time of each run in sessions M and B-E relative to session M | Fig. 16 |
| `replication_summary_main.csv`, `replication_crossovers_main.csv`, `replication_extra_main.csv` | per-session summaries, crossover dimensions and further replicated quantities | Tables 19-20 |
| `scaling_sparse_{banded,lowfill}_sessionB.csv`, `krylov_sparse_{banded,lowfill}_sessionB.csv`, `session_effect.csv` | scaling tables of session B; CPU-session over GPU-session wall time on identical instances (sessions A and B) | not used in the paper |
