# Certified classical and quantum KKT solves inside CVXOPT

Companion notebook of the paper *Certified Classical and Quantum Linear Solves inside a Production Interior-Point Method: Equilibration, Low-Precision Refinement, and Conditions for a Quantum Advantage*.

The notebook replaces the KKT factorisation of `cvxopt.solvers.coneqp` through its `kktsolver` callback. Every replacement, classical or quantum, runs under the same acceptance test, the same classical-fallback protocol and the same accounting. The notebook contains

* instance generators: synthetic QP families with a known primal-dual solution, a subset of the Maros-Mészáros QP collection and, optionally, SDPLIB problems solved with `conelp`;
* classical KKT solvers: LU and reduced (CHOLMOD) direct references, equilibrated MINRES with a baseline and a latency-optimised kernel (NumPy or CuPy), CG on the reduced and null-space systems, lazy refactorisation;
* a gate-level statevector simulator, phase-estimation HHL, and a polynomial linear solver (qubitised walk with a linear combination of unitaries) for the augmented and the null-space target systems;
* every experiment of the paper, the figures, a summary report and a CSV export of all tables.

All quantum results are classical simulations of ideal circuits with oracle block encoding and state preparation.

## Running the notebook

There are two builds. They differ only in the default device in the configuration cell.

| build | Kaggle accelerator | measured run time, profile `medium` |
|---|---|---|
| `quantum_kkt_cvxopt_cpu.ipynb` | None | 4.7 h |
| `quantum_kkt_cvxopt_gpu.ipynb` | GPU T4 | 5.8 h |

Switch Internet access on in the Kaggle session settings. The notebook downloads the Maros-Mészáros and SDPLIB files at run time and installs Qiskit for the simulator cross-check; without Internet the affected experiments are reported as `skipped` and everything else runs.

The problem sizes are chosen by `RUN_PROFILE` in the configuration cell or by the environment variable `QKKT_PROFILE`: `smoke` (a few minutes, exercises every code path), `small` (default), `medium` (used for the paper) and `large`. All output goes to `/kaggle/working/results/` (`./kaggle_working_local/results/` outside Kaggle) and is compressed into `results_bundle_<cpu|gpu>_<profile>.zip` at the end.

## Notebook sections and the paper

| section | experiment id | paper |
|---|---|---|
| 1-9 | configuration, generators, KKT assembly, solvers, runner | Sections 3, 4 and 6 |
| 10 | `E1_contract_tiny` | contract check of the callback, Section 3.1, Table 4 |
| 10 | `E2_main` | acceptance-test ablation, Section 7.1, Table 5, Fig. 2 |
| 10 | `E3_scaling` | scaling, Section 7.3, Table 6, Figs. 4-5 |
| 10, 11 | `E4_noise`, `E4b_tolerance_sweep` | precision of a Newton direction, Section 7.6, Table 8, Fig. 8 |
| 12 | `E8_maros_meszaros` | real-world QPs, Section 7.5, Table 7, Fig. 7 |
| 16 | `Q1_captured`, `Q2_qiskit_crosscheck`, `Q3_integrated` | HHL, Section 8.2, Fig. 9(a) |
| 17 | `Q4_regularization_qp`, `Q4_regularization_sdplib` | regularised target with refinement; not reported in the paper |
| 18 | `D1_dequantisation`, `E5_outer_refinement`, `E6_gpu_matvec` | Sections 7.4, 7.6 and 7.7 |
| 19 | `E7_spectrum_deflation` | conditioning along the IPM run, Section 7.2, Fig. 3 |
| 20 | `Q5ab_cks_captured` | polynomial solver and refinement cost, Sections 8.3-8.4, Tables 10-11, Figs. 9-11 |
| 20 | `Q5c_integrated_cks`, `Q5e_shot_sweep` | quantum solvers inside the IPM, Sections 8.5-8.7, Tables 12-14, Figs. 12-14 |
| 21 | `S1_sdplib` | SDPLIB, Appendix B, Table 18 |
