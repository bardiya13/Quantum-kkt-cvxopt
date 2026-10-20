## 8. KKT solvers and the solve protocol

Every KKT solve of a method other than the unmodified baseline goes through `KKTSolveEngine.solve`:

1. **Base solve** with the method's solver (table below).
2. **Verify** the direction on the original system with the method's acceptance test (Section 3.4). Krylov methods stop as soon as the test holds; it is evaluated every `check_every` iterations, because checking every iteration costs as much as the iteration itself.
3. **Refine** at most `max_refine` times with the same solver on the residual.
4. **Fall back** to a direct solve of the reduced quasi-definite system (CHOLMOD) if the test still fails and the method allows it. Fallbacks are counted and logged as `classical_fallback`; a run counts as fallback-free only with zero fallbacks.

The reference solver `exact` never refines, never falls back and does not stop a run on a large residual; only a non-finite direction raises `ArithmeticError`, as in CVXOPT itself. Every solve log carries all three test values (`relres`, `backward_err`, `block_ratio_max` and the per-block ratios), the estimate of $\|\tilde K\|_2$ and the timing components, so the acceptance tests can be re-evaluated offline.

| kind | solver |
|---|---|
| `exact` | dense LU or SuperLU of $\tilde K$ |
| `exact_reduced` | CHOLMOD on the reduced system $[[S, A^\top], [A, 0]]$ with a dense Schur complement |
| `noise` | exact solve plus an injected relative error (isotropic or per block) |
| `minres` | SciPy MINRES, optional Ruiz scaling |
| `minres_base` | MINRES written against NumPy or CuPy, optional warm start (baseline kernel) |
| `minres_fused` | MINRES with one host synchronisation per iteration and fused vector updates (latency-optimised kernel) |
| `cg_reduced` | CG on $S$ for $p = 0$ |
| `oss_cg` | CG on the Jacobi-scaled null-space system $Z^\top S Z$ |
| `lsqr_sqrt` | LSQR on the square-root factor $B$ with $B^\top B = S$ |
| `gmres_stalelu` | GMRES preconditioned with the factors of an earlier IPM iteration |

The cell ends with self-tests of both MINRES kernels against a dense solve, on the GPU as well when one is available.
