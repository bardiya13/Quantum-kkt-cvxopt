## 9. Instrumented `kktsolver` and the benchmark runner

`make_kktsolver` wraps a solver in the callback contract of Section 3.2. For every factorisation it records the assembly time, $\kappa_2(\tilde K)$ (reference runs only) and the spread of the NT scaling; for every solve, the test values before and after refinement, refinements, fallback, Krylov iterations and the timing components.

`run_instance` calls `solvers.coneqp` (or `conelp`) with the same algorithm parameters for every method, parses the per-iteration table that CVXOPT prints, and recomputes the feasibility and optimality measures from the returned iterate. This includes a valid duality gap: the Lagrange dual function evaluated at $z_+ = \max(z, 0)$ is a lower bound on the optimal value when $P \succ 0$, so `relgap_valid` does not depend on the solver's own report.

A run is classified at three levels:

* `accepted`: the final iterate passes the thresholds in `CONFIG["accept"]`;
* `pure_method_success`: accepted with no classical fallback;
* `strict_method_success`: in addition, every direction passed the acceptance test (only relevant when fallback is disabled).

Times are wall-clock seconds (`time.perf_counter`). `wall_total` is the whole `coneqp` call and `wall_method = wall_total - t_diag` excludes the condition-number diagnostic. `t_assemble`, `t_factor`, `t_solve`, `t_verify`, `t_refine` and `t_fallback` are summed over all callback calls, so `wall_method` minus their sum is the time spent in CVXOPT's own iteration.

`method_specs` defines every classical method; the quantum methods are added in Sections 16 and 20.
