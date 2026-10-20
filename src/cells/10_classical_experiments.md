## 10. Classical experiments

* **E1, contract check.** `custom_exact` must reproduce the iterations and objective of the unmodified solver on the tiny instances. Their KKT systems are captured here for the quantum experiments.
* **E2, acceptance-test ablation** on `dense_kappa` and `dense_noeq`: the unmodified solver, the exact references, the relative-residual, backward-error, block and combined tests with and without Ruiz scaling, CG on the reduced system, LSQR on the square-root system, lazy refactorisation, and both MINRES kernels. `full_ablation` (on in `medium` and `large`) runs every method defined in Section 9; otherwise an essential subset.
* **E3, scaling** on `sparse_banded`, `sparse_lowfill` and `sparse_scale`, with several seeds per size (fewer at the largest sizes), the null-space CG up to $n = 3\,200$ and peak memory for every run.
* **E4, precision of a direction**: exact solves with an injected isotropic error ($\delta = 10^{-8} \dots 10^{-6}$) or block-relative error ($\delta = 10^{-1} \dots 10^{-3}$).

Time budgets are per method. When a method exceeds `max_run_seconds` on one instance, it is skipped on the larger instances of the same family, and a power law through the last two sizes skips a run that would exceed the budget before it starts. The other methods are not affected. A running CVXOPT call cannot be interrupted, so the budget acts between runs.
