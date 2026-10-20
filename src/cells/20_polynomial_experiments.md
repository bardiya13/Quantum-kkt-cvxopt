## 20. Polynomial-solver experiments (Q5)

* **Q5a.** On the captured KKT systems: forward error of HHL (largest clock register, ideal readout) against the polynomial solver at $\epsilon = 10^{-3}, 10^{-6}, 10^{-10}$, with walk steps, qubits, $\|c\|_1$ and post-selection probability.
* **Q5b.** Total cost (samples, block-encoding queries) of refining a finite-shot solve to a residual of $10^{-10}$, over circuit precision $\epsilon$ and samples per round, for both polynomials.
* **Q5c.** The polynomial solvers inside `coneqp` without classical fallback, with the combined test and the accounting of Q3, together with HHL at the same refinement budget, and the null-space solver `lcu_oss_*`.
* **Q5d.** On the same captured systems, the null-space target: dimension $n - p$ instead of $n + m + p$, Jacobi-scaled $\kappa$, the shifted-Chebyshev degree next to the odd-polynomial degree of the augmented system, qubits, and the forward error of the full KKT direction (`oss_target_comparison.csv`).
* **Q5e.** The finite-shot operating point ($\epsilon = 0.3$) over sampling seeds × accepted samples per round, so the success rate comes with a spread.

A run whose polynomial degree exceeds `max_degree` is reported as over the simulation budget; it is never replaced by a classical solve.
