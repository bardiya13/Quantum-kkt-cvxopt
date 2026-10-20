## 18. Diagnostics

* **D1, dequantisation probe.** For every captured system, the fraction of singular directions that a truncated solve needs to reproduce the Newton direction to 30 %, 10 % and 1 %, and the stable rank. A low-rank structure would let a sampling-based classical algorithm compete with a quantum solver.
* **E5, optimisation-level iterative refinement** (Gleixner et al.). The QP is solved to $10^{-3}$, shifted and scaled around the current point by $\Delta$, and the correction QP is solved again, five times. Inner solvers: CVXOPT's own, exact solves with injected error, loosely accepted MINRES (combined test, $\tau = 10^{-5}$), and HHL without fallback. The question is whether an outer loop makes a low-accuracy KKT solver sufficient.
* **E6, mat-vec benchmark.** CPU and GPU time of one sparse KKT mat-vec at every size of `sparse_banded`; `skipped` without a GPU.

Instances without equality constraints also log $\kappa(S)$ of the reduced system and $\kappa(B) = \sqrt{\kappa(S)}$ of its square-root factor next to $\kappa(\tilde K)$ and $\kappa(D\tilde K D)$ (Section 9), so the choice of target system is measured.
