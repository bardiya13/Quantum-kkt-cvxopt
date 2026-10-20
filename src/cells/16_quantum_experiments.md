## 16. HHL experiments

* **Q1, captured systems.** Clock qubits × scaling × reciprocal constant on the KKT systems captured in E1, with up to `ir_rounds` HHL refinement rounds on the true system and all three readouts.
* **Q2, Qiskit cross-check** of the simulator on the same circuit: statevector fidelity, transpiled depth and CX count, and the Aer shot sampler. Reported as `skipped` if Qiskit is not available.
* **Q3, HHL inside `coneqp`** with the combined test ($\eta = 0.1$, $\tau = 10^{-8}$, so $\omega \le 10^{-7}$) on the original system: without scaling and with $C = \Delta$, with Ruiz scaling and known $\kappa$, with and without fallback, with emulated amplitude amplification, and with the block test alone as an ablation. Each run records the directions accepted directly, accepted after refinement, used although rejected (only without fallback), fallbacks and failures.
* **Resource estimates** for the HHL circuits (qubits, gates, evolution time, post-selection probability, repetitions with and without amplitude amplification). These depend on the stated oracle assumptions and are kept apart from the measurements.
