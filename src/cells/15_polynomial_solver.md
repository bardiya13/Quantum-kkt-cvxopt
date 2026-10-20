## 15. Polynomial linear solver

HHL's error is set by the clock register ($\propto 2^{-n_c}$) and is systematic, so refinement cannot remove it: with ten clock qubits and ideal readout the forward error stays at $10^{-3}$ to $10^{-2}$, against the roughly $10^{-7}$ that the interior-point method needs (Section 16).

Childs, Kothari and Somma (*SIAM J. Comput.* 46, 2017) approximate $A^{-1}$ by an odd Chebyshev polynomial of degree $O(\kappa\log(\kappa/\epsilon))$ and implement it as a linear combination of unitaries over a qubitised walk. The cost of precision is then $\log(1/\epsilon)$. QSVT (Gilyén et al. 2019) realises the same polynomial with fewer ancillas. Two polynomials are available:

* `poly = "optimal"` (default): the minimax-optimal odd polynomial of Sünderhauf et al. (arXiv:2507.15537). If its double-precision evaluation misses the requested accuracy or its degree exceeds 60 000, the CKS expansion is used and the run is labelled accordingly.
* `poly = "cks"`: the CKS expansion.

The remaining error of a finite-shot readout is zero-mean sampling noise, which iterative refinement on the true system contracts geometrically, as in the iterative-refinement QIPM literature (Mohammadisiahroudi, Wu, Augustino, Terlaky et al.). The default operating point is therefore a low-precision circuit ($\epsilon = 0.3$) with few samples per refinement round, in line with the fixed-precision analysis of Harkness et al. (arXiv:2609.10250).

**Simulated.** A dense Hermitian unitary block encoding of $A/\|A\|$, the walk $W$ with $\Pi W^k \Pi = T_k(A)$ (asserted), LCU post-selection with its exact success probability, and the readout of Section 14 (statevector, raw shots, emulated amplitude amplification). SELECT only controls on the index register, so the post-selected output is accumulated term by term; the self-test compares this with an explicit index-register statevector. **Not simulated:** gate synthesis of the block encoding and of PREP, amplitude amplification, hardware noise. The reported walk steps, qubit counts and success probabilities are those of the full circuit.

**Simulation shortcut.** For a fixed matrix the output is $p(A/\alpha)\,b$ for every right-hand side. The walk is simulated once per KKT matrix and compared with the spectral form $V p(\Lambda) V^\top$ of the same polynomial (difference logged as `q_walk_check_diff`, required below $10^{-7}$); the second right-hand side and all refinement rounds reuse $p(\lambda_i)$. This affects only the cost of the classical simulation.

**SPD target.** For the null-space system $Z^\top S Z$ the polynomial is the shifted Chebyshev expansion of $1/x$ on $[1/\kappa, 1]$, with $c_0 = \sqrt\kappa$, $c_k = 2\sqrt\kappa(-1)^k\rho^k$ and $\rho = (\sqrt\kappa-1)/(\sqrt\kappa+1)$. Its degree is $O(\sqrt\kappa\log(\kappa/\epsilon))$, the counterpart of the $\sqrt\kappa$ of CG. The walk acts on a block encoding of the shifted matrix $(2A/\alpha - (1+a)I)/(1-a)$, so that every $T_k$ is available; the LCU normalisation of about 3 for building that encoding from one of $A$ is reported, not simulated.

`block_encoding_costs` reports the normalisation of two standard encodings (sparse access, Pauli LCU) for comparison with the ideal $\alpha = \|A\|_2$ used in the simulation.
