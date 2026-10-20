## 14. HHL

### Circuit

Registers, little-endian: ancilla `anc` (1 qubit), clock `clk` ($n_c$), data `dat` ($n_d = \log_2 N$), reference flag `flg` (1).

1. **Input.** $\tilde K$ (real symmetric, indefinite), optionally Ruiz-scaled, is padded to $N = 2^{n_d}$ with $\hat\lambda_{\max} I$. The padding lies inside the spectral range, so $\kappa$ is unchanged and the padded components of the solution are zero.
2. **Spectral bound.** $\hat\lambda_{\max} \ge \|\tilde K\|_2$ comes from 40 power iterations with a safety factor 1.15. The evolution time $t = 2\pi(2^{n_c-1}-1)/(2^{n_c}\hat\lambda_{\max})$ maps every eigenvalue in $[-\hat\lambda_{\max}, \hat\lambda_{\max}]$ to a signed clock value $k_s \in [-(2^{n_c-1}-1),\, 2^{n_c-1}-1]$ (two's complement), which handles the negative eigenvalues of the saddle-point matrix. With clock resolution $\Delta = 2\pi/(2^{n_c} t)$ the rotation for $k_s$ is $R_y(2\arcsin(C/(k_s\Delta)))$, the exact reciprocal (Qiskit's `ExactReciprocal` with `neg_vals=True`). `reciprocal = "resolution"` uses $C = \Delta$; `"kappa_known"` uses $C = k_{\min}\Delta$ with $k_{\min} = \lfloor |\lambda|_{\min}/\Delta \rfloor$, the standard HHL assumption that $\kappa$ is known.
3. **State preparation.** The data and flag registers are prepared in $|v\rangle \propto |0\rangle_{\text{flg}}|\tilde r\rangle + |1\rangle_{\text{flg}}|\tilde K r_{\text{ref}}\rangle$ with a known positive reference vector $r_{\text{ref}}$ scaled so that $\|\tilde K r_{\text{ref}}\| = \|\tilde r\|$. HHL is linear, so the post-selected output is $\propto |0\rangle|x\rangle + |1\rangle|r_{\text{ref}}\rangle$ with a common scale.
4. **Phase estimation** with $U = e^{i\tilde K t}$. The powers $U^{2^j}$ are formed classically by `expm` and repeated squaring ($O(N^3)$, part of `t_prep`); on hardware they would come from Hamiltonian simulation with total evolution time $2(2^{n_c}-1)t\|\tilde K\|$.
5. **Eigenvalue inversion** with $2^{n_c}-1$ multi-controlled $R_y$ rotations on `anc`.
6. **Uncomputation** of phase estimation. Measuring `anc = 1` selects $\propto \sum_j \beta_j (C/\tilde\lambda_j)|u_j\rangle$.

### Readout of a signed, scaled vector

Basis-state probabilities give $|x_i|^2$ up to a global scale. With the reference branch, two measurement settings recover signs and scale, each post-selected on `anc = 1`:

* **Z setting:** $P(\text{flg}=0, i) \propto x_i^2$ and $P(\text{flg}=1, i) \propto r_{\text{ref},i}^2$, so $\|x\| = \|r_{\text{ref}}\|\sqrt{P(\text{flg}=0)/P(\text{flg}=1)}$ and $|x_i| = \|x\|\sqrt{P(\text{flg}=0,i)/P(\text{flg}=0)}$.
* **X setting** (H on `flg` first): $P(\text{flg}=0,i) - P(\text{flg}=1,i) \propto x_i r_{\text{ref},i}$, whose sign is that of $x_i$.

Three readouts are recorded:

* `statevector`: the amplitudes of the `anc = 1, clk = 0` branch, a diagnostic that separates the error of the algorithm from the sampling error;
* `shots`: raw finite shots including post-selection;
* `aa_shots`: a fixed number of accepted samples drawn from the distribution conditioned on `anc = 1`. This is what ideal amplitude amplification would deliver at about $1/\sqrt{P}$ circuit repetitions per sample (Brassard, Høyer, Mosca and Tapp 2002). The amplification circuit is not simulated; its overhead is reported next to the raw-shot equivalent $1/P$.

The leakage $P(\text{anc}=1, \text{clk}\neq 0)$ is measured, and the finite-shot readout marginalises over the clock register as hardware would.

### Inside `coneqp`

The `hhl` solver kind runs inside the protocol of Section 8: each KKT solve is one HHL run, followed by the acceptance test on the full system, up to `max_refine` HHL runs on the residual, and a classical fallback if allowed. Runs with `fallback = False` show what the interior-point method does with quantum directions alone. The cell ends with a functional test on a $4\times4$ indefinite system.
