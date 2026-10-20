## 3. Formulation

### 3.1 Problem and optimality conditions

`cvxopt.solvers.coneqp` solves, for the linear cone used here,

$$
\min_x\; \tfrac12 x^\top P x + q^\top x \quad\text{s.t.}\quad Gx + s = h,\; Ax = b,\; s \ge 0,
\qquad P = P^\top \succeq 0,\; x\in\mathbb R^n,\; s\in\mathbb R^m,\; b\in\mathbb R^p .
$$

With multipliers $z \ge 0$ and $y$ the KKT conditions are

$$
P x + q + G^\top z + A^\top y = 0,\qquad Ax = b,\qquad Gx + s = h,\qquad s\ge 0,\; z \ge 0,\qquad s_i z_i = 0 .
$$

CVXOPT's primal-dual path-following method (Vandenberghe 2010) uses a Mehrotra-type predictor-corrector and the Nesterov-Todd scaling $W$; for the nonnegative orthant $W = \mathrm{diag}(d)$ with $d_i = \sqrt{s_i/z_i}$. CVXOPT passes the scaling as the dictionary `W` (`W['d']`, `W['di'] = 1/d`, and `v`, `beta`, `r`, `rti` for second-order and semidefinite cones).

### 3.2 The KKT system replaced by the callback

Each interior-point iteration computes an affine and a combined direction, each with one solve of

$$
\begin{bmatrix} P & A^\top & G^\top \\ A & 0 & 0 \\ G & 0 & -W^\top W \end{bmatrix}
\begin{bmatrix} u_x \\ u_y \\ u_z \end{bmatrix} =
\begin{bmatrix} b_x \\ b_y \\ b_z \end{bmatrix},
$$

plus one solve at initialisation with $W = I$. The callback contract (CVXOPT user guide, *Exploiting structure*, and `misc.kkt_ldl`) is: `f = kktsolver(W)` returns a function `f(x, y, z)` that receives $b_x, b_y, b_z$ in `x, y, z` and overwrites them with $u_x$, $u_y$ and $W u_z$. With $w = W u_z$ and the third block row multiplied by $W^{-\top}$, this is the symmetric system that CVXOPT's own `'ldl'` solver factors and that every method in this notebook forms, measures and solves:

$$
\tilde K \tilde u = \tilde r,\qquad
\tilde K = \begin{bmatrix} P & A^\top & G^\top W^{-1} \\ A & 0 & 0 \\ W^{-\top} G & 0 & -I \end{bmatrix},\quad
\tilde u = \begin{bmatrix} u_x \\ u_y \\ w \end{bmatrix},\quad
\tilde r = \begin{bmatrix} b_x \\ b_y \\ W^{-\top} b_z \end{bmatrix}.
$$

$\tilde K$ is symmetric indefinite of dimension $N = n + p + m$. Eliminating $w$ gives the reduced system with $S = P + G^\top W^{-1}W^{-\top}G$, which is positive definite and is what CVXOPT's default `'chol2'` solver factors. Neither $\kappa(\tilde K)$ nor $\kappa(S)$ is controlled by $\kappa(P)$ alone, because $d_i \to 0$ for active and $d_i \to \infty$ for inactive constraints as $\mu \to 0$ (paper, Section 3).

CVXOPT's `refinement` option wraps any `kktsolver` in an outer refinement loop. It is fixed for every method (0 for QPs, the CVXOPT default for cone LPs), so that an inexact solver is never corrected silently; the refinement and fallback of the solvers below are logged separately.

### 3.3 Stopping conditions

With `abstol` $=10^{-7}$, `reltol` $=10^{-6}$ and `feastol` $=10^{-7}$, CVXOPT returns `status = 'optimal'` when

$$
\text{pres} = \max\!\Big(\tfrac{\|Ax-b\|_2}{\max(1,\|b\|_2)},\, \tfrac{\|Gx+s-h\|_2}{\max(1,\|h\|_2)}\Big)\le \texttt{feastol},\qquad
\text{dres} = \tfrac{\|Px+q+G^\top z+A^\top y\|_2}{\max(1,\|q\|_2)} \le \texttt{feastol},
$$

and either $s^\top z \le \texttt{abstol}$ or the relative gap is at most `reltol`; otherwise it stops with `'unknown'` after `maxiters`. The runner recomputes all these quantities from the returned $(x, y, s, z)$, together with a valid duality gap and the objective error against the known optimum.

### 3.4 Acceptance tests for an inexact KKT solve

Let $e = \tilde K\tilde u - \tilde r$ be the residual of a computed direction and $\tau$ the tolerance of the method. The notebook implements five tests, all evaluated on the original (unscaled, unregularised) system. The relative-residual and backward-error tests accept at $10\,\tau$ (`accept_factor` in the configuration); tolerances in method names are the values of $\tau$.

* `relres`: $\|e\|/\|\tilde r\| \le 10\,\tau$. In floating point the smallest attainable residual is of order $u\,(\|\tilde K\|\|\tilde u\| + \|\tilde r\|)$ with unit round-off $u$. Late on the central path $\|\tilde K\|\|\tilde u\| \gg \|\tilde r\|$, so this test can be out of reach even for an LU solve (figure `04_conditioning_across_iterations.png`), and it does not distinguish the blocks of the right-hand side.
* `relres_floor`: `relres`, or accept when the backward error is already at the level of a direct solve ($10^{-14}$).
* `backward`: the normwise backward error $\omega = \|e\| / (\|\tilde K\|_2\|\tilde u\| + \|\tilde r\|) \le 10\,\tau$ (Higham, *Accuracy and Stability of Numerical Algorithms*, Thm. 7.1). A direction with backward error $\omega$ solves a system whose matrix and right-hand side are perturbed by at most $\omega$ in relative norm. $\|\tilde K\|_2$ is estimated once per factorisation by 20 power iterations.
* `block`: block-wise forcing $\|e_b\| \le \eta \max(\|\tilde r_b\|, \epsilon_{\text{abs}}\rho_b)$ for $b \in \{x, y, z\}$ with $\rho_x = \max(1,\|q\|)$, $\rho_y = \max(1,\|b\|)$, $\rho_z = \max(1,\|h\|)$ and $\epsilon_{\text{abs}} = 10^{-9}$. The $x$ and $y$ blocks of $\tilde r$ are the dual and primal residuals that CVXOPT drives to zero; the $z$ block scales with $\sqrt{s^\top z}$, so the test tightens along the path.
* `combined`: `block` with $\eta = 0.1$ and `backward` together, i.e. $\max(\beta/\eta,\ \omega/(10\,\tau)) \le 1$. This is the test of the paper (Section 4.1, Eq. (5)); $\tau = 10^{-10}$ for the classical `*_comb*` methods and $10^{-9}$ for the quantum solvers, so $\omega \le 10^{-9}$ and $\omega \le 10^{-8}$.

For any computed $\tilde u$, $\|\delta u\|/\|\tilde u\| \le \kappa(\tilde K)\,\|e\|/(\|\tilde K\|\|\tilde u\|)$. A backward error $\omega$ therefore bounds the forward error by about $\kappa\,\omega$; a relative residual $\rho$ bounds it by $\kappa\rho\,\|\tilde r\|/(\|\tilde K\|\|\tilde u\|)$; the block test bounds the block residuals and gives no forward-error bound by itself.

The tests are evaluated every `check_every` Krylov iterations and after the solve. The protocol around them (refinement, fallback, accounting) is implemented in `KKTSolveEngine.solve`, Section 8.

### 3.5 Regularised target for the quantum solvers

$\tilde K_\delta = \tilde K + \delta J$ with $J = \mathrm{diag}(I_n, -I_{p+m})$, applied to the Ruiz-scaled system, keeps the quasi-definite structure and bounds $\kappa(\tilde K_\delta) \lesssim (1+\delta)/\delta$. The direction is refined on the true system, so the error introduced by $\delta$ is measured (Section 17).
