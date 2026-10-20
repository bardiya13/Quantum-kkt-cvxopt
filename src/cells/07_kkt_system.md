## 7. KKT assembly, conditioning and scaling

`KKTAssembler` forms $\tilde K$ with the block layout and the $W^{-\top}G$ scaling of `cvxopt.misc.kkt_ldl`; for semidefinite cones it uses the same packed storage with $\sqrt2$-scaled off-diagonals (`misc.scale`, `misc.pack`). Every solver therefore sees the matrix that CVXOPT's own `'ldl'` option factors. The cell ends with a check against `misc.kkt_ldl` on `tiny_B`.

$\kappa_2(\tilde K)$ is computed by an SVD at every factorisation of the reference runs up to dimension `cond_limit`. This diagnostic is timed separately and excluded from the method timings; above the limit the field is `NaN`.

Ruiz equilibration $\tilde K_s = D\tilde K D$ (six sweeps of $d_i \leftarrow d_i/\sqrt{\|(D\tilde K D)_{i,:}\|_\infty}$) keeps the matrix symmetric, so MINRES and the quantum solvers can both use it; the solution is $\tilde u = D\tilde u_s$. For a quantum solver $D$ costs only row norms, and since the IPM reads the solution out classically anyway, applying $D$ to the result is a classical rescaling.

`reduced_spd` eliminates $w$ for $p = 0$ and returns $S = P + G^\top W^{-1}W^{-\top}G$ with the maps between the full and the reduced system.
