## 5. Synthetic QP instances

For given $(n, m, p)$, a target condition number $\kappa_P$, a density and a seed:

1. $P = Q\,\mathrm{diag}(\lambda)\,Q^\top$ with $\lambda$ log-spaced in $[1, \kappa_P]$ for the dense families, so $\kappa_2(P) = \kappa_P$. The sparse families use $P = \mathrm{diag}(\lambda) + E$ with a sparse symmetric $E$ whose Gershgorin radius is at most $0.1\,\lambda_{\min}$; then $P \succ 0$ and $\kappa_2(P)$ is within a factor 1.3 of $\kappa_P$. The measured $\kappa_2(P)$ is stored.
2. $G$ ($m\times n$) and $A$ ($p \times n$, full row rank, checked) are random dense or sparse matrices.
3. $x^\star$ is random. An active set $\mathcal S$ with $|\mathcal S| = \lceil \text{active\_frac}\cdot m\rceil \le n - p - 1$ gets $s^\star_i = 0,\; z^\star_i \in [0.5, 2]$, the other rows $s^\star_i \in [0.5, 2],\; z^\star_i = 0$ (strict complementarity); $y^\star$ is random.
4. $q = -(Px^\star + G^\top z^\star + A^\top y^\star)$, $h = Gx^\star + s^\star$, $b = Ax^\star$. Then $(x^\star, y^\star, z^\star, s^\star)$ satisfies the KKT conditions exactly and $f^\star = \tfrac12 x^{\star\top} P x^\star + q^\top x^\star$ is the optimal value.

| family | structure | role |
|---|---|---|
| `dense_kappa`, `dense_noeq` | dense, $\kappa_P \in \{1, 10^3, 10^6\}$ | acceptance-test ablation, precision of a direction |
| `sparse_scale` | 5 % random density | small sizes only: $G^\top G$ fills in completely |
| `sparse_banded` | banded $P$, 4 nonzeros per row of $G$ and $A$ in random columns | high-fill scaling family |
| `sparse_lowfill` | as above, nonzeros within a window of 13 columns | low-fill control: $P + G^\top W^{-2} G$ is banded and sparse Cholesky is $O(n)$ |

The three tiny instances `tiny_A` ($n=2$, $m=2$), `tiny_B` ($n=3$, $m=4$, $p=1$) and `tiny_C` ($n=4$, $m=4$) have KKT dimension 4, 8 and 8; their KKT systems are the ones passed to the quantum solvers.
