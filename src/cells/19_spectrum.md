## 19. Conditioning along one interior-point run (E7)

The total number of MINRES iterations per run grows from about 1 500 to 38 000 as the KKT dimension grows from 310 to 79 360. E7 captures every $W$ and right-hand side of one run of `sparse_banded` and computes, per IPM iteration:

1. $\kappa$ of the Ruiz-scaled KKT matrix, the share of eigenvalues near $\pm 1$ and the number of small eigenvalues, to separate ill-conditioning from the eigenvalue distribution; and which test binds (iterations with the backward-error test alone against the combined test, and the block with the largest ratio);
2. MINRES on the matrix-free, Jacobi-scaled reduced system $[[P + \hat G^\top\hat G, A^\top], [A, 0]]$, whose vectors are 2.8 times shorter. $\kappa(S)$ contains $W^{-2}$, so the reduced target should win early in the run and lose late;
3. at the last IPM iteration, MINRES with the $k$ eigenvectors of smallest $|\lambda|$ deflated exactly. This bounds what any recycling or deflation method can gain, since those use only approximations of these vectors.
