## 6. Maros-Mészáros QPs

A subset of the Maros-Mészáros convex QP test set (Maros and Mészáros, *Optim. Methods Softw.* 11, 1999), read from the `.mat` files of [`qpsolvers/maros_meszaros_qpbenchmark`](https://github.com/qpsolvers/maros_meszaros_qpbenchmark). These files store $\min \tfrac12 x^\top P x + q^\top x + r$ subject to $l \le Ax \le u$, with variable bounds as rows of $A$. Conversion to the `coneqp` form:

* rows with $l = u$ become equality rows, every other finite bound becomes one inequality row, free rows are dropped;
* redundant equality rows are removed by a pivoted QR of $A^\top$ (the number is recorded);
* $P$ is symmetrised and kept sparse.

No optimal value is assumed. The objective of the unmodified `coneqp` run is the reference for `obj_rel_err`, and an instance that the baseline cannot solve is reported as such for every method. Many of these $P$ are only positive semidefinite, so the acceptance of the final iterate uses CVXOPT's relative gap instead of the recomputed dual value.

The files are looked up in `search_dirs` (for example a Kaggle dataset under `/kaggle/input`) and downloaded otherwise. A file that cannot be found makes the instance `skipped`.
