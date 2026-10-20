# Synthetic QP families with a known primal-dual solution (x*, y*, z*, s*) and prescribed kappa(P).
def _sparse_random(rng, rows, cols, density):
    nnz = max(rows, int(round(density * rows * cols)))
    I = np.concatenate([np.arange(rows), rng.integers(0, rows, nnz - rows)])      # every row gets >= 1 entry
    J = np.concatenate([rng.integers(0, cols, rows), rng.integers(0, cols, nnz - rows)])
    V = rng.standard_normal(nnz)
    M = sp.coo_matrix((V, (I, J)), shape=(rows, cols)).tocsr()
    M.sum_duplicates()
    return M

def _banded_rows(rng, rows, cols, nnz_per_row):
    """Sparse matrix with exactly nnz_per_row random nonzeros per row (values ~ N(0,1)/sqrt(nnz))."""
    k = min(nnz_per_row, cols)
    J = np.concatenate([rng.choice(cols, k, replace=False) for _ in range(rows)])
    I = np.repeat(np.arange(rows), k); V = rng.standard_normal(rows * k) / np.sqrt(k)
    return sp.coo_matrix((V, (I, J)), shape=(rows, cols)).tocsr()

def _local_rows(rng, rows, cols, nnz_per_row, halfwidth):
    """Sparse matrix whose row i has nnz_per_row nonzeros inside a window of half-width `halfwidth` around column i*cols/rows.
    G'G and A'A are then banded and the Cholesky factor of P + G'W^-2 G has O(n) nonzeros: the low-fill control case in which
    a sparse direct solver should win."""
    k = min(nnz_per_row, cols, 2 * halfwidth + 1); J = np.empty(rows * k, dtype=np.int64)
    for i in range(rows):
        c = int(round(i * (cols - 1) / max(rows - 1, 1))); lo = max(0, min(c - halfwidth, cols - (2 * halfwidth + 1))); hi = min(cols, lo + 2 * halfwidth + 1)
        J[i * k:(i + 1) * k] = lo + rng.choice(hi - lo, k, replace=False)
    I = np.repeat(np.arange(rows), k); V = rng.standard_normal(rows * k) / np.sqrt(k)
    return sp.coo_matrix((V, (I, J)), shape=(rows, cols)).tocsr()

def make_qp_instance(inst_id, family, n, m, p, kappa_P, density, active_frac, seed, structure=None, nnz_per_row=4, bandwidth=3):
    rng = rng_for(family, inst_id, n, m, p, kappa_P, density, seed)
    sparse = density < 0.5
    structure = structure or ("dense" if not sparse else "random_sparse")
    lam = np.logspace(0, np.log10(kappa_P), n) if n > 1 else np.array([1.0])
    if structure == "banded":
        # P = diag(lam) + symmetric band of half-width `bandwidth`, Gershgorin radius <= 0.1*lam_min  => P > 0 and kappa within ~1.2x of kappa_P
        rng_b = rng_for(family, inst_id, "banded", n, m, p, kappa_P, seed)
        diags = [lam]; offs = [0]
        for k in range(1, bandwidth + 1):
            v = rng_b.uniform(-1, 1, n - k) * (0.1 * lam.min() / (2 * bandwidth)); diags += [v, v]; offs += [k, -k]
        P = sp.diags(diags, offs, format="csr")
        G = _banded_rows(rng_b, m, n, nnz_per_row)
        A = _banded_rows(rng_b, p, n, nnz_per_row) if p > 0 else None
        sparse = True
    elif structure == "banded_local":
        rng_b = rng_for(family, inst_id, "banded_local", n, m, p, kappa_P, seed)
        diags = [lam]; offs = [0]
        for k in range(1, bandwidth + 1):
            v = rng_b.uniform(-1, 1, n - k) * (0.1 * lam.min() / (2 * bandwidth)); diags += [v, v]; offs += [k, -k]
        P = sp.diags(diags, offs, format="csr")
        G = _local_rows(rng_b, m, n, nnz_per_row, 2 * bandwidth)
        A = _local_rows(rng_b, p, n, nnz_per_row, 2 * bandwidth) if p > 0 else None
        sparse = True
    elif not sparse:
        Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
        P = (Q * lam) @ Q.T
        P = 0.5 * (P + P.T)
        G = rng.standard_normal((m, n)) / np.sqrt(n)
        A = rng.standard_normal((p, n)) / np.sqrt(n) if p > 0 else None
    else:
        E = _sparse_random(rng, n, n, density / 2)
        E = 0.5 * (E + E.T)
        row_abs = np.asarray(abs(E).sum(axis=1)).ravel()
        scale = 0.1 * lam.min() / max(row_abs.max(), 1e-300)
        P = (sp.diags(lam) + scale * E).tocsr()
        G = (_sparse_random(rng, m, n, density) / np.sqrt(max(1, density * n))).tocsr()
        A = (_sparse_random(rng, p, n, max(density, 3.0 / n)) / np.sqrt(max(1, density * n))).tocsr() if p > 0 else None
    # A must have full row rank (checked densely when small enough)
    if p > 0 and p * n <= 4_000_000:
        Ad = A.toarray() if sp.issparse(A) else A
        if np.linalg.matrix_rank(Ad) < p:
            raise ValueError(f"{inst_id}: A is rank deficient")
    x_star = rng.standard_normal(n)
    n_active = min(int(math.ceil(active_frac * m)), max(0, n - p - 1))
    active = np.zeros(m, dtype=bool)
    active[rng.choice(m, n_active, replace=False)] = True
    s_star = np.where(active, 0.0, rng.uniform(0.5, 2.0, m))
    z_star = np.where(active, rng.uniform(0.5, 2.0, m), 0.0)
    y_star = rng.standard_normal(p) if p > 0 else np.zeros(0)
    Px = P @ x_star
    q = -(Px + G.T @ z_star + (A.T @ y_star if p > 0 else 0.0))
    h = G @ x_star + s_star
    b = A @ x_star if p > 0 else None
    f_star = 0.5 * x_star @ Px + q @ x_star
    Pd = P.toarray() if sp.issparse(P) else P
    kappa_meas = float(np.linalg.cond(Pd)) if n <= 2000 else float("nan")
    nnz_kkt = (P.nnz if sp.issparse(P) else n * n) + 2 * (G.nnz if sp.issparse(G) else m * n) + 2 * ((A.nnz if sp.issparse(A) else p * n) if p > 0 else 0) + m
    inst = dict(id=inst_id, family=family, kind="coneqp", n=n, m=m, p=p, kkt_dim=n + m + p,
                kappa_P_target=float(kappa_P), kappa_P_measured=kappa_meas, density=float(density), seed=int(seed), structure=structure, nnz_kkt=int(nnz_kkt),
                sparse=sparse, P=P, q=q, G=G, h=h, A=A, b=b, dims={"l": m, "q": [], "s": []},
                x_star=x_star, y_star=y_star, z_star=z_star, s_star=s_star, f_star=float(f_star),
                n_active=int(n_active), known_solution=True)
    return inst

def cvx_data(inst):
    """CVXOPT matrices for an instance (cached)."""
    if "_cvx" not in inst:
        sparse = inst["sparse"]
        d = {"P": cvx_matrix_from(inst["P"], sparse) if inst["P"] is not None else None,
             "q": to_cvx(inst["q"]), "G": cvx_matrix_from(inst["G"], sparse), "h": to_cvx(inst["h"]),
             "A": cvx_matrix_from(inst["A"], sparse) if inst["p"] > 0 else None,
             "b": to_cvx(inst["b"]) if inst["p"] > 0 else None}
        inst["_cvx"] = d
    return inst["_cvx"]

def build_family(name, spec):
    out = []
    for n in spec["n_list"]:
        m = int(round(spec["m_ratio"] * n)); p = int(round(spec["p_ratio"] * n))
        seeds = spec["seeds"] if n < spec.get("seeds_large_from_n", 10 ** 9) else spec.get("seeds_large", spec["seeds"])     # fewer seeds at the largest sizes
        for kappa in spec["kappa_list"]:
            for seed in seeds:
                inst_id = f"{name}_n{n}_k{int(np.log10(kappa))}_s{seed}"
                out.append(make_qp_instance(inst_id, name, n, m, p, kappa, spec["density"], spec["active_frac"], seed, structure=spec.get("structure"),
                                            nnz_per_row=spec.get("nnz_per_row", 4), bandwidth=spec.get("bandwidth", 3)))
    return out

def sanity_instances():
    return [make_qp_instance("tiny_A", "sanity", 2, 2, 0, 1e0, 1.0, 0.5, 0),
            make_qp_instance("tiny_B", "sanity", 3, 4, 1, 1e1, 1.0, 0.5, 1),
            make_qp_instance("tiny_C", "sanity", 4, 4, 0, 1e2, 1.0, 0.5, 2)]

INSTANCES = {}
for inst in sanity_instances():
    INSTANCES[inst["id"]] = inst
for fam, spec in CONFIG["families"].items():
    for inst in build_family(fam, spec):
        INSTANCES[inst["id"]] = inst
summary = pd.DataFrame([{k: inst[k] for k in ("id", "family", "n", "m", "p", "kkt_dim", "kappa_P_target", "kappa_P_measured", "density", "structure", "nnz_kkt", "sparse", "n_active")} for inst in INSTANCES.values()])
print(f"{len(INSTANCES)} instances generated")
print(summary.groupby("family").agg(count=("id", "size"), n_min=("n", "min"), n_max=("n", "max"), kkt_dim_max=("kkt_dim", "max"), nnz_kkt_max=("nnz_kkt", "max"), structure=("structure", "first")).to_string())
summary.to_csv(os.path.join(RESULTS_DIR, "instances.csv"), index=False)
