# Maros-Meszaros QPs: locate or download the .mat file, convert to coneqp form, register as family "maros_meszaros".
MM_CFG = CONFIG["maros_meszaros"]; MM_INSTANCES = {}; MM_LOG = []; MM_INF = 1e19

def find_mm_file(name):
    fname = f"{name}.mat"
    for d in MM_CFG["search_dirs"]:
        if not os.path.isdir(d): continue
        p = os.path.join(d, fname)
        if os.path.isfile(p): return p, "local"
        for root, dirs, files in os.walk(d):
            if fname in files: return os.path.join(root, fname), "local"
            if root[len(d):].count(os.sep) >= 3: dirs[:] = []
    cache = os.path.join(CONFIG["output_root"], "maros_meszaros_cache"); os.makedirs(cache, exist_ok=True); p = os.path.join(cache, fname)
    if os.path.isfile(p) and os.path.getsize(p) > 0: return p, "cache"
    if MM_CFG.get("download_if_missing", True):
        import urllib.request
        try:
            urllib.request.urlretrieve(MM_CFG["download_url"].format(name=name), p)
            if os.path.getsize(p) > 0: return p, "download"
        except Exception as e:
            MM_LOG.append(f"{name}: download failed ({type(e).__name__}: {str(e)[:80]})")
    return None, None

def mm_to_instance(name, path):
    import scipy.io
    d = scipy.io.loadmat(path)
    P = sp.csr_matrix(d["P"]).astype(float); A = sp.csr_matrix(d["A"]).astype(float); q = np.asarray(d["q"], dtype=float).ravel()
    l = np.asarray(d["l"], dtype=float).ravel(); u = np.asarray(d["u"], dtype=float).ravel(); r0 = float(np.asarray(d.get("r", 0.0)).ravel()[0]) if "r" in d else 0.0
    n = P.shape[0]; P = (0.5 * (P + P.T)).tocsr(); P.sum_duplicates()
    l = np.where(l <= -MM_INF, -np.inf, l); u = np.where(u >= MM_INF, np.inf, u)          # the .mat files encode "no bound" as +-1e20
    tol_eq = 1e-10 * np.maximum(1.0, np.abs(l)); eq = np.isfinite(l) & np.isfinite(u) & (u - l <= tol_eq)
    lo = np.isfinite(l) & ~eq; up = np.isfinite(u) & ~eq; free = ~np.isfinite(l) & ~np.isfinite(u)
    Aeq = A[eq]; beq = 0.5 * (l[eq] + u[eq]); p = int(eq.sum()); removed = 0
    if p > 0 and p * n <= 4_000_000:
        Ad = Aeq.toarray(); rk = int(np.linalg.matrix_rank(Ad))
        if rk < p:                                       # keep an independent subset of the equality rows (pivoted QR of A')
            _, _, piv = sla.qr(Ad.T, mode="economic", pivoting=True); keep = np.sort(piv[:rk]); Aeq = Aeq[keep]; beq = beq[keep]; removed = p - rk; p = rk
    G = sp.vstack([-A[lo], A[up]], format="csr") if (lo.any() or up.any()) else sp.csr_matrix((0, n))
    h = np.concatenate([-l[lo], u[up]]); m = G.shape[0]
    rank_ok, gram_cond = True, np.nan
    if n <= 2500:                                    # coneqp needs rank([P; A; G]) = n: test the n x n Gram matrix P'P + A'A + G'G (dense, moderate n only)
        Gm = (P.T @ P + G.T @ G + (Aeq.T @ Aeq if p else 0)).toarray(); gram_cond = float(np.linalg.cond(Gm)); rank_ok = bool(np.isfinite(gram_cond) and gram_cond < 1e14)
    nnz_kkt = int(P.nnz + 2 * G.nnz + 2 * (Aeq.nnz if p else 0) + m)
    inst = dict(id=f"mm_{name}", family="maros_meszaros", kind="coneqp", n=int(n), m=int(m), p=int(p), kkt_dim=int(n + m + p), kappa_P_target=np.nan, kappa_P_measured=np.nan,
                density=float(A.nnz / max(1, A.shape[0] * n)), seed=0, structure="real", nnz_kkt=nnz_kkt, sparse=True, P=P, q=q, G=G, h=h, A=(Aeq.tocsr() if p else None), b=(beq if p else None),
                dims={"l": int(m), "q": [], "s": []}, x_star=None, y_star=None, z_star=None, s_star=None, f_star=np.nan, n_active=-1, known_solution=False, solution_source="",
                no_dual_recompute=True, obj_offset=r0, mm_name=name, mm_rows_total=int(A.shape[0]), mm_rows_free=int(free.sum()), mm_eq_rows_removed=int(removed), mm_nnz_P=int(P.nnz), mm_nnz_A=int(A.nnz),
                mm_P_zero=bool(P.nnz == 0), mm_rank_ok=rank_ok, mm_gram_cond=gram_cond)
    return inst

if not MM_CFG.get("enabled", True):
    REC.status("MM_load", "skipped", "disabled in CONFIG")
else:
    t_mm = time.perf_counter(); missing = []
    for name in MM_CFG["instances"]:
        path, src = find_mm_file(name)
        if path is None: missing.append(name); continue
        try:
            inst = mm_to_instance(name, path); inst["source"] = src
            if inst["kkt_dim"] > MM_CFG.get("max_kkt_dim", 10 ** 9): REC.status(f"MM_{name}", "skipped", f"kkt_dim {inst['kkt_dim']} > max_kkt_dim"); continue
            if inst["m"] == 0: REC.status(f"MM_{name}", "skipped", "no inequality rows after conversion"); continue
            if not inst["mm_rank_ok"]: REC.status(f"MM_{name}", "skipped", f"rank([P; A; G]) < n (Gram condition {inst['mm_gram_cond']:.1e}): violates the coneqp assumption; free variables without curvature"); continue
            MM_INSTANCES[inst["id"]] = inst; INSTANCES[inst["id"]] = inst
        except Exception as e:
            REC.failure(f"mm_{name}", "parse", repr(e)[:300]); MM_LOG.append(f"{name}: {repr(e)[:120]}")
    for msg in MM_LOG: print("  [mm]", msg)
    if missing: REC.status("MM_missing", "skipped", f"not found / not downloadable (Internet off?): {missing}")
    if MM_INSTANCES:
        mm_summary = pd.DataFrame([{k: inst[k] for k in ("id", "family", "n", "m", "p", "kkt_dim", "kappa_P_target", "kappa_P_measured", "density", "structure", "nnz_kkt", "sparse", "n_active")} for inst in MM_INSTANCES.values()])
        summary = pd.concat([summary, mm_summary], ignore_index=True); summary.to_csv(os.path.join(RESULTS_DIR, "instances.csv"), index=False)
        mm_detail = pd.DataFrame([{k: inst[k] for k in ("id", "mm_name", "n", "m", "p", "kkt_dim", "nnz_kkt", "mm_nnz_P", "mm_nnz_A", "mm_rows_total", "mm_rows_free", "mm_eq_rows_removed", "mm_P_zero", "mm_gram_cond", "source")} for inst in MM_INSTANCES.values()])
        mm_detail.to_csv(os.path.join(RESULTS_DIR, "maros_meszaros_instances.csv"), index=False)
        print(f"Maros-Meszaros: {len(MM_INSTANCES)} instances loaded in {time.perf_counter() - t_mm:.1f}s"); print(mm_detail.to_string(index=False))
        REC.status("MM_load", "done", f"{len(MM_INSTANCES)} instances ({len(missing)} missing)")
    else:
        REC.status("MM_load", "skipped", f"no instance could be loaded: missing={missing}")
