# E7: spectrum of the scaled KKT matrix along one IPM run, augmented against reduced target, exact-deflation bound.
def e7_spectrum(n_target):
    cands = sorted([i for i in INSTANCES.values() if i["family"] == "sparse_banded" and i["n"] <= n_target], key=lambda i: i["n"])
    if not cands: return pd.DataFrame(), pd.DataFrame()
    inst = cands[-1]; n, p, m = inst["n"], inst["p"], inst["m"]; P, G = inst["P"].tocsr(), inst["G"].tocsr(); A = inst["A"].tocsr() if p else None
    rs = {"factor_calls": 0, "solve_calls": 0, "factors": [], "solves": [], "captured": [], "compute_cond": False, "capture": False, "cond_scaled": False}
    base = make_kktsolver(inst, SPECS["custom_exact_reduced"], rs); asm0 = KKTAssembler(inst); cap = []
    def kkt(W):
        f = base(W); rec = {"di": to_np(W["di"]).copy(), "rhs": []}; cap.append(rec)
        def solve(x, y, z):
            rec["rhs"].append(asm0.rhs(x, y, z, W).copy()); f(x, y, z)
        return solve
    cd = cvx_data(inst); solvers.options.clear(); solvers.options.update({"show_progress": False, "maxiters": CONFIG["cvxopt"]["maxiters"]})
    sol = solvers.coneqp(cd["P"], cd["q"], cd["G"], cd["h"], inst["dims"], cd["A"], cd["b"], kktsolver=kkt); solvers.options.clear()
    sl = [q for q in (slice(0, n), slice(n, n + p), slice(n + p, n + p + m)) if q.stop > q.start]
    rows, last = [], None
    for k, rec in enumerate(cap):
        di = rec["di"]; Gh = (sp.diags(di) @ G).tocsr()
        K = sp.bmat([[P, A.T, Gh.T], [A, None, None], [Gh, None, -sp.eye(m)]], format="csr") if p else sp.bmat([[P, Gh.T], [Gh, -sp.eye(m)]], format="csr")
        nK = norm2_estimate(K); d = ruiz_equilibrate(K); Ks = apply_scaling(K, d).tocsr()
        def crit(u, r):
            e = K @ u - r; be = np.linalg.norm(e) / (nK * np.linalg.norm(u) + np.linalg.norm(r))
            return be <= 1e-9 and max(np.linalg.norm(e[q]) / max(np.linalg.norm(r[q]), 1e-9) for q in sl) <= 0.1
        ev = np.linalg.eigvalsh(Ks.toarray()) if K.shape[0] <= CONFIG["cond_limit"] * 3 else None
        a = np.abs(ev) if ev is not None else None
        # reduced, matrix-free, Jacobi-scaled
        dS = P.diagonal() + np.asarray(Gh.multiply(Gh).sum(axis=0)).ravel(); e1 = 1.0 / np.sqrt(dS)
        if p: AD = A @ sp.diags(e1); e2 = 1.0 / np.sqrt(np.maximum(np.asarray(AD.multiply(AD).sum(axis=1)).ravel(), 1e-300)); E = np.concatenate([e1, e2])
        else: E = e1
        GhT = Gh.T.tocsr(); AT = A.T.tocsr() if p else None
        def Mv(v):
            vx = v[:n]; t = P @ vx + GhT @ (Gh @ vx)
            return np.concatenate([t + AT @ v[n:], A @ vx]) if p else t
        ia = ir = ib = 0; ybind = []
        def be_only(u, r):
            e = K @ u - r; return np.linalg.norm(e) / (nK * np.linalg.norm(u) + np.linalg.norm(r)) <= 1e-9
        for r in rec["rhs"]:
            xb, _, its, _, _ = minres_fused_kernel(lambda v: Ks @ v, d * r, np, maxiter=50 * K.shape[0], stop_fn=lambda xk: be_only(d * xk, r), check_every=4, adaptive_div=10 ** 9); ib += its
            eb = K @ (d * xb) - r; ybind.append(max(range(len(sl)), key=lambda j: np.linalg.norm(eb[sl[j]]) / max(np.linalg.norm(r[sl[j]]), 1e-9)))
            _, _, its, _, _ = minres_fused_kernel(lambda v: Ks @ v, d * r, np, maxiter=50 * K.shape[0], stop_fn=lambda xk: crit(d * xk, r), check_every=4, adaptive_div=10 ** 9); ia += its
            bz = r[n + p:]; b = np.concatenate([r[:n] + GhT @ bz, r[n:n + p]])
            def full(xk):
                w_ = E * xk; ux = w_[:n]; return np.concatenate([ux, w_[n:], Gh @ ux - bz])
            _, _, its, _, _ = minres_fused_kernel(lambda v: E * Mv(E * v), E * b, np, maxiter=50 * K.shape[0], stop_fn=lambda xk: crit(full(xk), r), check_every=4, adaptive_div=10 ** 9); ir += its
        rows.append({"instance_id": inst["id"], "kkt_dim": K.shape[0], "ipm_iteration": k, "w_spread": float(di.max() / di.min()), "kappa_scaled": float(a.max() / a.min()) if a is not None else np.nan,
                     "n_eig_below_0.1": int((a < 0.1).sum()) if a is not None else -1, "n_eig_below_0.01": int((a < 0.01).sum()) if a is not None else -1,
                     "frac_eig_in_0.9_1.1": float(((a > 0.9) & (a < 1.1)).mean()) if a is not None else np.nan, "minres_its_backward_only": ib, "binding_block_when_backward_holds": "/".join("xyz"[j] if p else "xz"[j] for j in ybind), "minres_its_augmented": ia, "minres_its_reduced_jacobi": ir})
        last = (K, nK, d, Ks, rec, crit)
    drows = []
    if last is not None and last[0].shape[0] <= CONFIG["cond_limit"] * 3:
        K, nK, d, Ks, rec, crit = last; ev, V = np.linalg.eigh(Ks.toarray()); o = np.argsort(np.abs(ev)); N = len(ev); r = rec["rhs"][-1]; b = d * r
        for kdef in [0, 5, 10, 25, 50, 100, 200]:
            if kdef >= N // 4: break
            Z = V[:, o[:kdef]]; lam = ev[o[:kdef]]; x0 = Z @ ((Z.T @ b) / lam) if kdef else np.zeros(N)
            bb = b - Ks @ x0; bb = bb - Z @ (Z.T @ bb) if kdef else bb
            op = (lambda v: (lambda w_: w_ - Z @ (Z.T @ w_))(Ks @ v)) if kdef else (lambda v: Ks @ v)
            _, _, its, _, _ = minres_fused_kernel(op, bb, np, maxiter=50 * N, stop_fn=lambda xk: crit(d * (x0 + xk), r), check_every=4, adaptive_div=10 ** 9)
            drows.append({"instance_id": inst["id"], "kkt_dim": N, "n_deflated": kdef, "frac_deflated": kdef / N, "minres_its": its, "kappa_remaining": float(np.abs(ev).max() / np.abs(ev[o[kdef]]))})
    return pd.DataFrame(rows), pd.DataFrame(drows)

try:
    with Timer() as _t7:
        SPEC7, DEFL7 = e7_spectrum({"smoke": 200, "small": 400}.get(RUN_PROFILE, 800))
    SPEC7.to_csv(os.path.join(RESULTS_DIR, "spectrum_along_ipm.csv"), index=False); DEFL7.to_csv(os.path.join(RESULTS_DIR, "deflation_bound.csv"), index=False)
    print(SPEC7.round(3).to_string(index=False)); print(); print(DEFL7.round(4).to_string(index=False))
    REC.status("E7_spectrum_deflation", "done" if len(SPEC7) else "skipped", f"{len(SPEC7)} IPM iterations, {len(DEFL7)} deflation levels in {_t7.t:.0f}s")
except Exception as _e:
    SPEC7, DEFL7 = pd.DataFrame(), pd.DataFrame(); REC.status("E7_spectrum_deflation", "failed", f"{type(_e).__name__}: {_e}"[:300]); traceback.print_exc()
