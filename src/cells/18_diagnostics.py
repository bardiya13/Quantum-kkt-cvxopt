# Diagnostics: D1 dequantisation probe, E5 optimisation-level refinement, E6 CPU/GPU mat-vec benchmark.

# D1: fraction of singular directions a truncated solve needs to reproduce the captured Newton direction
rows = []
for inst_id, caps in CAPTURED.items():
    for cap in caps:
        K = cap["K"]; r = cap["solves"][0]["rhs"]; u = cap["solves"][0]["u"]; N = K.shape[0]
        U_, sv, Vt = np.linalg.svd(K); Utb = U_.T @ r; nu = max(np.linalg.norm(u), 1e-300); need = {}
        for dlt in (0.3, 0.1, 0.01):
            need[dlt] = N
            for k in range(1, N + 1):
                if np.linalg.norm(Vt[:k].T @ (Utb[:k] / sv[:k]) - u) / nu <= dlt: need[dlt] = k; break
        rows.append({"instance_id": inst_id, "factor_index": cap["factor_index"], "kkt_dim": N, "kappa": float(sv[0] / max(sv[-1], 1e-300)),
                     "stable_rank": float((sv ** 2).sum() / sv[0] ** 2), **{f"frac_for_{d}": need[d] / N for d in need}})
DEQ = pd.DataFrame(rows); DEQ.to_csv(os.path.join(RESULTS_DIR, "dequantisation_probe.csv"), index=False)
if len(DEQ):
    print(f"D1 dequantisation probe on {len(DEQ)} captured systems: median fraction of singular directions needed for 30% / 10% / 1% accuracy = "
          f"{DEQ['frac_for_0.3'].median():.2f} / {DEQ['frac_for_0.1'].median():.2f} / {DEQ['frac_for_0.01'].median():.2f}; median stable rank / N = {(DEQ.stable_rank / DEQ.kkt_dim).median():.2f}")
REC.status("D1_dequantisation", "done" if len(DEQ) else "skipped", f"{len(DEQ)} systems")


# E5: optimisation-level iterative refinement (loose inner solves, shift-and-scale outer loop)
def outer_refinement(inst, spec=None, rounds=5, inner_tol=1e-3, label="cvxopt_default", target=1e-7):
    """x <- x + d/Delta, y <- y + yhat/Delta, z <- w/Delta, where (d, yhat, w) solves the shifted/scaled QP
         min 1/2 d'Pd + Delta (P x + q + A'y)' d   s.t.  G d <= Delta (h - G x),  A d = Delta (b - A x)
    to the loose tolerance inner_tol. The reformulation is exact; only the inner accuracy is low. `spec` selects the inner KKT solver."""
    P, q, G, h, A, b, p = inst["P"], inst["q"], inst["G"], inst["h"], inst["A"], inst["b"], inst["p"]
    x = np.zeros(inst["n"]); y = np.zeros(p); Delta = 1.0; out = []; tot_it = 0; tot = {"solves": 0, "fallback": 0, "rejected_used": 0, "refine": 0, "krylov": 0}
    for k in range(rounds):
        sub = {kk: vv for kk, vv in inst.items() if kk != "_cvx"}
        sub.update({"q": Delta * (P @ x + q + (A.T @ y if p else 0.0)), "h": Delta * (h - G @ x), "b": (Delta * (b - A @ x)) if p else None, "known_solution": False})
        cd = cvx_data(sub); st = {"factor_calls": 0, "solve_calls": 0, "factors": [], "solves": [], "captured": [], "compute_cond": True, "capture": False, "cond_scaled": True}     # also kappa of the Ruiz-scaled inner systems
        kkt = make_kktsolver(sub, spec, st) if spec is not None else None
        solvers.options.clear(); solvers.options.update({"show_progress": False, "abstol": inner_tol, "reltol": inner_tol, "feastol": inner_tol, "maxiters": 60, "refinement": 0})
        try:
            sol = solvers.coneqp(cd["P"], cd["q"], cd["G"], cd["h"], inst["dims"], cd["A"], cd["b"], kktsolver=kkt)
        except Exception as e:
            out.append({"instance_id": inst["id"], "inner_solver": label, "round": k, "status": f"error: {type(e).__name__}"}); break
        finally:
            solvers.options.clear()
        x = x + to_np(sol["x"]) / Delta; y = y + (to_np(sol["y"]) / Delta if p else 0.0); z = to_np(sol["z"]) / Delta; tot_it += sol["iterations"]
        S_ = st["solves"]; tot["solves"] += len(S_); tot["fallback"] += sum(s_["fallback"] for s_ in S_); tot["refine"] += sum(s_["n_refine"] for s_ in S_); tot["krylov"] += sum(s_.get("krylov_iters", 0) for s_ in S_)
        tot["rejected_used"] += sum((not s_["accepted"]) and (not s_["fallback"]) for s_ in S_)
        m = recompute_metrics_qp(inst, {"x": x, "y": y, "z": z, "s": np.maximum(h - G @ x, 0.0)}); err = max(m["pres_rec"], m["dres_rec"], m["relgap_valid"])
        kap = [f["kappa"] for f in st["factors"] if np.isfinite(f["kappa"])]; kaps_ = [f["kappa_scaled"] for f in st["factors"] if np.isfinite(f.get("kappa_scaled", np.nan))]
        out.append({"instance_id": inst["id"], "inner_solver": label, "inner_tol": inner_tol, "round": k, "status": sol["status"], "Delta": Delta, "inner_iterations": sol["iterations"], "cum_iterations": tot_it,
                    "pres": m["pres_rec"], "dres": m["dres_rec"], "relgap_valid": m["relgap_valid"], "cone_viol": m["cone_viol"], "obj_rel_err": m["obj_rel_err"], "err": err,
                    "kappa_max_inner": max(kap) if kap else np.nan, "kappa_scaled_max_inner": max(kaps_) if kaps_ else np.nan, "cum_solves": tot["solves"], "cum_fallback": tot["fallback"], "cum_rejected_used": tot["rejected_used"], "cum_refine": tot["refine"], "cum_krylov": tot["krylov"],
                    "reached_target": bool(err <= target and m["obj_rel_err"] <= CONFIG["accept"]["obj_rel_err"])})
        if err <= 1e-13: break
        Delta = float(min(Delta * 1e2, 1.0 / max(err, 1e-14)))
    return out

loose = {"criterion": "combined", "eta": 0.1, "tol_rule": "fixed", "tol": 1e-5, "scaling": "ruiz", "fallback": False, "max_refine": 4}
INNER = [("cvxopt_default", None),
         ("exact_noise_iso_1e-4", {"name": "or_noise_1e-4", "kind": "noise", "noise_model": "iso", "delta": 1e-4, "fallback": False, "max_refine": 0, "criterion": "relres"}),
         ("exact_noise_iso_1e-3", {"name": "or_noise_1e-3", "kind": "noise", "noise_model": "iso", "delta": 1e-3, "fallback": False, "max_refine": 0, "criterion": "relres"}),
         ("minres_base_loose_1e-5", {"name": "or_minres_base_loose", "kind": "minres_base", "backend": "cpu", "warm_start": True, **loose})]
ref_rows = []
or_insts = [i for i in INSTANCES.values() if i["family"] == "dense_kappa" and i["seed"] == 0 and i["n"] <= 20] + [INSTANCES[t] for t in ("tiny_A", "tiny_B", "tiny_C")]
for inst in or_insts:
    for label, spec in INNER: ref_rows += outer_refinement(inst, spec, label=label)
if QCFG_OUTER := CONFIG["quantum"]["enabled"]:
    for tid in CONFIG["quantum"]["integrated"]["instances"]:
        for nc in sorted({8, max(CONFIG["quantum"]["n_clock_list"])}):
            if int(math.ceil(math.log2(INSTANCES[tid]["kkt_dim"]))) + nc + 2 > CONFIG["quantum"]["max_qubits"]: continue
            hspec = {"name": f"or_hhl_nc{nc}", "kind": "hhl", "recon": "statevector", "n_clock": nc, "reciprocal": "kappa_known", "reg": 0.0, **loose}
            ref_rows += outer_refinement(INSTANCES[tid], hspec, label=f"hhl_ideal_nc{nc}_nofallback")
OREF = pd.DataFrame(ref_rows); OREF.to_csv(os.path.join(RESULTS_DIR, "outer_refinement.csv"), index=False)
if len(OREF) and "err" in OREF:
    print("E5 outer refinement, inner tolerance 1e-3: median max(pres, dres, relgap) by round and inner solver")
    print(OREF.pivot_table(index="round", columns="inner_solver", values="err", aggfunc="median").to_string())
    last = OREF.sort_values("round").groupby(["instance_id", "inner_solver"]).tail(1)
    print(last.groupby("inner_solver").agg(instances=("instance_id", "size"), reached_1e7=("reached_target", "mean"), cum_iters=("cum_iterations", "median"), cum_krylov=("cum_krylov", "median"), kappa_max=("kappa_max_inner", "median"), kappa_scaled_max=("kappa_scaled_max_inner", "median"),
                                           rejected_used=("cum_rejected_used", "sum"), fallback=("cum_fallback", "sum")).round(3).to_string())
REC.status("E5_outer_refinement", "done", f"{OREF.instance_id.nunique() if len(OREF) else 0} instances x {OREF.inner_solver.nunique() if len(OREF) else 0} inner solvers")


# E6: CPU and GPU mat-vec time on the sparse KKT matrices (W = I)
mv_rows = []
if GPU_OK:
    for inst in sorted([i for i in INSTANCES.values() if i["family"] == "sparse_banded" and i["seed"] == 0], key=lambda i: i["kkt_dim"]):
        asm = KKTAssembler(inst); W1 = {"d": matrix(1.0, (inst["m"], 1)), "di": matrix(1.0, (inst["m"], 1)), "v": [], "beta": [], "r": [], "rti": []}
        K = asm.build(W1); v = np.random.default_rng(0).standard_normal(K.shape[0]); Kd = to_device(K); vd = DEVICE_XP.asarray(v); reps = 200
        t0 = time.perf_counter(); [K @ v for _ in range(reps)]; t_cpu = (time.perf_counter() - t0) / reps
        Kd @ vd; dev_sync(); t0 = time.perf_counter(); [Kd @ vd for _ in range(reps)]; dev_sync(); t_gpu = (time.perf_counter() - t0) / reps
        mv_rows.append({"instance_id": inst["id"], "kkt_dim": K.shape[0], "nnz": int(K.nnz), "t_matvec_cpu": t_cpu, "t_matvec_gpu": t_gpu, "speedup_gpu": t_cpu / t_gpu, "device": GPU_STATUS.get("name")})
    MV = pd.DataFrame(mv_rows); MV.to_csv(os.path.join(RESULTS_DIR, "gpu_matvec_benchmark.csv"), index=False); print(MV.round(7).to_string(index=False))
    REC.status("E6_gpu_matvec", "done", f"{len(MV)} sizes on {GPU_STATUS.get('name')}")
else:
    MV = pd.DataFrame(); REC.status("E6_gpu_matvec", "skipped", f"device={CONFIG['device']}; {GPU_STATUS.get('reason') or 'GPU not requested'}")
REC.flush(min_interval=0)
