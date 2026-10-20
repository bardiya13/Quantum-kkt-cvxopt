# Instrumented kktsolver, the benchmark runner, recomputed metrics and the acceptance of a run.
CAPTURED = {}

def make_kktsolver(inst, spec, run_state):
    asm = KKTAssembler(inst); run_state["asm"] = asm
    compute_cond = run_state.get("compute_cond", False) and asm.dim <= CONFIG["cond_limit"]
    capture = run_state.get("capture", False) and asm.dim <= CONFIG["capture"]["max_dim"]
    def kktsolver(W):
        fi = run_state["factor_calls"]; run_state["factor_calls"] += 1
        with Timer() as t_asm:
            K = asm.build(W)
        proxy = nt_spread(W)
        kappa, kappa_scaled, t_diag = float("nan"), float("nan"), 0.0
        if compute_cond:
            with Timer() as td:
                kappa = cond_exact(K)
                if spec.get("scaling", "ruiz") == "ruiz" or run_state.get("cond_scaled"):
                    kappa_scaled = cond_exact(apply_scaling(K, ruiz_equilibrate(K)))
            t_diag = td.t
        kappa_S, kappa_B = float("nan"), float("nan")
        if compute_cond and asm.l_only and asm.p == 0 and inst.get("P") is not None:          # kappa of the reduced system S and of its square root B
            with Timer() as td2:
                try:
                    S_, _, _ = reduced_spd(asm, W); Sd = S_.toarray() if sp.issparse(S_) else S_
                    kappa_S = float(np.linalg.cond(Sd)); kappa_B = float(np.sqrt(kappa_S))      # kappa(B) = sqrt(kappa(S)) for S = B'B
                except Exception:
                    pass
            t_diag += td2.t
        switched = bool(spec.get("switch_policy")) and run_state.get("switched", False)
        eff_spec = dict(spec, kind="exact", name=spec["name"]) if switched else spec
        with Timer() as t_fac:
            engine = KKTSolveEngine(eff_spec, K, asm, W)
        kappa_reduced = float("nan")
        if compute_cond and hasattr(engine.impl, "compute_kappa"):        # kappa of the method's own target system (null-space methods)
            with Timer() as td3:
                try: kappa_reduced = float(engine.impl.compute_kappa())
                except Exception: pass
            t_diag += td3.t
        frec = {"factor_index": fi, "t_assemble": t_asm.t, "t_factor": t_fac.t, "t_diag": t_diag, "kappa": kappa, "kappa_scaled": kappa_scaled, "kappa_S": kappa_S, "kappa_B": kappa_B, "kappa_reduced": kappa_reduced,
                "nt_proxy": proxy, "kkt_dim": asm.dim, "nK_est": engine.nK, "did_factor": bool(getattr(engine.impl, "did_factor", True))}
        run_state["factors"].append(frec)
        if capture:
            Kd = K.toarray() if sp.issparse(K) else K.copy()
            run_state["captured"].append({"factor_index": fi, "K": Kd, "kappa": kappa, "nt_proxy": proxy, "di": to_np(W["di"]).copy() if W["di"].size[0] else np.zeros(0), "solves": []})
        def solve(x, y, z):
            si = run_state["solve_calls"]; run_state["solve_calls"] += 1
            r = asm.rhs(x, y, z, W)
            u, log = engine.solve(r, {"factor_index": fi, "solve_index": si, "nt_proxy": proxy})
            log.update({"factor_index": fi, "solve_index": si, "kappa": kappa, "nt_proxy": proxy, "after_switch": switched})
            if switched: log["method_used"] = "exact_after_switch"
            if spec.get("switch_policy") and not switched and (log["fallback"] or not log["accepted"]):
                run_state["switched"] = True                     # first failed acceptance -> exact solver for the rest of the run
            run_state["solves"].append(log)
            if log["solve_failed"]:
                raise ArithmeticError("non-finite KKT direction")
            asm.write_back(u, x, y, z)
            if capture:
                run_state["captured"][-1]["solves"].append({"rhs": r.copy(), "u": u.copy(), "solve_index": si})
        return solve
    return kktsolver

_PROG_RE = re.compile(r"^\s*(\d+):\s+(.*)$")
_FLOAT_RE = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")
def parse_progress(text):
    rows = []
    for line in text.splitlines():
        m = _PROG_RE.match(line)
        if not m: continue
        vals = [float(v) for v in _FLOAT_RE.findall(m.group(2))]
        if len(vals) < 5: continue
        rows.append({"iter": int(m.group(1)), "pcost": vals[0], "dcost": vals[1], "gap": vals[2], "pres": vals[3], "dres": vals[4], "k_over_t": vals[5] if len(vals) > 5 else float("nan")})
    return rows

def recompute_metrics_qp(inst, sol):
    P, q, G, h, A, b = inst["P"], inst["q"], inst["G"], inst["h"], inst["A"], inst["b"]
    x, s, z = to_np(sol["x"]), to_np(sol["s"]), to_np(sol["z"])
    y = to_np(sol["y"]) if inst["p"] > 0 else np.zeros(0)
    Px = P @ x
    pobj = 0.5 * x @ Px + q @ x
    r_dual = Px + q + G.T @ z + (A.T @ y if inst["p"] > 0 else 0.0)
    r_eq = (A @ x - b) if inst["p"] > 0 else np.zeros(0)
    r_in = G @ x + s - h
    resy0 = max(1.0, np.linalg.norm(b)) if inst["p"] > 0 else 1.0
    resz0 = max(1.0, np.linalg.norm(h))
    pres = max(np.linalg.norm(r_eq) / resy0, np.linalg.norm(r_in) / resz0)
    dres = np.linalg.norm(r_dual) / max(1.0, np.linalg.norm(q))
    cone_viol = max(0.0, -s.min() if s.size else 0.0, -z.min() if z.size else 0.0)
    constr_viol = max(0.0, float((G @ x - h).max())) / resz0 if inst["m"] > 0 else 0.0
    gap = float(s @ z)
    zp = np.maximum(z, 0.0); v = q + G.T @ zp + (A.T @ y if inst["p"] > 0 else 0.0)
    try:
        if inst.get("no_dual_recompute"): raise ValueError("dual value not recomputed for this instance (P may be singular)")
        Pd = P.toarray() if sp.issparse(P) else P
        if Pd.shape[0] <= 3000:
            c, low = sla.cho_factor(Pd); dual_val = -0.5 * v @ sla.cho_solve((c, low), v) - h @ zp - (b @ y if inst["p"] > 0 else 0.0)
        else:
            f = spla.splu(sp.csc_matrix(P)); dual_val = -0.5 * v @ f.solve(v) - h @ zp - (b @ y if inst["p"] > 0 else 0.0)
        valid_gap = pobj - dual_val
    except Exception:
        dual_val, valid_gap = float("nan"), float("nan")
    out = {"pobj": float(pobj), "dobj_valid": float(dual_val), "gap_rec": gap, "relgap_valid": float(abs(valid_gap) / max(1.0, abs(pobj))),
           "pres_rec": float(pres), "dres_rec": float(dres), "cone_viol": float(cone_viol), "constraint_viol": float(constr_viol), "mu_final": gap / max(1, inst["m"])}
    if inst.get("known_solution"):
        out["obj_rel_err"] = float(abs(pobj - inst["f_star"]) / max(1.0, abs(inst["f_star"]))); out["x_rel_err"] = float(np.linalg.norm(x - inst["x_star"]) / max(1e-12, np.linalg.norm(inst["x_star"])))
    else:
        out["obj_rel_err"] = float("nan"); out["x_rel_err"] = float("nan")
    return out
METRIC_FUNCS = {"coneqp": recompute_metrics_qp}

def acceptance(rec, overrides=None):
    a = dict(CONFIG["accept"], **(overrides or {})); fails = []           # per-run overrides: tolerance sweep E4b
    if not (rec.get("pres_rec", np.inf) <= a["pres"]): fails.append("pres")
    if not (rec.get("dres_rec", np.inf) <= a["dres"]): fails.append("dres")
    rg = rec.get("relgap_valid", np.nan)
    if np.isnan(rg): rg = rec.get("relgap_cvx", np.nan)
    if np.isnan(rg) and np.isfinite(rec.get("gap_rec", np.nan)): rg = abs(rec["gap_rec"]) / max(1.0, abs(rec.get("pobj", 0.0)))     # CVXOPT reports no relative gap when pcost ~ 0
    if not (rg <= a["relgap"]): fails.append("relgap")
    if not (rec.get("cone_viol", np.inf) <= a["cone_viol"]): fails.append("cone")
    oe = rec.get("obj_rel_err", np.nan)
    if not np.isnan(oe) and not (oe <= a["obj_rel_err"]): fails.append("obj")
    if rec.get("error"): fails.append("error")
    return (len(fails) == 0), ",".join(fails)

def solver_options_for(inst, spec):
    c = dict(CONFIG["cvxopt"], **spec.get("cvxopt_overrides", {}))          # per-run tolerance overrides (E4b)
    return {"show_progress": True, "maxiters": spec.get("maxiters", c["maxiters"]), "abstol": c["abstol"], "reltol": c["reltol"], "feastol": c["feastol"],
            "refinement": c["refinement_qp"] if not inst["dims"]["s"] and not inst["dims"]["q"] else c["refinement_sdp"]}

def run_instance(inst, spec, repeat=0, compute_cond=False, capture=False, cond_scaled=False, experiment=""):
    run_id = f"{inst['id']}__{spec['name']}__r{repeat}"
    run_state = {"factor_calls": 0, "solve_calls": 0, "factors": [], "solves": [], "captured": [], "compute_cond": compute_cond, "capture": capture, "cond_scaled": cond_scaled}
    kkt = None if spec["kind"] == "baseline" else make_kktsolver(inst, spec, run_state)
    opts = solver_options_for(inst, spec)
    solvers.options.clear(); solvers.options.update(opts)
    cd = cvx_data(inst); buf = io.StringIO(); err = None; sol = None; singular = False
    t0 = time.perf_counter(); pm = PeakMemory()
    try:
        with contextlib.redirect_stdout(buf), pm:
            if inst["kind"] == "coneqp":
                sol = solvers.coneqp(cd["P"], cd["q"], cd["G"], cd["h"], inst["dims"], cd["A"], cd["b"], kktsolver=kkt)
            else:
                sol = solvers.conelp(cd["q"], cd["G"], cd["h"], inst["dims"], cd["A"], cd["b"], kktsolver=kkt)
    except ArithmeticError as e:
        err = (f"cvxopt ArithmeticError (its own factorisation failed): {e}" if kkt is None else f"terminated: non-finite KKT direction ({e})")[:300]; singular = True
    except Exception as e:
        err = f"{type(e).__name__}: {e}"[:300]
    wall = time.perf_counter() - t0
    solvers.options.clear()
    prog = parse_progress(buf.getvalue())
    status = sol["status"] if sol else (("cvxopt_arithmetic_error" if kkt is None else "unknown_singular_kkt") if singular else "error")
    rec = {"run_id": run_id, "code_version": CODE_VERSION, "config_hash": CONFIG_HASH, "experiment": experiment, "family": inst["family"], "instance_id": inst["id"], "kind": inst["kind"],
           "structure": inst.get("structure", ""), "n": inst["n"], "m": inst["m"], "p": inst["p"], "kkt_dim": inst["kkt_dim"], "nnz_kkt": inst.get("nnz_kkt", np.nan),
           "kappa_P_target": inst.get("kappa_P_target"), "kappa_P_measured": inst.get("kappa_P_measured"), "density": inst.get("density"), "seed": inst.get("seed"),
           "method": spec["name"], "method_kind": spec["kind"], "criterion": spec.get("criterion", "" if spec["kind"] in ("baseline", "exact") else "backward"),
           "tol_rule": spec.get("tol_rule", ""), "target_tol": spec.get("tol", spec.get("eta", np.nan)), "scaling": spec.get("scaling", "none" if spec["kind"] in ("baseline", "exact", "noise") else "ruiz"),
           "regularization": spec.get("reg", 0.0), "fallback_enabled": bool(spec.get("fallback", CONFIG["inexact"]["fallback"])), "repeat": repeat,
           "refinement_option": opts["refinement"], "maxiters_option": opts["maxiters"], "status": status, "ipm_tol_level": float(spec.get("tol_level", CONFIG["cvxopt"]["feastol"])),
           "feastol_option": opts["feastol"], "sweep_samples": spec.get("sweep_samples", np.nan), "sweep_seed": spec.get("sweep_seed", np.nan), "solution_source": inst.get("solution_source", "generator"),
           "iterations": int(sol["iterations"]) if sol else (prog[-1]["iter"] if prog else -1), "error": err or "", "wall_total": wall, **pm.fields()}
    if sol is not None:
        rec.update(METRIC_FUNCS[inst["kind"]](inst, sol))
        rec.update({"gap_cvx": safe_float(sol.get("gap")), "relgap_cvx": safe_float(sol.get("relative gap")), "pres_cvx": safe_float(sol.get("primal infeasibility")), "dres_cvx": safe_float(sol.get("dual infeasibility"))})
    F, S = run_state["factors"], run_state["solves"]
    for k in ["t_assemble", "t_factor", "t_diag"]: rec[k] = float(sum(f[k] for f in F))
    for k in ["t_solve", "t_verify", "t_refine", "t_fallback", "t_check"]: rec[k] = float(sum(s.get(k, 0.0) for s in S))
    rec["wall_method"] = wall - rec["t_diag"]
    rec["n_factor_calls"] = len(F); rec["n_solve_calls"] = len(S)
    rec["n_refine_total"] = int(sum(s["n_refine"] for s in S)); rec["n_fallback_total"] = int(sum(s["fallback"] for s in S))
    rec["n_solves_accepted_without_fallback"] = int(sum((s["accepted"] and not s["fallback"]) for s in S)); rec["n_solves_failed"] = int(sum(s.get("solve_failed", False) for s in S))
    rec["n_solves_accepted_direct"] = int(sum((s["accepted"] and not s["fallback"] and s["n_refine"] == 0) for s in S))
    rec["n_solves_accepted_refined"] = int(sum((s["accepted"] and not s["fallback"] and s["n_refine"] > 0) for s in S))
    rec["n_solves_unaccepted_used"] = int(sum(((not s["accepted"]) and not s["fallback"] and not s.get("solve_failed", False)) for s in S))   # only possible when fallback is disabled
    rec["max_rel_resid_final"] = float(max([s["rel_resid_final"] for s in S], default=np.nan)); rec["max_backward_err_final"] = float(max([s.get("backward_err_final", np.nan) for s in S], default=np.nan))
    rec["max_block_ratio_final"] = float(max([s.get("block_ratio_max_final", np.nan) for s in S], default=np.nan))
    rec["krylov_iters_total"] = int(sum(s.get("krylov_iters", 0) for s in S))
    n_after = int(sum(bool(s.get("after_switch", False)) for s in S)); n_own = len(S) - n_after - rec["n_fallback_total"]
    rec["n_solves_after_switch"] = n_after; rec["n_solves_by_method"] = int(max(n_own, 0))
    rec["utilisation"] = float(max(n_own, 0) / len(S)) if S else np.nan        # fraction of KKT solves delivered by the method under test (Amdahl bound on any kernel speed-up)
    rec["n_factorizations"] = int(sum(f.get("did_factor", True) for f in F) + sum(bool(s.get("refactored_now", False)) for s in S)) if spec["kind"] != "baseline" else int(rec["iterations"]) + 1
    rec["warm_started_solves"] = int(sum(bool(s.get("warm_started", False)) for s in S)); rec["backend"] = spec.get("backend", "cpu")
    rec["backend_used"] = ",".join(sorted({str(s_.get("backend")) for s_ in S if s_.get("backend") is not None})) or "cpu"; rec["n_checks_total"] = int(sum(s_.get("n_checks", 0) or 0 for s_ in S))
    rec["fuse_mode"] = next((s_.get("fuse_mode") for s_ in S if s_.get("fuse_mode")), ""); rec["t_transfer"] = float(sum({s_["factor_index"]: (s_.get("t_transfer") or 0.0) for s_ in S}.values()))
    kap = [f["kappa"] for f in F if np.isfinite(f["kappa"])]
    rec["kappa_first"] = kap[0] if kap else np.nan; rec["kappa_last"] = kap[-1] if kap else np.nan; rec["kappa_max"] = max(kap) if kap else np.nan
    kaps = [f["kappa_scaled"] for f in F if np.isfinite(f["kappa_scaled"])]; rec["kappa_scaled_max"] = max(kaps) if kaps else np.nan
    kapr = [f.get("kappa_reduced", np.nan) for f in F if np.isfinite(f.get("kappa_reduced", np.nan))]; rec["kappa_reduced_max"] = max(kapr) if kapr else np.nan
    rec["n_fallback_reduced"] = int(sum(1 for s in S if s.get("fallback_backend"))); rec["reduced_dim"] = next((s.get("reduced_dim") for s in S if s.get("reduced_dim") is not None), np.nan)
    rec["accepted"], rec["accept_fail_reasons"] = acceptance(rec, spec.get("accept_overrides"))
    rec["pure_method_success"] = bool(rec["accepted"] and rec["n_fallback_total"] == 0)                      # no classical fallback
    rec["strict_method_success"] = bool(rec["accepted"] and rec["n_fallback_total"] == 0 and rec["n_solves_unaccepted_used"] == 0)   # ... and every direction passed the acceptance test
    REC.add("benchmark_results", rec)
    fmap = {f["factor_index"]: f for f in F}; smap = {}
    for s in S: smap.setdefault(s["factor_index"], []).append(s)
    for row in prog:
        f = fmap.get(row["iter"] + 1, {}); ss = smap.get(row["iter"] + 1, [])
        REC.add("iteration_logs", {"run_id": run_id, "instance_id": inst["id"], "method": spec["name"], "repeat": repeat, **row, "kappa": f.get("kappa", np.nan), "kappa_scaled": f.get("kappa_scaled", np.nan), "kappa_S": f.get("kappa_S", np.nan), "kappa_B": f.get("kappa_B", np.nan), "kappa_reduced": f.get("kappa_reduced", np.nan),
                                   "nt_proxy": f.get("nt_proxy", np.nan), "n_solves": len(ss), "n_fallback": int(sum(s["fallback"] for s in ss)), "n_refine": int(sum(s["n_refine"] for s in ss)),
                                   "max_rel_resid_final": max([s["rel_resid_final"] for s in ss], default=np.nan), "max_backward_err_final": max([s.get("backward_err_final", np.nan) for s in ss], default=np.nan),
                                   "max_block_ratio_final": max([s.get("block_ratio_max_final", np.nan) for s in ss], default=np.nan)})
    for s in S:
        REC.add("linear_solve_logs", {"run_id": run_id, "instance_id": inst["id"], "method": spec["name"], "method_kind": spec["kind"], "repeat": repeat, **s})
    if err or not rec["accepted"]:
        REC.failure(run_id, "optimization", err or f"acceptance failed: {rec['accept_fail_reasons']}", method=spec["name"], instance_id=inst["id"], status=rec["status"], n_fallback=rec["n_fallback_total"])
    if capture and run_state["captured"]:
        CAPTURED[inst["id"]] = run_state["captured"]
    REC.flush()
    return rec, sol, run_state

def method_specs():
    cfg = CONFIG["inexact"]; specs = [{"name": "cvxopt_default", "kind": "baseline"}, {"name": "custom_exact", "kind": "exact"}]
    # relative-residual test, with and without Ruiz scaling
    specs.append({"name": "minres_relres_1e-06_noscale", "kind": "minres", "criterion": "relres", "tol_rule": "fixed", "tol": cfg["relres_tol"], "scaling": "none"})
    specs.append({"name": "minres_relres_1e-06", "kind": "minres", "criterion": "relres", "tol_rule": "fixed", "tol": cfg["relres_tol"], "scaling": "ruiz"})
    # backward-error test: tolerance sweep and an unscaled ablation
    for t in cfg["be_tols"]:
        specs.append({"name": f"minres_be_{t:.0e}", "kind": "minres", "criterion": "backward", "tol_rule": "fixed", "tol": t, "scaling": "ruiz"})
    specs.append({"name": "minres_be_1e-10_noscale", "kind": "minres", "criterion": "backward", "tol_rule": "fixed", "tol": 1e-10, "scaling": "none"})
    # relative residual or a backward error at direct-solver level
    specs.append({"name": "minres_rrfloor_1e-06", "kind": "minres", "criterion": "relres_floor", "tol_rule": "fixed", "tol": cfg["relres_tol"], "be_floor": cfg.get("be_floor", 1e-14), "scaling": "ruiz"})
    specs.append({"name": "minres_rrfloor_1e-06_noscale", "kind": "minres", "criterion": "relres_floor", "tol_rule": "fixed", "tol": cfg["relres_tol"], "be_floor": cfg.get("be_floor", 1e-14), "scaling": "none"})
    # combined test: block forcing and backward error
    specs.append({"name": "minres_comb_0.1_1e-10", "kind": "minres", "criterion": "combined", "eta": 0.1, "tol_rule": "fixed", "tol": 1e-10, "scaling": "ruiz"})
    # block forcing: eta sweep and an unscaled ablation
    for eta in cfg["block_etas"]:
        specs.append({"name": f"minres_block_{eta:g}", "kind": "minres", "criterion": "block", "eta": eta, "scaling": "ruiz"})
    specs.append({"name": "minres_block_0.1_noscale", "kind": "minres", "criterion": "block", "eta": 0.1, "scaling": "none"})
    # backward-error tolerance tied to the NT-scaling spread
    specs.append({"name": "minres_adaptive_nt_be", "kind": "minres", "criterion": "backward", "tol_rule": "adaptive_nt", "scaling": "ruiz", **cfg["adaptive_nt"]})
    specs.append({"name": "cg_reduced_be_1e-10", "kind": "cg_reduced", "criterion": "backward", "tol_rule": "fixed", "tol": 1e-10, "scaling": "ruiz", "requires_p0": True})
    # methods with the combined test
    comb = {"criterion": "combined", "eta": 0.1, "tol_rule": "fixed", "tol": 1e-10, "scaling": "ruiz"}
    specs.append({"name": "custom_exact_reduced", "kind": "exact_reduced"})
    specs.append({"name": "gmres_stalelu_comb", "kind": "gmres_stalelu", **comb})
    specs.append({"name": "minres_base_comb_cpu", "kind": "minres_base", "backend": "cpu", "warm_start": False, **comb})
    specs.append({"name": "minres_base_comb_ws_cpu", "kind": "minres_base", "backend": "cpu", "warm_start": True, **comb})
    specs.append({"name": "minres_base_comb_ws_switch_cpu", "kind": "minres_base", "backend": "cpu", "warm_start": True, "switch_policy": True, **comb})
    specs.append({"name": "minres_fused_comb_cpu", "kind": "minres_fused", "backend": "cpu", **comb})
    # the y-block test binds late in the run (r_y ~ 1e-15, absolute target 0.1 * floor): ablation with floor 1e-8 instead of 1e-9
    specs.append({"name": "minres_fused_comb_floor8_cpu", "kind": "minres_fused", "backend": "cpu", "block_abs_floor": 1e-8, **comb})
    specs.append({"name": "minres_fused_comb_switch_cpu", "kind": "minres_fused", "backend": "cpu", "switch_policy": True, **comb})    # exact solver after the first rejected direction
    specs.append({"name": "oss_cg_comb", "kind": "oss_cg", "max_n": cfg.get("oss_max_n", 3200), **comb})                        # CG on the null-space system
    specs.append({"name": "minres_fused_comb_gpu", "kind": "minres_fused", "backend": "gpu", "requires_gpu": True, **comb})
    specs.append({"name": "minres_fused_comb_gpu_mm", "kind": "minres_fused", "backend": "gpu", "requires_gpu": True, "min_kkt_dim": 2000, **comb})   # Maros-Meszaros runs: GPU only from KKT dimension 2000
    specs.append({"name": "minres_fused_comb_auto", "kind": "minres_fused", "backend": "auto", "requires_gpu": True, **comb})       # GPU from gpu_auto_min_dim on, CPU below
    specs.append({"name": "minres_base_comb_gpu", "kind": "minres_base", "backend": "gpu", "warm_start": False, "requires_gpu": True, **comb})
    specs.append({"name": "minres_base_comb_ws_gpu", "kind": "minres_base", "backend": "gpu", "warm_start": True, "requires_gpu": True, **comb})
    specs.append({"name": "lsqr_sqrt_comb", "kind": "lsqr_sqrt", "requires_p0": True, "requires_dense_P": True, **comb})
    for d in cfg["noise_iso"]:
        specs.append({"name": f"exact_noise_iso_{d:.0e}", "kind": "noise", "noise_model": "iso", "delta": d, "fallback": False, "max_refine": 0, "criterion": "relres"})
    for d in cfg["noise_block"]:
        specs.append({"name": f"exact_noise_block_{d:.0e}", "kind": "noise", "noise_model": "block", "delta": d, "fallback": False, "max_refine": 0, "criterion": "block"})
    return {s["name"]: s for s in specs}

SPECS = method_specs()
print(f"{len(SPECS)} methods:", ", ".join(SPECS))
