# Q5: polynomial solver. Q5a/Q5d captured systems (augmented and null-space target), Q5b refinement cost grid, Q5c runs
# inside coneqp without fallback (with HHL at the same refinement budget), Q5e finite-shot runs over sampling seeds.
CKSCFG = QCFG.get("cks", {}); Q5A, Q5B, Q5D = [], [], []
def poly_tag(p_):
    p_ = str(p_); return "not-run(cap)" if "not generated" in p_ else ("cks-fallback" if "fallback" in p_ else p_)
if not QCFG.get("enabled", True) or not CKSCFG:
    REC.status("Q5_cks", "skipped", "quantum disabled")
else:
    t5 = time.perf_counter(); n_ir = 0
    for inst_id, caps in CAPTURED.items():
        if inst_id.startswith("sdplib") or not caps: continue
        for fi in sorted({0, len(caps) // 2, len(caps) - 1}):
            K = np.asarray(caps[fi]["K"], dtype=float)
            if K.shape[0] > 64 or not caps[fi]["solves"]: continue
            r = caps[fi]["solves"][-1]["rhs"]; d = ruiz_equilibrate(K); Ks = apply_scaling(K, d); lam, _ = spectral_bound(Ks, "exact", rng_for("q5", inst_id, fi))
            Ap, _, N0 = pad_to_pow2(Ks, np.zeros(K.shape[0]), lam); rp = np.zeros(Ap.shape[0]); rp[:N0] = d * r; u_true = np.linalg.solve(K, r); nu = np.linalg.norm(u_true); bec = block_encoding_costs(Ap)
            base = {"instance_id": inst_id, "factor_index": fi, "kkt_dim": K.shape[0], "N_padded": Ap.shape[0], "kappa_original": float(np.linalg.cond(K)),
                    "be_inflation_sparse_access": bec["alpha_sparse"] / bec["alpha_ideal"], "be_inflation_pauli_lcu": bec["alpha_pauli"] / bec["alpha_ideal"], "pauli_density": bec["pauli_terms"] / bec["pauli_terms_possible"]}
            nc = max(n_ for n_ in QCFG["n_clock_list"] if int(math.log2(Ap.shape[0])) + n_ + 2 <= QCFG["max_qubits"])
            hs = HHLSystem(Ap, nc, rng_for("q5h", inst_id, fi), spectral="exact", c_mode="kappa_known"); ho = hs.run(rp)
            Q5A.append({**base, "solver": f"hhl_nc{nc}", "poly": "", "eps_target": np.nan, "kappa_target": hs.kappa, "n_qubits": ho["n_qubits"], "oracle_queries": 2 * (2 ** nc - 1), "l1_norm": np.nan,
                        "p_success": ho["p_postselect_exact"], "rel_err_ideal": float(np.linalg.norm(d * ho["x_ideal"][:N0] - u_true) / nu), "t_sim": ho["t_circuit_sim"], "status": "ok"})
            for eps in CKSCFG["eps_list"]:
                for poly in ["cks", "optimal"]:
                    cs = CKSSystem(Ap, eps, rng_for("q5c", inst_id, fi, eps), max_degree=CKSCFG["max_degree"], poly=poly)
                    row = {**base, "solver": "lcu_walk", "poly": cs.poly, "eps_target": eps, "kappa_target": cs.kappa, "n_qubits": cs.n_qubits, "oracle_queries": cs.degree, "l1_norm": cs.l1}
                    if cs.over_budget: Q5A.append({**row, "p_success": np.nan, "rel_err_ideal": np.nan, "t_sim": np.nan, "status": "degree_cap (simulation budget)"}); continue
                    co = cs.run(rp); Q5A.append({**row, "p_success": co["p_postselect_exact"], "rel_err_ideal": float(np.linalg.norm(d * co["x_ideal"][:N0] - u_true) / nu), "t_sim": co["t_circuit_sim"], "status": "ok"})
            # Q5d: the same captured system on the null-space SPD target: kappa, dimension, qubits, degree, forward error (ideal readout)
            try:
                inst_ = INSTANCES[inst_id]; W_ = {"di": matrix(np.asarray(caps[fi]["di"], dtype=float).reshape(-1, 1)), "d": matrix(1.0 / np.asarray(caps[fi]["di"], dtype=float).reshape(-1, 1)), "v": [], "beta": [], "r": [], "rti": []}
                asm_ = KKTAssembler(inst_); eng_ = KKTSolveEngine({"name": "q5d", "kind": "exact", "criterion": "combined", "eta": 0.1, "tol": 1e-9, "tol_rule": "fixed"}, K, asm_, W_)
                oss_ = OSSKind({"name": "q5d_oss", "max_n": 10 ** 9}, K, asm_, W_, eng_); M_ = oss_.matrix(); kap_red = float(np.linalg.cond(M_)); lam_mx = float(np.linalg.eigvalsh(M_).max())
                Mp_, _, N0r = pad_to_pow2(M_, np.zeros(M_.shape[0]), lam_mx); xp_, b_ = oss_.reduced_rhs(r); rq = np.zeros(Mp_.shape[0]); rq[:N0r] = b_
                kap_aug_s = float(np.linalg.cond(Ks)); drow = {**base, "dim_reduced": M_.shape[0], "N_padded_reduced": Mp_.shape[0], "kappa_augmented_scaled": kap_aug_s, "kappa_reduced_jacobi": kap_red, "sqrt_kappa_reduced": math.sqrt(kap_red)}
                for eps in CKSCFG["eps_list"]:
                    drow[f"degree_augmented_eps{eps:.0e}"] = polynomial_degree(kap_aug_s * 1.05, eps, "optimal"); drow[f"degree_oss_eps{eps:.0e}"] = polynomial_degree_spd(kap_red * 1.05, eps)
                    ss = SPDPolySystem(Mp_, eps, rng_for("q5d", inst_id, fi, eps), max_degree=CKSCFG["max_degree"])
                    row = {**base, "solver": "lcu_oss", "poly": ss.poly, "eps_target": eps, "kappa_target": ss.kappa, "n_qubits": ss.n_qubits, "oracle_queries": ss.degree, "l1_norm": ss.l1, "target": "null_space_spd", "dim_target": M_.shape[0]}
                    if ss.over_budget: Q5A.append({**row, "p_success": np.nan, "rel_err_ideal": np.nan, "t_sim": np.nan, "status": "degree_cap (simulation budget)"}); continue
                    so = ss.run(rq); v_ = so["x_ideal"][:N0r]; u_ = oss_.full(v_, r, xp_)
                    Q5A.append({**row, "p_success": so["p_postselect_exact"], "rel_err_ideal": float(np.linalg.norm(u_ - u_true) / nu), "t_sim": so["t_circuit_sim"], "status": "ok"})
                    drow[f"p_success_oss_eps{eps:.0e}"] = so["p_postselect_exact"]; drow[f"fwd_err_oss_eps{eps:.0e}"] = float(np.linalg.norm(u_ - u_true) / nu)
                drow["n_qubits_oss_eps0.3"] = SPDPolySystem(Mp_, 0.3, rng_for("q5d", inst_id, fi, "q"), max_degree=10 ** 9).n_qubits
                nq_aug = int(math.log2(Ap.shape[0])) + 3 + max(1, int(math.ceil(math.log2((polynomial_degree(kap_aug_s * 1.05, 0.3, "optimal") + 1) // 2 + 1)))); drow["n_qubits_augmented_eps0.3"] = nq_aug
                Q5D.append(drow)
            except Exception as e_:
                REC.failure(f"{inst_id}__q5d_{fi}", "q5d_oss", repr(e_)[:300], instance_id=inst_id); print("  Q5d failed:", inst_id, fi, repr(e_)[:160])
            # Q5b: cost of refinement to a 1e-10 residual against circuit precision and samples per round (emulated AA readout)
            if n_ir >= CKSCFG["ir_max_systems"]: continue
            n_rows_before = len(Q5B)
            for poly, eps in [("cks", 1e-6)] + [("optimal", e_) for e_ in CKSCFG["ir_eps_list"]]:
                cs = CKSSystem(Ap, eps, rng_for("q5b", inst_id, fi, eps, poly), max_degree=CKSCFG["max_degree"], poly=poly)
                if cs.over_budget: continue
                aa = 1.0 / math.sqrt(max(cs.run(rp)["p_postselect_exact"], 1e-300))
                for spr in CKSCFG["ir_samples_per_round"]:
                    u = np.zeros_like(r); hist = []
                    for rd in range(CKSCFG["ir_rounds"]):
                        res = r - K @ u; rp2 = np.zeros(Ap.shape[0]); rp2[:N0] = d * res; xa = cs.run(rp2, eff_shots=max(spr // 2, 1))["x_aa"][:N0]
                        if not np.all(np.isfinite(xa)): break
                        u = u + d * xa; hist.append(float(np.linalg.norm(K @ u - r) / np.linalg.norm(r)))
                        if hist[-1] < 1e-10 or hist[-1] > 1e6: break
                    if not hist: continue
                    Q5B.append({**base, "poly": cs.poly, "eps_circuit": eps, "kappa_target": cs.kappa, "walk_steps": cs.degree, "p_success": 1.0 / aa ** 2, "samples_per_round": spr, "rounds": len(hist), "resid_final": hist[-1],
                                "reached_1e-10": bool(hist[-1] < 1e-10), "contraction_per_round": float((hist[-1] / hist[0]) ** (1.0 / max(len(hist) - 1, 1))) if len(hist) > 1 else np.nan,
                                "total_samples": spr * len(hist), "block_encoding_queries_total_with_AA": float(len(hist) * spr * aa * cs.degree)})
            n_ir += int(len(Q5B) > n_rows_before)                      # a system counts only if at least one configuration was within the simulation budget
    Q5A, Q5B, Q5D = pd.DataFrame(Q5A), pd.DataFrame(Q5B), pd.DataFrame(Q5D)
    Q5A.to_csv(os.path.join(RESULTS_DIR, "qls_comparison_hhl_vs_cks.csv"), index=False); Q5B.to_csv(os.path.join(RESULTS_DIR, "cks_refinement.csv"), index=False); Q5D.to_csv(os.path.join(RESULTS_DIR, "oss_target_comparison.csv"), index=False)
    if len(Q5D):
        print("Q5d - augmented (Ruiz-scaled, indefinite) vs null-space (Jacobi-scaled, SPD) target on the same captured KKT systems:")
        cols_ = ["instance_id", "factor_index", "kkt_dim", "dim_reduced", "kappa_augmented_scaled", "kappa_reduced_jacobi"] + [c_ for c_ in Q5D.columns if c_.startswith("degree_")] + ["n_qubits_augmented_eps0.3", "n_qubits_oss_eps0.3"]
        print(Q5D[cols_].to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    if len(Q5A):
        a_ = Q5A.assign(solver=lambda x: np.where(x.solver == "lcu_walk", "LCU " + x.poly.map(poly_tag) + " eps=" + x.eps_target.map(lambda v: f"{v:.0e}"), x.solver))
        print("Q5a - forward error (ideal readout) on captured KKT systems:")
        print(a_.groupby("solver").agg(systems_run=("rel_err_ideal", "count"), err_median=("rel_err_ideal", "median"), err_max=("rel_err_ideal", "max"), queries_median=("oracle_queries", "median"), l1_median=("l1_norm", "median"),
              p_success_median=("p_success", "median")).to_string(float_format=lambda v: f"{v:.2e}"))
    if len(Q5B):
        print("\nQ5b - cost of refining to a 1e-10 residual (median over systems; queries include ideal amplitude amplification):")
        print(Q5B.groupby(["poly", "eps_circuit", "samples_per_round"]).agg(systems=("rounds", "size"), reached=("reached_1e-10", "mean"), rounds=("rounds", "median"), walk_steps=("walk_steps", "median"),
              total_samples=("total_samples", "median"), queries=("block_encoding_queries_total_with_AA", "median")).to_string(float_format=lambda v: f"{v:.3g}"))
    REC.status("Q5ab_cks_captured", "done", f"{len(Q5A)} solver x system rows, {len(Q5B)} refinement rows in {time.perf_counter() - t5:.0f}s")

    # Q5c: inside coneqp, no classical fallback
    icfg = QCFG["integrated"]; t5 = time.perf_counter(); n5 = 0
    common = {"fallback": False, "max_refine": CKSCFG["integrated_max_refine"], "criterion": icfg["accept_criterion"], "eta": icfg["accept_eta"], "tol_rule": "fixed", "tol": CKSCFG.get("integrated_be_tol", icfg["accept_be_tol"]), "scaling": "ruiz"}
    jobs = [(i_, {"name": rs["name"], "kind": "cks", "eps": rs["eps"], "poly": rs.get("poly"), "recon": rs["recon"], "eff_shots": rs.get("eff_shots", 0), "maxiters": icfg["maxiters"], **common})
            for i_ in list(icfg["instances"]) + [x for x in CKSCFG["integrated_extra_instances"] if x in INSTANCES] for rs in CKSCFG["integrated_runs"]]
    jobs += [(i_, {"name": rs["name"], "kind": "hhl", "n_clock": rs["n_clock"], "recon": rs["recon"], "eff_shots": rs.get("eff_shots", 0), "shots": 0, "reciprocal": "kappa_known", "reg": 0.0, "maxiters": CKSCFG.get("hhl_fair_maxiters", 20), **common})
             for i_ in list(icfg["instances"]) + [x for x in CKSCFG.get("hhl_fair_extra_instances", []) if x in INSTANCES] for rs in CKSCFG.get("hhl_fair_runs", [])
             if int(math.ceil(math.log2(INSTANCES[i_]["kkt_dim"]))) + rs["n_clock"] + 2 <= QCFG["max_qubits"] and (i_ in icfg["instances"] or rs["name"] in CKSCFG.get("hhl_fair_extra_runs", [rs["name"]]))]
    # the same instances with the null-space polynomial solver
    jobs += [(i_, {"name": rs["name"], "kind": "lcu_oss", "eps": rs["eps"], "recon": rs["recon"], "eff_shots": rs.get("eff_shots", 0), "maxiters": icfg["maxiters"], **common})
             for i_ in list(icfg["instances"]) + [x for x in CKSCFG["integrated_extra_instances"] if x in INSTANCES] for rs in CKSCFG.get("oss_runs", [])]
    for inst_id, spec in jobs:
        try:
            rec, sol, st = run_instance(INSTANCES[inst_id], spec, repeat=0, compute_cond=True, experiment="Q5_integrated_cks"); n5 += 1
            print(f"  {inst_id} {spec['name']}: status={rec['status']} iters={rec['iterations']} accepted={rec['accepted']} solves={rec['n_solve_calls']} refinement_rounds={rec['n_refine_total']} "
                  f"unaccepted_used={rec['n_solves_unaccepted_used']} fallback={rec['n_fallback_total']} strict={rec['strict_method_success']} obj_rel_err={rec.get('obj_rel_err', float('nan')):.2e} wall={rec['wall_method']:.1f}s")
        except Exception as e:
            REC.failure(f"{inst_id}__{spec['name']}", "integrated_cks", repr(e)[:300], method=spec["name"], instance_id=inst_id); print("  integrated run failed:", repr(e)[:200])
    REC.status("Q5c_integrated_cks", "done", f"{n5} runs in {time.perf_counter() - t5:.0f}s"); REC.flush(min_interval=0)

    # Q5e: finite-shot runs over sampling seeds x accepted samples per refinement round
    sw = CKSCFG.get("shot_sweep"); t5e = time.perf_counter(); n5e = 0
    if sw:
        sw_insts = (list(icfg["instances"]) + [x for x in CKSCFG["integrated_extra_instances"] if x in INSTANCES])[:sw["max_instances"]]
        for inst_id in sw_insts:
            for spr in sw["samples_per_round"]:
                for sd in sw["seeds"]:
                    spec = {"name": f"lcu_opt_eps{sw['eps']:g}_aa{spr}_seed{sd}_ir_nofb", "kind": "cks", "eps": sw["eps"], "poly": "optimal", "recon": "aa_shots", "eff_shots": max(spr // 2, 1), "maxiters": icfg["maxiters"],
                            "sweep_samples": spr, "sweep_seed": sd, **common}
                    try:
                        rec, sol, st = run_instance(INSTANCES[inst_id], spec, repeat=0, compute_cond=False, experiment="Q5e_shot_sweep"); n5e += 1
                    except Exception as e:
                        REC.failure(f"{inst_id}__{spec['name']}", "shot_sweep", repr(e)[:300], method=spec["name"], instance_id=inst_id); print("  shot-sweep run failed:", repr(e)[:200])
        REC.flush(min_interval=0); d5e = REC.df("benchmark_results"); d5e = d5e[d5e.experiment == "Q5e_shot_sweep"]
        if len(d5e):
            print("Q5e - strict success rate / median IPM iterations / median refinement rounds over instances x seeds, by accepted samples per round:")
            print(d5e.groupby("sweep_samples").agg(runs=("run_id", "size"), strict=("strict_method_success", "mean"), accepted=("accepted", "mean"), ipm_its=("iterations", "median"), refine=("n_refine_total", "median"),
                                                  rejected_used=("n_solves_unaccepted_used", "sum"), wall=("wall_method", "median")).round(3).to_string())
            print("  per-instance spread across seeds (strict successes / seeds):"); print(d5e.pivot_table(index="instance_id", columns="sweep_samples", values="strict_method_success", aggfunc="sum").to_string())
    REC.status("Q5e_shot_sweep", "done" if n5e else "skipped", f"{n5e} runs in {time.perf_counter() - t5e:.0f}s")
