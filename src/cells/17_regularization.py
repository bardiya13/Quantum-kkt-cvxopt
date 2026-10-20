# Q4: regularisation of the ill-conditioned late-path systems. delta is added to the Ruiz-scaled system and the direction is
# refined on the true system; the corrector is LU of the regularised system (classical diagnostic) or HHL (ideal statevector).
RCFG = QCFG["reg_experiment"]

def reg_experiment_on_system(inst_id, cap, asm_n, corrector="classical_lu", n_clock=None):
    K = cap["K"]; r = cap["solves"][0]["rhs"]; u_exact = cap["solves"][0]["u"]; N0 = K.shape[0]
    d = ruiz_equilibrate(K); Ks = apply_scaling(K, d); kappa_s = cond_exact(Ks, limit=RCFG["classical_max_dim"])
    if not (kappa_s >= RCFG["min_kappa_scaled"]): return "well_conditioned"
    nK = np.linalg.norm(K, 2); nr = np.linalg.norm(r); nu = max(np.linalg.norm(u_exact), 1e-300); J = np.ones(N0); J[asm_n:] = -1.0
    base = {"experiment": f"Q4_regularization_{'quantum' if corrector == 'hhl' else 'classical'}", "instance_id": inst_id, "factor_index": cap["factor_index"], "kappa_kkt": cap["kappa"],
            "kkt_dim": N0, "scaling": "ruiz", "reciprocal_mode": "kappa_known" if corrector == "hhl" else "", "corrector": ("hhl_ideal_statevector" if corrector == "hhl" else "classical_lu_regularized"),
            "engine": "minisim" if corrector == "hhl" else "classical_lu", "n_clock": n_clock if corrector == "hhl" else np.nan, "kappa_original": float(np.linalg.cond(K)), "kappa_scaled": float(kappa_s)}
    for delta in RCFG["deltas"]:
        Kd = Ks + delta * np.diag(J); kappa_d = cond_exact(Kd, limit=RCFG["classical_max_dim"])
        t0 = time.perf_counter()
        if corrector == "hhl":
            rng = rng_for("reg", inst_id, cap["factor_index"], delta, n_clock)
            lam, _ = spectral_bound(Kd, QCFG["spectral_bound"], rng); Ap, _, _ = pad_to_pow2(Kd, np.zeros(N0), lam)
            n_q = int(round(math.log2(Ap.shape[0]))) + n_clock + 2
            if n_q > QCFG["max_qubits"]:
                q_row(base, regularization=delta, status="skipped", reason=f"needs {n_q} qubits > max_qubits", kappa_target=kappa_d); continue
            try: sysm = HHLSystem(Ap, n_clock, rng, spectral=QCFG["spectral_bound"], c_mode="kappa_known")
            except ResourceLimit as e:
                q_row(base, regularization=delta, status="skipped", reason=str(e), kappa_target=kappa_d); continue
            t_prep = time.perf_counter() - t0
            def correct(e):
                rp = np.zeros(Ap.shape[0]); rp[:N0] = d * e; out = sysm.run(rp); return d * out["x_ideal"][:N0], out
        else:
            lu = sla.lu_factor(Kd, check_finite=False); t_prep = time.perf_counter() - t0
            def correct(e): return d * sla.lu_solve(lu, d * e, check_finite=False), None
        u = np.zeros(N0); t_rounds = 0.0; p_post = np.nan; t_circ = 0.0
        for rd in range(RCFG["ir_rounds"] + 1):
            t1 = time.perf_counter(); du, out = correct(r - K @ u); u = u + du; t_rounds += time.perf_counter() - t1
            if out is not None: p_post = out["p_postselect_exact"]; t_circ += out["t_circuit_sim"]
            e = K @ u - r; res = float(np.linalg.norm(e) / nr)
            q_row(base, regularization=delta, kappa_target=float(kappa_d), status="ok", reason="", recon="ideal_statevector" if corrector == "hhl" else "classical", ir_round=rd,
                  rel_resid_original=res, backward_err_original=float(np.linalg.norm(e) / (nK * np.linalg.norm(u) + nr)), rel_err_original=float(np.linalg.norm(u - u_exact) / nu),
                  t_prep=t_prep, t_rounds_cum=t_rounds, t_circuit_sim=t_circ, p_postselect_exact=p_post, n_qubits=(sysm.n_qubits if corrector == "hhl" else np.nan))
            if res < 1e-13 or not np.isfinite(res): break
    return "done"

def run_reg_experiment(capture_dict, label):
    n_sys, n_skipped = 0, 0
    for inst_id, caps in capture_dict.items():
        if inst_id not in INSTANCES: continue
        asm_n = INSTANCES[inst_id]["n"]
        for cap in caps:
            if cap["K"].shape[0] > RCFG["classical_max_dim"]: n_skipped += 1; continue
            st = reg_experiment_on_system(inst_id, cap, asm_n, corrector="classical_lu")
            if st != "done": continue
            n_sys += 1
            N0 = cap["K"].shape[0]
            if 2 ** int(math.ceil(math.log2(N0))) <= RCFG["quantum_max_dim"]:
                for nc in RCFG["quantum_n_clock"]: reg_experiment_on_system(inst_id, cap, asm_n, corrector="hhl", n_clock=nc)
    REC.status(f"Q4_regularization_{label}", "done", f"{n_sys} ill-conditioned systems (kappa_scaled >= {RCFG['min_kappa_scaled']}); {n_skipped} above classical_max_dim")
    REC.flush(min_interval=0)

if QCFG["enabled"] and CAPTURED:
    run_reg_experiment({k: v for k, v in CAPTURED.items() if not k.startswith("sdplib")}, "qp")
    q4 = REC.df("quantum_results"); q4 = q4[q4.experiment.str.startswith("Q4_")]
    if len(q4):
        print("Q4 (classical LU corrector): median residual on the true system by refinement round x delta")
        c4 = q4[q4.corrector == "classical_lu_regularized"]
        print(c4.pivot_table(index="ir_round", columns="regularization", values="rel_resid_original", aggfunc="median").to_string())
        print("kappa of the regularised scaled target (median) by delta:", c4[c4.ir_round == 0].groupby("regularization").kappa_target.median().round(1).to_dict())
        h4 = q4[q4.corrector == "hhl_ideal_statevector"]
        if len(h4):
            print("Q4 (HHL ideal-statevector corrector): median residual by round x delta, per n_clock")
            print(h4.pivot_table(index=["n_clock", "ir_round"], columns="regularization", values="rel_resid_original", aggfunc="median").to_string())
else:
    REC.status("Q4_regularization_qp", "skipped", "quantum disabled or nothing captured")
