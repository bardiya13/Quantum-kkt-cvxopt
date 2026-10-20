# HHL experiments: Q1 captured systems, Q2 Qiskit cross-check, Q3 HHL inside coneqp, resource estimates.
QCFG = CONFIG["quantum"]

def q_row(base, **kw):
    r = dict(base); r.update(kw); REC.add("quantum_results", r); return r

def select_factor_indices(caps):
    idx = [c["factor_index"] for c in caps]
    if QCFG["captured_iters"] != "auto": return [i for i in QCFG["captured_iters"] if i in idx]
    L = len(idx); return sorted({idx[0], idx[L // 4], idx[L // 2], idx[(3 * L) // 4], idx[-1]})

def evaluate_captured_system(inst_id, cap, n_clock, scaling, c_mode, reg, shots_list, eff_shots_list, sampling_repeats, ir_rounds, asm_n):
    K = cap["K"]; solve0 = cap["solves"][0]; r = solve0["rhs"]; u_exact = solve0["u"]; N0 = K.shape[0]
    base = {"experiment": "Q1_captured", "instance_id": inst_id, "factor_index": cap["factor_index"], "kappa_kkt": cap["kappa"], "nt_proxy": cap["nt_proxy"], "kkt_dim": N0,
            "n_clock": n_clock, "scaling": scaling, "reciprocal_mode": c_mode, "regularization": reg, "engine": "minisim"}
    J = np.ones(N0); J[asm_n:] = -1.0
    Kt = K + reg * np.diag(J) if reg else K
    d = ruiz_equilibrate(Kt) if scaling == "ruiz" else None
    Ks = apply_scaling(Kt, d) if scaling == "ruiz" else Kt
    rng = rng_for("captured", inst_id, cap["factor_index"], n_clock, scaling, c_mode, reg)
    lam, _ = spectral_bound(Ks, QCFG["spectral_bound"], rng)
    Ap, _, _ = pad_to_pow2(Ks, np.zeros(N0), lam)
    n_q = int(round(math.log2(Ap.shape[0]))) + n_clock + 2
    if n_q > QCFG["max_qubits"] or 16 * 2 ** n_q > QCFG["max_state_bytes"]:
        q_row(base, status="skipped", reason=f"needs {n_q} qubits > max_qubits={QCFG['max_qubits']}", n_qubits=n_q); return None
    try:
        sysm = HHLSystem(Ap, n_clock, rng, spectral=QCFG["spectral_bound"], c_mode=c_mode)
    except ResourceLimit as e:
        q_row(base, status="skipped", reason=str(e)); return None
    nK = np.linalg.norm(K, 2); nr = np.linalg.norm(r)
    def solve_target(rhs):                      # one HHL run for a right-hand side of the original system; returns (u, out)
        rs = d * rhs if scaling == "ruiz" else rhs
        rp = np.zeros(Ap.shape[0]); rp[:N0] = rs
        out = sysm.run(rp); up = out["x_ideal"][:N0]
        return (d * up if scaling == "ruiz" else up), out
    def measures(u):
        e = K @ u - r; return {"rel_resid_original": float(np.linalg.norm(e) / nr), "backward_err_original": float(np.linalg.norm(e) / (nK * np.linalg.norm(u) + nr)), "rel_err_original": float(relnorm(u - u_exact, u_exact))}
    common = dict(status="ok", reason="", k_min=None, n_qubits=sysm.n_qubits, N_padded=Ap.shape[0], kappa_target=sysm.kappa, kappa_original=float(np.linalg.cond(K)),
                  lam_max_est=sysm.lam_max_est, lam_min_abs=sysm.lam_min_abs, clock_resolution=sysm.Delta, n_eig_below_resolution=sysm.n_eig_below_resolution,
                  n_eig_unresolved_10pct=sysm.n_eig_unresolved_10pct, t_prep=sysm.t_prep, t_diag=sysm.t_diag, t_spectral_assumption=sysm.t_spectral_assumption,
                  n_mc_rotations=len(sysm.thetas), evolution_time_x_norm=2 * (2 ** n_clock - 1) * sysm.t * sysm.lam_max_est)
    common["k_min"] = sysm.k_min
    # ideal statevector, iterative refinement on the true system
    u, out = solve_target(r); xe = out["x_exact"]; nx = max(np.linalg.norm(xe), 1e-300)
    common.update(p_postselect_exact=out["p_postselect_exact"], p_leak_exact=out["p_leak_exact"], gates_highlevel=out["n_gates_highlevel"], depth_highlevel=out["depth_highlevel"],
                  gate_counts=json.dumps(out["gate_counts"]), t_circuit_sim=out["t_circuit_sim"], t_readout_probs=out["t_readout_probs"], ideal_imag_fraction=out["ideal_imag_fraction"],
                  expected_repetitions_no_AA=1.0 / max(out["p_postselect_exact"], 1e-300), expected_repetitions_with_AA=1.0 / math.sqrt(max(out["p_postselect_exact"], 1e-300)),
                  scale_from_postselection_rel_err=abs(out["scale_from_postselection"] - nx) / nx)
    q_row(base, recon="ideal_statevector", ir_round=0, shots_per_setting=0, sampling_repeat=0, rel_err_target=float(np.linalg.norm(out["x_ideal"] - xe) / nx), **measures(u), **common)
    for rd in range(1, ir_rounds + 1):
        du, out2 = solve_target(r - K @ u); u = u + du; m = measures(u)
        q_row(base, recon="ideal_statevector", ir_round=rd, shots_per_setting=0, sampling_repeat=0, rel_err_target=np.nan, **m, **common)
        if m["rel_resid_original"] < 1e-12: break
    # finite-shot readouts at round 0
    for shots in shots_list:
        for rec in reconstruct_from_shots(sysm, out, shots, sampling_repeats, rng):
            if rec["status"] != "ok":
                q_row(base, recon="shots", ir_round=0, shots_per_setting=shots, sampling_repeat=rec["repeat"], status="failed", reason=rec["status"], postselect_rate_empirical=rec["postselect_rate"], **{k: v for k, v in common.items() if k not in ("status", "reason")}); continue
            us = rec["x_rec"][:N0]; us = d * us if scaling == "ruiz" else us; uls = rec["x_rec_ls"][:N0]; uls = d * uls if scaling == "ruiz" else uls
            q_row(base, recon="shots", ir_round=0, shots_per_setting=shots, sampling_repeat=rec["repeat"], rel_err_target=rec["rel_err"], rel_err_ls_rescale=rec["rel_err_ls"], rel_err_original=relnorm(us - u_exact, u_exact),
                  rel_resid_original=relnorm(K @ us - r, r), rel_resid_original_ls=relnorm(K @ uls - r, r), scale_rel_err=rec["scale_rel_err"], n_sign_errors=rec["n_sign_errors"],
                  postselect_rate_empirical=rec["postselect_rate"], postselected_shots=rec["postselected_shots"], t_reconstruct=rec["t_reconstruct"], **common)
    for eff in eff_shots_list:
        for rec in reconstruct_aa_emulated(sysm, out, eff, sampling_repeats, rng):
            if rec["status"] != "ok":
                q_row(base, recon="aa_shots", ir_round=0, shots_per_setting=eff, sampling_repeat=rec["repeat"], status="failed", reason=rec["status"], **{k: v for k, v in common.items() if k not in ("status", "reason")}); continue
            us = rec["x_rec"][:N0]; us = d * us if scaling == "ruiz" else us
            q_row(base, recon="aa_shots", ir_round=0, shots_per_setting=eff, sampling_repeat=rec["repeat"], rel_err_target=rec["rel_err"], rel_err_ls_rescale=rec["rel_err_ls"], rel_err_original=relnorm(us - u_exact, u_exact),
                  rel_resid_original=relnorm(K @ us - r, r), scale_rel_err=rec["scale_rel_err"], n_sign_errors=rec["n_sign_errors"], aa_overhead_per_sample=rec["aa_overhead_per_sample"],
                  raw_shots_equivalent_no_AA=rec["raw_shots_equivalent_no_AA"], postselected_shots=rec["postselected_shots"], t_reconstruct=rec["t_reconstruct"], **common)
    return out, sysm, Ap, N0, d

Q1_KEEP = {}
if not QCFG["enabled"]:
    REC.status("Q1_captured", "skipped", "quantum disabled in CONFIG")
elif not CAPTURED:
    REC.status("Q1_captured", "skipped", "no captured KKT systems")
else:
    t_q1 = time.perf_counter(); n_eval = 0
    for inst_id, caps in CAPTURED.items():
        if not inst_id.startswith("tiny"): continue
        asm_n = INSTANCES[inst_id]["n"]
        for fi in select_factor_indices(caps):
            cap = caps[fi]
            for n_clock in QCFG["n_clock_list"]:
                grid = [(sc, cm, 0.0) for sc in QCFG["scalings"] for cm in QCFG["reciprocal_modes"]]      # regularisation: Q4, Section 17
                for sc, cm, rg in grid:
                    res = evaluate_captured_system(inst_id, cap, n_clock, sc, cm, rg, QCFG["shots_list"], QCFG["eff_shots_list"], QCFG["sampling_repeats"], QCFG["ir_rounds"], asm_n)
                    n_eval += 1
                    if res is not None and sc == "ruiz" and cm == "kappa_known" and rg == 0.0: Q1_KEEP[(inst_id, fi, n_clock)] = res
    REC.flush(min_interval=0)
    REC.status("Q1_captured", "done", f"{n_eval} configurations in {time.perf_counter() - t_q1:.0f}s")
    qdf = REC.df("quantum_results"); ok = qdf[(qdf.experiment == "Q1_captured") & (qdf.status == "ok")]
    idl = ok[(ok.recon == "ideal_statevector") & (ok.ir_round == 0)]
    print("\nIdeal HHL error on the original system (median) by n_clock x (scaling, reciprocal, reg):")
    print(idl.pivot_table(index="n_clock", columns=["scaling", "reciprocal_mode", "regularization"], values="rel_err_original", aggfunc="median").round(4).to_string())
    print("\nPostselection probability (median):"); print(idl.pivot_table(index="n_clock", columns=["scaling", "reciprocal_mode", "regularization"], values="p_postselect_exact", aggfunc="median").round(5).to_string())
    irr = ok[(ok.recon == "ideal_statevector") & (ok.scaling == "ruiz") & (ok.reciprocal_mode == "kappa_known")]
    print("\nIterative refinement with HHL (delta = 0), residual on the true system (median) by round, largest n_clock:")
    print(irr[irr.n_clock == irr.n_clock.max()].groupby("ir_round").rel_resid_original.median().round(9).to_string())
    aa = ok[(ok.recon == "aa_shots") & (ok.scaling == "ruiz") & (ok.reciprocal_mode == "kappa_known") & (ok.regularization == 0.0)]
    print("\nAA-emulated readout error (median) by accepted samples per setting x n_clock:"); print(aa.pivot_table(index="shots_per_setting", columns="n_clock", values="rel_err_target", aggfunc="median").round(4).to_string())


# Q2: the same circuit in Qiskit (Ruiz scaling, kappa-known reciprocal)
def qiskit_crosscheck(inst_id, fi, n_clock, res):
    out, sysm, Ap, N0, d = res
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit.library import StatePreparation, UnitaryGate, QFTGate, RYGate
    from qiskit.quantum_info import Statevector
    base = {"experiment": "Q2_qiskit_crosscheck", "instance_id": inst_id, "factor_index": fi, "n_clock": n_clock, "scaling": "ruiz", "reciprocal_mode": sysm.c_mode, "regularization": 0.0, "engine": "qiskit",
            "kkt_dim": N0, "N_padded": Ap.shape[0], "n_qubits": sysm.n_qubits, "kappa_target": sysm.kappa}
    t0 = time.perf_counter(); qc = QuantumCircuit(sysm.n_qubits)
    b = out["b"]; r_ref, Ar = sysm.reference(b); v = np.concatenate([b, Ar]); v = v / np.linalg.norm(v)
    qc.append(StatePreparation(v.astype(complex)), sysm.dat + [sysm.flg])
    for q in sysm.clk: qc.h(q)
    for j, q in enumerate(sysm.clk): qc.append(UnitaryGate(sysm.U_pow[j]).control(1, annotated=False), [q] + sysm.dat)
    qc.append(QFTGate(sysm.n_c).inverse(), sysm.clk)
    for k, th in sysm.thetas.items(): qc.append(RYGate(th).control(sysm.n_c, ctrl_state=k, annotated=False), sysm.clk + [sysm.anc])
    qc.append(QFTGate(sysm.n_c), sysm.clk)
    for j in reversed(range(sysm.n_c)): qc.append(UnitaryGate(sysm.U_pow_dag[j]).control(1, annotated=False), [sysm.clk[j]] + sysm.dat)
    for q in sysm.clk: qc.h(q)
    t_build = time.perf_counter() - t0; t0 = time.perf_counter(); sv = Statevector.from_instruction(qc); t_sv = time.perf_counter() - t0
    sim = MiniSim(sysm.n_qubits); sim.prepare(v.astype(complex), sysm.dat + [sysm.flg])
    for q in sysm.clk: sim.h(q)
    for j, q in enumerate(sysm.clk): sim.cu(sysm.U_pow[j], q, sysm.dat)
    sim.iqft(sysm.clk)
    for k, th in sysm.thetas.items(): sim.mcry(th, sysm.clk, [(k >> j) & 1 for j in range(sysm.n_c)], sysm.anc)
    sim.qft(sysm.clk)
    for j in reversed(range(sysm.n_c)): sim.cu(sysm.U_pow_dag[j], sysm.clk[j], sysm.dat)
    for q in sysm.clk: sim.h(q)
    row = dict(base, status="ok", reason="", statevector_fidelity_vs_minisim=float(abs(np.vdot(sv.data, sim.psi)) ** 2), t_qiskit_build=t_build, t_qiskit_statevector=t_sv, qiskit_version=QISKIT_STATUS.get("version"))
    try:
        t0 = time.perf_counter(); tq = transpile(qc, basis_gates=["u", "cx"], optimization_level=1, seed_transpiler=GLOBAL_SEED); ops = dict(tq.count_ops())
        row.update(transpiled_depth=tq.depth(), transpiled_cx=ops.get("cx", 0), transpiled_u=ops.get("u", 0), transpiled_total=sum(ops.values()), t_transpile=time.perf_counter() - t0)
    except Exception as e:
        row.update(transpile_status=f"failed: {e!r}"[:200])
    if QISKIT_STATUS.get("aer"):
        try:
            from qiskit_aer import AerSimulator
            meas = [sysm.anc, sysm.flg] + sysm.dat
            def run_counts(circ, shots):
                c = circ.copy(); c.measure_all(); simr = AerSimulator(seed_simulator=GLOBAL_SEED); tc = transpile(c, simr)
                counts = simr.run(tc, shots=shots).result().get_counts(); p = np.zeros(2 ** len(meas))
                for bitstr, cnt in counts.items():
                    bits = bitstr.replace(" ", "")[::-1]; p[sum(int(bits[q]) << i for i, q in enumerate(meas))] += cnt
                return p / shots
            shots = int(QCFG["shots_list"][-1]); t0 = time.perf_counter()
            pZ = run_counts(qc, shots); qcX = qc.copy(); qcX.h(sysm.flg); pX = run_counts(qcX, shots); t_aer = time.perf_counter() - t0
            cZ = (pZ * shots).reshape(sysm.N, 2, 2); cX = (pX * shots).reshape(sysm.N, 2, 2)
            rec = reconstruct_from_counts(sysm, out, cZ, cX, shots, 0, "aer_shots")
            row.update(aer_shots_per_setting=shots, aer_rel_err=rec.get("rel_err", np.nan), aer_postselect_rate=rec.get("postselect_rate", np.nan), aer_status=rec["status"], t_aer_sampling=t_aer)
        except Exception as e:
            row.update(aer_status=f"failed: {e!r}"[:200])
    q_row(base, **{k: v for k, v in row.items() if k not in base}); return row

if not QCFG["enabled"] or not CAPTURED:
    REC.status("Q2_qiskit_crosscheck", "skipped", "quantum disabled or nothing captured")
elif not QISKIT_STATUS.get("available"):
    REC.status("Q2_qiskit_crosscheck", "skipped", f"qiskit not available: {QISKIT_STATUS.get('reason')}")
else:
    n_cc = 0; t_cc = time.perf_counter()
    for (inst_id, fi, n_clock), res in list(Q1_KEEP.items()):
        if n_clock > QCFG["qiskit_max_clock"] or res[1].n_qubits > QCFG["qiskit_max_qubits"]: continue
        if fi != select_factor_indices(CAPTURED[inst_id])[0]: continue
        try:
            row = qiskit_crosscheck(inst_id, fi, n_clock, res); n_cc += 1
            print(f"  qiskit check {inst_id} factor {fi} n_clock={n_clock}: fidelity={row['statevector_fidelity_vs_minisim']:.6f} depth={row.get('transpiled_depth')} cx={row.get('transpiled_cx')} aer_rel_err={row.get('aer_rel_err', float('nan')):.3e}")
        except Exception as e:
            q_row({"experiment": "Q2_qiskit_crosscheck", "instance_id": inst_id, "factor_index": fi, "n_clock": n_clock, "engine": "qiskit"}, status="failed", reason=repr(e)[:300]); print("  qiskit check failed:", repr(e)[:200])
    REC.status("Q2_qiskit_crosscheck", "done" if n_cc else "failed", f"{n_cc} circuits cross-checked in {time.perf_counter() - t_cc:.0f}s")
REC.flush(min_interval=0)


# Q3: HHL inside coneqp (combined test; block test alone as an ablation)
if not QCFG["enabled"]:
    REC.status("Q3_integrated", "skipped", "quantum disabled")
else:
    icfg = QCFG["integrated"]; n_int = 0; t_int = time.perf_counter()
    for inst_id in icfg["instances"]:
        inst = INSTANCES[inst_id]
        for rs in icfg["runs"]:
            if rs.get("instances_only") and inst_id not in rs["instances_only"]: continue
            spec = {"name": rs["name"], "kind": "hhl", "recon": rs["recon"], "n_clock": rs["n_clock"], "shots": rs.get("shots", 0), "eff_shots": rs.get("eff_shots", 0), "fallback": rs["fallback"],
                    "max_refine": icfg["max_refine"], "criterion": rs.get("criterion", icfg["accept_criterion"]), "eta": icfg["accept_eta"], "tol_rule": "fixed", "tol": icfg["accept_be_tol"],
                    "maxiters": icfg["maxiters"], "scaling": rs["scaling"], "reciprocal": rs["reciprocal"], "reg": rs["reg"], "switch_policy": bool(rs.get("switch_policy", False))}
            n_q = int(math.ceil(math.log2(inst["kkt_dim"]))) + rs["n_clock"] + 2
            if n_q > QCFG["max_qubits"]:
                REC.status(f"Q3_{inst_id}_{rs['name']}", "skipped", f"needs {n_q} qubits > max_qubits"); continue
            try:
                rec, sol, st = run_instance(inst, spec, repeat=0, compute_cond=True, experiment="Q3_integrated"); n_int += 1
                print(f"  {inst_id} {rs['name']}: status={rec['status']} iters={rec['iterations']} accepted={rec['accepted']} solves={rec['n_solve_calls']} direct={rec['n_solves_accepted_direct']} "
                      f"refined={rec['n_solves_accepted_refined']} unaccepted_used={rec['n_solves_unaccepted_used']} fallback={rec['n_fallback_total']} strict_quantum={rec['strict_method_success']} "
                      f"wall={rec['wall_method']:.1f}s obj_rel_err={rec.get('obj_rel_err', float('nan')):.2e}")
            except Exception as e:
                REC.failure(f"{inst_id}__{rs['name']}", "integrated_quantum", repr(e)[:300], method=rs["name"], instance_id=inst_id); print("  integrated run failed:", repr(e)[:200])
    REC.status("Q3_integrated", "done", f"{n_int} runs in {time.perf_counter() - t_int:.0f}s"); REC.flush(min_interval=0)
    dfq = REC.df("benchmark_results"); dfq = dfq[dfq.experiment == "Q3_integrated"]
    if len(dfq): print(dfq[["instance_id", "method", "criterion", "status", "iterations", "accepted", "n_solve_calls", "n_solves_accepted_direct", "n_solves_accepted_refined", "n_solves_unaccepted_used", "n_fallback_total", "strict_method_success", "obj_rel_err", "wall_method"]].to_string(index=False))


# resource estimates (depend on the stated assumptions)
qdf = REC.df("quantum_results")
if len(qdf) and "expected_repetitions_no_AA" in qdf:
    idl = qdf[(qdf.experiment == "Q1_captured") & (qdf.recon == "ideal_statevector") & (qdf.status == "ok") & (qdf.ir_round == 0)]
    if len(idl):
        RESOURCE_TABLE = idl.groupby(["instance_id", "n_clock", "scaling", "reciprocal_mode", "regularization"]).agg(
            kappa_target_med=("kappa_target", "median"), qubits=("n_qubits", "first"), highlevel_gates=("gates_highlevel", "first"), mc_rotations=("n_mc_rotations", "first"),
            evolution_time_x_norm=("evolution_time_x_norm", "median"), p_post_med=("p_postselect_exact", "median"), reps_no_AA_med=("expected_repetitions_no_AA", "median"),
            reps_with_AA_med=("expected_repetitions_with_AA", "median"), ideal_err_med=("rel_err_original", "median")).reset_index()
        RESOURCE_TABLE.to_csv(os.path.join(RESULTS_DIR, "quantum_resource_estimates.csv"), index=False)
        print("\nResource estimates written (assumption-dependent):", len(RESOURCE_TABLE), "rows")
