# Export: final CSV tables, instances and captured KKT systems as .npz, results/README.md, and the zip bundle.
REC.flush(final=True)

# instances and captured KKT systems
for inst_id, caps in CAPTURED.items():
    inst = INSTANCES[inst_id]
    data = {"K": np.stack([c["K"] for c in caps]), "factor_index": np.array([c["factor_index"] for c in caps]), "kappa": np.array([c["kappa"] for c in caps]), "nt_proxy": np.array([c["nt_proxy"] for c in caps]),
            "di": np.stack([c["di"] for c in caps]) if caps[0]["di"].size else np.zeros((len(caps), 0)), "rhs_first_solve": np.stack([c["solves"][0]["rhs"] for c in caps]), "u_first_solve": np.stack([c["solves"][0]["u"] for c in caps]),
            "rhs_all": np.array([np.stack([s["rhs"] for s in c["solves"]]) for c in caps], dtype=object), "u_all": np.array([np.stack([s["u"] for s in c["solves"]]) for c in caps], dtype=object),
            "n": inst["n"], "p": inst["p"], "m_packed": inst["kkt_dim"] - inst["n"] - inst["p"], "dims_l": inst["dims"]["l"], "dims_s": np.array(inst["dims"]["s"])}
    if inst["kind"] == "coneqp":
        data.update({"P": np.asarray(inst["P"].todense() if sp.issparse(inst["P"]) else inst["P"]), "q": inst["q"], "G": np.asarray(inst["G"].todense() if sp.issparse(inst["G"]) else inst["G"]), "h": inst["h"],
                     "A": np.asarray(inst["A"].todense() if (inst["p"] and sp.issparse(inst["A"])) else (inst["A"] if inst["p"] else np.zeros((0, inst["n"])))), "b": inst["b"] if inst["p"] else np.zeros(0),
                     "x_star": inst["x_star"], "y_star": inst["y_star"], "z_star": inst["z_star"], "s_star": inst["s_star"], "f_star": inst["f_star"]})
    else:
        data.update({"c": inst["q"], "G": inst["G"], "h": inst["h"], "block_sizes": np.array(inst["block_sizes"]), "f_star_sdplib": inst["f_star"]})
    np.savez_compressed(os.path.join(NPZ_DIR, f"{inst_id}_instance_and_kkt.npz"), **data)
for inst in list(INSTANCES.values()):
    if inst["family"] in ("dense_kappa", "dense_noeq") and inst["seed"] == 0 and inst["n"] <= 40:
        np.savez_compressed(os.path.join(NPZ_DIR, f"{inst['id']}.npz"), P=np.asarray(inst["P"]), q=inst["q"], G=np.asarray(inst["G"]), h=inst["h"], A=np.asarray(inst["A"]) if inst["p"] else np.zeros((0, inst["n"])),
                            b=inst["b"] if inst["p"] else np.zeros(0), x_star=inst["x_star"], y_star=inst["y_star"], z_star=inst["z_star"], s_star=inst["s_star"], f_star=inst["f_star"])
    if inst["family"] == "sparse_banded" and inst["seed"] == 0 and inst["n"] <= 400:
        sp.save_npz(os.path.join(NPZ_DIR, f"{inst['id']}_P.npz"), sp.csr_matrix(inst["P"])); sp.save_npz(os.path.join(NPZ_DIR, f"{inst['id']}_G.npz"), sp.csr_matrix(inst["G"]))
        if inst["p"]: sp.save_npz(os.path.join(NPZ_DIR, f"{inst['id']}_A.npz"), sp.csr_matrix(inst["A"]))
        np.savez_compressed(os.path.join(NPZ_DIR, f"{inst['id']}_vectors.npz"), q=inst["q"], h=inst["h"], b=inst["b"] if inst["p"] else np.zeros(0), x_star=inst["x_star"], y_star=inst["y_star"], z_star=inst["z_star"], s_star=inst["s_star"], f_star=inst["f_star"])
print("npz files:", len(os.listdir(NPZ_DIR)))

README = f"""# Results

Written by the notebook, version {CODE_VERSION}, profile `{CONFIG['run_profile']}`, device `{'gpu' if GPU_OK else 'cpu'}`, config hash `{CONFIG_HASH}`, seed {GLOBAL_SEED}, {time.strftime('%Y-%m-%d')}.
Times are wall-clock seconds; residuals and errors are relative unless the column name says otherwise. A missing value means that the quantity
was not computed for the method, was above a size limit (`cond_limit`, `max_qubits`), or that the run failed (`error`, `failures.csv`).
Skipped experiments and the reason are listed in `experiment_status.csv`. Failed runs are kept in every table.

## Identifiers

`run_id` = `<instance_id>__<method>__r<repeat>`. Benchmark rows carry `code_version`, `config_hash`, `experiment`, `instance_id`, `seed`, `method`,
`criterion`, `scaling`, `regularization` and `target_tol`. Solve logs and iteration logs join on `run_id`; quantum rows are identified by
`instance_id`, `factor_index` (the captured KKT system) and the solver settings. `instances.csv` lists n, m, p, kappa(P), structure and nnz.

## Methods

Tolerances in method names are the nominal tolerance tau; the relative-residual and backward-error tests accept at 10 tau.

* `cvxopt_default`: unmodified `coneqp` / `conelp`. `custom_exact`: LU of the full KKT matrix through the callback.
  `custom_exact_reduced`: CHOLMOD on the reduced quasi-definite system.
* `minres_relres_1e-06`, `minres_be_<tau>`, `minres_block_<eta>`, `minres_comb_0.1_1e-10`, `minres_rrfloor_1e-06`: SciPy MINRES with the
  relative-residual, backward-error, block, combined and floored relative-residual test, Ruiz-scaled; `_noscale` without scaling.
* `minres_adaptive_nt_be`: backward-error tolerance tied to the NT-scaling spread. `cg_reduced_be_1e-10`: CG on the reduced system (p = 0).
* `minres_base_comb_<cpu|gpu>`: baseline MINRES kernel with the combined test (eta = 0.1, tau = 1e-10); `_ws` warm start.
  `minres_fused_comb_<cpu|gpu|auto>`: latency-optimised kernel; `_auto` uses the GPU from KKT dimension {CONFIG['inexact']['gpu_auto_min_dim']};
  `_floor8` block floor 1e-8; `_gpu_mm` GPU from KKT dimension 2 000 (Maros-Meszaros runs).
* `_switch`: exact solver for the rest of the run after the first rejected direction (`utilisation` = share of solves by the method).
* `gmres_stalelu_comb`: GMRES preconditioned with the factors of an earlier IPM iteration (`n_factorizations` = factorisations made).
* `lsqr_sqrt_comb`: LSQR on the square-root factor (p = 0, dense P). `oss_cg_comb`: CG on the null-space system (`kappa_reduced_max`, `reduced_dim`).
* `exact_noise_iso_<delta>`, `exact_noise_block_<delta>`: exact direction plus an injected relative error (sensitivity only).
  In E4b the names are `<method>@tau<level>` with columns `ipm_tol_level` and `feastol_option`.
* `hhl_*` (Q3): HHL with `noscale_res` (no scaling, C = Delta) or `scaled_kk` (Ruiz, kappa known), `nc<k>` clock qubits, `fb` / `nofb`
  with or without classical fallback, `aa<n>` emulated amplitude amplification with n accepted samples per setting, `blockonly` block test alone.
  Without fallback a rejected direction is passed to CVXOPT and counted in `n_solves_unaccepted_used`.
* `lcu_opt_*`, `lcu_cks_*` (Q5c): polynomial solver with the optimal polynomial or the CKS expansion at circuit precision `eps`;
  `lcu_oss_*`: polynomial solver on the null-space target; `hhl_*_ir60_nofb`: HHL with the same refinement budget. Q5e rows carry
  `sweep_samples` and `sweep_seed`.

## Files

* `benchmark_results.csv`: one row per optimisation run, with timing components, solve counts (`n_solves_accepted_direct`,
  `n_solves_accepted_refined`, `n_solves_unaccepted_used`, `n_fallback_total`), recomputed metrics (`pres_rec`, `dres_rec`, `relgap_valid`,
  `obj_rel_err`, `x_rel_err`), CVXOPT's own values (`*_cvx`), condition numbers, peak memory (`mem_*`) and the three success levels
  `accepted`, `pure_method_success` and `strict_method_success`.
* `iteration_logs.csv`: CVXOPT's per-iteration table joined with the factorisation at that iterate.
* `linear_solve_logs.csv`: one row per KKT solve with all three test values before and after refinement, fallback, Krylov iterations,
  timings and, for the quantum solvers, the `q_*` columns (qubits, kappa of the target, post-selection probability, readout, walk steps).
* `quantum_results.csv`: HHL on captured systems (Q1), Qiskit cross-check (Q2) and regularisation (Q4). `recon` is `ideal_statevector`,
  `shots` or `aa_shots`; `ir_round` is the refinement round; errors are given on the target and on the original system. In Q4 the
  corrector `classical_lu_regularized` is a classical diagnostic, not a quantum solve.
* `qls_comparison_hhl_vs_cks.csv` (Q5a), `cks_refinement.csv` (Q5b), `oss_target_comparison.csv` (Q5d), `quantum_resource_estimates.csv`.
* `spectrum_along_ipm.csv`, `deflation_bound.csv` (E7), `outer_refinement.csv` (E5), `dequantisation_probe.csv` (D1),
  `gpu_matvec_benchmark.csv` (E6, GPU runs only), `performance_profiles.csv`, `maros_meszaros_instances.csv`.
* `summary_report.md`, `failures.csv`, `experiment_status.csv`, `config.json`, `environment.json`, `instances.csv`, `plots/`,
  `instances_npz/` (tiny instances with their captured KKT systems, small dense instances, sparse instances as SciPy `.npz`).
"""
with open(os.path.join(RESULTS_DIR, "README.md"), "w") as f: f.write(README)

bundle = os.path.join(CONFIG["output_root"], f"results_bundle_{'gpu' if GPU_OK else 'cpu'}_{RUN_PROFILE}.zip")
if os.path.exists(bundle): os.remove(bundle)
with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as zf:
    for root, _, files in os.walk(RESULTS_DIR):
        if root.endswith("partial"): continue
        for fn in files:
            full = os.path.join(root, fn); zf.write(full, os.path.relpath(full, CONFIG["output_root"]))
print(f"bundle: {bundle} ({os.path.getsize(bundle) / 2**20:.2f} MB)")
try:
    from IPython.display import FileLink, display
    display(FileLink(os.path.relpath(bundle) if not bundle.startswith("/kaggle") else bundle))
except Exception:
    pass
print(f"\nTotal time: {time.time() - NOTEBOOK_T0:.0f}s"); print(REC.df("experiment_status")[["experiment", "status", "reason"]].to_string(index=False))
