# Classical experiments: E1 contract check, E2 acceptance-test ablation, E3 scaling, E4 direction-error sensitivity.
def run_methods(insts, method_names, experiment, repeats=None, compute_cond_for=("custom_exact",), capture_ids=(), budget=None, predictive=None):
    """Runs methods x instances. Budgets are per method: a method that exceeds `budget` seconds on an instance is skipped only for
    larger instances of the same family; other methods are unaffected. The exact LU reference is skipped above limits.exact_max_dim.
    A running CVXOPT call cannot be interrupted safely (no subprocess isolation is used), so budgets act between runs and every run is flushed."""
    repeats = CONFIG["repeats"] if repeats is None else repeats
    budget = CONFIG["limits"]["max_run_seconds"] if budget is None else budget
    exact_max = CONFIG["limits"].get("exact_max_dim", 10 ** 9)
    gpu_only = [n_ for n_ in method_names if SPECS.get(n_, {}).get("requires_gpu")]
    if gpu_only and not GPU_OK:
        REC.status(f"{experiment}_gpu_methods", "skipped", f"{gpu_only}: device={CONFIG['device']}, GPU usable={GPU_OK} {GPU_STATUS.get('reason', '')}".strip())
        method_names = [n_ for n_ in method_names if n_ not in gpu_only]
    t_exp = time.perf_counter(); n_runs = 0; skipped = []; slow = {}; hist = {}          # (family, method) -> kkt_dim at which it exceeded the budget
    for inst in sorted(insts, key=lambda i: (i["family"], i["kkt_dim"])):
        for name in method_names:
            if name not in SPECS:
                REC.status(f"{experiment}_{name}", "skipped", "method not defined in this profile"); continue
            spec = SPECS[name]
            if spec.get("requires_p0") and inst["p"] > 0: continue
            if spec.get("requires_dense_P") and inst["n"] > 2000: continue
            if spec.get("max_n") and inst["n"] > spec["max_n"]: continue                       # dense null-space basis only up to oss_max_n
            if spec.get("min_kkt_dim") and inst["kkt_dim"] < spec["min_kkt_dim"]: continue
            if spec["kind"] == "exact" and inst["kkt_dim"] > exact_max:
                skipped.append((inst["id"], name, f"exact LU skipped above exact_max_dim={exact_max}")); continue
            if (inst["family"], name) in slow and inst["kkt_dim"] > slow[(inst["family"], name)]:
                skipped.append((inst["id"], name, f"method exceeded {budget}s at kkt_dim {slow[(inst['family'], name)]}")); continue
            hk = hist.get((inst["family"], name), [])
            if (CONFIG["limits"].get("predictive_budget", True) if predictive is None else predictive) and hk and inst["kkt_dim"] > hk[-1][0] and hk[-1][1] > 1.0:
                # power law through the last two sizes (exponent clipped to [1, 3.5]; 2.0 with one point): skip before spending the time
                ex = 2.0
                if len(hk) >= 2 and hk[-2][1] > 0 and hk[-1][0] > hk[-2][0]:
                    ex = float(np.clip(np.log(hk[-1][1] / hk[-2][1]) / np.log(hk[-1][0] / hk[-2][0]), 1.0, 3.5))
                pred = hk[-1][1] * (inst["kkt_dim"] / hk[-1][0]) ** ex
                if pred > CONFIG["limits"].get("budget_margin", 1.5) * budget:
                    skipped.append((inst["id"], name, f"predicted {pred:.0f}s (= {hk[-1][1]:.0f}s at kkt_dim {hk[-1][0]} x size^{ex:.2f}) > {CONFIG['limits'].get('budget_margin', 1.5)} x {budget}s budget")); continue
            for rep in range(repeats):
                rec, _, _ = run_instance(inst, spec, repeat=rep, compute_cond=(name in compute_cond_for and rep == 0),
                                         capture=(inst["id"] in capture_ids and name == "custom_exact" and rep == 0), cond_scaled=(name in compute_cond_for and rep == 0), experiment=experiment)
                n_runs += 1
                if rep == 0:
                    hl = hist.setdefault((inst["family"], name), [])
                    if hl and hl[-1][0] == inst["kkt_dim"]: hl[-1] = (inst["kkt_dim"], max(hl[-1][1], rec["wall_total"]))     # several seeds per size: keep the slowest
                    else: hl.append((inst["kkt_dim"], rec["wall_total"]))
                if rec["wall_total"] > budget: slow[(inst["family"], name)] = min(slow.get((inst["family"], name), 10 ** 9), inst["kkt_dim"])
    dt = time.perf_counter() - t_exp
    for inst_id, name, why in skipped:
        REC.status(f"{experiment}_{inst_id}_{name}", "skipped", why)
    print(f"[{experiment}] {n_runs} runs in {dt:.1f}s" + (f"; {len(skipped)} (instance, method) pairs skipped (per-method budget / size guard)" if skipped else ""))
    return dt

tiny_ids = CONFIG["capture"]["instances"]; tiny = [INSTANCES[i] for i in tiny_ids]
run_methods(tiny, ["cvxopt_default", "custom_exact"], "E1_contract_tiny", repeats=1, capture_ids=tiny_ids)
df = REC.df("benchmark_results")
for inst in tiny:
    sub = df[(df.instance_id == inst["id"]) & (df.experiment == "E1_contract_tiny")]
    base, cust = sub[sub.method == "cvxopt_default"].iloc[0], sub[sub.method == "custom_exact"].iloc[0]
    ok = (base["iterations"] == cust["iterations"]) and abs(base["pobj"] - cust["pobj"]) <= 1e-8 * max(1, abs(base["pobj"]))
    print(f"  {inst['id']}: baseline iters={base['iterations']} custom iters={cust['iterations']} |pobj diff|={abs(base['pobj']-cust['pobj']):.2e} obj_rel_err={base['obj_rel_err']:.2e} -> {'OK' if ok else 'MISMATCH'}")
    assert ok, "custom exact solver does not reproduce the baseline"
REC.status("E1_contract_tiny", "done"); print("captured KKT systems:", {k: len(v) for k, v in CAPTURED.items()})

main_insts = [i for i in INSTANCES.values() if i["family"] in ("dense_kappa", "dense_noeq")]
ESSENTIAL = ["cvxopt_default", "custom_exact", "custom_exact_reduced", "gmres_stalelu_comb", "minres_relres_1e-06", "minres_be_1e-12", "minres_comb_0.1_1e-10",
             "minres_base_comb_cpu", "minres_base_comb_ws_cpu", "minres_base_comb_ws_switch_cpu", "minres_base_comb_gpu", "minres_base_comb_ws_gpu",
             "minres_fused_comb_cpu", "minres_fused_comb_floor8_cpu", "minres_fused_comb_gpu", "minres_fused_comb_auto", "lsqr_sqrt_comb", "cg_reduced_be_1e-10", "oss_cg_comb"]
main_methods = [n for n in SPECS if not n.startswith("exact_noise")] if CONFIG["inexact"]["full_ablation"] else [n for n in ESSENTIAL if n in SPECS]
t_main = run_methods(main_insts, main_methods, "E2_main", compute_cond_for=("custom_exact", "oss_cg_comb"))
REC.status("E2_main", "done", f"{len(main_insts)} instances x {len(main_methods)} methods x {CONFIG['repeats']} repeats in {t_main:.0f}s")

scale_insts = [i for i in INSTANCES.values() if i["family"] in ("sparse_banded", "sparse_scale", "sparse_lowfill")]
# minres_base_* is the baseline kernel for the comparison with minres_fused_*. SciPy MINRES and the warm-start variants are not
# run at scale: warm starts save at most 4 % of the iterations on the sparse families and cost iterations on the dense ones.
scale_methods = ["cvxopt_default", "custom_exact", "custom_exact_reduced", "gmres_stalelu_comb", "minres_base_comb_cpu", "minres_fused_comb_cpu", "minres_fused_comb_floor8_cpu",
                 "minres_base_comb_gpu", "minres_fused_comb_gpu", "minres_fused_comb_auto", "oss_cg_comb"]       # oss_cg_comb: n <= oss_max_n
t_scale = run_methods(scale_insts, scale_methods, "E3_scaling", compute_cond_for=("custom_exact", "oss_cg_comb"))
REC.status("E3_scaling", "done", f"{len(scale_insts)} instances in {t_scale:.0f}s")

ns = sorted({j["n"] for j in main_insts if j["family"] == "dense_kappa"})
noise_insts = [i for i in main_insts if i["family"] == "dense_kappa" and i["n"] == ns[min(1, len(ns) - 1)]]
noise_methods = [n for n in SPECS if n.startswith("exact_noise")]
t_noise = run_methods(noise_insts, noise_methods, "E4_noise", repeats=1)
REC.status("E4_noise", "done", f"{len(noise_insts)} instances x {len(noise_methods)} noise settings")

REC.flush(final=False, min_interval=0)
df = REC.df("benchmark_results")
print("\nAcceptance by method (classical experiments):")
print(df[df.experiment.isin(["E1_contract_tiny", "E2_main", "E3_scaling"])].groupby("method").agg(runs=("run_id", "size"), accepted=("accepted", "mean"), pure=("pure_method_success", "mean"),
      iters=("iterations", "median"), wall_med=("wall_method", "median"), fallback=("n_fallback_total", "sum"), refine=("n_refine_total", "sum"), krylov=("krylov_iters_total", "median")).round(4).to_string())
