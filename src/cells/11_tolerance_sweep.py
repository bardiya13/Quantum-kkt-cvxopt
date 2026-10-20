# E4b: tolerated direction error as a function of the IPM tolerance level.
TS = CONFIG["tolerance_sweep"]
ts_insts = sorted([i for i in INSTANCES.values() if i["family"] in TS["families"] and i["n"] <= TS["n_max"]], key=lambda i: (i["n"], i["id"]))[:TS["max_instances"]]
def sweep_specs(tau):
    cvx_o = {"abstol": tau, "reltol": 10 * tau, "feastol": tau}; acc_o = {"pres": 10 * tau, "dres": 10 * tau, "relgap": 100 * tau, "obj_rel_err": 100 * tau, "cone_viol": 1e-8}
    out = []
    for d in TS["noise_iso"]:
        out.append({"name": f"exact_noise_iso_{d:.0e}@tau{tau:.0e}", "kind": "noise", "noise_model": "iso", "delta": d, "fallback": False, "max_refine": 0, "criterion": "relres"})
    for d in TS["noise_block"]:
        out.append({"name": f"exact_noise_block_{d:.0e}@tau{tau:.0e}", "kind": "noise", "noise_model": "block", "delta": d, "fallback": False, "max_refine": 0, "criterion": "block"})
    for m in TS["extra_methods"]:
        if m in SPECS: out.append(dict(SPECS[m], name=f"{m}@tau{tau:.0e}"))
    for s in out: s.update({"cvxopt_overrides": cvx_o, "accept_overrides": acc_o, "tol_level": tau, "base_method": s["name"].split("@")[0]})
    return out
t_ts = time.perf_counter(); n_ts = 0
for tau in TS["levels"]:
    for spec in sweep_specs(tau):
        for inst in ts_insts:
            if spec.get("requires_gpu") and not GPU_OK: continue
            run_instance(inst, spec, repeat=0, experiment="E4b_tolerance_sweep"); n_ts += 1
REC.status("E4b_tolerance_sweep", "done" if n_ts else "skipped", f"{n_ts} runs ({len(ts_insts)} instances x {len(TS['levels'])} tolerance levels) in {time.perf_counter() - t_ts:.0f}s")
REC.flush(final=False, min_interval=0)
df = REC.df("benchmark_results"); tsd = df[df.experiment == "E4b_tolerance_sweep"].copy()
if len(tsd):
    tsd["base"] = tsd.method.str.split("@").str[0]
    print("E4b - fraction accepted (thresholds scaled with tau) by inner solver x tolerance level tau:")
    print(tsd.pivot_table(index="base", columns="ipm_tol_level", values="accepted", aggfunc="mean").round(3).to_string())
    print("\nmedian IPM iterations:"); print(tsd.pivot_table(index="base", columns="ipm_tol_level", values="iterations", aggfunc="median").to_string())
