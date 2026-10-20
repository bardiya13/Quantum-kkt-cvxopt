# E8: Maros-Meszaros QPs.
mm_insts = sorted(MM_INSTANCES.values(), key=lambda i: i["kkt_dim"]) if MM_CFG.get("enabled", True) else []
if not mm_insts:
    REC.status("E8_maros_meszaros", "skipped", "no Maros-Meszaros instances loaded")
else:
    t_mm = time.perf_counter(); n_base = 0
    for inst in mm_insts:                                   # 1) reference run: unmodified coneqp; its solution becomes the reference for obj_rel_err / x_rel_err
        rec, sol, _ = run_instance(inst, SPECS["cvxopt_default"], repeat=0, experiment="E8_maros_meszaros"); n_base += 1
        if sol is not None and rec["status"] == "optimal" and rec["accepted"]:
            inst.update({"f_star": rec["pobj"], "x_star": to_np(sol["x"]), "known_solution": True, "solution_source": "cvxopt_default"}); rec["obj_rel_err"] = 0.0; rec["x_rel_err"] = 0.0; rec["solution_source"] = "cvxopt_default"
        else:
            inst.update({"known_solution": False, "solution_source": f"none (baseline {rec['status']}, accepted={rec['accepted']})"})
        print(f"  {inst['id']}: baseline status={rec['status']} iters={rec['iterations']} accepted={rec['accepted']} ({rec['accept_fail_reasons'] or 'ok'}) wall={rec['wall_method']:.2f}s pobj={rec.get('pobj', float('nan')):.6g}")
    others = [m for m in MM_CFG["methods"] if m != "cvxopt_default"]
    t_o = run_methods(mm_insts, others, "E8_maros_meszaros", repeats=1, budget=MM_CFG["budget_s"], predictive=False)
    REC.status("E8_maros_meszaros", "done", f"{len(mm_insts)} instances; baseline + {len(others)} methods in {time.perf_counter() - t_mm:.0f}s")
    REC.flush(final=False, min_interval=0)
    d8 = REC.df("benchmark_results"); d8 = d8[d8.experiment == "E8_maros_meszaros"]
    print("\nE8 by method:"); print(d8.groupby("method").agg(runs=("run_id", "size"), optimal=("status", lambda s_: int((s_ == "optimal").sum())), accepted=("accepted", "mean"), strict=("strict_method_success", "mean"),
                                                            iters=("iterations", "median"), wall=("wall_method", "median"), fallbacks=("n_fallback_total", "sum"), krylov=("krylov_iters_total", "median"), peak_mem_mb=("mem_rss_increase_mb", "median")).round(3).to_string())
    base_ok = set(d8[(d8.method == "cvxopt_default") & d8.accepted].instance_id)
    print(f"\ninstances the unmodified solver accepts: {len(base_ok)} of {len(mm_insts)}")
    piv = d8[d8.instance_id.isin(base_ok)].pivot_table(index="instance_id", columns="method", values="wall_method", aggfunc="first")
    if len(piv): print("wall time [s] on those instances:"); print(piv.round(2).to_string())
