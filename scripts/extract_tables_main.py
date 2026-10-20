"""Tables of the paper that rest on the main session M (results/tables/*_main.csv), and the replication data of sessions B-E.

    python scripts/extract_tables_main.py --main <bundle M> [--repB <bundle B>] [--repC <bundle C>] [--repD <bundle D>]
                                          [--repE <bundle E>] [--mmdiag mm_kappa_diag.csv] [--out results/tables]

Session M (GPU notebook, profile medium) supplies the scaling tables with several seeds, the GPU kernel, peak memory, the
Maros-Meszaros runs, performance profiles, the tolerance sweep, HHL on captured systems, the quantum runs inside coneqp,
the shot-seed sweep, the null-space target and the condition numbers before and after equilibration. Sessions B-E enter
the replication tables only. The tables that rest on session A are written by extract_tables_sessionA.py.
"""
import argparse
import os
import shutil

import numpy as np
import pandas as pd

from bundles import TABLES, root_of, read_table as rd

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--main", required=True)
for s_ in "BCDE":
    ap.add_argument(f"--rep{s_}")
ap.add_argument("--mmdiag", help="output of scripts/mm_kappa_diag.py")
ap.add_argument("--out", default=TABLES)
A = ap.parse_args(); T = A.out; os.makedirs(T, exist_ok=True)

M = root_of(A.main); B = rd(M, "benchmark_results.csv"); LS = rd(M, "linear_solve_logs.csv")
SPARSE = ["sparse_banded", "sparse_lowfill"]
E3 = B[(B.experiment == "E3_scaling") & B.family.isin(SPARSE)]

# scaling (all seeds x repeats)
for fam in SPARSE:
    x = E3[E3.family == fam]
    x.pivot_table(index="kkt_dim", columns="method", values="wall_method", aggfunc="median").to_csv(f"{T}/scaling_{fam}_main.csv")
    x.pivot_table(index="kkt_dim", columns="method", values="krylov_iters_total", aggfunc="median").to_csv(f"{T}/krylov_{fam}_main.csv")
rows = []
for fam in SPARSE:
    g0 = E3[E3.family == fam]; per = g0.groupby(["instance_id", "method"]).agg(w=("wall_method", "median"), N=("kkt_dim", "first"), seed=("seed", "first")).reset_index()
    base = per[per.method == "cvxopt_default"].set_index("instance_id").w; red = per[per.method == "custom_exact_reduced"].set_index("instance_id").w
    for m in ["cvxopt_default", "custom_exact_reduced", "gmres_stalelu_comb", "minres_fused_comb_cpu", "minres_fused_comb_gpu", "minres_fused_comb_auto", "minres_base_comb_cpu", "minres_base_comb_gpu", "oss_cg_comb"]:
        a = per[per.method == m].copy()
        if a.empty: continue
        a["vs_cvxopt"] = a.instance_id.map(base) / a.w; a["vs_reduced"] = a.instance_id.map(red) / a.w
        cpu = per[per.method == "minres_fused_comb_cpu"].set_index("instance_id").w; a["gpu_over_cpu"] = a.instance_id.map(cpu) / a.w
        for N, g in a.groupby("N"):
            rows.append({"family": fam, "method": m, "kkt_dim": int(N), "seeds": g.seed.nunique(), "wall_median": g.w.median(), "wall_min": g.w.min(), "wall_max": g.w.max(),
                         "speedup_vs_cvxopt_median": g.vs_cvxopt.median(), "speedup_vs_cvxopt_min": g.vs_cvxopt.min(), "speedup_vs_cvxopt_max": g.vs_cvxopt.max(),
                         "speedup_vs_reduced_median": g.vs_reduced.median(), "cpu_over_this_median": g.gpu_over_cpu.median(), "cpu_over_this_min": g.gpu_over_cpu.min(), "cpu_over_this_max": g.gpu_over_cpu.max()})
pd.DataFrame(rows).to_csv(f"{T}/scaling_seeds_main.csv", index=False)
fit = []
for fam in SPARSE:
    for m, g in E3[E3.family == fam].groupby("method"):
        s = g.groupby("kkt_dim").wall_method.median(); k = g.groupby("kkt_dim").krylov_iters_total.median()
        if len(s) >= 3: fit.append({"family": fam, "method": m, "wall_exponent": np.polyfit(np.log(s.index), np.log(s.values), 1)[0], "krylov_exponent": np.polyfit(np.log(k[k > 0].index), np.log(k[k > 0].values), 1)[0] if (k > 0).sum() >= 3 else np.nan,
                                    "N_max": int(s.index.max())})
pd.DataFrame(fit).to_csv(f"{T}/scaling_exponents_main.csv", index=False)
acc = E3.groupby("method").agg(runs=("run_id", "size"), accepted=("accepted", "mean"), strict=("strict_method_success", "mean"), fallbacks=("n_fallback_total", "sum"), relgap_max=("relgap_cvx", "max"), obj_err_max=("obj_rel_err", "max"),
                               cone_max=("cone_viol", "max")).reset_index(); acc.to_csv(f"{T}/scaling_accuracy_main.csv", index=False)
same = []
for fam in SPARSE:
    x = E3[E3.family == fam]; bi = x[x.method == "cvxopt_default"].set_index(["instance_id", "repeat"]).iterations
    for m, g in x.groupby("method"): same.append({"family": fam, "method": m, "same_iterations_as_cvxopt": float((g.set_index(["instance_id", "repeat"]).iterations == bi.reindex(g.set_index(["instance_id", "repeat"]).index)).mean())})
pd.DataFrame(same).to_csv(f"{T}/scaling_same_iterations_main.csv", index=False)

# GPU kernel
L = LS[LS.instance_id.astype(str).str.startswith("sparse_banded")]; rows = []
for m in ["minres_base_comb_cpu", "minres_fused_comb_cpu", "minres_base_comb_gpu", "minres_fused_comb_gpu"]:
    for kd, g in L[L.method == m].groupby("kkt_dim"): rows.append({"method": m, "kkt_dim": kd, "ms_per_iter": 1e3 * (g.t_solve + g.t_check).sum() / max(g.krylov_iters.sum(), 1)})
pd.DataFrame(rows).to_csv(f"{T}/kernel_per_iter_main.csv", index=False)
rd(M, "gpu_matvec_benchmark.csv").to_csv(f"{T}/gpu_matvec_main.csv", index=False)

# memory
x = B[B.experiment.isin(["E3_scaling", "E8_maros_meszaros"]) & B.mem_rss_increase_mb.notna()]
x.groupby(["experiment", "family", "method", "kkt_dim"]).agg(runs=("run_id", "size"), rss_increase_median=("mem_rss_increase_mb", "median"), rss_increase_max=("mem_rss_increase_mb", "max"), rss_peak_median=("mem_rss_peak_mb", "median"),
                                                           gpu_pool_peak_median=("mem_gpu_pool_peak_mb", "median"), gpu_pool_increase_median=("mem_gpu_pool_increase_mb", "median"), samples_median=("mem_samples", "median")).reset_index().to_csv(f"{T}/memory_main.csv", index=False)

# Maros-Meszaros
d8 = B[B.experiment == "E8_maros_meszaros"]; bo = d8[d8.method == "cvxopt_default"].set_index("instance_id"); ok = set(bo[bo.accepted].index)
rd(M, "maros_meszaros_instances.csv").to_csv(f"{T}/mm_instances.csv", index=False)
cols = ["instance_id", "kkt_dim", "nnz_kkt", "n", "m", "p", "method", "status", "accepted", "strict_method_success", "iterations", "wall_method", "n_fallback_total", "n_solves_unaccepted_used", "krylov_iters_total", "utilisation",
        "mem_rss_increase_mb", "mem_gpu_pool_peak_mb", "obj_rel_err", "relgap_cvx", "pres_rec", "dres_rec"]
d8[cols].to_csv(f"{T}/mm_runs.csv", index=False); rows = []
for m, g in d8.groupby("method"):
    g = g[g.instance_id.isin(ok)].set_index("instance_id"); r = (bo.wall_method.reindex(g.index) / g.wall_method).dropna()
    rows.append({"method": m, "instances": len(g), "accepted": g.accepted.mean(), "strict": g.strict_method_success.mean(), "instances_with_fallback": int((g.n_fallback_total > 0).sum()), "fallbacks": int(g.n_fallback_total.sum()),
                 "rejected_used": int(g.n_solves_unaccepted_used.sum()), "same_iterations": float((g.iterations == bo.iterations.reindex(g.index)).mean()), "wall_median": g.wall_method.median(), "wall_total": g.wall_method.sum(),
                 "speedup_vs_cvxopt_median": r.median(), "speedup_vs_cvxopt_min": r.min(), "speedup_vs_cvxopt_max": r.max(), "faster_than_cvxopt": int((r > 1).sum()), "krylov_median": g.krylov_iters_total.median(),
                 "utilisation_median": g.utilisation.median(), "obj_err_max": g.obj_rel_err.max(), "N_min": int(g.kkt_dim.min()), "N_max": int(g.kkt_dim.max())})
pd.DataFrame(rows).to_csv(f"{T}/mm_summary.csv", index=False)
w = d8[d8.method != "minres_fused_comb_gpu_mm"].pivot_table(index="instance_id", columns="method", values="wall_method", aggfunc="first")
w.idxmin(axis=1).value_counts().rename_axis("method").reset_index(name="fastest_on").to_csv(f"{T}/mm_fastest.csv", index=False)
if A.mmdiag and os.path.isfile(A.mmdiag) and os.path.abspath(A.mmdiag) != os.path.abspath(f"{T}/mm_kappa_diag.csv"):
    shutil.copyfile(A.mmdiag, f"{T}/mm_kappa_diag.csv")

# performance profiles (recomputed; the exact-LU contract check and the GPU-only subset run are excluded)
def profile(d, methods):
    g = d[d.method.isin(methods)].groupby(["instance_id", "method"]).agg(t=("wall_method", "median"), ok=("accepted", "all")).reset_index(); g.loc[~g.ok, "t"] = np.inf
    Tm = g.pivot_table(index="instance_id", columns="method", values="t", aggfunc="first").reindex(columns=[m for m in methods if m in set(g.method)])
    Tm = Tm.reindex(sorted(set(d.instance_id))).fillna(np.inf); R = Tm.div(Tm.min(axis=1), axis=0).replace([np.nan], np.inf)
    taus = np.unique(np.concatenate([[1.0], np.geomspace(1.0, 2 ** 12, 400)]))
    return pd.DataFrame([{"method": m, "tau": t, "rho": float((R[m] <= t).mean()), "instances": len(R)} for m in R.columns for t in taus]), R
PP = []; PR = []
for lab, d, meths in [("high-fill", E3[E3.family == "sparse_banded"], ["cvxopt_default", "custom_exact_reduced", "gmres_stalelu_comb", "minres_fused_comb_cpu", "minres_fused_comb_gpu", "minres_fused_comb_auto"]),
                      ("low-fill", E3[E3.family == "sparse_lowfill"], ["cvxopt_default", "custom_exact_reduced", "gmres_stalelu_comb", "minres_fused_comb_cpu", "minres_fused_comb_gpu", "minres_fused_comb_auto"]),
                      ("maros-meszaros", d8, ["cvxopt_default", "custom_exact_reduced", "gmres_stalelu_comb", "minres_fused_comb_cpu", "minres_fused_comb_switch_cpu"])]:
    pp, R = profile(d, meths); pp["set"] = lab; PP.append(pp)
    for m in R.columns:
        r = R[m]; PR.append({"set": lab, "method": m, "instances": len(r), "rho_1": float((r <= 1.0).mean()), "rho_2": float((r <= 2).mean()), "rho_4": float((r <= 4).mean()), "rho_16": float((r <= 16).mean()), "solved": float(np.isfinite(r).mean()),
                           "median_ratio": float(np.median(r[np.isfinite(r)])) if np.isfinite(r).any() else np.nan})
pd.concat(PP).to_csv(f"{T}/profiles_main.csv", index=False); pd.DataFrame(PR).to_csv(f"{T}/profiles_summary_main.csv", index=False)

# E4b tolerance sweep
t = B[B.experiment == "E4b_tolerance_sweep"].copy(); t["base"] = t.method.str.split("@").str[0]
t["model"] = np.where(t.base.str.contains("noise_iso"), "iso", np.where(t.base.str.contains("noise_block"), "block", "solver")); t["delta"] = t.base.str.extract(r"noise_(?:iso|block)_(.*)$")[0].astype(float)
e4 = B[B.experiment == "E4_noise"].copy(); e4["base"] = e4.method; e4["model"] = np.where(e4.base.str.contains("_iso_"), "iso", "block"); e4["delta"] = e4.base.str.split("_").str[-1].astype(float); e4["ipm_tol_level"] = 1e-7
tt = pd.concat([t, e4], ignore_index=True)
tt.groupby(["base", "model", "delta", "ipm_tol_level"], dropna=False).agg(runs=("run_id", "size"), instances=("instance_id", "nunique"), accepted=("accepted", "mean"), its_median=("iterations", "median")).reset_index().to_csv(f"{T}/tolerance_sweep_main.csv", index=False)

# quantum: integrated runs (Q3 + Q5c incl. null-space)
q = B[B.experiment.isin(["Q3_integrated", "Q5_integrated_cks"])].copy()
_lq = LS[LS.run_id.isin(q.run_id)]
def _cause(g):
    if "q_status" in g and (g.q_status.astype(str) == "degree_cap").any(): return "degree_cap"
    if "q_shot_status" in g and (g.q_shot_status.astype(str) == "postselection_failed").any(): return "readout_failed"
    return ""
q["fail_cause"] = q.run_id.map(_lq.groupby("run_id").apply(_cause)).fillna(""); q.to_csv(f"{T}/integrated_runs_main.csv", index=False)
l = LS[LS.run_id.isin(q.run_id)]; rows = []
for (e, m), g in q.groupby(["experiment", "method"]):
    lm = l[(l.method == m) & l.run_id.isin(g.run_id)]; lq = lm[lm.q_degree.notna()] if "q_degree" in lm else pd.DataFrame()
    sam = 2 * lq.q_eff_shots.fillna(0) * (1 + lq.n_refine) if len(lq) else pd.Series(dtype=float); qs = (sam * lq.q_degree / np.sqrt(lq.q_p_postselect.clip(lower=1e-300))).sum() if len(lq) else np.nan
    rows.append({"experiment": e, "method": m, "runs": len(g), "strict": int(g.strict_method_success.sum()), "cvxopt_optimal": int((g.status == "optimal").sum()), "strict_and_optimal": int((g.strict_method_success & (g.status == "optimal")).sum()),
                 "rejected_used": int(g.n_solves_unaccepted_used.sum()), "refine_rounds": int(g.n_refine_total.sum()), "kkt_solves": int(g.n_solve_calls.sum()), "fallbacks": int(g.n_fallback_total.sum()), "ipm_its_median": g.iterations.median(),
                 "steps_median": lq.q_degree.median() if len(lq) else np.nan, "steps_max": lq.q_degree.max() if len(lq) else np.nan, "qubits_max": lq.q_n_qubits.max() if len(lq) else np.nan, "psucc_min": lq.q_p_postselect.min() if len(lq) else np.nan,
                 "samples": sam.sum() if len(lq) else np.nan, "queries_with_AA": qs if (len(lq) and sam.sum() > 0) else np.nan, "kappa_target_median": lq.q_kappa_target.median() if len(lq) else np.nan,
                 "steps_augmented_equiv_median": lq.q_degree_augmented_equiv.median() if ("q_degree_augmented_equiv" in lq and len(lq)) else np.nan, "qubits_augmented_equiv_max": lq.q_n_qubits_augmented_equiv.max() if ("q_n_qubits_augmented_equiv" in lq and len(lq)) else np.nan,
                 "kappa_augmented_median": lq.q_kappa_augmented.median() if ("q_kappa_augmented" in lq and len(lq)) else np.nan})
pd.DataFrame(rows).to_csv(f"{T}/integrated_summary_main.csv", index=False)

# Q5e shot-seed sweep
q = B[B.experiment == "Q5e_shot_sweep"]; lq = LS[LS.run_id.isin(q.run_id)].merge(q[["run_id", "sweep_samples", "sweep_seed"]], on="run_id"); rows = []
for spr, g in q.groupby("sweep_samples"):
    ps = g.groupby("sweep_seed").strict_method_success.mean(); x = lq[(lq.sweep_samples == spr) & lq.q_degree.notna()]; sam = 2 * x.q_eff_shots.fillna(0) * (1 + x.n_refine); qq = sam * x.q_degree / np.sqrt(x.q_p_postselect.clip(lower=1e-300))
    pr = pd.DataFrame({"run_id": x.run_id, "s": sam, "q": qq}).groupby("run_id").sum()
    rows.append({"samples_per_round": int(spr), "runs": len(g), "instances": g.instance_id.nunique(), "seeds": g.sweep_seed.nunique(), "strict": int(g.strict_method_success.sum()), "strict_rate": g.strict_method_success.mean(),
                 "strict_rate_min_seed": ps.min(), "strict_rate_max_seed": ps.max(), "cvxopt_optimal": int((g.status == "optimal").sum()), "rejected_used": int(g.n_solves_unaccepted_used.sum()), "ipm_its_median": g.iterations.median(),
                 "refine_rounds_median": g.n_refine_total.median(), "samples_per_run_median": pr.s.median(), "queries_per_run_median": pr.q.median(), "psucc_min": x.q_p_postselect.min()})
pd.DataFrame(rows).to_csv(f"{T}/shots_main.csv", index=False)
q.pivot_table(index="instance_id", columns="sweep_samples", values="strict_method_success", aggfunc="sum").to_csv(f"{T}/shots_per_instance_main.csv")
acc = LS[LS.accepted & LS.run_id.isin(set(B[B.experiment.isin(["Q5e_shot_sweep", "Q5_integrated_cks"])].run_id))].merge(B[["run_id", "experiment", "sweep_samples", "iterations", "strict_method_success"]], on="run_id")
acc["readout"] = np.where(acc.method.str.contains("ideal"), "ideal", np.where(acc.sweep_samples.notna(), "S=" + acc.sweep_samples.fillna(0).astype(int).astype(str), "other"))
acc = acc[acc.method.str.startswith("lcu_opt") | (acc.experiment == "Q5e_shot_sweep")]
acc.groupby(["instance_id", "readout"]).agg(solves=("run_id", "size"), ipm_its=("iterations", "median"), beta_median=("block_ratio_max_final", "median"), beta_q90=("block_ratio_max_final", lambda s: s.quantile(.9)),
                                             omega_median=("backward_err_final", "median")).reset_index().to_csv(f"{T}/shots_accepted_accuracy_main.csv", index=False)

# null-space (OSS) target
o = rd(M, "oss_target_comparison.csv")
if len(o):
    o["stage"] = np.where(o.groupby("instance_id").factor_index.transform("min") == o.factor_index, "first", np.where(o.groupby("instance_id").factor_index.transform("max") == o.factor_index, "last", "middle"))
    o.to_csv(f"{T}/oss_q5d_main.csv", index=False)
qa = rd(M, "qls_comparison_hhl_vs_cks.csv"); qa[qa.solver == "lcu_oss"].to_csv(f"{T}/oss_q5a_main.csv", index=False)
x = B[B.experiment.isin(["E2_main", "E3_scaling"]) & (B.repeat == 0)]
oc = x[x.method == "oss_cg_comb"].groupby("instance_id").agg(family=("family", "first"), kkt_dim=("kkt_dim", "first"), n=("n", "first"), p=("p", "first"), cg=("krylov_iters_total", "median"), strict=("strict_method_success", "all"),
                                                            kappa_nullspace=("kappa_reduced_max", "max"), wall_cg=("wall_method", "median"))
oc = oc.join(x[x.method == "minres_fused_comb_cpu"].groupby("instance_id").agg(minres=("krylov_iters_total", "median"), wall_minres=("wall_method", "median")))
oc = oc.join(x[x.method == "custom_exact"].groupby("instance_id").kappa_scaled_max.max().rename("kappa_augmented")); oc["ratio"] = oc.minres / oc.cg.replace(0, np.nan)
oc.reset_index().to_csv(f"{T}/oss_classical_main.csv", index=False)

# E5 (scaled kappa of the inner systems)
rd(M, "outer_refinement.csv").to_csv(f"{T}/outer_E5_main.csv", index=False)

# replication across sessions
REP = []; SESS = {}
def per_instance(r):
    b = rd(r, "benchmark_results.csv"); e = b[(b.experiment == "E3_scaling") & b.family.isin(SPARSE)]
    return e.groupby(["family", "instance_id", "method"]).agg(w=("wall_method", "median"), N=("kkt_dim", "first")).reset_index(), b
main_pi, _ = per_instance(M)
for lab, z in [("B", A.repB), ("C", A.repC), ("D", A.repD), ("E", A.repE)]:
    if not z: continue
    r = root_of(z); pi, b = per_instance(r); SESS[lab] = b
    j = pi.merge(main_pi, on=["family", "instance_id", "method", "N"], suffixes=("_s", "_m")); j["ratio"] = j.w_s / j.w_m; j["session"] = lab; REP.append(j)
if REP:
    rep = pd.concat(REP, ignore_index=True); rep.to_csv(f"{T}/replication_main.csv", index=False)
    meths = ["cvxopt_default", "custom_exact_reduced", "gmres_stalelu_comb", "minres_fused_comb_cpu", "minres_fused_comb_gpu", "minres_base_comb_cpu", "minres_base_comb_gpu"]
    rep[rep.method.isin(meths)].groupby(["session", "family", "method"]).agg(pairs=("ratio", "size"), median=("ratio", "median"), q10=("ratio", lambda s: s.quantile(.1)), q90=("ratio", lambda s: s.quantile(.9)),
                                                                             min=("ratio", "min"), max=("ratio", "max")).reset_index().to_csv(f"{T}/replication_summary_main.csv", index=False)
    rows = []
    for lab, b in SESS.items():
        for fam in SPARSE:
            e = b[(b.experiment == "E3_scaling") & (b.family == fam)]
            if e.empty: continue
            per = e.groupby(["instance_id", "method"]).agg(w=("wall_method", "median"), N=("kkt_dim", "first")).reset_index(); base = per[per.method == "cvxopt_default"].set_index("instance_id").w
            for m in ["minres_fused_comb_cpu", "minres_fused_comb_gpu", "custom_exact_reduced"]:
                a = per[per.method == m].copy(); a["s"] = a.instance_id.map(base) / a.w
                for N, g in a.groupby("N"): rows.append({"session": lab, "family": fam, "method": m, "kkt_dim": int(N), "speedup_vs_cvxopt_median": g.s.median()})
            g = per[per.method.isin(["minres_fused_comb_cpu", "minres_fused_comb_gpu"])].pivot_table(index=["instance_id", "N"], columns="method", values="w").reset_index()
            if {"minres_fused_comb_cpu", "minres_fused_comb_gpu"} <= set(g.columns):
                for N, gg in g.groupby("N"): rows.append({"session": lab, "family": fam, "method": "gpu_over_cpu", "kkt_dim": int(N), "speedup_vs_cvxopt_median": (gg.minres_fused_comb_cpu / gg.minres_fused_comb_gpu).median()})
        q = b[b.experiment == "Q5_integrated_cks"]
        for m, g in q.groupby("method"): rows.append({"session": lab, "family": "quantum", "method": m, "kkt_dim": len(g), "speedup_vs_cvxopt_median": int(g.strict_method_success.sum())})
    pd.DataFrame(rows).to_csv(f"{T}/replication_crossovers_main.csv", index=False)

    # further replicated quantities: Maros-Meszaros, tolerance sweep and shot-seed sweep, for M and every replication session
    rows = []
    for lab, b in [("M", B)] + list(SESS.items()):
        d = b[b.experiment == "E8_maros_meszaros"]
        if len(d):
            w_ = d[d.method != "minres_fused_comb_gpu_mm"].pivot_table(index="instance_id", columns="method", values="wall_method", aggfunc="first")
            rows.append({"session": lab, "quantity": "mm_ldl_fastest", "value": int((w_.idxmin(axis=1) == "cvxopt_default").sum()), "of": len(w_)})
            mr = d[d.method == "minres_fused_comb_cpu"]; rows.append({"session": lab, "quantity": "mm_minres_fallback_instances", "value": int((mr.groupby("instance_id").n_fallback_total.max() > 0).sum()), "of": mr.instance_id.nunique()})
        t_ = b[b.experiment == "E4b_tolerance_sweep"].copy()
        if len(t_) and "ipm_tol_level" in t_:
            t_["base"] = t_.method.str.split("@").str[0]; iso = t_[t_.base.str.contains("noise_iso")].copy(); iso["delta"] = iso.base.str.extract(r"noise_iso_(.*)$")[0].astype(float)
            for tau, g in iso.groupby("ipm_tol_level"):
                ok_ = g.groupby("delta").accepted.mean(); full = ok_[ok_ >= 0.999]
                rows.append({"session": lab, "quantity": f"tol_largest_full_{tau:.0e}", "value": float(full.index.max()) if len(full) else np.nan, "of": g.instance_id.nunique()})
        s_ = b[b.experiment == "Q5e_shot_sweep"] if "sweep_samples" in b else None
        for spr, g in (s_.groupby("sweep_samples") if s_ is not None else []): rows.append({"session": lab, "quantity": f"shots_strict_{int(spr)}", "value": int(g.strict_method_success.sum()), "of": len(g)})
    pd.DataFrame(rows).to_csv(f"{T}/replication_extra_main.csv", index=False)

# HHL on captured systems (first solve, no refinement)
QR = rd(M, "quantum_results.csv"); q1 = QR[(QR.experiment == "Q1_captured") & (QR.ir_round == 0)]
q1[q1.recon == "ideal_statevector"].pivot_table(index=["scaling", "reciprocal_mode"], columns="n_clock", values="rel_err_original", aggfunc="median").to_csv(f"{T}/hhl_q1_ideal_main.csv")
sh = q1[q1.recon.isin(["shots", "aa_shots"])]; sh.pivot_table(index=["recon", "scaling", "reciprocal_mode", "n_clock"], columns="shots_per_setting", values="rel_err_original", aggfunc="median").to_csv(f"{T}/hhl_q1_shots_main.csv")
q1.groupby(["instance_id", "factor_index"]).agg(kkt_dim=("kkt_dim", "first"), kappa=("kappa_original", "first")).reset_index().to_csv(f"{T}/hhl_q1_systems_main.csv", index=False)

# condition number before and after equilibration (largest over the IPM run)
ce = B[(B.method == "custom_exact") & B.kappa_max.notna() & B.kappa_scaled_max.notna()]
eq = ce.groupby("instance_id").agg(family=("family", "first"), kkt_dim=("kkt_dim", "first"), kappa_unscaled=("kappa_max", "max"), kappa_equilibrated=("kappa_scaled_max", "max")).reset_index()
if A.mmdiag and os.path.isfile(A.mmdiag):
    dg = pd.read_csv(A.mmdiag); eq = pd.concat([eq, pd.DataFrame({"instance_id": dg.instance, "family": "maros_meszaros", "kkt_dim": dg.N, "kappa_unscaled": dg.kappa_max, "kappa_equilibrated": dg.kappa_scaled_max})], ignore_index=True)
eq["reduction"] = eq.kappa_unscaled / eq.kappa_equilibrated; eq.to_csv(f"{T}/equil_main.csv", index=False)

# classical work per KKT solve: MINRES iterations vs. refinement rounds of the quantum solver
x = B[B.experiment.isin(["E2_main", "E3_scaling"]) & (B.method == "minres_fused_comb_cpu")].copy(); x["per_solve"] = x.krylov_iters_total / x.n_solve_calls
rows = [{"solver": "minres", "family": f, "kkt_dim": int(N), "per_solve": g.per_solve.median(), "per_solve_min": g.per_solve.min(), "per_solve_max": g.per_solve.max(), "instances": g.instance_id.nunique()} for (f, N), g in x.groupby(["family", "kkt_dim"])]
qi = B[(B.experiment == "Q5_integrated_cks") & B.method.isin(["lcu_opt_eps0.3_ideal_ir_nofb", "lcu_opt_eps0.3_aa500_ir_nofb", "lcu_oss_eps0.3_ideal_ir_nofb"])].copy()
qi["per_solve"] = 1 + qi.n_refine_total / qi.n_solve_calls
mn = x.groupby("instance_id").per_solve.median()
for _, r in qi.iterrows(): rows.append({"solver": r.method, "family": r.family, "instance_id": r.instance_id, "kkt_dim": int(r.kkt_dim), "per_solve": r.per_solve, "minres_same_instance": mn.get(r.instance_id, np.nan), "strict": bool(r.strict_method_success)})
pd.DataFrame(rows).to_csv(f"{T}/rounds_vs_minres_main.csv", index=False)
print("tables written to", T)
