# Summary report: one table per experiment, computed from the result tables of this run, written to results/summary_report.md.
DF = REC.df("benchmark_results"); LS = REC.df("linear_solve_logs"); QD = REC.df("quantum_results"); ST = REC.df("experiment_status")
L = []
def add(s=""): L.append(s)

def _cell(v):
    if isinstance(v, (bool, np.bool_)): return str(bool(v))
    if isinstance(v, (int, np.integer)): return str(int(v))
    if isinstance(v, (float, np.floating)): return "" if np.isnan(v) else f"{v:.3g}"
    return str(v)

def table(df):
    if df is None or len(df) == 0: return "(no data)\n"
    df = df.reset_index() if not isinstance(df.index, pd.RangeIndex) else df
    cols = [str(c) for c in df.columns]
    rows = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] + ["| " + " | ".join(_cell(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(rows) + "\n"

def run_summary(d):
    return d.groupby("method").agg(runs=("run_id", "size"), accepted=("accepted", "mean"), fallback_free=("pure_method_success", "mean"), strict=("strict_method_success", "mean"),
                                   ipm_iters=("iterations", "median"), krylov_iters=("krylov_iters_total", "median"), fallbacks=("n_fallback_total", "sum"),
                                   wall_s=("wall_method", "median")).reindex([m for m in order_methods(list(d.method.unique()))])

def section(title, fn):
    add(f"## {title}\n")
    try: add(fn())
    except Exception as e: add(f"(not available: {type(e).__name__}: {e})\n")

add("# Summary report\n")
add(f"Version {CODE_VERSION}, profile `{CONFIG['run_profile']}`, device `{'gpu' if GPU_OK else 'cpu'}`, config hash `{CONFIG_HASH}`, seed {GLOBAL_SEED}, "
    f"{time.strftime('%Y-%m-%d %H:%M')}. {len(DF)} optimisation runs, {len(LS)} KKT solves, {len(QD)} quantum records. "
    f"CPU: {ENV['hardware'].get('cpu_model', '?')} ({ENV['hardware'].get('cpu_count')} cores); GPU: {GPU_STATUS.get('name') or 'none'}.\n")
add("Acceptance of the final iterate: " + ", ".join(f"{k} <= {v:g}" for k, v in CONFIG["accept"].items()) + ". "
    "`fallback_free` = accepted without classical fallback; `strict` = in addition every direction passed the acceptance test.\n")

section("Experiment status", lambda: table(ST[["experiment", "status", "reason"]]))
for exp, title in [("E1_contract_tiny", "E1, contract check"), ("E2_main", "E2, acceptance-test ablation"), ("E8_maros_meszaros", "E8, Maros-Meszaros QPs"),
                   ("E4_noise", "E4, injected direction error"), ("Q3_integrated", "Q3, HHL inside coneqp"),
                   ("Q5_integrated_cks", "Q5c, polynomial solver inside coneqp (no fallback)"), ("S1_sdplib", "S1, SDPLIB")]:
    if (DF.experiment == exp).any(): section(title, lambda exp=exp: table(run_summary(DF[DF.experiment == exp])))

def _e3():
    d = DF[DF.experiment == "E3_scaling"]; out = []
    for fam, g in d.groupby("family"):
        out.append(f"Median wall time [s], family `{fam}` (rows: method, columns: KKT dimension; fallback-free share in the last column)\n")
        piv = g.pivot_table(index="method", columns="kkt_dim", values="wall_method", aggfunc="median")
        piv["fallback_free"] = g.groupby("method").pure_method_success.mean(); out.append(table(piv.reindex([m for m in order_methods(list(piv.index))])))
    return "\n".join(out)
section("E3, scaling", _e3)

def _e4b():
    d = DF[DF.experiment == "E4b_tolerance_sweep"].copy(); d["base"] = d.method.str.split("@").str[0]
    return table(d.pivot_table(index="base", columns="ipm_tol_level", values="accepted", aggfunc="mean"))
if (DF.experiment == "E4b_tolerance_sweep").any(): section("E4b, share of accepted runs by IPM tolerance level", _e4b)

def _q5e():
    d = DF[DF.experiment == "Q5e_shot_sweep"]
    return table(d.groupby("sweep_samples").agg(runs=("run_id", "size"), strict=("strict_method_success", "mean"), accepted=("accepted", "mean"),
                                                ipm_iters=("iterations", "median"), refinement_rounds=("n_refine_total", "median")))
if (DF.experiment == "Q5e_shot_sweep").any(): section("Q5e, finite-shot polynomial solver over sampling seeds", _q5e)

def _q1():
    q = QD[(QD.experiment == "Q1_captured") & (QD.status == "ok") & (QD.recon == "ideal_statevector") & (QD.ir_round == 0)]
    return table(q.pivot_table(index="n_clock", columns=["scaling", "reciprocal_mode"], values="rel_err_original", aggfunc="median"))
section("Q1, HHL forward error on captured systems (ideal readout, median)", _q1)

def _q4():
    q = QD[QD.experiment.astype(str).str.startswith("Q4_regularization") & (QD.status == "ok")]
    last = q.sort_values("ir_round").groupby(["experiment", "instance_id", "factor_index", "regularization"]).tail(1)
    return table(last.groupby(["experiment", "regularization"]).agg(systems=("instance_id", "size"), final_rel_resid=("rel_resid_original", "median"), rounds=("ir_round", "median")))
section("Q4, regularised target with refinement (residual after the last round, median)", _q4)

section("Q5a, forward error of the quantum solvers on captured systems (ideal readout, median)", lambda: table(
    Q5A[Q5A.status == "ok"].groupby(["solver", "poly", "eps_target"], dropna=False).agg(systems=("rel_err_ideal", "size"), rel_err=("rel_err_ideal", "median"),
                                                                                         walk_steps=("oracle_queries", "median"), p_success=("p_success", "median"))))
section("Q5b, cost of refinement to a 1e-10 residual", lambda: table(
    Q5B.groupby(["poly", "eps_circuit", "samples_per_round"]).agg(systems=("rounds", "size"), reached=("reached_1e-10", "mean"), rounds=("rounds", "median"),
                                                                  total_samples=("total_samples", "median"), queries=("block_encoding_queries_total_with_AA", "median"))))
section("Q5d, augmented against null-space target", lambda: table(Q5D))
section("E5, outer refinement: median max(pres, dres, relgap) by round", lambda: table(OREF.pivot_table(index="round", columns="inner_solver", values="err", aggfunc="median")))
section("E7, spectrum along one IPM run", lambda: table(SPEC7.drop(columns=["instance_id"], errors="ignore")))
section("E7, exact-deflation bound at the last IPM iteration", lambda: table(DEFL7.drop(columns=["instance_id"], errors="ignore")))
section("D1, dequantisation probe (median)", lambda: table(DEQ.drop(columns=["instance_id", "factor_index"], errors="ignore").median().to_frame("median")))

REPORT = "\n".join(L)
with open(os.path.join(RESULTS_DIR, "summary_report.md"), "w") as f: f.write(REPORT)
print(f"summary_report.md: {len(REPORT.splitlines())} lines")
