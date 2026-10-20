# Diagnostic figures, written to results/plots/. A figure whose data is missing in this run is skipped with a printed reason.
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150, "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.6, "axes.edgecolor": "#c3c2b7",
                     "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 7,
                     "legend.frameon": False, "lines.linewidth": 2, "lines.markersize": 5, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})
_CAT = {"cvxopt_default": "#0b0b0b", "custom_exact": "#eb6834", "minres_relres_1e-06_noscale": "#898781", "minres_relres_1e-06": "#52514e",
        "minres_be_1e-10_noscale": "#e87ba4", "minres_block_0.1_noscale": "#eda100", "minres_adaptive_nt_be": "#4a3aa7", "cg_reduced_be_1e-10": "#008300",
        "custom_exact_reduced": "#c98500", "minres_fused_comb_cpu": "#5598e7", "minres_fused_comb_floor8_cpu": "#1baf7a", "minres_fused_comb_gpu": "#eb6834", "minres_fused_comb_auto": "#9085e9", "gmres_stalelu_comb": "#008300", "minres_base_comb_cpu": "#2a78d6", "minres_base_comb_ws_cpu": "#0d366b", "minres_base_comb_ws_switch_cpu": "#1baf7a",
        "minres_base_comb_gpu": "#e87ba4", "minres_base_comb_ws_gpu": "#d03b3b", "lsqr_sqrt_comb": "#4a3aa7", "minres_rrfloor_1e-06": "#d55181", "minres_rrfloor_1e-06_noscale": "#f2b3c9", "minres_comb_0.1_1e-10": "#c98500",
        "minres_fused_comb_switch_cpu": "#6a5bd0", "oss_cg_comb": "#0a6b4a", "minres_fused_comb_gpu_mm": "#eb6834"}
_BLUES = ["#9ec5f4", "#5598e7", "#2a78d6", "#0d366b"]; _GREENS = ["#8fd3b6", "#1baf7a", "#0a6b4a"]; _VIOLETS = ["#b3a9ec", "#9085e9", "#6a5bd0", "#4a3aa7", "#2f2470", "#1c1545"]
def method_color(name):
    if name in _CAT: return _CAT[name]
    if name.startswith("minres_be_"):
        tols = sorted({float(n.split("_")[-1]) for n in SPECS if n.startswith("minres_be_") and not n.endswith("noscale")} | {float(name.split("_")[-1])}, reverse=True)
        return _BLUES[min(len(_BLUES) - 1, tols.index(float(name.split("_")[-1])) * (len(_BLUES) - 1) // max(1, len(tols) - 1))]
    if name.startswith("minres_block_"):
        etas = sorted({float(n.split("_")[-1]) for n in SPECS if n.startswith("minres_block_") and not n.endswith("noscale")} | {float(name.split("_")[-1])}, reverse=True)
        return _GREENS[min(len(_GREENS) - 1, etas.index(float(name.split("_")[-1])) * (len(_GREENS) - 1) // max(1, len(etas) - 1))]
    if name.startswith("hhl"):
        names = [r["name"] for r in CONFIG["quantum"]["integrated"]["runs"]]
        return _VIOLETS[names.index(name) % len(_VIOLETS)] if name in names else "#4a3aa7"
    if name.startswith("exact_noise_iso"): return "#e34948"
    if name.startswith("exact_noise_block"): return "#8b1a1a"
    return "#898781"
def method_style(name): return {"linestyle": "--" if name == "cvxopt_default" else "-", "marker": "D" if name.startswith("hhl") else "o"}
def savefig(fig, name):
    path = os.path.join(PLOTS_DIR, name); fig.tight_layout(); fig.savefig(path); plt.close(fig); print("  saved", os.path.relpath(path, CONFIG["output_root"]))
def with_data(fn):
    def wrapper(*a, **k):
        try: fn(*a, **k)
        except Exception as e: print(f"  plot {fn.__name__} skipped: {type(e).__name__}: {e}")
    return wrapper
def order_methods(names):
    pref = ["cvxopt_default", "custom_exact", "custom_exact_reduced", "gmres_stalelu_comb", "minres_fused_comb_cpu", "minres_fused_comb_gpu", "minres_fused_comb_auto", "minres_base_comb_cpu", "minres_base_comb_ws_cpu", "minres_base_comb_ws_switch_cpu", "minres_base_comb_gpu", "minres_base_comb_ws_gpu", "lsqr_sqrt_comb", "minres_relres_1e-06_noscale", "minres_relres_1e-06", "minres_rrfloor_1e-06_noscale", "minres_rrfloor_1e-06"] + \
           sorted([n for n in names if n.startswith("minres_be_") and not n.endswith("noscale")], key=lambda m: -float(m.split("_")[-1])) + ["minres_be_1e-10_noscale"] + \
           sorted([n for n in names if n.startswith("minres_block_") and not n.endswith("noscale")], key=lambda m: -float(m.split("_")[-1])) + ["minres_block_0.1_noscale", "minres_comb_0.1_1e-10", "minres_adaptive_nt_be", "cg_reduced_be_1e-10"]
    return [m for m in pref if m in set(names)] + [m for m in names if m not in pref]

DF = REC.df("benchmark_results"); IT = REC.df("iteration_logs"); LS = REC.df("linear_solve_logs"); QD = REC.df("quantum_results")

@with_data
def plot_runtime_vs_size():
    d = DF[(DF.experiment == "E3_scaling")]
    if d.empty: raise ValueError("no E3 data")
    fams = [f for f in ["sparse_banded", "sparse_scale"] if f in set(d.family)]
    fig, axes = plt.subplots(1, len(fams) + 1, figsize=(5.5 * (len(fams) + 1), 4.3))
    for ax, fam in zip(axes, fams):
        g0 = d[d.family == fam]
        for meth, g in g0.groupby("method"):
            s = g.groupby("kkt_dim")["wall_method"].agg(["median", "min", "max"]).reset_index()
            ax.plot(s.kkt_dim, s["median"], color=method_color(meth), label=meth, **method_style(meth)); ax.fill_between(s.kkt_dim, s["min"], s["max"], color=method_color(meth), alpha=0.12, linewidth=0)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("KKT dimension n+m+p"); ax.set_ylabel("wall time per coneqp call [s] (median, min–max)")
        ax.set_title(f"Runtime vs. size, family {fam}"); ax.legend()
    ax = axes[-1]; fam = fams[0]; big = d[(d.family == fam) & (d.kkt_dim == d[d.family == fam].kkt_dim.max())]
    comps = ["t_assemble", "t_factor", "t_solve", "t_check", "t_verify", "t_refine", "t_fallback"]; rows = []
    for meth, g in big.groupby("method"):
        r = g[comps].median(); r["cvxopt_iteration"] = max(0.0, g["wall_method"].median() - r.sum()); rows.append(r.rename(meth))
    comp_df = pd.DataFrame(rows).reindex(order_methods(list(pd.DataFrame(rows).index))); bottom = np.zeros(len(comp_df))
    cols = ["#9ec5f4", "#2a78d6", "#0d366b", "#e87ba4", "#1baf7a", "#eda100", "#e34948", "#898781"]
    for c, colr in zip(comps + ["cvxopt_iteration"], cols):
        ax.bar(comp_df.index, comp_df[c], bottom=bottom, color=colr, label=c, width=0.6, edgecolor="#fcfcfb", linewidth=1); bottom += comp_df[c].values
    ax.set_ylabel("seconds (median)"); ax.set_title(f"Time components, {fam}, KKT dim {int(big.kkt_dim.max())}"); ax.tick_params(axis="x", rotation=60); ax.legend(ncol=2)
    savefig(fig, "01_runtime_vs_size.png")

@with_data
def plot_accuracy_vs_runtime():
    d = DF[DF.experiment == "E2_main"].copy()
    if d.empty: raise ValueError("no E2 data")
    d["final_err"] = d[["pres_rec", "dres_rec", "obj_rel_err"]].max(axis=1).clip(lower=1e-16)
    fig, ax = plt.subplots(figsize=(8, 5))
    for meth in order_methods(list(d.method.unique())):
        g = d[d.method == meth]; ax.scatter(g.wall_method, g.final_err, s=16, color=method_color(meth), label=meth, alpha=0.75, edgecolors="none")
    ax.axhline(max(CONFIG["accept"]["pres"], CONFIG["accept"]["obj_rel_err"]), color="#e34948", linewidth=1, linestyle=":", label="acceptance threshold")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("wall time per coneqp call [s]"); ax.set_ylabel("max(pres, dres, objective rel. error)"); ax.set_title("Accuracy vs. runtime (dense families, every run)"); ax.legend(ncol=2, fontsize=6)
    savefig(fig, "02_accuracy_vs_runtime.png")

@with_data
def plot_convergence():
    d = DF[(DF.experiment == "E2_main") & (DF.family == "dense_kappa") & (DF.repeat == 0)]
    if d.empty: raise ValueError("no E2 data")
    inst_id = d[d.kappa_P_target == d.kappa_P_target.max()].sort_values("n").instance_id.iloc[-1]
    it = IT[(IT.instance_id == inst_id) & (IT.repeat == 0)]
    keep = ["cvxopt_default", "custom_exact", "minres_relres_1e-06_noscale", "minres_relres_1e-06", "minres_rrfloor_1e-06", "minres_be_1e-08", "minres_be_1e-12", "minres_block_0.1", "minres_comb_0.1_1e-10"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for meth in keep:
        g = it[it.method == meth]
        if g.empty: continue
        axes[0].plot(g["iter"], g["gap"].clip(lower=1e-16), color=method_color(meth), label=meth, **method_style(meth))
        axes[1].plot(g["iter"], np.maximum(g["pres"], g["dres"]).clip(lower=1e-16), color=method_color(meth), label=meth, **method_style(meth))
    axes[0].set_yscale("log"); axes[0].set_xlabel("IPM iteration"); axes[0].set_ylabel("duality gap sᵀz (CVXOPT report)"); axes[0].set_title(f"Convergence by iteration, {inst_id}")
    axes[1].set_yscale("log"); axes[1].set_xlabel("IPM iteration"); axes[1].set_ylabel("max(pres, dres)"); axes[1].set_title("Residuals by iteration"); axes[1].legend(fontsize=6)
    savefig(fig, "03_convergence_by_iteration.png")

@with_data
def plot_conditioning():
    it = IT[(IT.method == "custom_exact") & (IT.repeat == 0) & IT.kappa.notna()].merge(DF[DF.method == "custom_exact"][["run_id", "kappa_P_target", "family", "n"]], on="run_id", how="left")
    it = it[it.family.isin(["dense_kappa", "sanity"])]
    if it.empty: raise ValueError("no conditioning data")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2)); kps = sorted(it.kappa_P_target.dropna().unique())
    ramp = [plt.cm.Blues(0.35 + 0.6 * i / max(1, len(kps) - 1)) for i in range(len(kps))]
    for kp, colr in zip(kps, ramp):
        med = it[it.kappa_P_target == kp].groupby("iter")[["kappa", "kappa_scaled", "nt_proxy"]].median()
        axes[0].plot(med.index, med.kappa, color=colr, label=f"κ(P)={kp:.0e}: κ(K̃)")
        if med.kappa_scaled.notna().any(): axes[0].plot(med.index, med.kappa_scaled, color=colr, linestyle="--", label=f"κ(P)={kp:.0e}: κ(DK̃D) Ruiz")
        axes[1].plot(med.index, med.nt_proxy, color=colr, label=f"κ(P)={kp:.0e}")
    axes[0].set_yscale("log"); axes[0].set_xlabel("IPM iteration"); axes[0].set_ylabel("κ₂ of the scaled KKT matrix (median)"); axes[0].set_title("Conditioning along the path"); axes[0].legend(fontsize=6)
    axes[1].set_yscale("log"); axes[1].set_xlabel("IPM iteration"); axes[1].set_ylabel("NT-scaling spread min(d)/max(d)"); axes[1].set_title("μ proxy (adaptive_nt)"); axes[1].legend(fontsize=6)
    ex = LS[(LS.method_kind == "exact") & LS.kappa.notna()]
    axes[2].scatter(ex.kappa, ex.rel_resid_final.clip(lower=1e-17), s=8, color="#898781", label="relative residual ‖K̃ũ−r̃‖/‖r̃‖", edgecolors="none")
    axes[2].scatter(ex.kappa, ex.backward_err_final.clip(lower=1e-17), s=8, color="#2a78d6", label="backward error", edgecolors="none")
    axes[2].set_xscale("log"); axes[2].set_yscale("log"); axes[2].set_xlabel("κ₂(K̃)"); axes[2].set_ylabel("residual measure of the LU solve"); axes[2].set_title("Attainability: LU residual vs. conditioning"); axes[2].legend(fontsize=6)
    savefig(fig, "04_conditioning_across_iterations.png")

@with_data
def plot_criterion_ablation():
    d = DF[(DF.experiment == "E2_main") & (~DF.method.isin(["cvxopt_default", "custom_exact"]))]
    if d.empty: raise ValueError("no E2 data")
    meths = order_methods(list(d.method.unique()))
    s = d.groupby("method").agg(pure=("pure_method_success", "mean"), acc=("accepted", "mean"), fb=("n_fallback_total", "mean"), kry=("krylov_iters_total", "median"), wall=("wall_method", "median"), it=("iterations", "median")).reindex(meths)
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.4)); x = np.arange(len(meths)); cols = [method_color(m) for m in meths]
    axes[0].bar(x - 0.2, s.acc, 0.38, color="#c3c2b7", label="accepted (fallback allowed)"); axes[0].bar(x + 0.2, s.pure, 0.38, color=cols, label="accepted with zero fallbacks")
    axes[0].set_ylim(0, 1.05); axes[0].set_ylabel("fraction of runs"); axes[0].set_title("Acceptance (dense families)"); axes[0].legend(fontsize=6)
    axes[1].bar(x, s.fb, color=cols); axes[1].set_ylabel("mean classical fallbacks per run"); axes[1].set_title("Fallbacks")
    axes[2].bar(x, s.kry, color=cols); axes[2].set_yscale("log"); axes[2].set_ylabel("median Σ Krylov iterations per run"); axes[2].set_title("Krylov work")
    axes[3].bar(x, s.wall, color=cols); axes[3].set_yscale("log"); axes[3].set_ylabel("median wall time [s]"); axes[3].set_title("Runtime")
    for ax in axes: ax.set_xticks(x); ax.set_xticklabels(meths, rotation=70, fontsize=7)
    fig.suptitle("Acceptance-test ablation (relres = relative residual; rrfloor = relres with direct-solver floor; be = backward error; block = block forcing; comb = block and backward; noscale = no Ruiz scaling)", fontsize=8)
    savefig(fig, "05_criterion_ablation.png")

@with_data
def plot_variability():
    d = DF[(DF.experiment == "E2_main")]
    if d.empty: raise ValueError("no E2 data")
    meths = order_methods(list(d.method.unique())); fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    data = [d[d.method == m].wall_method.values for m in meths]; bkw = dict(patch_artist=True, widths=0.6, showfliers=True, flierprops={"markersize": 3, "markeredgecolor": "#898781"})
    try: bp = axes[0].boxplot(data, tick_labels=meths, **bkw)
    except TypeError: bp = axes[0].boxplot(data, labels=meths, **bkw)
    for patch, m in zip(bp["boxes"], meths): patch.set_facecolor(method_color(m)); patch.set_alpha(0.55); patch.set_edgecolor(method_color(m))
    axes[0].set_yscale("log"); axes[0].set_ylabel("wall time per coneqp call [s]"); axes[0].set_title("Runtime variability across instances and seeds"); axes[0].tick_params(axis="x", rotation=70, labelsize=7)
    piv = d.pivot_table(index="method", columns="kappa_P_target", values="pure_method_success", aggfunc="mean").reindex(meths)
    im = axes[1].imshow(piv.values, cmap="Blues", vmin=0, vmax=1, aspect="auto"); axes[1].set_yticks(range(len(meths))); axes[1].set_yticklabels(meths, fontsize=7)
    axes[1].set_xticks(range(piv.shape[1])); axes[1].set_xticklabels([f"{c:.0e}" for c in piv.columns]); axes[1].set_xlabel("κ(P)"); axes[1].set_title("Fallback-free success rate by κ(P)"); axes[1].grid(False)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]): axes[1].text(j, i, f"{piv.values[i, j]:.2f}", ha="center", va="center", fontsize=7, color="#0b0b0b" if piv.values[i, j] < 0.6 else "#ffffff")
    fig.colorbar(im, ax=axes[1], label="fraction of runs"); savefig(fig, "06_variability_and_success.png")

@with_data
def plot_noise():
    nz = DF[DF.experiment == "E4_noise"].copy()
    if nz.empty: raise ValueError("no noise data")
    nz["delta"] = nz.method.str.split("_").str[-1].astype(float); nz["model"] = np.where(nz.method.str.contains("_iso_"), "isotropic", "block-relative")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for model, colr in [("isotropic", "#e34948"), ("block-relative", "#8b1a1a")]:
        s = nz[nz.model == model].groupby("delta").agg(acc=("accepted", "mean"), it=("iterations", "median")).reset_index()
        if s.empty: continue
        axes[0].plot(s.delta, s.acc, marker="o", color=colr, label=model); axes[1].plot(s.delta, s.it, marker="o", color=colr, label=model)
    axes[0].set_xscale("log"); axes[0].set_ylim(-0.05, 1.05); axes[0].set_xlabel("injected relative direction error δ"); axes[0].set_ylabel("fraction accepted"); axes[0].set_title("Noise-injection sensitivity (not a quantum method)"); axes[0].legend()
    axes[1].set_xscale("log"); axes[1].set_xlabel("δ"); axes[1].set_ylabel("median IPM iterations"); axes[1].set_title("Iterations under injected direction error"); axes[1].legend()
    savefig(fig, "07_noise_sensitivity.png")

@with_data
def plot_quantum_error_vs_clock():
    q = QD[(QD.experiment == "Q1_captured") & (QD.status == "ok") & (QD.recon == "ideal_statevector") & (QD.ir_round == 0)]
    if q.empty: raise ValueError("no Q1 data")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    settings = [("none", "resolution", 0.0, "#898781", "no scaling, C=Δ"), ("ruiz", "resolution", 0.0, "#5598e7", "Ruiz, C=Δ"), ("none", "kappa_known", 0.0, "#eda100", "no scaling, C=k_min·Δ"), ("ruiz", "kappa_known", 0.0, "#0d366b", "Ruiz, C=k_min·Δ")]
    for rg in sorted(set(q.regularization) - {0.0}): settings.append(("ruiz", "kappa_known", rg, "#1baf7a" if rg < 1e-3 else "#0a6b4a", f"Ruiz, κ-known, δ={rg:g} (before refinement)"))
    for sc, cm, rg, colr, lab in settings:
        g = q[(q.scaling == sc) & (q.reciprocal_mode == cm) & (q.regularization == rg)]
        if g.empty: continue
        s = g.groupby("n_clock").rel_err_original.quantile([0.25, 0.5, 0.75]).unstack()
        axes[0].plot(s.index, s[0.5].clip(lower=1e-16), marker="o", color=colr, label=lab); axes[0].fill_between(s.index, s[0.25].clip(lower=1e-16), s[0.75].clip(lower=1e-16), color=colr, alpha=0.12, linewidth=0)
    axes[0].set_yscale("log"); axes[0].set_xlabel("clock qubits n_c"); axes[0].set_ylabel("ideal-statevector rel. error, original system (median, IQR)"); axes[0].set_title("HHL intrinsic error vs. clock register and settings"); axes[0].legend(fontsize=6)
    g = q[(q.scaling == "ruiz") & (q.reciprocal_mode == "kappa_known") & (q.regularization == 0.0)]
    sc_ = axes[1].scatter(g.kappa_original, g.rel_err_original.clip(lower=1e-16), c=g.n_clock, cmap="Blues", s=22, edgecolors="#52514e", linewidths=0.4)
    axes[1].set_xscale("log"); axes[1].set_yscale("log"); axes[1].set_xlabel("κ₂ of the original KKT system"); axes[1].set_ylabel("ideal rel. error (Ruiz, κ known)"); axes[1].set_title("Error vs. conditioning (colour = n_c)"); fig.colorbar(sc_, ax=axes[1], label="clock qubits")
    savefig(fig, "08_quantum_error_vs_clock.png")

@with_data
def plot_quantum_readout():
    q = QD[(QD.experiment == "Q1_captured") & (QD.status == "ok") & (QD.scaling == "ruiz") & (QD.reciprocal_mode == "kappa_known") & (QD.regularization == 0.0)]
    if q.empty: raise ValueError("no Q1 data")
    nc = q.n_clock.max(); idl = q[(q.recon == "ideal_statevector") & (q.ir_round == 0) & (q.n_clock == nc)]
    res = set(map(tuple, idl[idl.rel_err_target <= 0.5][["instance_id", "factor_index"]].values))
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))
    for recon, colr, lab in [("aa_shots", "#4a3aa7", "AA-emulated: accepted samples per setting"), ("shots", "#eb6834", "raw shots per setting (postselection included)")]:
        g = q[(q.recon == recon) & (q.n_clock == nc) & np.array([tuple(v) in res for v in q[["instance_id", "factor_index"]].values], dtype=bool)]
        if g.empty: continue
        s = g.groupby("shots_per_setting").rel_err_target.quantile([0.25, 0.5, 0.75]).unstack()
        axes[0].plot(s.index, s[0.5], marker="o", color=colr, label=lab); axes[0].fill_between(s.index, s[0.25], s[0.75], color=colr, alpha=0.15, linewidth=0)
    axes[0].axhline(idl[[tuple(v) in res for v in idl[["instance_id", "factor_index"]].values]].rel_err_target.median() if res else idl.rel_err_target.median(), color="#0b0b0b", linestyle="--", linewidth=1, label="ideal-statevector error (median)")
    axes[0].set_xscale("log"); axes[0].set_yscale("log"); axes[0].set_xlabel("samples per measurement setting"); axes[0].set_ylabel("rel. error of reconstructed vector (median, IQR)")
    axes[0].set_title(f"Readout error vs. budget, n_c={nc}, {len(res)} resolvable systems", fontsize=9); axes[0].legend(fontsize=6)
    aa = q[q.recon == "aa_shots"]
    if not aa.empty:
        s = aa.groupby(["n_clock", "shots_per_setting"]).aa_overhead_per_sample.median().unstack()
        for ncc in s.index: axes[1].plot(s.columns, s.loc[ncc], marker="o", color=plt.cm.Blues(0.35 + 0.6 * (list(s.index).index(ncc)) / max(1, len(s.index) - 1)), label=f"n_c={ncc}")
        axes[1].set_xscale("log"); axes[1].set_yscale("log"); axes[1].set_xlabel("accepted samples per setting"); axes[1].set_ylabel("expected circuit repetitions per accepted sample = 1/√P (median)"); axes[1].set_title("Amplitude-amplification overhead (assumption-dependent)"); axes[1].legend(fontsize=6)
    ideal_all = QD[(QD.experiment == "Q1_captured") & (QD.status == "ok") & (QD.recon == "ideal_statevector") & (QD.ir_round == 0)]
    sc_ = axes[2].scatter(ideal_all.kappa_target, ideal_all.p_postselect_exact.clip(lower=1e-9), c=ideal_all.n_clock, cmap="Blues", s=20, edgecolors="#52514e", linewidths=0.3)
    axes[2].set_xscale("log"); axes[2].set_yscale("log"); axes[2].set_xlabel("κ of the HHL target system"); axes[2].set_ylabel("P(anc = 1) exact"); axes[2].set_title("Postselection probability (all settings; colour = n_c)"); fig.colorbar(sc_, ax=axes[2], label="clock qubits")
    savefig(fig, "09_quantum_readout_and_postselection.png")

@with_data
def plot_quantum_regularization():
    q = QD[QD.experiment.str.startswith("Q4_regularization") & (QD.status == "ok")]
    if q.empty: raise ValueError("no Q4 data")
    corrs = list(q.corrector.unique()); fig, axes = plt.subplots(1, len(corrs) + 1, figsize=(5.6 * (len(corrs) + 1), 4.3))
    cols = ["#0d366b", "#1baf7a", "#eda100", "#e34948", "#4a3aa7"]
    for ax, corr in zip(axes, corrs):
        g0 = q[q.corrector == corr]
        for i, rg in enumerate(sorted(g0.regularization.unique())):
            s = g0[g0.regularization == rg].groupby("ir_round").rel_resid_original.quantile([0.25, 0.5, 0.75]).unstack()
            ax.plot(s.index, s[0.5].clip(lower=1e-17), marker="o", color=cols[i % 5], label=f"δ = {rg:g}"); ax.fill_between(s.index, s[0.25].clip(lower=1e-17), s[0.75].clip(lower=1e-17), color=cols[i % 5], alpha=0.12, linewidth=0)
        ax.set_yscale("log"); ax.set_xlabel("refinement round"); ax.set_ylabel("‖K̃ũ−r̃‖/‖r̃‖, original system (median, IQR)")
        ax.set_title(f"corrector: {corr}" + (" (classical diagnostic)" if "classical" in corr else " (ideal statevector)") + f", {g0[['instance_id','factor_index']].drop_duplicates().shape[0]} systems", fontsize=9); ax.legend(fontsize=7)
    ax = axes[-1]; g1 = q[(q.corrector == corrs[0]) & (q.ir_round == 0)]
    for i, rg in enumerate(sorted(g1.regularization.unique())):
        gg = g1[g1.regularization == rg]; ax.scatter(gg.kappa_scaled, gg.kappa_target, s=22, color=cols[i % 5], label=f"δ = {rg:g}", edgecolors="none")
    lim = [g1.kappa_scaled.min() * 0.5, g1.kappa_scaled.max() * 2]; ax.plot(lim, lim, color="#c3c2b7", linewidth=1)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("κ of the Ruiz-scaled system"); ax.set_ylabel("κ of the regularised scaled target"); ax.set_title("Regularisation bounds κ by ≈ (1+δ)/δ"); ax.legend(fontsize=7)
    savefig(fig, "10_regularization_refinement.png")

@with_data
def plot_integrated():
    ls = LS[LS.method.str.startswith("hhl")]
    if ls.empty: raise ValueError("no integrated HHL solves")
    runs = list(ls.run_id.unique())[:6]; ncol = min(3, len(runs)); nrow = int(math.ceil(len(runs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 4.0 * nrow), squeeze=False)
    for ax, rid in zip(axes.ravel(), runs):
        g = ls[ls.run_id == rid].sort_values("solve_index")
        ax.plot(g.solve_index, g.crit_initial.clip(lower=1e-16), marker="D", color="#4a3aa7", label="criterion value of the HHL direction (before refinement)")
        ax.plot(g.solve_index, g.crit_final.clip(lower=1e-16), marker="o", color="#1baf7a", label="criterion value of the direction used")
        rej = g[(~g.accepted) & (~g.fallback)]
        if len(rej): ax.scatter(rej.solve_index, rej.crit_final.clip(lower=1e-16), marker="s", facecolors="none", edgecolors="#eda100", s=70, label="rejected direction used (fallback off)", zorder=6)
        fb = g[g.fallback]
        if len(fb): ax.scatter(fb.solve_index, fb.crit_initial.clip(lower=1e-16), marker="x", color="#e34948", s=50, label="classical fallback", zorder=5)
        ax.axhline(g.accept_rtol.iloc[0], color="#c3c2b7", linestyle=":", label="acceptance threshold")
        ax.set_yscale("log"); ax.set_xlabel("KKT solve index"); ax.set_ylabel("criterion value (combined: max of block/η and backward/tol)"); ax.set_title(rid.replace("__", "\n"), fontsize=7); ax.legend(fontsize=5)
    for ax in axes.ravel()[len(runs):]: ax.axis("off")
    savefig(fig, "11_hhl_inside_coneqp.png")

@with_data
def plot_quantum_simtime():
    q = QD[(QD.experiment == "Q1_captured") & (QD.status == "ok") & (QD.recon == "ideal_statevector") & (QD.ir_round == 0)]
    if q.empty: raise ValueError("no Q1 data")
    fig, ax = plt.subplots(figsize=(6.5, 4.2)); s = q.groupby("n_qubits")[["t_prep", "t_circuit_sim", "t_readout_probs"]].median()
    for col, colr in [("t_prep", "#9ec5f4"), ("t_circuit_sim", "#2a78d6"), ("t_readout_probs", "#0d366b")]: ax.plot(s.index, s[col], marker="o", color=colr, label=col)
    ax.set_yscale("log"); ax.set_xlabel("total qubits"); ax.set_ylabel("seconds (median), classical statevector simulation"); ax.set_title("Simulator time components vs. circuit size"); ax.legend()
    savefig(fig, "12_quantum_simulation_time.png")

@with_data
def plot_sdplib():
    d = DF[DF.experiment == "S1_sdplib"]
    if d.empty: raise ValueError("no SDPLIB data")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2)); piv = d.pivot_table(index="instance_id", columns="method", values="wall_method", aggfunc="median")
    piv.plot.bar(ax=axes[0], color=[method_color(c) for c in piv.columns], width=0.8, edgecolor="#fcfcfb"); axes[0].set_yscale("log"); axes[0].set_ylabel("wall time [s]"); axes[0].set_title("SDPLIB: runtime per method"); axes[0].legend(fontsize=6)
    k = d[d.method == "custom_exact"].set_index("instance_id")[["kappa_first", "kappa_max"]]
    k.plot.bar(ax=axes[1], color=["#9ec5f4", "#0d366b"], width=0.7, edgecolor="#fcfcfb"); axes[1].set_yscale("log"); axes[1].set_ylabel("κ₂(K̃)"); axes[1].set_title("SDPLIB: KKT conditioning (first vs. worst iteration)")
    savefig(fig, "13_sdplib.png")

@with_data
def plot_refinement_and_matvec():
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.4))
    o = OREF if "OREF" in globals() else pd.DataFrame()
    if len(o) and "err" in o:
        cols = ["#0b0b0b", "#e34948", "#eb6834", "#2a78d6", "#4a3aa7", "#9085e9", "#1baf7a"]
        for i, (lab, g) in enumerate(o.groupby("inner_solver")):
            m = g.groupby("round").err.median(); axes[0].plot(m.index, m.values.clip(1e-17), marker="o", color=cols[i % len(cols)], label=lab)
        axes[0].axhline(1e-7, color="#c3c2b7", linestyle=":"); axes[0].set_yscale("log"); axes[0].set_xlabel("outer refinement round (inner tolerance 1e-3)"); axes[0].set_ylabel("max(pres, dres, valid rel. gap), median")
        axes[0].set_title("E5: optimisation-level refinement by inner KKT solver"); axes[0].legend(fontsize=6)
    d = DF[(DF.experiment == "E3_scaling") & (DF.family == "sparse_banded") & DF.method.str.startswith(("minres_base", "minres_fused", "minres_comb", "gmres_stalelu"))]
    for meth, g in d.groupby("method"):
        m = g.groupby("kkt_dim").krylov_iters_total.median()
        if (m > 0).any(): axes[1].plot(m.index, m.values, marker="o", color=method_color(meth), label=meth)
    axes[1].set_xscale("log"); axes[1].set_yscale("log"); axes[1].set_xlabel("KKT dimension"); axes[1].set_ylabel("Σ Krylov iterations per run (median)"); axes[1].set_title("Krylov work by method, sparse_banded"); axes[1].legend(fontsize=6)
    mv = MV if "MV" in globals() else pd.DataFrame()
    if len(mv):
        axes[2].plot(mv.kkt_dim, mv.t_matvec_cpu, marker="o", color="#2a78d6", label="CPU mat-vec"); axes[2].plot(mv.kkt_dim, mv.t_matvec_gpu, marker="o", color="#d03b3b", label="GPU mat-vec")
        axes[2].set_xscale("log"); axes[2].set_yscale("log"); axes[2].set_xlabel("KKT dimension"); axes[2].set_ylabel("seconds per sparse mat-vec"); axes[2].set_title("E6: CPU vs GPU mat-vec"); axes[2].legend()
    else:
        axes[2].axis("off"); axes[2].set_title("E6 GPU mat-vec: skipped (no GPU in this run)")
    savefig(fig, "14_outer_refinement_and_matvec.png")

@with_data
def plot_kernels_and_spectrum():
    fig, axes = plt.subplots(1, 4, figsize=(22, 4.4)); S_ = DF[DF.experiment == "E3_scaling"]
    for ax, fam, ttl in [(axes[0], "sparse_banded", "high-fill sparse family"), (axes[1], "sparse_lowfill", "low-fill control family")]:
        d = S_[(S_.family == fam) & S_.method.isin(["cvxopt_default", "custom_exact_reduced", "minres_base_comb_cpu", "minres_fused_comb_cpu", "minres_base_comb_gpu", "minres_fused_comb_gpu"])]
        for meth, g in d.groupby("method"):
            m = g.groupby("kkt_dim").wall_method.median(); ax.plot(m.index, m.values, marker="o", color=method_color(meth), label=meth, linestyle="--" if meth in ("minres_base_comb_cpu", "minres_base_comb_gpu") else "-")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("KKT dimension"); ax.set_ylabel("wall time [s] (median)"); ax.set_title(ttl + " (dashed = baseline kernel)"); ax.legend(fontsize=6)
    if "SPEC7" in globals() and len(SPEC7):
        ax = axes[2]; ax.plot(SPEC7.ipm_iteration, SPEC7.minres_its_augmented, marker="o", color="#2a78d6", label="MINRES its, augmented (Ruiz)"); ax.plot(SPEC7.ipm_iteration, SPEC7.minres_its_reduced_jacobi, marker="o", color="#e34948", label="MINRES its, reduced (Jacobi, matrix-free)")
        ax.set_yscale("log"); ax.set_xlabel("IPM iteration"); ax.set_ylabel("Krylov iterations (2 solves)"); ax2 = ax.twinx(); ax2.plot(SPEC7.ipm_iteration, SPEC7.kappa_scaled, marker="s", color="#0b0b0b", linestyle=":", label="κ(scaled KKT)"); ax2.set_yscale("log"); ax2.set_ylabel("κ of Ruiz-scaled KKT")
        ax.set_title("E7: target system and κ along one IPM run"); ax.legend(fontsize=6, loc="upper left"); ax2.legend(fontsize=6, loc="lower right")
    else: axes[2].axis("off")
    if "DEFL7" in globals() and len(DEFL7):
        axes[3].plot(DEFL7.n_deflated, DEFL7.minres_its, marker="o", color="#1baf7a"); axes[3].set_yscale("log"); axes[3].set_xlabel("exactly deflated smallest-|λ| eigenvectors"); axes[3].set_ylabel("MINRES iterations"); axes[3].set_title("E7: deflation upper bound (last IPM iteration)")
    else: axes[3].axis("off")
    savefig(fig, "15_kernels_lowfill_spectrum.png")

def performance_profile(d, methods, time_col="wall_method"):
    """Dolan-More performance profile: per instance the ratio r = t_m / min_m' t_m' (runs that were not accepted count as failed: r = inf);
    rho_m(tau) = fraction of instances with r <= tau.  Returns (long DataFrame [method, tau, rho], per-instance ratio table)."""
    g = d[d.method.isin(methods)].groupby(["instance_id", "method"]).agg(t=(time_col, "median"), ok=("accepted", "all")).reset_index()
    g.loc[~g.ok, "t"] = np.inf; T = g.pivot_table(index="instance_id", columns="method", values="t", aggfunc="first").reindex(columns=[m for m in methods if m in set(g.method)])
    T = T.dropna(how="all"); tmin = T.min(axis=1); R = T.div(tmin, axis=0); R[np.isinf(R) | R.isna()] = np.inf
    fin = R.values[np.isfinite(R.values)]; tmax = float(fin.max()) if fin.size else 1.0; taus = np.unique(np.concatenate([[1.0], np.geomspace(1.0, max(2.0, 1.05 * tmax), 300)]))
    rows = [{"method": m, "tau": t, "rho": float((R[m] <= t).mean()), "instances": int(len(R))} for m in R.columns for t in taus]
    return pd.DataFrame(rows), R

PP_ALL = []
@with_data
def plot_performance_profiles():
    panels = [("E3_scaling", "sparse_banded", "sparse high-fill family (all seeds)"), ("E3_scaling", "sparse_lowfill", "sparse low-fill family (all seeds)"), ("E8_maros_meszaros", "maros_meszaros", "Maros-Meszaros (real-world QPs)")]
    meths = ["cvxopt_default", "custom_exact", "custom_exact_reduced", "gmres_stalelu_comb", "minres_fused_comb_cpu", "minres_fused_comb_switch_cpu", "minres_fused_comb_gpu", "minres_fused_comb_gpu_mm", "minres_fused_comb_auto"]
    have = [(e, f, t) for e, f, t in panels if len(DF[(DF.experiment == e) & (DF.family == f)])]
    if not have: raise ValueError("no data for performance profiles")
    fig, axes = plt.subplots(1, len(have), figsize=(5.6 * len(have), 4.3), squeeze=False); axes = axes[0]
    for ax, (e, f, ttl) in zip(axes, have):
        d = DF[(DF.experiment == e) & (DF.family == f)]; pp, R = performance_profile(d, meths)
        if pp.empty: ax.axis("off"); continue
        pp["experiment"] = e; pp["family"] = f; PP_ALL.append(pp)
        for m, g in pp.groupby("method", sort=False): ax.step(g.tau, g.rho, where="post", color=method_color(m), label=m, linewidth=1.8)
        ax.set_xscale("log", base=2); ax.set_ylim(-0.02, 1.02); ax.set_xlabel("τ  (time within a factor τ of the fastest accepted method)"); ax.set_ylabel("fraction of instances ρ(τ)"); ax.set_title(f"Performance profile: {ttl}, {pp.instances.iloc[0]} instances"); ax.legend(fontsize=6, loc="lower right")
    savefig(fig, "16_performance_profiles.png")
    if PP_ALL: pd.concat(PP_ALL, ignore_index=True).to_csv(os.path.join(RESULTS_DIR, "performance_profiles.csv"), index=False)

@with_data
def plot_seeds_mm_memory():
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.4))
    # (a) seed-to-seed variability of the time ratio MINRES / CVXOPT on the sparse families
    ax = axes[0]; S_ = DF[(DF.experiment == "E3_scaling") & DF.family.isin(["sparse_banded", "sparse_lowfill"])]
    for fam, mk in [("sparse_banded", "o"), ("sparse_lowfill", "s")]:
        for meth, colr in [("minres_fused_comb_cpu", "#5598e7"), ("minres_fused_comb_gpu", "#eb6834"), ("custom_exact_reduced", "#c98500")]:
            a_ = S_[(S_.family == fam) & (S_.method == meth)].groupby(["instance_id", "kkt_dim"]).wall_method.median().reset_index(); b_ = S_[(S_.family == fam) & (S_.method == "cvxopt_default")].groupby("instance_id").wall_method.median()
            if a_.empty or b_.empty: continue
            a_["ratio"] = a_.wall_method / a_.instance_id.map(b_); a_ = a_.dropna()
            if a_.empty: continue
            ax.scatter(a_.kkt_dim, a_.ratio, color=colr, marker=mk, s=18, alpha=0.6, label=f"{meth} / cvxopt, {fam}")
            med = a_.groupby("kkt_dim").ratio.median(); ax.plot(med.index, med.values, color=colr, linestyle="-" if fam == "sparse_banded" else ":", linewidth=1.2)
    ax.axhline(1.0, color="#0b0b0b", linewidth=0.8, linestyle="--"); ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("KKT dimension"); ax.set_ylabel("wall-time ratio vs cvxopt_default (one point per seed)"); ax.set_title("Seed-to-seed variability (E3)"); ax.legend(fontsize=5.5)
    # (b) Maros-Meszaros: time of each method vs the unmodified solver
    ax = axes[1]; d8 = DF[DF.experiment == "E8_maros_meszaros"]
    if len(d8):
        b_ = d8[d8.method == "cvxopt_default"].set_index("instance_id").wall_method
        for meth, g in d8[d8.method != "cvxopt_default"].groupby("method"):
            g = g.set_index("instance_id"); x_ = b_.reindex(g.index); ok = g.accepted.values
            ax.scatter(x_[ok], g.wall_method[ok], color=method_color(meth), s=22, label=f"{meth} (accepted)"); ax.scatter(x_[~ok], g.wall_method[~ok], color=method_color(meth), s=30, marker="x", label=f"{meth} (not accepted)" if (~ok).any() else None)
        lim = [max(1e-3, min(b_.min(), d8.wall_method.min()) * 0.7), max(b_.max(), d8.wall_method.max()) * 1.4]; ax.plot(lim, lim, color="#0b0b0b", linewidth=0.8, linestyle="--"); ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("cvxopt_default wall time [s]"); ax.set_ylabel("method wall time [s]"); ax.set_title(f"Maros-Meszaros ({d8.instance_id.nunique()} QPs): below the diagonal = faster"); ax.legend(fontsize=5.5)
    else: ax.axis("off"); ax.set_title("E8: no Maros-Meszaros runs")
    # (c) peak memory increase per run vs size
    ax = axes[2]; M_ = DF[DF.experiment.isin(["E3_scaling", "E8_maros_meszaros"]) & DF.mem_rss_increase_mb.notna()] if "mem_rss_increase_mb" in DF else pd.DataFrame()
    if len(M_):
        for meth in ["cvxopt_default", "custom_exact", "custom_exact_reduced", "minres_fused_comb_cpu", "minres_fused_comb_gpu"]:
            g = M_[(M_.method == meth) & (M_.family == "sparse_banded")].groupby("kkt_dim").mem_rss_increase_mb.median()
            if len(g): ax.plot(g.index, g.values.clip(0.1), marker="o", color=method_color(meth), label=f"{meth} (RSS increase)")
        gg = M_[(M_.method == "minres_fused_comb_gpu") & (M_.family == "sparse_banded")]
        if len(gg) and "mem_gpu_pool_peak_mb" in gg and gg.mem_gpu_pool_peak_mb.notna().any():
            g = gg.groupby("kkt_dim").mem_gpu_pool_peak_mb.median(); ax.plot(g.index, g.values.clip(0.1), marker="s", linestyle=":", color="#eb6834", label="minres_fused_comb_gpu (CuPy pool peak)")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("KKT dimension"); ax.set_ylabel("peak memory [MB] (median over runs)"); ax.set_title("Peak memory per coneqp call (high-fill family)"); ax.legend(fontsize=6)
    else: ax.axis("off")
    savefig(fig, "17_seeds_maros_meszaros_memory.png")

@with_data
def plot_tolerance_shots_nullspace():
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.4))
    ax = axes[0]; tsd = DF[DF.experiment == "E4b_tolerance_sweep"].copy()
    if len(tsd):
        tsd["base"] = tsd.method.str.split("@").str[0]; tsd["delta"] = tsd.base.str.extract(r"noise_(?:iso|block)_(.*)$")[0].astype(float)
        for model, colr in [("iso", "#e34948"), ("block", "#8b1a1a")]:
            g = tsd[tsd.base.str.contains(f"_{model}_")]
            for j, (tau, gt) in enumerate(sorted(g.groupby("ipm_tol_level"), key=lambda kv: kv[0])):
                s = gt.groupby("delta").accepted.mean().sort_index(); ax.plot(s.index, s.values, marker="o", color=colr, alpha=0.35 + 0.3 * j, label=f"{model} noise, τ={tau:.0e}")
        ax.set_xscale("log"); ax.set_ylim(-0.05, 1.05); ax.set_xlabel("injected relative direction error δ"); ax.set_ylabel("fraction accepted (thresholds scaled with τ)"); ax.set_title("E4b: tolerated direction error vs IPM tolerance τ"); ax.legend(fontsize=5.5)
    else: ax.axis("off"); ax.set_title("E4b: no data")
    ax = axes[1]; d5e = DF[DF.experiment == "Q5e_shot_sweep"]
    if len(d5e):
        g = d5e.groupby("sweep_samples").agg(strict=("strict_method_success", "mean"), lo=("strict_method_success", lambda x: x.groupby(d5e.loc[x.index, "sweep_seed"]).mean().min()), hi=("strict_method_success", lambda x: x.groupby(d5e.loc[x.index, "sweep_seed"]).mean().max()), its=("iterations", "median"), ref=("n_refine_total", "median")).sort_index()
        ax.plot(g.index, g.strict, marker="o", color="#4a3aa7", label="strict success rate (mean over seeds x instances)"); ax.fill_between(g.index, g.lo, g.hi, color="#4a3aa7", alpha=0.15, linewidth=0, label="range over sampling seeds")
        ax.set_xscale("log"); ax.set_ylim(-0.05, 1.05); ax.set_xlabel("accepted samples per refinement round (both settings)"); ax.set_ylabel("strict success rate"); ax2 = ax.twinx(); ax2.plot(g.index, g.ref, marker="s", linestyle=":", color="#c98500", label="median refinement rounds per run"); ax2.set_ylabel("refinement rounds per run")
        ax.set_title(f"Q5e: finite-shot polynomial solver, {d5e.instance_id.nunique()} QPs x {d5e.sweep_seed.nunique()} seeds"); ax.legend(fontsize=6, loc="lower right"); ax2.legend(fontsize=6, loc="upper left")
    else: ax.axis("off"); ax.set_title("Q5e: no data")
    ax = axes[2]; q5d = Q5D if "Q5D" in globals() and len(Q5D) else pd.DataFrame()
    if len(q5d):
        ax.scatter(q5d.kappa_augmented_scaled, q5d.kappa_reduced_jacobi, color="#0d366b", s=24, label="κ: null-space (Jacobi) vs augmented (Ruiz)")
        lim = [min(q5d.kappa_augmented_scaled.min(), q5d.kappa_reduced_jacobi.min()) * 0.7, max(q5d.kappa_augmented_scaled.max(), q5d.kappa_reduced_jacobi.max()) * 1.4]; ax.plot(lim, lim, color="#0b0b0b", linewidth=0.8, linestyle="--")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("κ of the Ruiz-scaled augmented KKT matrix"); ax.set_ylabel("κ of the Jacobi-scaled null-space matrix ZᵀSZ")
        c3 = [c_ for c_ in q5d.columns if c_.startswith("degree_augmented_eps")]
        if c3:
            e_ = c3[0].split("eps")[1]; ax2 = ax.twinx(); ax2.scatter(q5d.kappa_augmented_scaled, q5d[f"degree_augmented_eps{e_}"] / q5d[f"degree_oss_eps{e_}"], color="#1baf7a", marker="^", s=24, label=f"degree ratio augmented / null-space at ε={e_}"); ax2.set_yscale("log"); ax2.set_ylabel("walk-step ratio (odd polynomial / SPD polynomial)"); ax2.legend(fontsize=6, loc="lower right")
        ax.set_title("Q5d: target system for the polynomial solver"); ax.legend(fontsize=6, loc="upper left")
    else: ax.axis("off"); ax.set_title("Q5d: no data")
    savefig(fig, "18_tolerance_shots_nullspace.png")

for f in [plot_kernels_and_spectrum, plot_refinement_and_matvec, plot_runtime_vs_size, plot_accuracy_vs_runtime, plot_convergence, plot_conditioning, plot_criterion_ablation, plot_variability, plot_noise,
          plot_quantum_error_vs_clock, plot_quantum_readout, plot_quantum_regularization, plot_integrated, plot_quantum_simtime, plot_sdplib,
          plot_performance_profiles, plot_seeds_mm_memory, plot_tolerance_shots_nullspace]:
    f()
print("plots done:", sorted(os.listdir(PLOTS_DIR)))
