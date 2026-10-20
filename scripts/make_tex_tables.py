"""LaTeX bodies of the tables of the paper (paper/tex/*.tex) from the derived tables, so that no number is copied by hand.

    python scripts/make_tex_tables.py [--tables results/tables] [--out paper/tex]
"""
import argparse
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--tables", default=os.path.join(ROOT, "results", "tables"))
ap.add_argument("--out", default=os.path.join(ROOT, "paper", "tex"))
args = ap.parse_args(); T = args.tables; O = args.out; os.makedirs(O, exist_ok=True)
def n3(v, d=None):
    if v is None or (isinstance(v, float) and not np.isfinite(v)): return "--"
    a = abs(v)
    if d is not None: return f"{v:.{d}f}"
    if a >= 100: return f"{v:,.0f}".replace(",", "\\,")
    if a >= 10: return f"{v:.1f}"
    if a >= 1: return f"{v:.2f}"
    if a >= 0.01: return f"{v:.2f}" if a >= 0.1 else f"{v:.3f}"
    return f"{v:.4f}"
def sci(v, p=1):
    if not np.isfinite(v): return "--"
    m, e = f"{v:.{p}e}".split("e"); e = int(e); return f"${m}\\times10^{{{e}}}$" if e not in (0,) else f"${m}$"
def th(v): return f"{int(round(v)):,}".replace(",", "\\,")
def w(name, s): open(os.path.join(O, name), "w").write(s); print("wrote", name)
# scaling table (session M)
S = pd.read_csv(f"{T}/scaling_seeds_main.csv"); rows = []
for fam, Ns, lab in [("sparse_banded", [310, 620, 1240, 2480, 4960, 9920, 19840, 39680, 79360], "\\texttt{sparse\\_banded} (high fill; 5 instances per size up to $N=9\\,920$, 2 beyond)"),
                     ("sparse_lowfill", [2480, 4960, 9920, 19840, 39680, 79360], "\\texttt{sparse\\_lowfill} (low fill; 3 instances per size up to $N=9\\,920$, 1 beyond)")]:
    rows.append(f"\\multicolumn{{9}}{{l}}{{{lab}}}\\\\")
    x = S[S.family == fam]
    for N in Ns:
        g = x[x.kkt_dim == N].set_index("method"); cell = lambda m: n3(g.loc[m, "wall_median"]) if m in g.index else "--"
        r = g.loc["minres_fused_comb_cpu"] if "minres_fused_comb_cpu" in g.index else None
        rat = "--"
        if r is not None and np.isfinite(r.speedup_vs_cvxopt_median):
            rat = f"${r.speedup_vs_cvxopt_median:.2f}\\times$" + (f" \\,({r.speedup_vs_cvxopt_min:.2f}--{r.speedup_vs_cvxopt_max:.2f})" if r.speedup_vs_cvxopt_max > r.speedup_vs_cvxopt_min + 1e-9 else "")
        rows.append(f"{N:,}".replace(",", "\\,") + " & " + " & ".join(cell(m) for m in ["cvxopt_default", "custom_exact_reduced", "gmres_stalelu_comb"]) + " & " + " & ".join(cell(m) for m in ["minres_fused_comb_cpu", "minres_fused_comb_gpu"]) + " & " + " & ".join(cell(m) for m in ["minres_base_comb_cpu", "minres_base_comb_gpu"]) + f" & {rat}\\\\")
    rows.append("\\midrule" if fam == "sparse_banded" else "")
w("scaling_rows.tex", "\n".join(r for r in rows if r) + "\n")
# Maros-Meszaros summary
M = pd.read_csv(f"{T}/mm_summary.csv").set_index("method"); F = pd.read_csv(f"{T}/mm_fastest.csv").set_index("method").fastest_on
lab = {"cvxopt_default": "LDL$\\T$ (\\textsc{cvxopt}, unmodified)", "custom_exact_reduced": "reduced system, CHOLMOD", "gmres_stalelu_comb": "lazy refactorisation", "minres_fused_comb_cpu": "Ruiz + MINRES, CPU",
       "minres_fused_comb_switch_cpu": "Ruiz + MINRES, switch to direct", "minres_fused_comb_gpu_mm": "Ruiz + MINRES, GPU ($N\\ge2\\,400$)"}
rows = []
for m in lab:
    r = M.loc[m]; fast = int(F.get(m, 0)) if m != "minres_fused_comb_gpu_mm" else "--"
    sp_ = "1" if m == "cvxopt_default" else f"{r.speedup_vs_cvxopt_median:.3f}" if r.speedup_vs_cvxopt_median < 0.1 else f"{r.speedup_vs_cvxopt_median:.2f}"
    rows.append(f"{lab[m]} & {int(r.instances)} & {100*r.strict:.0f} & {int(r.instances_with_fallback)} & {int(r.fallbacks)} & {100*r.same_iterations:.0f} & {n3(r.wall_median)} & {n3(r.wall_total)} & {sp_} & {fast}\\\\")
w("mm_rows.tex", "\n".join(rows) + "\n")
# tolerance sweep
TS = pd.read_csv(f"{T}/tolerance_sweep_main.csv"); rows = []
taus = [1e-8, 1e-7, 1e-6, 1e-4]
for mdl, deltas, lab_ in [("iso", [1e-8, 1e-7, 1e-6, 1e-5], "isotropic"), ("block", [1e-3, 1e-2], "block-wise")]:
    for d in deltas:
        cells = []
        for t in taus:
            g = TS[(TS.model == mdl) & np.isclose(TS.delta, d) & np.isclose(TS.ipm_tol_level, t)]
            cells.append(f"{100*g.accepted.iloc[0]:.0f}" if len(g) else "--")
        rows.append(f"{lab_} & $10^{{{int(round(np.log10(d)))}}}$ & " + " & ".join(cells) + "\\\\")
rows.append("\\midrule")
for base, lab_ in [("minres_fused_comb_cpu", "Ruiz + MINRES, combined test"), ("cvxopt_default", "unmodified solver")]:
    cells = []
    for t in taus:
        g = TS[(TS.base == base) & np.isclose(TS.ipm_tol_level, t)]
        cells.append(f"{100*g.accepted.iloc[0]:.0f} ({g.its_median.iloc[0]:.0f})" if len(g) else "--")
    rows.append(f"\\multicolumn{{2}}{{l}}{{{lab_}}} & " + " & ".join(cells) + "\\\\")
w("tolerance_rows.tex", "\n".join(rows) + "\n")
# shot-seed sweep
SH = pd.read_csv(f"{T}/shots_main.csv"); rows = []
for _, r in SH.iterrows():
    rng_ = f"{r.strict_rate_min_seed:.2f}" if abs(r.strict_rate_max_seed - r.strict_rate_min_seed) < 1e-9 else f"{r.strict_rate_min_seed:.2f}--{r.strict_rate_max_seed:.2f}"
    rows.append(th(r.samples_per_round) + f" & {int(r.strict)}/{int(r.runs)} & {rng_} & {int(r.cvxopt_optimal)} & {int(r.rejected_used)} & {r.ipm_its_median:.1f} & {r.refine_rounds_median:.0f} & {sci(r.samples_per_run_median)} & {sci(r.queries_per_run_median)}\\\\")
w("shots_rows.tex", "\n".join(rows) + "\n")
# integrated quantum table
I = pd.read_csv(f"{T}/integrated_summary_main.csv").set_index("method")
spec = [("\\emph{refinement budget of 4 rounds per solve}", None, None, None),
        ("HHL $n_c=8$ (Ruiz, $\\kappa$-known)", "ideal", "hhl_scaled_kk_nc8_nofb", ""), ("HHL $n_c=10$", "ideal", "hhl_scaled_kk_nc10_nofb", ""), ("HHL $n_c=12$", "ideal", "hhl_scaled_kk_nc12_nofb", ""),
        ("HHL $n_c=8$ (fallback on)", "$S=1000$", "hhl_scaled_kk_nc8_aa1000_fb", "fb"),
        ("\\emph{refinement budget of 60 rounds per solve}", None, None, None),
        ("HHL $n_c=8$", "ideal", "hhl_nc8_ideal_ir60_nofb", ""), ("HHL $n_c=10$", "ideal", "hhl_nc10_ideal_ir60_nofb", ""), ("HHL $n_c=10$", "$S=250$", "hhl_nc10_aa500_ir60_nofb", ""),
        ("LCU augmented, optimal, $\\eps=10^{-10}$", "ideal", "lcu_opt_eps1e-10_ideal_nofb", ""), ("LCU augmented, optimal, $\\eps=0.3$", "ideal", "lcu_opt_eps0.3_ideal_ir_nofb", ""),
        ("LCU augmented, optimal, $\\eps=0.3$", "$S=250$", "lcu_opt_eps0.3_aa500_ir_nofb", ""), ("LCU augmented, CKS, $\\eps=10^{-6}$", "$S=10\\,000$", "lcu_cks_eps1e-6_aa2e4_ir_nofb", ""),
        ("LCU null space, $\\eps=0.3$", "ideal", "lcu_oss_eps0.3_ideal_ir_nofb", ""), ("LCU null space, $\\eps=0.3$", "$S=250$", "lcu_oss_eps0.3_aa500_ir_nofb", "")]
rows = []
for lab_, ro, m, flag in spec:
    if m is None: rows.append(f"\\multicolumn{{8}}{{l}}{{{lab_}}}\\\\"); continue
    r = I.loc[m]; strict = f"\\textbf{{{int(r.strict)}}}" if r.strict == r.runs and r.runs >= 8 else f"{int(r.strict)}"
    rej = f"0 ({int(r.fallbacks)} fb)" if flag == "fb" else f"{int(r.rejected_used)}"
    steps = th(r.steps_median) if np.isfinite(r.steps_median) else "--"
    rows.append(f"{lab_} & {ro} & {int(r.runs)} & {strict} & {int(r.cvxopt_optimal)} & {rej} & {th(r.refine_rounds)} & {steps}\\\\")
w("integrated_rows.tex", "\n".join(rows) + "\n")
# null-space vs augmented on captured systems
Q = pd.read_csv(f"{T}/oss_q5d_main.csv"); A5 = pd.read_csv(f"{T}/oss_q5a_main.csv"); rows = []
for _, r in Q.iterrows():
    fe = A5[(A5.instance_id == r.instance_id) & (A5.factor_index == r.factor_index) & np.isclose(A5.eps_target, 0.3)].rel_err_ideal
    name = "\\texttt{" + r.instance_id.replace("_", "\\_") + "}"
    rat = r['degree_augmented_eps3e-01'] / r['degree_oss_eps3e-01']
    rows.append(f"{name} & {r.stage} & {int(r.kkt_dim)}/{int(r.dim_reduced)} & {sci(r.kappa_augmented_scaled)} & {sci(r.kappa_reduced_jacobi)} & {th(r['degree_augmented_eps3e-01'])} & {th(r['degree_oss_eps3e-01'])} & {n3(rat) if rat < 100 else th(rat)} & {sci(r['p_success_oss_eps3e-01'])} & {sci(fe.iloc[0]) if len(fe) else '--'}\\\\")
w("oss_rows.tex", "\n".join(rows) + "\n")
# replication summary
R = pd.read_csv(f"{T}/replication_summary_main.csv"); rows = []
labm = {"cvxopt_default": "LDL$\\T$ (\\textsc{cvxopt})", "custom_exact_reduced": "reduced, CHOLMOD", "gmres_stalelu_comb": "lazy refactorisation", "minres_fused_comb_cpu": "MINRES, CPU", "minres_fused_comb_gpu": "MINRES, GPU", "minres_base_comb_cpu": "MINRES, CPU, baseline kernel", "minres_base_comb_gpu": "MINRES, GPU, baseline kernel"}
sessions = ["B", "C", "D", "E"]
for fam, flab in [("sparse_banded", "\\texttt{sparse\\_banded}"), ("sparse_lowfill", "\\texttt{sparse\\_lowfill}")]:
    rows.append(f"\\multicolumn{{5}}{{l}}{{{flab}}}\\\\")
    for m, ml in labm.items():
        cells = []
        for s_ in sessions:
            g = R[(R.session == s_) & (R.family == fam) & (R.method == m)]
            cells.append(f"{g['median'].iloc[0]:.2f} ({g.q10.iloc[0]:.2f}--{g.q90.iloc[0]:.2f})" if len(g) else "--")
        rows.append(f"{ml} & " + " & ".join(cells) + "\\\\")
w("replication_rows.tex", "\n".join(rows) + "\n")
C = pd.read_csv(f"{T}/replication_crossovers_main.csv")
Sm = pd.read_csv(f"{T}/scaling_seeds_main.csv"); Im = I
def main_val(fam, meth, N):
    if meth == "gpu_over_cpu": g = Sm[(Sm.family == fam) & (Sm.method == "minres_fused_comb_gpu") & (Sm.kkt_dim == N)]; return g.cpu_over_this_median.iloc[0] if len(g) else np.nan
    g = Sm[(Sm.family == fam) & (Sm.method == meth) & (Sm.kkt_dim == N)]; return g.speedup_vs_cvxopt_median.iloc[0] if len(g) else np.nan
qty = [("MINRES CPU / LDL$\\T$, \\texttt{sparse\\_banded}, $N=4\\,960$", "sparse_banded", "minres_fused_comb_cpu", 4960), ("MINRES CPU / LDL$\\T$, \\texttt{sparse\\_banded}, $N=9\\,920$", "sparse_banded", "minres_fused_comb_cpu", 9920),
       ("MINRES CPU / LDL$\\T$, \\texttt{sparse\\_banded}, $N=19\\,840$", "sparse_banded", "minres_fused_comb_cpu", 19840), ("GPU / CPU MINRES, \\texttt{sparse\\_banded}, $N=19\\,840$", "sparse_banded", "gpu_over_cpu", 19840),
       ("GPU / CPU MINRES, \\texttt{sparse\\_banded}, $N=79\\,360$", "sparse_banded", "gpu_over_cpu", 79360), ("CHOLMOD / LDL$\\T$, \\texttt{sparse\\_lowfill}, $N=39\\,680$", "sparse_lowfill", "custom_exact_reduced", 39680),
       ("MINRES GPU / LDL$\\T$, \\texttt{sparse\\_lowfill}, $N=79\\,360$", "sparse_lowfill", "minres_fused_comb_gpu", 79360)]
rows = []
for lab_, fam, meth, N in qty:
    cells = [n3(main_val(fam, meth, N), 2)]
    for s_ in sessions:
        g = C[(C.session == s_) & (C.family == fam) & (C.method == meth) & (C.kkt_dim == N)]; cells.append(f"{g.speedup_vs_cvxopt_median.iloc[0]:.2f}" if len(g) else "--")
    rows.append(f"{lab_} & " + " & ".join(cells) + "\\\\")
qm = C[C.family == "quantum"]
for m, lab_ in [("lcu_opt_eps1e-10_ideal_nofb", "strict: LCU $\\eps=10^{-10}$, ideal"), ("lcu_opt_eps0.3_ideal_ir_nofb", "strict: LCU $\\eps=0.3$, ideal"), ("lcu_opt_eps0.3_aa500_ir_nofb", "strict: LCU $\\eps=0.3$, $S=250$"),
               ("lcu_cks_eps1e-6_aa2e4_ir_nofb", "strict: LCU CKS, $S=10^4$"), ("lcu_oss_eps0.3_ideal_ir_nofb", "strict: null space, ideal"), ("lcu_oss_eps0.3_aa500_ir_nofb", "strict: null space, $S=250$")]:
    cells = [f"{int(Im.loc[m, 'strict'])}/{int(Im.loc[m, 'runs'])}"]
    for s_ in sessions:
        g = qm[(qm.session == s_) & (qm.method == m)]
        cells.append(f"{int(g.speedup_vs_cvxopt_median.iloc[0])}/{int(g.kkt_dim.iloc[0])}" if len(g) else "--")
    rows.append(f"{lab_} & " + " & ".join(cells) + "\\\\")
w("crossover_rows.tex", "\n".join(rows) + "\n")
# further replicated quantities (Maros-Meszaros, tolerance sweep, shot sweep), appended to the crossover table
X = pd.read_csv(f"{T}/replication_extra_main.csv")
def xv(sess, q, fmt):
    g = X[(X.session == sess) & (X.quantity == q)]
    return fmt(g.iloc[0]) if len(g) and np.isfinite(g.value.iloc[0]) else "--"
cnt = lambda r: f"{int(r.value)}/{int(r.of)}"
pw = lambda r: f"$10^{{{int(round(np.log10(r.value)))}}}$"
rows = [r for r in open(os.path.join(O, "crossover_rows.tex")).read().splitlines() if r]
extra = [("Maros--M\\'esz\\'aros: LDL$\\T$ fastest", "mm_ldl_fastest", cnt), ("Maros--M\\'esz\\'aros: MINRES with fallback", "mm_minres_fallback_instances", cnt),
         ("largest error tolerated on all instances, $\\tau=10^{-6}$", "tol_largest_full_1e-06", pw), ("largest error tolerated on all instances, $\\tau=10^{-4}$", "tol_largest_full_1e-04", pw),
         ("shot-seed sweep, strict at 250 samples/round", "shots_strict_250", cnt), ("shot-seed sweep, strict at $4\\,000$ samples/round", "shots_strict_4000", cnt)]
for lab_, q_, f_ in extra: rows.append(f"{lab_} & " + " & ".join(xv(s_, q_, f_) for s_ in ["M"] + sessions) + "\\\\")
w("crossover_rows.tex", "\n".join(rows) + "\n")
# acceptance-test ablation (session A)
AB = pd.read_csv(f"{T}/ablation_E2_sessionA.csv").set_index("method")
crit = [("relative residual $10^{-6}$", "none", "minres_relres_1e-06_noscale"), ("backward error $10^{-10}$", "none", "minres_be_1e-10_noscale"), ("block $\\eta=0.1$", "none", "minres_block_0.1_noscale"),
        ("relative residual $10^{-6}$", "Ruiz", "minres_relres_1e-06"), ("backward error $10^{-6}$", "Ruiz", "minres_be_1e-06"), ("backward error $10^{-8}$", "Ruiz", "minres_be_1e-08"),
        ("backward error $10^{-10}$", "Ruiz", "minres_be_1e-10"), ("backward error $10^{-12}$", "Ruiz", "minres_be_1e-12"), ("block $\\eta=0.5$", "Ruiz", "minres_block_0.5"),
        ("block $\\eta=0.1$", "Ruiz", "minres_block_0.1"), ("block $\\eta=0.01$", "Ruiz", "minres_block_0.01"), ("combined ($\\eta=0.1$, $\\tau_\\omega=10^{-10}$)", "Ruiz", "minres_comb_0.1_1e-10"),
        ("combined, latency-optimised kernel", "Ruiz", "minres_fused_comb_cpu"), ("adaptive backward error (NT spread)", "Ruiz", "minres_adaptive_nt_be"), None,
        ("LDL$\\T$ (\\textsc{cvxopt}, unmodified)", "--", "cvxopt_default"), ("reduced system, CHOLMOD", "--", "custom_exact_reduced")]
rows = []
for c_ in crit:
    if c_ is None: rows.append("\\midrule"); continue
    l_, sc_, m_ = c_; r = AB.loc[m_]
    rows.append(f"{l_} & {sc_} & {100*r.accepted:.0f} & {100*r.pure:.0f} & {int(r.fallbacks)} & {th(r.krylov)} & {r.wall:.4f}".rstrip("0") + "\\\\" if r.wall < 0.01 else f"{l_} & {sc_} & {100*r.accepted:.0f} & {100*r.pure:.0f} & {int(r.fallbacks)} & {th(r.krylov)} & {r.wall:.3f}\\\\")
w("criteria_rows.tex", "\n".join(rows) + "\n")
# condition number before and after equilibration (session M; Maros-Meszaros from the offline diagnostic)
EQ = pd.read_csv(f"{T}/equil_main.csv")
def dopt(k, eps=0.3):
    a = 1 / k; n = np.ceil((np.log(1 / eps) + np.log(k) + np.log(1 + a)) / (np.log(1 + a) - np.log(1 - a))); return 2 * n - 1
EQ["dred"] = dopt(EQ.kappa_unscaled) / dopt(EQ.kappa_equilibrated); rows = []
for fam, lab_ in [("sanity", "\\texttt{sanity} (\\texttt{tiny\\_A}--\\texttt{tiny\\_C})"), ("dense_kappa", "\\texttt{dense\\_kappa}"), ("dense_noeq", "\\texttt{dense\\_noeq}"), ("sparse_scale", "\\texttt{sparse\\_scale}"),
                  ("sparse_banded", "\\texttt{sparse\\_banded}"), ("sparse_lowfill", "\\texttt{sparse\\_lowfill}"), ("sdplib", "SDPLIB (\\texttt{conelp})"), ("maros_meszaros", "Maros--M\\'esz\\'aros (diagnosed)")]:
    g = EQ[EQ.family == fam]
    if g.empty: continue
    rows.append(f"{lab_} & {len(g)} & {th(g.kkt_dim.min())}--{th(g.kkt_dim.max())} & {sci(g.kappa_unscaled.median())} & {sci(g.kappa_equilibrated.median())} & {sci(g.reduction.median())} & {sci(g.reduction.min())}--{sci(g.reduction.max())} & {sci(g.dred.median())}\\\\")
w("equil_rows.tex", "\n".join(rows) + "\n")
# SDPLIB per instance (session A)
SD = pd.read_csv(f"{T}/sdplib_instances.csv"); rows = []
def yn(v): return "--" if (v is None or (isinstance(v, float) and np.isnan(v))) else ("yes" if bool(v) else "no")
def fb(v): return "--" if (v is None or (isinstance(v, float) and np.isnan(v))) else f"{int(v)}"
for _, r in SD.iterrows():
    k_ = r.get("custom_exact__kappa_scaled", np.nan)
    rows.append(f"\\texttt{{{r.instance}}} & {th(r.kkt_dim)} & {int(r.cvxopt_default__its)} & {yn(r.cvxopt_default__acc)} & {yn(r.get('minres_comb_0.1_1e-10__acc'))} & {fb(r.get('minres_comb_0.1_1e-10__fb'))} & {yn(r.get('minres_be_1e-12__acc'))} & {fb(r.get('minres_be_1e-12__fb'))} & {fb(r.get('minres_block_0.1__fb'))} & {sci(k_) if np.isfinite(k_) else '--'}\\\\")
w("sdplib_rows.tex", "\n".join(rows) + "\n")
# resource reductions attributable to each methodological component
qb = pd.read_csv(f"{T}/q5b.csv"); key = ["instance_id", "factor_index"]
cs = qb[(qb.poly == "cks") & (qb.eps_circuit < 1e-5)].copy(); cs["c0"] = cs.contraction_per_round / np.sqrt(cs.kkt_dim / cs.samples_per_round); C0 = cs.c0.median()
ref = qb[(qb.poly == "cks") & (qb.eps_circuit == 1e-6) & (qb.samples_per_round == qb.samples_per_round.max()) & qb["reached_1e-10"]].set_index(key)
def red(po, ep, spr):
    g = qb[(qb.poly == po) & np.isclose(qb.eps_circuit, ep) & (qb.samples_per_round == spr)].set_index(key); cm = g.index.intersection(ref.index)
    return (ref.loc[cm, "block_encoding_queries_total_with_AA"] / g.loc[cm, "block_encoding_queries_total_with_AA"]).median()
qa = pd.read_csv(f"{T}/q5a.csv"); lw = qa[(qa.solver == "lcu_walk") & (qa.status == "ok")]
pq = lw.pivot_table(index=["instance_id", "factor_index", "eps_target"], columns="poly", values="oracle_queries", aggfunc="first")[["cks", "optimal"]].dropna()
pp = lw.pivot_table(index=["instance_id", "factor_index", "eps_target"], columns="poly", values="p_success", aggfunc="first")[["cks", "optimal"]].loc[pq.index]
dr = pq.cks / pq.optimal; cr = dr * (pp.optimal / pp.cks)          # walk steps per successful sample without amplitude amplification: d / p_succ
IS = I
aug, oss = IS.loc["lcu_opt_eps0.3_ideal_ir_nofb"], IS.loc["lcu_oss_eps0.3_ideal_ir_nofb"]; shot = IS.loc["lcu_opt_eps0.3_aa500_ir_nofb"]
samples_per_solve = shot.samples / shot.kkt_solves; Nq = 32; eps_f = 1e-8; direct = C0 ** 2 * Nq / eps_f ** 2          # samples per solve (both settings): c0 sqrt(N / samples_per_round) <= eps_f
eqb = EQ[EQ.family == "sparse_banded"]; eqd = EQ[EQ.family == "dense_kappa"]
rows = [f"per-iteration Ruiz equilibration & $\\kappa$ in $d$ and $\\norm{{c}}_1$ & $d\\propto\\kappa\\log(\\kappa/\\eps)$, $\\norm{{c}}_1=\\tilde\\Theta(\\kappa)$ & $\\kappa$: {sci(eqb.reduction.median())} (\\texttt{{sparse\\_banded}}), {sci(eqd.reduction.median())} (\\texttt{{dense\\_kappa}}); $d$: {sci(eqb.dred.median())}, {sci(eqd.dred.median())} & Table~\\ref{{tab:equil}}\\\\",
        f"optimal vs.\\ CKS polynomial & $d$, $\\norm{{c}}_1$ & same order, smaller constant & walk steps ${dr.median():.2f}\\times$; per accepted sample ${cr.median():.1f}\\times$ ({len(dr)} pairs) & Sec.~\\ref{{sec:qpoly}}\\\\",
        f"low-precision circuit ($\\eps=0.3$) + refinement vs.\\ high-precision circuit & samples, rounds, $d$ & Eq.~\\eqref{{eq:qcost}} & queries ${red('optimal', 0.3, 500):.0f}\\times$ (500 samples/round), ${red('optimal', 0.3, 200):.0f}\\times$ (200) & Sec.~\\ref{{sec:qcost}}\\\\",
        f"refinement vs.\\ direct readout to $\\eps_f=10^{{-8}}$ & samples per KKT solve & $\\Theta(N/\\eps_f^2)\\to\\Theta(N\\log(1/\\eps_f))$ & {sci(direct)} (model, $c_0={C0:.2f}$, $N=32$) vs.\\ {sci(samples_per_solve)} (measured) & Cor.~\\ref{{cor:precision}}\\\\",
        f"positive definite null-space target & $d$ & $\\sqrt\\kappa$ instead of $\\kappa$ & walk steps ${aug.steps_median / oss.steps_median:.1f}\\times$ inside the IPM; $p_{{\\text{{succ}}}}$ {sci(oss.psucc_min)} vs.\\ {sci(aug.psucc_min)} & Sec.~\\ref{{sec:oss}}\\\\"]
w("resources_rows.tex", "\n".join(rows) + "\n")
# inversion polynomials, CKS against optimal (the eps = 1e-2 rows only for kappa = 5)
PT = pd.read_csv(f"{T}/poly_table.csv"); PT = PT[(PT.kappa == 5) | ~np.isclose(PT.eps, 1e-2)]
def eps_tex(e): return f"{e:g}" if e >= 0.1 else f"$10^{{{int(round(np.log10(e)))}}}$"
def l1(v): return f"{v:.1f}" if v < 30 else th(v)
rows = [f"{th(r.kappa)} & {eps_tex(r.eps)} & {th(r.d_cks)} & {th(r.d_opt)} & {l1(r.l1_cks)} & {l1(r.l1_opt)} & {r.ratio_d:.2f} & {r.ratio_cost:.1f}\\\\" for r in PT.itertuples()]
w("poly_rows.tex", "\n".join(rows) + "\n")
# cost of refinement to a 1e-10 residual (session A); reduction paired against the first row
rows = []
for po, ep, spr in [("cks", 1e-6, 20000), ("cks", 1e-6, 500), ("cks", 1e-6, 200), ("optimal", 1e-2, 500), ("optimal", 0.3, 4000), ("optimal", 0.3, 500), ("optimal", 0.3, 200)]:
    g = qb[(qb.poly == po) & np.isclose(qb.eps_circuit, ep) & (qb.samples_per_round == spr)]
    e_ = eps_tex(ep) if ep < 0.1 else f"${ep:g}$"
    rows.append(f"{'CKS' if po == 'cks' else po} & {e_} & {th(spr)} & {g.rounds.median():g} & {th(g.walk_steps.median())} & {sci(g.block_encoding_queries_total_with_AA.median())} & ${red(po, ep, spr):.0f}\\times$\\\\")
w("qcost_rows.tex", "\n".join(rows) + "\n")
open(os.path.join(O, "constants.tex"), "w").write(f"\\newcommand{{\\cZero}}{{{C0:.2f}}}\n\\newcommand{{\\directSamples}}{{{sci(direct)}}}\n\\newcommand{{\\refinedSamples}}{{{sci(samples_per_solve)}}}\n")
print("c0 =", C0, "pairs", len(dr), "steps ratio", dr.median(), "per-sample ratio", cr.median(), "direct", direct, "refined/solve", samples_per_solve)
