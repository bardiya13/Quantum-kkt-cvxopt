"""Every figure of the paper (PDF and PNG) from the derived tables.

    python scripts/make_figures.py [--tables results/tables] [--out paper/fig]
"""
import argparse
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patheffects as pe

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--tables", default=os.path.join(ROOT, "results", "tables"))
ap.add_argument("--out", default=os.path.join(ROOT, "paper", "fig"))
args = ap.parse_args(); T = args.tables; F = args.out; os.makedirs(F, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix", "font.size": 8.5, "axes.titlesize": 9.2, "axes.titleweight": "bold", "axes.labelsize": 8.5,
                     "legend.fontsize": 7, "figure.dpi": 150, "axes.grid": True, "grid.alpha": 0.35, "grid.linewidth": 0.5, "grid.linestyle": ":", "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.7, "xtick.direction": "out", "ytick.direction": "out", "xtick.major.size": 3, "ytick.major.size": 3, "legend.frameon": False, "pdf.fonttype": 42, "lines.linewidth": 1.5, "lines.markersize": 3.6})
# categorical colours (blue, orange, aqua, yellow, magenta, violet), checked for colour-vision deficiency; near-black is the baseline
C = {"cvxopt_default": "#1b1b1b", "custom_exact_reduced": "#1baf7a", "gmres_stalelu_comb": "#eda100", "minres_fused_comb_cpu": "#2a78d6", "minres_fused_comb_gpu": "#eb6834", "minres_base_comb_cpu": "#2a78d6", "minres_base_comb_gpu": "#eb6834",
     "minres_fused_comb_auto": "#4a3aa7", "minres_fused_comb_switch_cpu": "#e87ba4", "oss_cg_comb": "#4a3aa7", "custom_exact": "#8c8c8c", "grey": "#8c8c8c", "purple": "#4a3aa7", "amber": "#eda100", "green": "#1baf7a",
     "blue": "#2a78d6", "red": "#eb6834", "magenta": "#e87ba4", "black": "#1b1b1b", "teal": "#1baf7a", "good": "#0ca30c", "warning": "#fab219", "critical": "#d03b3b", "ink": "#333333"}
MK = {"cvxopt_default": "o", "custom_exact_reduced": "^", "gmres_stalelu_comb": "v", "minres_fused_comb_cpu": "o", "minres_fused_comb_gpu": "s", "minres_fused_comb_auto": "D", "minres_fused_comb_switch_cpu": "P", "oss_cg_comb": "X", "minres_base_comb_cpu": "o", "minres_base_comb_gpu": "s"}
LAB = {"cvxopt_default": "CVXOPT default (sparse LDL$^{\\top}$)", "custom_exact_reduced": "reduced system, CHOLMOD", "gmres_stalelu_comb": "lazy refactorisation (GMRES + stale LU)", "minres_fused_comb_cpu": "Ruiz + MINRES, CPU",
       "minres_fused_comb_gpu": "Ruiz + MINRES, GPU (T4)", "minres_base_comb_cpu": "Ruiz + MINRES, CPU, baseline kernel", "minres_base_comb_gpu": "Ruiz + MINRES, GPU, baseline kernel", "minres_fused_comb_auto": "Ruiz + MINRES, auto CPU/GPU",
       "minres_fused_comb_switch_cpu": "MINRES, switch to direct after 1st rejection", "oss_cg_comb": "null-space CG"}
def save(fig, name): fig.savefig(os.path.join(F, name + ".pdf"), bbox_inches="tight", metadata={"CreationDate": None}); fig.savefig(os.path.join(F, name + ".png"), bbox_inches="tight", dpi=220); plt.close(fig)
def panel(ax, s): ax.set_title(s, loc="left")
def note(ax, xy, text, xytext, color="#333333", fs=7, arrow=True):
    ax.annotate(text, xy=xy, xytext=xytext, textcoords="offset points", fontsize=fs, color="#333333", ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=color, lw=0.6, shrinkA=0, shrinkB=2) if arrow else None, path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])
sb = pd.read_csv(os.path.join(T, "scaling_sparse_banded_main.csv"), index_col=0); lf = pd.read_csv(os.path.join(T, "scaling_sparse_lowfill_main.csv"), index_col=0)
SEEDS = pd.read_csv(os.path.join(T, "scaling_seeds_main.csv"))
def seedband(fam, m, col):
    g = SEEDS[(SEEDS.family == fam) & (SEEDS.method == m)].set_index("kkt_dim").sort_index() if len(SEEDS) else pd.DataFrame(); return g

# Fig. 1 (fig_summary): summary figure of the introduction
IS = pd.read_csv(os.path.join(T, "integrated_summary_main.csv")).set_index("method") if os.path.exists(os.path.join(T, "integrated_summary_main.csv")) else None
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), constrained_layout=True, gridspec_kw={"width_ratios": [1.0, 1.05]})
ax = axes[0]
for t, fam, lab, col, mk, ls in [(sb, "sparse_banded", "sparse_banded, CPU", C["blue"], "o", "-"), (sb, "sparse_banded", "sparse_banded, GPU", C["red"], "s", "-"), (lf, "sparse_lowfill", "sparse_lowfill, CPU", C["blue"], "o", "--"), (lf, "sparse_lowfill", "sparse_lowfill, GPU", C["red"], "s", "--")]:
    m = "minres_fused_comb_gpu" if "GPU" in lab else "minres_fused_comb_cpu"; ref = t[["cvxopt_default", "custom_exact_reduced"]].min(axis=1); r = (ref / t[m]).dropna()
    ax.plot(r.index, r.values, marker=mk, color=col, ls=ls, label=lab, mfc=col if ls == "-" else "white")
ax.axhline(1, color="#555555", lw=0.8); ax.text(290, 13, "iterative faster", fontsize=6.8, color="#333333"); ax.text(290, 0.55, "direct faster", fontsize=6.8, color="#333333")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("KKT dimension $N$"); ax.set_ylabel("speed-up over the faster direct solver"); panel(ax, "(a) equilibrated MINRES vs. factorisation"); ax.legend(loc="lower right", ncol=1, fontsize=6.3); ax.set_xlim(250, 1.3e5)
r_hi = (sb[["cvxopt_default", "custom_exact_reduced"]].min(axis=1) / sb["minres_fused_comb_cpu"]).dropna(); r_lo = (lf[["cvxopt_default", "custom_exact_reduced"]].min(axis=1) / lf["minres_fused_comb_gpu"]).dropna()
note(ax, (r_hi.index[-1], r_hi.iloc[-1]), f"{r_hi.iloc[-1]:.1f}×", (-16, 9), C["blue"]); note(ax, (r_lo.index[-1], r_lo.iloc[-1]), f"{r_lo.iloc[-1]:.1f}×", (-16, 9), C["red"])
ax = axes[1]
def cnt(m): return (int(IS.loc[m, "strict"]), int(IS.loc[m, "runs"])) if IS is not None and m in IS.index else (0, 0)
rows = [("HHL, $n_c=8$, 4 rounds", "hhl_scaled_kk_nc8_nofb", "hhl", False), ("HHL, $n_c=12$, 4 rounds", "hhl_scaled_kk_nc12_nofb", "hhl", False), ("HHL, $n_c=10$, 60 rounds", "hhl_nc10_ideal_ir60_nofb", "hhl", False),
        ("HHL, $n_c=10$, 250 samples", "hhl_nc10_aa500_ir60_nofb", "hhl", True), ("augmented, $\\varepsilon=10^{-10}$", "lcu_opt_eps1e-10_ideal_nofb", "aug", False), ("augmented, $\\varepsilon=0.3$ + refinement", "lcu_opt_eps0.3_ideal_ir_nofb", "aug", False),
        ("augmented, $\\varepsilon=0.3$, 250 samples", "lcu_opt_eps0.3_aa500_ir_nofb", "aug", True), ("augmented (CKS), $10^4$ samples", "lcu_cks_eps1e-6_aa2e4_ir_nofb", "aug", True),
        ("null space, $\\varepsilon=0.3$ + refinement", "lcu_oss_eps0.3_ideal_ir_nofb", "oss", False), ("null space, $\\varepsilon=0.3$, 250 samples", "lcu_oss_eps0.3_aa500_ir_nofb", "oss", True)]
fam_col = {"hhl": C["grey"], "aug": C["blue"], "oss": C["purple"]}; y = np.arange(len(rows))[::-1]
for yi, (lab, m, fam_, shots) in zip(y, rows):
    k, n = cnt(m)
    if n == 0: continue
    ax.barh(yi, 100 * k / n, color=fam_col[fam_], height=0.62, alpha=0.95, hatch="////" if shots else None, edgecolor="white", lw=0.0); ax.text(100 * k / n + 2, yi, f"{k}/{n}", va="center", fontsize=6.8, color="#333333")
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=6.6); ax.set_xlim(0, 118); ax.set_xlabel("strict success inside the IPM, no classical fallback [%]"); panel(ax, "(b) simulated quantum KKT solvers"); ax.grid(axis="y", visible=False)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=C["grey"], label="HHL"), Patch(color=C["blue"], label="polynomial, augmented"), Patch(color=C["purple"], label="polynomial, null space"), Patch(facecolor="#bbbbbb", hatch="////", edgecolor="white", label="finite-shot readout")],
          loc="upper center", bbox_to_anchor=(0.42, -0.2), fontsize=6.0, ncol=2)
save(fig, "fig_summary")

# Fig. 4 (fig_scaling): scaling with the seed range as a band, peak memory
MEM = pd.read_csv(os.path.join(T, "memory_main.csv")) if os.path.exists(os.path.join(T, "memory_main.csv")) else pd.DataFrame()
fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.85), constrained_layout=True, gridspec_kw={"width_ratios": [1, 1, 0.9]})
for ax, t, fam, ttl in [(axes[0], sb, "sparse_banded", "(a) sparse_banded (high fill)"), (axes[1], lf, "sparse_lowfill", "(b) sparse_lowfill (low fill)")]:
    for m in ["cvxopt_default", "custom_exact_reduced", "gmres_stalelu_comb", "minres_fused_comb_cpu", "minres_fused_comb_gpu"]:
        if m not in t: continue
        ax.plot(t.index, t[m], marker=MK[m], color=C[m], label=LAB[m])
        g = seedband(fam, m, C[m])
        if len(g): ax.fill_between(g.index, g.wall_min, g.wall_max, color=C[m], alpha=0.15, lw=0)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("KKT dimension $N$"); panel(ax, ttl)
axes[0].set_ylabel("wall time, whole IPM run [s]"); axes[1].sharey(axes[0])
c19 = sb.loc[19840]; note(axes[0], (19840, c19["cvxopt_default"]), f"{c19['cvxopt_default']:.0f} s", (-24, 8), C["black"]); note(axes[0], (19840, c19["minres_fused_comb_cpu"]), f"{c19['minres_fused_comb_cpu']:.1f} s", (6, -10), C["blue"])
note(axes[0], (79360, sb.loc[79360, "minres_fused_comb_gpu"]), f"{sb.loc[79360, 'minres_fused_comb_gpu']:.0f} s", (5, -9), C["red"]); note(axes[1], (79360, lf.loc[79360, "minres_fused_comb_gpu"]), f"{lf.loc[79360, 'minres_fused_comb_gpu']:.0f} s", (5, -9), C["red"])
note(axes[1], (79360, lf.loc[79360, "custom_exact_reduced"]), f"{lf.loc[79360, 'custom_exact_reduced']:.0f} s", (-24, 8), C["green"])
ax = axes[2]
if len(MEM):
    mm_ = MEM[(MEM.experiment == "E3_scaling") & (MEM.family == "sparse_banded")]
    for m in ["cvxopt_default", "custom_exact_reduced", "gmres_stalelu_comb", "minres_fused_comb_cpu"]:
        g = mm_[mm_.method == m].set_index("kkt_dim").rss_increase_median; g = g[(g.index >= 9920) & (g >= 1.0)]      # below ~1 MB the growth is under the sampling resolution (allocator reuse)
        ax.plot(g.index, g, marker=MK[m], color=C[m], label=LAB[m])
    g = mm_[mm_.method == "minres_fused_comb_gpu"].set_index("kkt_dim").gpu_pool_peak_median; g = g[g.index >= 9920]; ax.plot(g.index, g, marker="s", color=C["red"], ls="--", label="GPU MINRES, device memory pool")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("KKT dimension $N$ (sparse_banded)"); ax.set_ylabel("peak memory growth during the run [MB]"); panel(ax, "(c) memory"); ax.set_ylim(1, 2000)
    ax.set_xticks([1e4, 2e4, 4e4, 8e4]); ax.set_xticklabels(["$10^4$", "$2\\cdot10^4$", "$4\\cdot10^4$", "$8\\cdot10^4$"]); ax.minorticks_off(); ax.set_xlim(8000, 1e5)
h_, l_ = axes[0].get_legend_handles_labels(); h2, l2 = axes[2].get_legend_handles_labels(); h_ += [h2[-1]]; l_ += [l2[-1]]
fig.legend(h_, l_, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.13), fontsize=6.4)
save(fig, "fig_scaling")

# Fig. 6 (fig_gpu): GPU kernel
k = pd.read_csv(os.path.join(T, "kernel_per_iter_main.csv")); mv = pd.read_csv(os.path.join(T, "gpu_matvec_main.csv"))
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), constrained_layout=True)
for m, ls in [("minres_base_comb_cpu", "--"), ("minres_fused_comb_cpu", "-"), ("minres_base_comb_gpu", "--"), ("minres_fused_comb_gpu", "-")]:
    g = k[k.method == m]; axes[0].plot(g.kkt_dim, g.ms_per_iter, ls=ls, marker=MK[m], color=C[m], label=LAB[m], mfc=C[m] if ls == "-" else "white")
axes[0].plot(mv.kkt_dim, 1e3 * mv.t_matvec_gpu, ":", color=C["red"], lw=1.2, label="GPU sparse mat-vec alone"); axes[0].plot(mv.kkt_dim, 1e3 * mv.t_matvec_cpu, ":", color=C["blue"], lw=1.2, label="CPU sparse mat-vec alone")
axes[0].set_xscale("log"); axes[0].set_yscale("log"); axes[0].set_xlabel("KKT dimension $N$"); axes[0].set_ylabel("ms per MINRES iteration"); panel(axes[0], "(a) cost of one MINRES iteration")
g5 = k[k.method == "minres_fused_comb_gpu"].set_index("kkt_dim").ms_per_iter; g4 = k[k.method == "minres_base_comb_gpu"].set_index("kkt_dim").ms_per_iter
note(axes[0], (620, g4.loc[620]), f"{g5.min():.2f}–{g4.max():.2f} ms, flat: latency-bound", (-5, 14), C["red"]); note(axes[0], (2480, 1e3 * mv.loc[mv.kkt_dim == 2480, "t_matvec_gpu"].iloc[0]), f"mat-vec {1e3 * mv.t_matvec_gpu.median():.2f} ms", (5, -9), C["red"])
h_, l_ = axes[0].get_legend_handles_labels(); axes[0].legend(h_, l_, loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=6, ncol=2, frameon=False)
axes[1].plot(sb.index, sb["minres_fused_comb_cpu"] / sb["minres_fused_comb_gpu"], marker="s", color=C["red"], label="whole IPM run (latency-optimised kernel)")
axes[1].plot(sb.index, sb["minres_base_comb_cpu"] / sb["minres_base_comb_gpu"], marker="s", ls="--", mfc="white", color=C["red"], label="whole IPM run (baseline kernel)")
axes[1].plot(mv.kkt_dim, mv.speedup_gpu, marker="o", color=C["grey"], label="sparse mat-vec alone")
axes[1].axhline(1, color="#555555", lw=0.8); axes[1].axvline(19840, color="#555555", lw=0.6, ls=":"); axes[1].text(21000, 0.12, "break-even\n$N\\approx2\\times10^4$", fontsize=6.5, color="#333333")
axes[1].set_xscale("log"); axes[1].set_yscale("log"); axes[1].set_xlabel("KKT dimension $N$"); axes[1].set_ylabel("CPU time / GPU time"); panel(axes[1], "(b) GPU speed-up (T4 vs. 4 vCPU)"); axes[1].legend(loc="upper left", fontsize=6.5)
note(axes[1], (79360, (sb["minres_fused_comb_cpu"] / sb["minres_fused_comb_gpu"]).loc[79360]), f"{(sb['minres_fused_comb_cpu'] / sb['minres_fused_comb_gpu']).loc[79360]:.1f}×", (6, 0), C["red"]); note(axes[1], (79360, mv.speedup_gpu.iloc[-1]), f"{mv.speedup_gpu.iloc[-1]:.1f}×", (6, 0), C["grey"])
save(fig, "fig_gpu")

# Fig. 3 (fig_spectrum): spectrum along the IPM run
sp = pd.read_csv(os.path.join(T, "spectrum_E7.csv")); de = pd.read_csv(os.path.join(T, "deflation_E7.csv"))
fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7), constrained_layout=True)
ax = axes[0]; ax.plot(sp.ipm_iteration, sp.w_spread, marker="o", color=C["black"], label="NT spread $\\max w/\\min w$"); ax.plot(sp.ipm_iteration, sp.kappa_scaled, marker="s", color=C["blue"], label="$\\kappa(D\\tilde K D)$, Ruiz-scaled")
ax.set_yscale("log"); ax.set_xlabel("IPM iteration"); panel(ax, "(a) conditioning"); ax.legend(loc="upper left"); ax.axhspan(80, 180, color=C["blue"], alpha=0.08); ax.text(4.2, 250, "80–180", fontsize=7, color=C["blue"])
ax = axes[1]; ax.plot(sp.ipm_iteration, sp.minres_its_augmented, marker="o", color=C["blue"], label="augmented, combined test"); ax.plot(sp.ipm_iteration, sp.minres_its_backward_only, marker="o", ls="--", color=C["blue"], alpha=0.75, label="augmented, backward test only")
ax.plot(sp.ipm_iteration, sp.minres_its_reduced_jacobi, marker="^", color=C["red"], label="reduced, matrix-free, Jacobi")
ax.set_yscale("log"); ax.set_xlabel("IPM iteration"); ax.set_ylabel("MINRES iterations (2 solves)"); panel(ax, "(b) Krylov work"); ax.legend(fontsize=6, loc="upper left"); note(ax, (8, sp.minres_its_augmented.iloc[-1]), "20×", (-14, 9), C["blue"])
ax = axes[2]; ax.plot(de.n_deflated, de.minres_its, marker="o", color=C["green"]); ax.set_xscale("symlog", linthresh=5); ax.set_yscale("log"); ax.set_xlabel("deflated eigenvectors $k$ (smallest $|\\lambda|$)"); ax.set_ylabel("MINRES iterations"); panel(ax, "(c) exact-deflation bound")
for x_, y_ in zip(de.n_deflated, de.minres_its): ax.annotate(f"{de.minres_its.iloc[0]/y_:.1f}×", (x_, y_), textcoords="offset points", xytext=(4, 3), fontsize=6.5, color=C["green"])
ax.axhspan(de.minres_its.iloc[0] / 1.63, de.minres_its.iloc[0] / 1.38, color=C["grey"], alpha=0.18); ax.text(30, de.minres_its.iloc[0] / 1.5, "recycled MINRES\n(measured)", fontsize=6.2, color="#555555", va="center")
save(fig, "fig_spectrum")

# Fig. 2 (fig_criteria): acceptance tests
ab = pd.read_csv(os.path.join(T, "ablation_E2_sessionA.csv")).set_index("method")
order = [("minres_relres_1e-06_noscale", "relative residual $10^{-6}$"), ("minres_be_1e-10_noscale", "backward error $10^{-10}$"), ("minres_block_0.1_noscale", "block $\\eta=0.1$"), None,
         ("minres_relres_1e-06", "relative residual $10^{-6}$"), ("minres_be_1e-06", "backward error $10^{-6}$"), ("minres_be_1e-08", "backward error $10^{-8}$"), ("minres_be_1e-10", "backward error $10^{-10}$"), ("minres_be_1e-12", "backward error $10^{-12}$"),
         ("minres_block_0.5", "block $\\eta=0.5$"), ("minres_block_0.1", "block $\\eta=0.1$"), ("minres_block_0.01", "block $\\eta=0.01$"), ("minres_comb_0.1_1e-10", "combined ($\\eta=0.1$ and $10^{-10}$)"), ("minres_adaptive_nt_be", "adaptive backward (NT spread)")]
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.3), gridspec_kw={"width_ratios": [1.25, 1]}, constrained_layout=True)
ys = []; yy = len(order) - 1
for o in order:
    ys.append(yy if o is not None else None); yy -= 1
for (o, yv) in zip(order, ys):
    if o is None: continue
    m, l = o; col = C["red"] if ys.index(yv) < 3 else C["blue"]
    axes[0].barh(yv + 0.19, 100 * ab.loc[m, "accepted"], height=0.36, color="#d8d8d8"); axes[0].barh(yv - 0.19, 100 * ab.loc[m, "pure"], height=0.36, color=col)
    axes[0].text(101, yv - 0.19, f"{100*ab.loc[m,'pure']:.0f}", va="center", fontsize=6.3, color=col); fb = int(ab.loc[m, "fallbacks"])
    if fb: axes[0].text(2, yv + 0.19, f"{fb} fallbacks", va="center", fontsize=5.8, color="#444444")
    axes[1].barh(yv, ab.loc[m, "krylov"], color=col, height=0.62); axes[1].text(ab.loc[m, "krylov"] * 1.04, yv, f"{ab.loc[m,'krylov']:.0f}", va="center", fontsize=6.3, color=col)
labels_y = [yv for yv in ys if yv is not None]; labels = [o[1] for o in order if o is not None]
axes[0].set_yticks(labels_y); axes[0].set_yticklabels(labels, fontsize=7); axes[0].set_xlim(0, 112); axes[0].set_xlabel("% of 136 runs reaching the final IPM tolerance"); panel(axes[0], "(a) reliability by acceptance test")
axes[0].text(112, ys[1], "unscaled", rotation=270, va="center", fontsize=7, color=C["red"]); axes[0].text(112, (ys[5] + ys[-1]) / 2, "Ruiz-equilibrated", rotation=270, va="center", fontsize=7, color=C["blue"])
from matplotlib.patches import Patch
axes[0].legend(handles=[Patch(color="#d8d8d8", label="with classical fallback"), Patch(color=C["blue"], label="fallback-free (coloured)")], loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=6.3); axes[0].grid(axis="y", visible=False)
axes[1].set_yticks(labels_y); axes[1].set_yticklabels([]); axes[1].set_xscale("log"); axes[1].set_xlim(500, 9000); axes[1].set_xticks([600, 1000, 2000, 5000]); axes[1].set_xticklabels(["600", "1000", "2000", "5000"]); axes[1].minorticks_off(); axes[1].set_xlabel("median $\\Sigma$ Krylov iterations per IPM run"); panel(axes[1], "(b) cost"); axes[1].grid(axis="y", visible=False)
save(fig, "fig_criteria")

# Fig. 8 (fig_noise_outer): direction-error tolerance, tolerance sweep, outer refinement
o = pd.read_csv(os.path.join(T, "outer_E5.csv")); TS = pd.read_csv(os.path.join(T, "tolerance_sweep_main.csv"))
fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7), constrained_layout=True)
d7 = TS[TS.ipm_tol_level == 1e-7]
for mdl, col_, mk, lab_ in [("iso", C["blue"], "o", "isotropic relative error"), ("block", C["red"], "s", "block-wise relative error")]:
    g = d7[d7.model == mdl].sort_values("delta"); axes[0].plot(g.delta, 100 * g.accepted, marker=mk, color=col_, label=lab_)
axes[0].axvline(1e-7, color="#555555", lw=0.6, ls=":"); axes[0].text(1.25e-7, 45, "solver\ntolerance", fontsize=6.3, color="#333333")
axes[0].set_xscale("log"); axes[0].set_xlabel("injected error per Newton direction"); axes[0].set_ylabel("% runs reaching the final tolerance"); panel(axes[0], "(a) default tolerance ($\\tau=10^{-7}$)"); axes[0].legend(loc="center right", fontsize=6.3); axes[0].set_ylim(-5, 108)
ramp = {1e-8: "#0d366b", 1e-6: "#2a78d6", 1e-4: "#86b6ef"}          # single-hue ordinal ramp: darker = tighter tolerance
for tau, col_ in ramp.items():
    g = TS[(TS.ipm_tol_level == tau) & (TS.model == "iso")].sort_values("delta"); axes[1].plot(g.delta, 100 * g.accepted, marker="o", color=col_, label=f"$\\tau=10^{{{int(np.log10(tau))}}}$")
    ok_ = g[g.accepted >= 0.999]
    if len(ok_): axes[1].annotate("", xy=(ok_.delta.max(), 100), xytext=(ok_.delta.max(), 112), arrowprops=dict(arrowstyle="-|>", color=col_, lw=0.8))
axes[1].set_xscale("log"); axes[1].set_xlabel("injected isotropic error $\\delta$"); axes[1].set_ylabel("% of 30 runs reaching $10\\tau$"); panel(axes[1], "(b) tolerance sweep"); axes[1].legend(loc="lower left", fontsize=6.3); axes[1].set_ylim(-5, 118)
pv = o.pivot_table(index="round", columns="inner_solver", values="err", aggfunc="median")
lab = {"cvxopt_default": "exact inner solve", "minres_base_loose_1e-5": "MINRES, backward $10^{-4}$", "exact_noise_iso_1e-4": "exact + $10^{-4}$ error", "exact_noise_iso_1e-3": "exact + $10^{-3}$ error", "hhl_ideal_nc8_nofallback": "HHL, 8 clock qubits", "hhl_ideal_nc10_nofallback": "HHL, 10 clock qubits"}
colr = {"cvxopt_default": C["black"], "minres_base_loose_1e-5": C["blue"], "exact_noise_iso_1e-4": C["amber"], "exact_noise_iso_1e-3": C["red"], "hhl_ideal_nc8_nofallback": C["purple"], "hhl_ideal_nc10_nofallback": C["green"]}
mkr = {"cvxopt_default": "o", "minres_base_loose_1e-5": "s", "exact_noise_iso_1e-4": "^", "exact_noise_iso_1e-3": "v", "hhl_ideal_nc8_nofallback": "D", "hhl_ideal_nc10_nofallback": "P"}
for c_ in [c for c in lab if c in pv.columns]: axes[2].plot(pv.index, pv[c_].clip(1e-17), marker=mkr[c_], color=colr[c_], label=lab[c_])
axes[2].axhline(1e-7, color="#555555", lw=0.6, ls=":"); axes[2].set_yscale("log"); axes[2].set_xlabel("outer refinement round"); axes[2].set_ylabel("max(pres, dres, rel. gap), median"); panel(axes[2], "(c) optimisation-level refinement"); axes[2].legend(fontsize=5.6, loc="lower left")
save(fig, "fig_noise_outer")

# Fig. 9 (fig_qaccuracy): accuracy of the quantum solvers
h = pd.read_csv(os.path.join(T, "hhl_q1_ideal_main.csv")); a = pd.read_csv(os.path.join(T, "q5a.csv"))
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), constrained_layout=True)
xs = [c for c in h.columns if c not in ("scaling", "reciprocal_mode")]
for (sc, rm), lab_, col_ in [(("none", "resolution"), "no scaling, $C=\\Delta$", "#b0b0b0"), (("none", "kappa_known"), "no scaling, $\\kappa$ known", "#6b6b6b"), (("ruiz", "resolution"), "Ruiz, $C=\\Delta$", C["purple"]), (("ruiz", "kappa_known"), "Ruiz, $\\kappa$ known", C["red"])]:
    r = h[(h.scaling == sc) & (h.reciprocal_mode == rm)].iloc[0]; axes[0].plot([float(x) for x in xs], [r[x] for x in xs], marker="o", color=col_, label=lab_)
axes[0].axhspan(1e-8, 1e-7, color=C["blue"], alpha=0.12); axes[0].text(7.2, 2.2e-7, "what the IPM needs ($10^{-7}$)", fontsize=6.5, color=C["blue"]); axes[0].set_yscale("log"); axes[0].set_ylim(3e-8, 3)
axes[0].set_xlabel("clock (phase-estimation) qubits $n_c$"); axes[0].set_ylabel("relative forward error (ideal readout)"); panel(axes[0], "(a) HHL on captured KKT systems"); axes[0].legend(loc="upper right", fontsize=6.5)
ok = a[(a.solver == "lcu_walk") & (a.status == "ok") & a.poly.isin(["cks", "optimal"])]
for po, col_, lab_ in [("cks", C["blue"], "polynomial solver, CKS expansion"), ("optimal", C["red"], "polynomial solver, optimal polynomial")]:
    g = ok[ok.poly == po].groupby("eps_target").rel_err_ideal.agg(["median", "min", "max"]); axes[1].plot(g.index, g["median"], marker="o", color=col_, label=lab_); axes[1].fill_between(g.index, g["min"], g["max"], color=col_, alpha=0.12, lw=0)
axes[1].plot([1e-10, 0.3], [1e-10, 0.3], ":", color="#555555", label="error $=$ requested $\\varepsilon$"); axes[1].axhspan(1e-8, 1e-7, color=C["blue"], alpha=0.12)
hh = a[a.solver.str.startswith("hhl")]
for s_, txt in [("hhl_nc10", "HHL, 10 clock qubits"), ("hhl_nc8", "HHL, 8 clock qubits")]:
    g = hh[hh.solver == s_]
    if len(g): axes[1].axhline(g.rel_err_ideal.median(), color=C["purple"], lw=0.8, ls="--"); axes[1].text(1.5e-10, g.rel_err_ideal.median() * 1.6, txt + " (median)", fontsize=6.3, color=C["purple"])
axes[1].set_xscale("log"); axes[1].set_yscale("log"); axes[1].set_xlabel("requested polynomial accuracy $\\varepsilon$"); axes[1].set_ylabel("relative forward error (ideal readout)"); panel(axes[1], "(b) polynomial solver (12 systems, $\\kappa\\leq10^5$)"); axes[1].legend(fontsize=6.3, loc="lower right")
save(fig, "fig_qaccuracy")

# Fig. 11 (fig_qcost): cost of refinement
b = pd.read_csv(os.path.join(T, "q5b.csv")); key = ["instance_id", "factor_index"]
ref = b[(b.poly == "cks") & (b.eps_circuit == 1e-6) & (b.samples_per_round == b.samples_per_round.max()) & b["reached_1e-10"]].set_index(key)
fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7), constrained_layout=True)
cfg = [("cks", 1e-6, C["blue"], "CKS expansion, $\\varepsilon=10^{-6}$ (high-precision baseline)"), ("optimal", 1e-2, C["amber"], "optimal polynomial, $\\varepsilon=10^{-2}$"), ("optimal", 0.3, C["red"], "optimal polynomial, $\\varepsilon=0.3$ (proposed)")]
best = None
for po, ep, col_, lab_ in cfg:
    g = b[(b.poly == po) & (b.eps_circuit == ep)].set_index(key); r = []
    for spr, gg in g.groupby("samples_per_round"):
        cm = gg.index.intersection(ref.index); r.append((spr, (ref.loc[cm, "block_encoding_queries_total_with_AA"] / gg.loc[cm, "block_encoding_queries_total_with_AA"]).median(), gg.loc[cm, "rounds"].median(), gg.loc[cm, "block_encoding_queries_total_with_AA"].median()))
    r = pd.DataFrame(r, columns=["spr", "ratio", "rounds", "q"]); axes[0].plot(r.spr, r.q, marker="o", color=col_, label=lab_); axes[1].plot(r.spr, r.rounds, marker="o", color=col_, label=lab_); axes[2].plot(r.spr, r.ratio, marker="o", color=col_, label=lab_)
    if po == "optimal" and ep == 0.3: best = r
axes[0].set_xscale("log"); axes[0].set_yscale("log"); axes[0].set_xlabel("samples per refinement round"); axes[0].set_ylabel("block-encoding queries to residual $10^{-10}$"); panel(axes[0], "(a) total cost"); axes[0].legend(fontsize=6, loc="upper left")
axes[1].set_xscale("log"); axes[1].set_xlabel("samples per refinement round"); axes[1].set_ylabel("refinement rounds (median)"); panel(axes[1], "(b) rounds needed")
axes[2].set_xscale("log"); axes[2].set_yscale("log"); axes[2].set_xlabel("samples per refinement round"); axes[2].set_ylabel("query reduction vs. high-precision baseline"); panel(axes[2], "(c) paired reduction, 9 systems"); axes[2].axhline(1, color="#555555", lw=0.7)
if best is not None:
    for spr, val in zip(best.spr, best.ratio):
        if spr in (200, 500): note(axes[2], (spr, val), f"{val:.0f}×", (6, 5), C["red"])
save(fig, "fig_qcost")

# Fig. 10 (fig_poly): polynomial degrees
pt = pd.read_csv(os.path.join(T, "poly_table.csv"))
fig, ax = plt.subplots(figsize=(3.5, 2.5), constrained_layout=True)
for e, col_ in [(0.3, C["red"]), (0.01, C["amber"]), (1e-6, C["blue"])]:
    g = pt[np.isclose(pt.eps, e)]; ax.plot(g.kappa, g.d_cks, ls="--", marker="o", color=col_, alpha=0.8); ax.plot(g.kappa, g.d_opt, ls="-", marker="o", color=col_, label=(f"$\\varepsilon=10^{{{int(np.log10(e))}}}$" if e < 0.1 else f"$\\varepsilon={e:g}$"))
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("condition number $\\kappa$"); ax.set_ylabel("polynomial degree (walk steps)"); panel(ax, "dashed: CKS expansion, solid: optimal"); ax.legend(loc="upper left"); note(ax, (1000, 8113), "3.0× fewer", (-62, -14), C["red"]); note(ax, (1000, 20725), "2.5× fewer", (-62, -14), C["blue"])
save(fig, "fig_poly")

# Fig. 12 (fig_qinstances): quantum solvers inside coneqp, per instance
q = pd.read_csv(os.path.join(T, "integrated_runs_main.csv" if os.path.exists(os.path.join(T, "integrated_runs_main.csv")) else "q5c_q3_runs.csv"), low_memory=False); q5 = q[q.experiment == "Q5_integrated_cks"]
meths = [("lcu_opt_eps1e-10_ideal_nofb", "augmented, $\\varepsilon=10^{-10}$, ideal"), ("lcu_opt_eps0.3_ideal_ir_nofb", "augmented, $\\varepsilon=0.3$, ideal + IR"), ("lcu_opt_eps0.3_aa500_ir_nofb", "augmented, $\\varepsilon=0.3$, 250 samples"),
         ("lcu_cks_eps1e-6_aa2e4_ir_nofb", "augmented CKS, $10^4$ samples"), ("lcu_oss_eps0.3_ideal_ir_nofb", "null space, $\\varepsilon=0.3$, ideal + IR"), ("lcu_oss_eps0.3_aa500_ir_nofb", "null space, $\\varepsilon=0.3$, 250 samples"),
         ("hhl_nc10_ideal_ir60_nofb", "HHL $n_c=10$, ideal + IR"), ("hhl_nc10_aa500_ir60_nofb", "HHL $n_c=10$, 250 samples"), ("hhl_nc8_ideal_ir60_nofb", "HHL $n_c=8$, ideal + IR")]
insts = ["tiny_A", "tiny_B", "tiny_C", "dense_noeq_n10_k0_s0", "dense_kappa_n10_k0_s0", "dense_kappa_n10_k0_s1", "dense_kappa_n10_k3_s0", "dense_kappa_n10_k3_s1"]
fig, ax = plt.subplots(figsize=(7.2, 3.5), constrained_layout=True); ax.grid(False)
for j, inst in enumerate(insts):
    for i, (m, _) in enumerate(meths):
        r = q5[(q5.instance_id == inst) & (q5.method == m)]
        if not len(r): ax.text(j, i, "not run", ha="center", va="center", fontsize=6, color="#999999"); continue
        r = r.iloc[0]
        if r.strict_method_success: col, txt = C["good"], ("yes" if r.status == "optimal" else "yes*")
        elif r.n_solves_unaccepted_used > 0: col, txt = C["critical"], f"{int(r.n_solves_unaccepted_used)}"
        elif str(r.get("fail_cause", "")) == "readout_failed": col, txt = C["critical"], "readout"
        else: col, txt = C["warning"], "cap"
        ax.add_patch(plt.Rectangle((j - 0.45, i - 0.42), 0.9, 0.84, color=col, alpha=0.85, lw=0)); ax.text(j, i, txt, ha="center", va="center", fontsize=7, color="white" if col != C["warning"] else "#1b1b1b", fontweight="bold")
kap = q5[q5.method == "lcu_opt_eps0.3_ideal_ir_nofb"].groupby("instance_id").kappa_max.max(); N_ = q5.groupby("instance_id").kkt_dim.first()
def kfmt(v):
    m_, e_ = f"{v:.0e}".split("e"); e_ = int(e_); return (f"{m_}\\cdot10^{{{e_}}}" if e_ >= 2 else f"{v:.0f}")
ax.set_xticks(range(len(insts))); ax.set_xticklabels([f"{i.replace('dense_kappa_','dense_kappa_\n').replace('dense_noeq_','dense_noeq_\n')}\n$N={int(N_[i])}$\n$\\kappa_{{\\max}}={kfmt(kap[i])}$" for i in insts], fontsize=6)
ax.set_yticks(range(len(meths))); ax.set_yticklabels([m[1] for m in meths], fontsize=6.8); ax.set_xlim(-0.6, len(insts) - 0.4); ax.set_ylim(-0.6, len(meths) - 0.4); ax.invert_yaxis()
for s_ in ["top", "right", "left", "bottom"]: ax.spines[s_].set_visible(False)
ax.tick_params(length=0); panel(ax, "strict success per instance and solver, no classical fallback")
save(fig, "fig_qinstances")

# Fig. 16 (fig_replication): replication across sessions
if os.path.exists(os.path.join(T, "replication_main.csv")):
    rep = pd.read_csv(os.path.join(T, "replication_main.csv")); sess_mk = {"B": "o", "C": "s", "D": "^", "E": "D"}; sess_lab = {"B": "B: GPU, medium", "C": "C: GPU, small", "D": "D: CPU, small", "E": "E: CPU, medium"}
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True, constrained_layout=True)
    for ax, fam, ttl in [(axes[0], "sparse_banded", "(a) sparse_banded"), (axes[1], "sparse_lowfill", "(b) sparse_lowfill")]:
        for m in ["cvxopt_default", "custom_exact_reduced", "minres_fused_comb_cpu", "minres_fused_comb_gpu"]:
            for sname, mk in sess_mk.items():
                g = rep[(rep.family == fam) & (rep.method == m) & (rep.session == sname)].groupby("N").ratio.median()
                if len(g): ax.plot(g.index, g.values, marker=mk, ms=3.6, lw=0.9, color=C[m], mfc="white", alpha=0.95, label=None)
        ax.axhline(1, color="#555555", lw=0.8); ax.axhspan(0.9, 1.1, color="#999999", alpha=0.10, lw=0); ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("KKT dimension $N$"); panel(ax, ttl)
    axes[0].set_ylabel("wall time, other session / main session"); axes[0].set_ylim(0.5, 2.0); axes[0].set_yticks([0.5, 0.7, 1.0, 1.4, 2.0]); axes[0].set_yticklabels(["0.5", "0.7", "1", "1.4", "2"]); axes[0].minorticks_off()
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], color=C[m], lw=1.2, label=LAB[m]) for m in ["cvxopt_default", "custom_exact_reduced", "minres_fused_comb_cpu", "minres_fused_comb_gpu"]] + [Line2D([], [], color="#555555", marker=mk, mfc="white", lw=0, label=sess_lab[s_]) for s_, mk in sess_mk.items()]
    fig.legend(handles=hs, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.16), fontsize=6.3)
    save(fig, "fig_replication")

# Fig. 5 (fig_profiles): performance profiles
if os.path.exists(os.path.join(T, "profiles_main.csv")):
    PPm = pd.read_csv(os.path.join(T, "profiles_main.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7), sharey=True, constrained_layout=True)
    for ax, (st, ttl) in zip(axes, [("high-fill", "(a) sparse_banded, 36 instances"), ("low-fill", "(b) sparse_lowfill, 21 instances"), ("maros-meszaros", "(c) Maros–Mészáros, 48 QPs")]):
        g0 = PPm[PPm.set == st]
        for m, g in g0.groupby("method", sort=False):
            g = g.sort_values("tau"); ax.step(g.tau, g.rho, where="post", color=C[m], lw=1.5, ls="--" if m == "minres_fused_comb_auto" else "-", label=LAB[m])
        ax.set_xscale("log", base=2); ax.set_xlim(1, 2 ** 10); ax.set_ylim(-0.02, 1.03); ax.set_xlabel("$\\tau$ (within a factor $\\tau$ of the fastest)"); panel(ax, ttl)
    axes[0].set_ylabel("share of instances $\\rho(\\tau)$")
    h_, l_ = [], []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in l_: h_.append(h); l_.append(l)
    fig.legend(h_, l_, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.15), fontsize=6.3)
    save(fig, "fig_profiles")

# Fig. 7 (fig_realqp): Maros-Meszaros QPs
if os.path.exists(os.path.join(T, "mm_runs.csv")):
    mr = pd.read_csv(os.path.join(T, "mm_runs.csv")); b0 = mr[mr.method == "cvxopt_default"].set_index("instance_id").wall_method
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), constrained_layout=True)
    ax = axes[0]
    for m in ["custom_exact_reduced", "minres_fused_comb_cpu", "minres_fused_comb_switch_cpu"]:
        g = mr[mr.method == m].set_index("instance_id"); x_ = b0.reindex(g.index); fb = g.n_fallback_total > 0
        ax.scatter(x_[~fb], g.wall_method[~fb], s=14, marker=MK[m], color=C[m], label=LAB[m] + (" (no fallback)" if m != "custom_exact_reduced" else ""), zorder=3)
        if fb.any(): ax.scatter(x_[fb], g.wall_method[fb], s=16, marker=MK[m], facecolors="white", edgecolors=C[m], lw=0.9, label=LAB[m] + " (with fallback)", zorder=3)
    lim = [5e-3, 600]; ax.plot(lim, lim, color="#555555", lw=0.8); ax.plot(lim, [10 * v for v in lim], color="#999999", lw=0.6, ls=":"); ax.text(60, 300, "10× slower", fontsize=6.3, color="#333333", rotation=38, ha="center"); ax.text(60, 30, "equal", fontsize=6.3, color="#333333", rotation=38, ha="center")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim); ax.set_xlabel("CVXOPT default (sparse LDL$^{\\top}$) [s]"); ax.set_ylabel("method [s]"); panel(ax, "(a) whole-run time, 48 QPs"); ax.legend(fontsize=5.7, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    ax = axes[1]
    if os.path.exists(os.path.join(T, "mm_kappa_diag.csv")):
        dg = pd.read_csv(os.path.join(T, "mm_kappa_diag.csv"))
        ax.axvspan(80, 180, color=C["blue"], alpha=0.12, lw=0); ax.text(210, 2.2, "sparse_banded\n($\\kappa$ 80–180)", fontsize=6.2, color="#333333")
        ax.scatter(dg.kappa_scaled_max, dg.fallbacks.clip(lower=0.3), s=22, color=C["blue"], zorder=3)
        offs = {"QE226": (-30, 7), "QBEACONF": (-4, 7), "CVXQP3_S": (-18, -11), "DUAL1": (-8, 8), "PRIMAL4": (-14, -11), "CVXQP1_S": (-6, 8), "QSC205": (-34, 7), "QSHARE1B": (4, 3)}
        for _, r in dg.iterrows(): ax.annotate(r.instance, (r.kappa_scaled_max, max(r.fallbacks, 0.3)), textcoords="offset points", xytext=offs.get(r.instance, (4, 3)), fontsize=6, color="#333333")
        ax.set_xscale("log"); ax.set_yscale("symlog", linthresh=1); ax.set_ylim(-0.1, 250); ax.set_xlabel("max $\\kappa(D\\tilde KD)$ along the IPM run"); ax.set_ylabel("MINRES solves needing fallback"); panel(ax, "(b) why MINRES falls back on real QPs")
    save(fig, "fig_realqp")

# Fig. 13 (fig_shots): finite-shot runs over sampling seeds
if os.path.exists(os.path.join(T, "shots_main.csv")):
    sh = pd.read_csv(os.path.join(T, "shots_main.csv")); acc = pd.read_csv(os.path.join(T, "shots_accepted_accuracy_main.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), constrained_layout=True)
    ax = axes[0]; ax.plot(sh.samples_per_round, 100 * sh.strict_rate, marker="o", color=C["blue"], label="strict success (8 QPs × 5 seeds)")
    ax.fill_between(sh.samples_per_round, 100 * sh.strict_rate_min_seed, 100 * sh.strict_rate_max_seed, color=C["blue"], alpha=0.15, lw=0, label="range over sampling seeds")
    for _, r in sh.iterrows(): ax.annotate(f"{int(r.strict)}/{int(r.runs)}", (r.samples_per_round, 100 * r.strict_rate), textcoords="offset points", xytext=(4, -10), fontsize=6.5, color="#333333")
    ax.set_xscale("log"); ax.set_ylim(0, 108); ax.set_xlabel("accepted samples per refinement round"); ax.set_ylabel("% runs strict, no fallback"); panel(ax, "(a) success vs. samples per round"); ax.legend(fontsize=6.3, loc="lower right")
    ax = axes[1]; order_ = ["ideal", "S=100", "S=250", "S=1000", "S=4000"]; xs_ = {k_: i for i, k_ in enumerate(order_)}; a2 = acc[acc.readout.isin(order_)]
    for inst, g in a2.groupby("instance_id"):
        ax.plot([xs_[r] for r in g.readout], g.beta_median.clip(lower=1e-7), color="#bbbbbb", lw=0.6, marker="o", ms=2.4, zorder=1)
    med = a2.groupby("readout").agg(b=("beta_median", "median"), its=("ipm_its", "median")).reindex(order_)
    ax.plot(range(len(order_)), med.b, color=C["blue"], marker="o", lw=1.6, zorder=3, label="median over instances")
    for i, (r, row) in enumerate(med.iterrows()): ax.text(i, 0.42, f"{row.its:.0f} IPM its", ha="center", fontsize=6.0, color="#333333")
    ax.axhline(0.1, color="#555555", lw=0.6, ls=":"); ax.text(0.05, 0.13, "threshold $\\eta=0.1$", fontsize=6.2, color="#333333", ha="left")
    ax.set_yscale("log"); ax.set_xticks(range(len(order_))); ax.set_xticklabels(order_, fontsize=6.8); ax.set_ylim(1e-7, 1.0); ax.set_ylabel("block ratio $\\beta$ of accepted directions"); panel(ax, "(b) accepted directions sit near the threshold"); ax.legend(fontsize=6.3, loc="lower right")
    save(fig, "fig_shots")

# Fig. 14 (fig_oss): null-space target
if os.path.exists(os.path.join(T, "oss_q5d_main.csv")):
    od = pd.read_csv(os.path.join(T, "oss_q5d_main.csv")); oc = pd.read_csv(os.path.join(T, "oss_classical_main.csv"))
    stage_col = {"first": "#86b6ef", "middle": "#2a78d6", "last": "#0d366b"}                 # ordinal: IPM progress
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.75), constrained_layout=True)
    for ax, (xc, yc, ttl, xl, yl) in zip(axes[:2], [("kappa_augmented_scaled", "kappa_reduced_jacobi", "(a) condition number", "$\\kappa$, augmented (Ruiz)", "$\\kappa$, null space (Jacobi)"),
                                                   ("degree_augmented_eps3e-01", "degree_oss_eps3e-01", "(b) walk steps at $\\varepsilon=0.3$", "odd polynomial, augmented", "shifted Chebyshev, null space")]):
        for st, col_ in stage_col.items():
            g = od[od.stage == st]; ax.scatter(g[xc], g[yc], s=20, color=col_, label=f"{st} captured IPM iteration", zorder=3, edgecolors="white", lw=0.5)
        lo_, hi_ = min(od[xc].min(), od[yc].min()) * 0.5, max(od[xc].max(), od[yc].max()) * 2; ax.plot([lo_, hi_], [lo_, hi_], color="#555555", lw=0.7); ax.set_xlim(lo_, hi_); ax.set_ylim(lo_, hi_)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel(xl); ax.set_ylabel(yl); panel(ax, ttl)
    axes[0].legend(fontsize=5.8, loc="lower right"); axes[1].text(0.97, 0.04, "below the line: null space cheaper", transform=axes[1].transAxes, fontsize=6.2, ha="right", color="#333333")
    ax = axes[2]; fam_mk = {"dense_kappa": ("o", C["blue"]), "dense_noeq": ("s", C["red"]), "sparse_banded": ("^", C["green"]), "sparse_lowfill": ("v", C["amber"]), "sparse_scale": ("D", C["purple"])}
    for fam, (mk, col_) in fam_mk.items():
        g = oc[oc.family == fam].groupby("kkt_dim").ratio.median()
        if len(g): ax.plot(g.index, g.values, marker=mk, color=col_, label=fam)
    ax.axhline(1, color="#555555", lw=0.8); ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("KKT dimension $N$"); ax.set_ylabel("MINRES (augmented) / CG (null space) iterations"); panel(ax, "(c) classical Krylov work"); ax.legend(fontsize=5.8, loc="upper right")
    ax.set_yticks([0.6, 1, 2, 3]); ax.set_yticklabels(["0.6", "1", "2", "3"]); ax.minorticks_off()
    save(fig, "fig_oss")

# Fig. 15 (fig_qresources): quantum resources: equilibration, readout model, classical work per KKT solve
if os.path.exists(os.path.join(T, "equil_main.csv")):
    EQ = pd.read_csv(os.path.join(T, "equil_main.csv")); RB = pd.read_csv(os.path.join(T, "q5b.csv")); RV = pd.read_csv(os.path.join(T, "rounds_vs_minres_main.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.95), constrained_layout=True, gridspec_kw={"width_ratios": [1.05, 1, 1]})
    ax = axes[0]
    famsty = [("dense_kappa", C["blue"], "o"), ("dense_noeq", C["blue"], "s"), ("sanity", C["blue"], "^"), ("sparse_banded", C["red"], "o"), ("sparse_lowfill", C["red"], "s"), ("sparse_scale", C["red"], "^"),
              ("sdplib", C["teal"], "D"), ("maros_meszaros", C["amber"], "v")]
    for fam, col_, mk in famsty:
        g = EQ[EQ.family == fam]
        if len(g): ax.scatter(g.kappa_unscaled, g.kappa_equilibrated, s=13, marker=mk, color=col_, edgecolors="white", lw=0.4, zorder=3, label={"sdplib": "SDPLIB", "maros_meszaros": "Maros–Mészáros"}.get(fam, fam))
    xs_ = np.array([1, 1e26])
    ax.plot(xs_, xs_, color="#555555", lw=0.8, ls="-", label="no reduction"); ax.plot(xs_, xs_ / 1e3, color="#555555", lw=0.7, ls="--", label="$10^{3}\\times$ lower"); ax.plot(xs_, xs_ / 1e6, color="#555555", lw=0.7, ls=":", label="$10^{6}\\times$ lower")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(1, 1e26); ax.set_ylim(1, 1e20); ax.set_xlabel("max $\\kappa(\\tilde K)$, unscaled"); ax.set_ylabel("max $\\kappa(D\\tilde KD)$, equilibrated"); panel(ax, "(a) effect of equilibration")
    ax.legend(fontsize=5.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, handletextpad=0.25, columnspacing=0.7)
    ax = axes[1]; RB = RB.assign(x=np.sqrt(RB.kkt_dim / RB.samples_per_round))
    c0_ = None
    for (po, ep), col_, mk, lab_ in [(("cks", 1e-6), C["blue"], "o", "CKS, $\\varepsilon=10^{-6}$"), (("optimal", 1e-2), C["red"], "s", "optimal, $\\varepsilon=10^{-2}$"), (("optimal", 0.3), C["teal"], "D", "optimal, $\\varepsilon=0.3$")]:
        g = RB[(RB.poly == po) & np.isclose(RB.eps_circuit, ep)]
        ax.scatter(g.x, g.contraction_per_round, s=11, marker=mk, color=col_, edgecolors="white", lw=0.4, zorder=3, label=lab_)
        if po == "cks": c0_ = (g.contraction_per_round / g.x).median()
    xx = np.geomspace(0.012, 0.8, 50); ax.plot(xx, c0_ * xx, color=C["ink"], lw=1.0, label=f"$\\xi={c0_:.2f}\\sqrt{{N/(2S)}}$"); ax.axhline(0.3, color=C["teal"], lw=0.7, ls=":")
    ax.text(0.013, 0.36, "$\\varepsilon=0.3$", fontsize=6.2, color="#333333")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("$\\sqrt{N/(2S)}$ ($2S$: accepted samples per round)"); ax.set_ylabel("residual contraction per round $\\xi$"); panel(ax, "(b) readout model, 9 systems"); ax.legend(fontsize=5.8, loc="lower right")
    ax = axes[2]; mr_ = RV[RV.solver == "minres"]
    for fam, col_, mk in [("dense_kappa", C["blue"], "o"), ("sparse_banded", C["red"], "o"), ("sparse_lowfill", C["red"], "s")]:
        g = mr_[mr_.family == fam].sort_values("kkt_dim"); ax.plot(g.kkt_dim, g.per_solve, marker=mk, color=col_, mfc=col_ if mk == "o" else "white", label=f"MINRES, {fam}")
    qq = RV[RV.solver.isin(["lcu_opt_eps0.3_ideal_ir_nofb", "lcu_opt_eps0.3_aa500_ir_nofb"])]
    for m_, mfc_, lab_ in [("lcu_opt_eps0.3_ideal_ir_nofb", C["teal"], "refined LCU, ideal readout"), ("lcu_opt_eps0.3_aa500_ir_nofb", "white", "refined LCU, $S=250$")]:
        g = qq[qq.solver == m_]; ax.scatter(g.kkt_dim * (1.0 if mfc_ != "white" else 1.12), g.per_solve, s=14, marker="D", color=C["teal"], facecolors=mfc_, edgecolors=C["teal"], lw=0.8, zorder=3, label=lab_)
    hi_q = qq.per_solve.max(); lo_q = qq.per_solve.min()
    ax.fill_between([40, 1e5], lo_q, hi_q, color=C["teal"], alpha=0.12, lw=0); ax.text(60, hi_q * 1.15, "model (Thm. 1, $S\\propto N$): independent of $N$", fontsize=5.8, color="#333333")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(3, 1.2e5); ax.set_ylim(2, 4000); ax.set_xlabel("KKT dimension $N$"); ax.set_ylabel("classical mat-vecs per KKT solve"); panel(ax, "(c) classical work per Newton solve")
    ax.legend(fontsize=5.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, handletextpad=0.3, columnspacing=0.7)
    save(fig, "fig_qresources")
print("figures written to", F)
