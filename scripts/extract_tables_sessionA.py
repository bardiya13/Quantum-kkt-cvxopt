"""Tables of the paper that rest on session A (CPU notebook, profile medium): the acceptance-test ablation, the spectrum
and deflation study, the injected-error experiment at the default tolerance, outer refinement, the captured-system
experiments of the polynomial solver (accuracy, degree, refinement cost) and SDPLIB. Session B (GPU, medium) supplies
the auxiliary kernel and session-effect tables.

    python scripts/extract_tables_sessionA.py --sessionA <bundle A> --sessionB <bundle B> [--out results/tables]
"""
import argparse
import math
import os

import numpy as np
import pandas as pd

from bundles import TABLES, root_of, read_table

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--sessionA", required=True)
ap.add_argument("--sessionB", required=True)
ap.add_argument("--out", default=TABLES)
A = ap.parse_args(); T = A.out; os.makedirs(T, exist_ok=True)
RA = root_of(A.sessionA); RB = root_of(A.sessionB)
BA, BB = read_table(RA, "benchmark_results.csv"), read_table(RB, "benchmark_results.csv")
# classical
e = BA[BA.experiment == "E2_main"]
e.groupby("method").agg(runs=("run_id", "size"), accepted=("accepted", "mean"), pure=("pure_method_success", "mean"), fallbacks=("n_fallback_total", "sum"), krylov=("krylov_iters_total", "median"), wall=("wall_method", "median"), ipm=("iterations", "median")).reset_index().to_csv(f"{T}/ablation_E2_sessionA.csv", index=False)
for fam in ["sparse_banded", "sparse_lowfill"]:
    x = BB[(BB.experiment == "E3_scaling") & (BB.family == fam)]
    x.pivot_table(index="kkt_dim", columns="method", values="wall_method", aggfunc="median").to_csv(f"{T}/scaling_{fam}_sessionB.csv")
    x.pivot_table(index="kkt_dim", columns="method", values="krylov_iters_total", aggfunc="median").to_csv(f"{T}/krylov_{fam}_sessionB.csv")
a = BA[(BA.experiment == "E3_scaling") & (BA.family == "sparse_banded")].groupby(["method", "kkt_dim"]).wall_method.median(); b = BB[(BB.experiment == "E3_scaling") & (BB.family == "sparse_banded")].groupby(["method", "kkt_dim"]).wall_method.median()
j = pd.concat([a, b], axis=1, keys=["cpu_sess", "gpu_sess"]).dropna(); j["ratio"] = j.cpu_sess / j.gpu_sess; j.to_csv(f"{T}/session_effect.csv")
BA[BA.experiment == "E4_noise"].groupby("method").agg(runs=("run_id", "size"), accepted=("accepted", "mean"), ipm=("iterations", "median")).reset_index().to_csv(f"{T}/noise_E4.csv", index=False)
for name, out in [("spectrum_along_ipm.csv", "spectrum_E7.csv"), ("deflation_bound.csv", "deflation_E7.csv"), ("outer_refinement.csv", "outer_E5.csv"), ("instances.csv", "instances.csv"), ("qls_comparison_hhl_vs_cks.csv", "q5a.csv"), ("cks_refinement.csv", "q5b.csv")]:
    read_table(RA, name).to_csv(f"{T}/{out}", index=False)
read_table(RB, "gpu_matvec_benchmark.csv").to_csv(f"{T}/gpu_matvec.csv", index=False)
BA[BA.experiment == "S1_sdplib"].groupby("method").agg(n=("run_id", "size"), acc=("accepted", "mean"), strict=("strict_method_success", "mean"), fb=("n_fallback_total", "sum")).reset_index().to_csv(f"{T}/sdplib.csv", index=False)
L = read_table(RB, "linear_solve_logs.csv"); L = L[L.instance_id.str.startswith("sparse_banded")]; rows = []
for m in ["minres_base_comb_cpu", "minres_fused_comb_cpu", "minres_base_comb_gpu", "minres_fused_comb_gpu"]:
    for kd, g in L[L.method == m].groupby("kkt_dim"): rows.append({"method": m, "kkt_dim": kd, "ms_per_iter": 1e3 * (g.t_solve + g.t_check).sum() / max(g.krylov_iters.sum(), 1)})
pd.DataFrame(rows).to_csv(f"{T}/kernel_per_iter.csv", index=False)
# quantum integrated
q = BA[BA.experiment.isin(["Q5_integrated_cks", "Q3_integrated"])]; q.to_csv(f"{T}/q5c_q3_runs.csv", index=False)
Lc = read_table(RA, "linear_solve_logs.csv"); l = Lc[Lc.method.isin(q.method.unique()) & Lc.q_degree.notna()]; cost = []
for m, g in l.groupby("method"):
    sam = 2 * g.q_eff_shots.fillna(0) * (1 + g.n_refine); qs = (sam * g.q_degree / np.sqrt(g.q_p_postselect.clip(lower=1e-300))).sum()
    cost.append({"method": m, "steps_median": g.q_degree.median(), "steps_max": g.q_degree.max(), "psucc_min": g.q_p_postselect.min(), "samples": sam.sum(), "queries": qs if sam.sum() > 0 else np.nan, "qubits_max": g.q_n_qubits.max()})
pd.DataFrame(cost).to_csv(f"{T}/q5c_cost.csv", index=False)
# polynomial degrees (analytic)
from scipy.stats import binom
from scipy.fft import dct
def cks(kappa, eps):
    b = int(math.ceil(kappa ** 2 * math.log(kappa / eps))); j0 = int(math.ceil(math.sqrt(b * math.log(4 * b / eps)))); j = np.arange(j0 + 1); return 4.0 * (-1.0) ** j * binom.sf(b + j, 2 * b, 0.5)
def opt(kappa, eps):
    a = 1 / kappa; n = max(1, int(math.ceil((math.log(1 / eps) + math.log(1 / a) + math.log(1 + a)) / (math.log(1 + a) - math.log(1 - a))))); al = (1 + a) / (2 * (1 - a)); d = 2 * n - 1
    xs = np.cos(np.pi * (np.arange(d + 1) + 0.5) / (d + 1)); y = (2 * xs ** 2 - (1 + a * a)) / (1 - a * a); L1 = (y + (1 - a) / (1 + a)) / al; L2 = (y * y + (1 - a) / (2 * (1 + a)) * y - 0.5) / al ** 2; Lm, Ln = (L1, L1) if n == 1 else (L1, L2)
    for _ in range(max(n - 2, 0)): Lm, Ln = Ln, y * Ln / al - Lm / (4 * al * al)
    pv = (1 - (-1) ** n * (1 + a) ** 2 / (4 * a) * Ln) / xs; ck = dct(pv, type=2) / (d + 1); return ck[1::2]
rows = []
for k in [5, 20, 100, 1000]:
    for e_ in [3e-1, 1e-2, 1e-6]:
        c0, c1 = cks(k, e_), opt(k, e_); rows.append({"kappa": k, "eps": e_, "d_cks": 2 * len(c0) - 1, "d_opt": 2 * len(c1) - 1, "l1_cks": np.abs(c0).sum(), "l1_opt": np.abs(c1).sum()})
pt = pd.DataFrame(rows); pt["ratio_d"] = pt.d_cks / pt.d_opt; pt["ratio_cost"] = pt.ratio_d * (pt.l1_cks / pt.l1_opt) ** 2; pt.to_csv(f"{T}/poly_table.csv", index=False)
# SDPLIB per instance
s = BA[BA.experiment == "S1_sdplib"].copy(); rows = []
for inst, g in s.groupby("instance_id"):
    row = {"instance": inst.replace("sdplib_", ""), "kkt_dim": int(g.kkt_dim.iloc[0]), "n": int(g.n.iloc[0]), "m": int(g.m.iloc[0]) if "m" in g else np.nan}
    for m, gg in g.groupby("method"):
        r = gg.iloc[0]; row[f"{m}__acc"] = bool(r.accepted); row[f"{m}__strict"] = bool(r.strict_method_success); row[f"{m}__fb"] = int(r.n_fallback_total); row[f"{m}__its"] = r.iterations; row[f"{m}__kappa_scaled"] = r.kappa_scaled_max
    rows.append(row)
pd.DataFrame(rows).sort_values("kkt_dim").to_csv(f"{T}/sdplib_instances.csv", index=False)
print("tables written to", T)
