"""Condition numbers along the IPM run for eight Maros-Meszaros QPs, before and after Ruiz equilibration, together with the
fallbacks of the combined-test MINRES on the same instances (the Maros-Meszaros rows of Table 15 and Fig. 7(b)).

    python scripts/mm_kappa_diag.py [--mm-dir DIR] [--out results/tables]

Runs cells 01-09 of the notebook (configuration up to the runner) with these instances, cond_limit = 2000 and the
synthetic families cut to their smallest size, then solves each instance with the exact solver (condition numbers at
every iteration) and with minres_fused_comb_cpu. The .mat files are read from --mm-dir or downloaded. About six
minutes on a 4-core CPU.
"""
import argparse
import glob
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INSTANCES = ["DUAL1", "PRIMAL4", "CVXQP1_S", "CVXQP3_S", "QSHARE1B", "QSC205", "QBEACONF", "QE226"]

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--mm-dir", help="directory with the Maros-Meszaros .mat files")
ap.add_argument("--out", default=os.path.join(ROOT, "results", "tables"))
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
os.environ.setdefault("QKKT_PROFILE", "small")
os.environ.setdefault("MPLBACKEND", "Agg")

cells = sorted(p for p in glob.glob(os.path.join(ROOT, "src", "cells", "*.py")) if os.path.basename(p)[:2] <= "09")
ns = {"__name__": "__main__"}
for path in cells:
    src = open(path, encoding="utf-8").read()
    if os.path.basename(path) == "01_config.py":
        # restrict the run before the configuration is hashed and written
        cut = src.index("CONFIG_HASH = ")
        src = src[:cut] + (
            f"CONFIG['maros_meszaros']['instances'] = {INSTANCES!r}\n"
            f"CONFIG['maros_meszaros']['search_dirs'] = {[args.mm_dir] if args.mm_dir else []!r} + CONFIG['maros_meszaros']['search_dirs']\n"
            "CONFIG['quantum']['qiskit_crosscheck'] = False; CONFIG['cond_limit'] = 2000\n"
            "for fam in CONFIG['families'].values(): fam['n_list'] = fam['n_list'][:1]\n") + src[cut:]
    exec(compile(src, path, "exec"), ns)

run_instance, SPECS, MM, to_np, pd = ns["run_instance"], ns["SPECS"], ns["MM_INSTANCES"], ns["to_np"], ns["pd"]
rows, path_rows = [], []
for inst in sorted(MM.values(), key=lambda i: i["kkt_dim"]):
    rec0, sol0, st0 = run_instance(inst, SPECS["custom_exact"], repeat=0, compute_cond=True, cond_scaled=True, experiment="diag")
    inst.update({"f_star": rec0["pobj"], "x_star": to_np(sol0["x"]), "known_solution": True})
    rec1, _, st1 = run_instance(inst, SPECS["minres_fused_comb_cpu"], repeat=0, experiment="diag")
    F0, S1 = st0["factors"], st1["solves"]
    for f in F0:
        path_rows.append({"instance": inst["mm_name"], "it": f["factor_index"], "kappa": f["kappa"], "kappa_scaled": f["kappa_scaled"], "nt": f["nt_proxy"]})
    fb = [s["factor_index"] for s in S1 if s["fallback"]]
    rows.append({"instance": inst["mm_name"], "N": inst["kkt_dim"], "n": inst["n"], "p": inst["p"], "nnzP_lt_n": inst["mm_nnz_P"] < inst["n"],
                 "ipm_its": rec0["iterations"], "kappa_max": max(f["kappa"] for f in F0), "kappa_scaled_first": F0[0]["kappa_scaled"],
                 "kappa_scaled_max": max(f["kappa_scaled"] for f in F0), "fallbacks": rec1["n_fallback_total"], "first_fallback_ipm_it": min(fb) if fb else -1,
                 "krylov_total": rec1["krylov_iters_total"], "maxiter_hits": int(sum(1 for s in S1 if s.get("krylov_info") == 1))})
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(os.path.join(args.out, "mm_kappa_diag.csv"), index=False)
pd.DataFrame(path_rows).to_csv(os.path.join(args.out, "mm_kappa_path.csv"), index=False)
print(pd.DataFrame(rows).to_string())
