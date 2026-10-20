# S1: SDPLIB problems with conelp (optional).
SDPLIB_OPT = {"truss1": -8.999996, "truss2": -123.3804, "truss3": -9.109996, "truss4": -9.009996, "truss5": -132.6357,
              "hinf1": 2.0326, "hinf2": 10.967, "hinf3": 56.9, "hinf4": 274.764, "hinf5": 363.0, "hinf6": 449.0, "hinf7": 391.0, "hinf8": 116.0,
              "hinf9": 236.25, "hinf10": 109.0, "hinf11": 65.9, "hinf12": 0.2, "hinf13": 46.0, "hinf14": 13.0, "hinf15": 25.0,
              "control1": 17.78463, "control2": 8.3, "control3": 13.63327, "control4": 19.79423, "qap5": -436.0, "qap6": -381.44,
              "theta1": 23.0, "theta2": 32.87917, "mcp100": 226.1574, "arch0": 0.566517, "gpp100": -44.9435}

def find_sdplib_file(name):
    fname = f"{name}.dat-s"
    for d in CONFIG["sdplib"]["search_dirs"]:
        if not os.path.isdir(d): continue
        p = os.path.join(d, fname)
        if os.path.isfile(p): return p
        for root, dirs, files in os.walk(d):
            if fname in files: return os.path.join(root, fname)
            if root[len(d):].count(os.sep) >= 3: dirs[:] = []
    return None

def download_sdplib(name):
    import urllib.request
    cache = os.path.join(CONFIG["output_root"], "sdplib_cache"); os.makedirs(cache, exist_ok=True)
    p = os.path.join(cache, f"{name}.dat-s")
    if os.path.isfile(p): return p
    try:
        urllib.request.urlretrieve(CONFIG["sdplib"]["download_url"].format(name=name), p)
        return p if os.path.getsize(p) > 0 else None
    except Exception as e:
        return None

def parse_sdpa_sparse(path):
    lines = [l for l in open(path) if l.strip() and l.lstrip()[0] not in '"*']
    clean = lambda s: re.sub(r"[{}(),]", " ", s)
    m = int(clean(lines[0]).split()[0]); nblocks = int(clean(lines[1]).split()[0])
    sizes = [int(float(t)) for t in clean(lines[2]).split()][:nblocks]
    c = np.array([float(t) for t in clean(lines[3]).split()])[:m]
    entries = []
    for l in lines[4:]:
        t = l.split()
        if len(t) < 5: continue
        entries.append((int(t[0]), int(t[1]), int(t[2]), int(t[3]), float(t[4])))
    return m, sizes, c, entries

def sdpa_to_instance(name, path):
    m, sizes, c, entries = parse_sdpa_sparse(path)
    lp_blocks = [(bi, -s) for bi, s in enumerate(sizes) if s < 0]
    sdp_blocks = [(bi, s) for bi, s in enumerate(sizes) if s > 0]
    off_l = {}; n_l = 0
    for bi, s in lp_blocks: off_l[bi] = n_l; n_l += s
    off_s = {}; pos = n_l
    for bi, s in sdp_blocks: off_s[bi] = pos; pos += s * s
    cdim = pos
    G = np.zeros((cdim, m)); h = np.zeros(cdim)
    for matno, blk, i, j, v in entries:
        bi = blk - 1
        if bi in off_l:
            if i != j: continue
            row = off_l[bi] + (i - 1)
            if matno == 0: h[row] = -v
            else: G[row, matno - 1] = -v
        else:
            s = sizes[bi]; base = off_s[bi]
            rows = {base + (j - 1) * s + (i - 1), base + (i - 1) * s + (j - 1)}
            for row in rows:
                if matno == 0: h[row] = -v
                else: G[row, matno - 1] = -v
    dims = {"l": n_l, "q": [], "s": [s for _, s in sdp_blocks]}
    kkt_dim = m + n_l + sum(s * (s + 1) // 2 for s in dims["s"])
    opt = SDPLIB_OPT.get(name)
    return dict(id=f"sdplib_{name}", family="sdplib", kind="conelp", n=m, m=cdim, p=0, kkt_dim=kkt_dim, kappa_P_target=np.nan, kappa_P_measured=np.nan,
                density=float(np.count_nonzero(G) / G.size), seed=0, sparse=False, P=None, q=c, G=G, h=h, A=None, b=None, dims=dims,
                f_star=opt if opt is not None else np.nan, known_solution=opt is not None, sdplib_name=name, block_sizes=sizes, path=path)

def recompute_metrics_sdp(inst, sol):
    x, s, z = to_np(sol["x"]), to_np(sol["s"]), to_np(sol["z"])
    c, G, h, dims = inst["q"], inst["G"], inst["h"], inst["dims"]
    pobj = float(c @ x); dobj = float(-h @ z)
    pres = np.linalg.norm(G @ x + s - h) / max(1.0, np.linalg.norm(h))
    dres = np.linalg.norm(G.T @ z + c) / max(1.0, np.linalg.norm(c))
    def min_eig_cone(v):
        worst = 0.0; l = dims["l"]
        if l: worst = min(worst, float(v[:l].min()))
        off = l
        for k in dims["s"]:
            M = v[off:off + k * k].reshape((k, k), order="F"); M = 0.5 * (M + M.T)
            worst = min(worst, float(np.linalg.eigvalsh(M).min())); off += k * k
        return worst
    cone_viol = max(0.0, -min_eig_cone(s), -min_eig_cone(z))
    constr_viol = max(0.0, -min_eig_cone(h - G @ x)) / max(1.0, np.linalg.norm(h))
    gap = float(s @ z)
    out = {"pobj": pobj, "dobj_valid": np.nan, "dobj_cvx": dobj, "gap_rec": gap, "relgap_valid": np.nan, "relgap_rec": gap / max(1.0, abs(pobj)),
           "pres_rec": float(pres), "dres_rec": float(dres), "cone_viol": float(cone_viol), "constraint_viol": float(constr_viol), "mu_final": gap / max(1, dims["l"] + sum(dims["s"]))}
    out["obj_rel_err"] = float(abs(pobj - inst["f_star"]) / max(1.0, abs(inst["f_star"]))) if inst.get("known_solution") else np.nan
    out["x_rel_err"] = np.nan
    return out
METRIC_FUNCS["conelp"] = recompute_metrics_sdp

SDPLIB_INSTANCES = {}
if not CONFIG["sdplib"]["enabled"]:
    REC.status("S1_sdplib", "skipped", "disabled in CONFIG")
else:
    missing = []
    for name in CONFIG["sdplib"]["instances"]:
        path = find_sdplib_file(name)
        src = "local"
        if path is None and CONFIG["sdplib"]["download_if_missing"]:
            path = download_sdplib(name); src = "download"
        if path is None:
            missing.append(name); continue
        try:
            inst = sdpa_to_instance(name, path); inst["source"] = src
            SDPLIB_INSTANCES[inst["id"]] = inst; INSTANCES[inst["id"]] = inst
        except Exception as e:
            REC.failure(f"sdplib_{name}", "parse", repr(e)[:300])
    if not SDPLIB_INSTANCES:
        REC.status("S1_sdplib", "skipped", f"no SDPLIB files found in {CONFIG['sdplib']['search_dirs']} and download failed (Internet off?)" if missing else "no instances configured")
    else:
        print("SDPLIB instances loaded:", ", ".join(f"{i['sdplib_name']}(m={i['n']}, blocks={i['block_sizes'] if len(i['block_sizes'])<8 else len(i['block_sizes'])}, kkt_dim={i['kkt_dim']}, {i['source']})" for i in SDPLIB_INSTANCES.values()))
        if missing: print("  not found / not downloadable:", missing)
        t_s = time.perf_counter(); n_runs = 0
        sd_specs = {k: SPECS[k] for k in ["cvxopt_default", "custom_exact", "minres_relres_1e-06", "minres_rrfloor_1e-06", "minres_be_1e-08", "minres_be_1e-12", "minres_block_0.1", "minres_comb_0.1_1e-10"] if k in SPECS}
        max_inexact_dim = CONFIG["sdplib"].get("max_kkt_dim_inexact", 300)
        for inst in SDPLIB_INSTANCES.values():
            for name, spec in sd_specs.items():
                if spec["kind"] != "baseline" and inst["kkt_dim"] > CONFIG["sdplib"]["max_kkt_dim_custom"]:
                    REC.status(f"S1_{inst['sdplib_name']}_{name}", "skipped", f"kkt_dim {inst['kkt_dim']} > max_kkt_dim_custom"); continue
                if spec["kind"] == "minres" and inst["kkt_dim"] > max_inexact_dim:
                    REC.status(f"S1_{inst['sdplib_name']}_{name}", "skipped", f"kkt_dim {inst['kkt_dim']} > max_kkt_dim_inexact={max_inexact_dim}"); continue
                try:
                    rec, sol, st = run_instance(inst, spec, repeat=0, compute_cond=(name == "custom_exact"),
                                                capture=(name == "custom_exact" and inst["sdplib_name"] in CONFIG["sdplib"]["quantum_instances"]),
                                                cond_scaled=(name == "custom_exact"), experiment="S1_sdplib")
                    n_runs += 1
                except Exception as e:
                    REC.failure(f"{inst['id']}__{name}", "sdplib_run", repr(e)[:300], method=name, instance_id=inst["id"])
        REC.status("S1_sdplib", "done", f"{n_runs} runs in {time.perf_counter() - t_s:.0f}s")
        dfs = REC.df("benchmark_results"); dfs = dfs[dfs.experiment == "S1_sdplib"]
        print(dfs[["instance_id", "method", "status", "iterations", "pobj", "obj_rel_err", "pres_rec", "dres_rec", "cone_viol", "kappa_max", "n_fallback_total", "wall_method", "accepted"]].round(6).to_string(index=False))
        # quantum prototype on captured SDP KKT systems (tiny ones only)
        qn = 0
        for inst in SDPLIB_INSTANCES.values():
            if inst["id"] in CAPTURED and QCFG["enabled"]:
                caps = CAPTURED[inst["id"]]
                for fi in select_factor_indices(caps):
                    for n_clock in QCFG["n_clock_list"]:
                        for sc, cm in [("none", "resolution"), ("ruiz", "kappa_known")]:
                            evaluate_captured_system(inst["id"], caps[fi], n_clock, sc, cm, 0.0, QCFG["shots_list"], QCFG["eff_shots_list"], QCFG["sampling_repeats"], QCFG["ir_rounds"], inst["n"]); qn += 1
        if qn:
            REC.status("Q1_captured_sdplib", "done", f"{qn} configurations on captured SDPLIB KKT systems")
            run_reg_experiment({k: v for k, v in CAPTURED.items() if k.startswith("sdplib")}, "sdplib")
            qq = REC.df("quantum_results"); qq = qq[(qq.instance_id.str.startswith("sdplib")) & (qq.recon == "ideal_statevector")]
            qq = qq[qq.ir_round == 0]
            print(qq[["instance_id", "factor_index", "kappa_kkt", "n_clock", "scaling", "reciprocal_mode", "status", "reason", "n_qubits", "p_postselect_exact", "rel_err_original"]].to_string(index=False))
    REC.flush(min_interval=0)
