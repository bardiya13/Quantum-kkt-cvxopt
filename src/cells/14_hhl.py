# HHL: circuit, readout of a signed and scaled vector, and the 'hhl' KKT-solver kind.
def pad_to_pow2(A, b, lam_pad):
    N0 = A.shape[0]; n_d = max(1, int(math.ceil(math.log2(N0)))); N = 2 ** n_d
    Ap = np.zeros((N, N)); Ap[:N0, :N0] = A
    if N > N0: Ap[N0:, N0:] = lam_pad * np.eye(N - N0)
    bp = np.zeros(N); bp[:N0] = b
    return Ap, bp, N0

def spectral_bound(A, method, rng):
    """Upper estimate of ||A||_2 by power iteration (oracle-compatible) or exact eigvalsh (diagnostic)."""
    if method == "exact":
        return float(np.abs(np.linalg.eigvalsh(A)).max()), "exact"
    q = CONFIG["quantum"]; v = rng.standard_normal(A.shape[0]); v /= np.linalg.norm(v); lam = 0.0
    for _ in range(q["power_iters"]):
        w = A @ v; lam = float(np.linalg.norm(w))
        if lam == 0: break
        v = w / lam
    return lam * q["power_safety"], "power_iteration"

class HHLSystem:
    """HHL circuit for a fixed real symmetric matrix A (N = 2^n_d). Built once per matrix; .run() solves for a given rhs."""
    def __init__(self, A, n_clock, rng, spectral="power", c_mode="resolution"):
        t0 = time.perf_counter()
        self.c_mode = c_mode
        A = np.asarray(A, dtype=float)
        assert A.shape[0] == A.shape[1] and np.allclose(A, A.T, atol=1e-10 * max(1, np.abs(A).max())), "A must be symmetric"
        self.N = A.shape[0]; self.n_d = int(round(math.log2(self.N))); assert 2 ** self.n_d == self.N
        self.n_c = int(n_clock); self.A = A; self.rng = rng
        self.anc = 0; self.clk = list(range(1, 1 + self.n_c)); self.dat = list(range(1 + self.n_c, 1 + self.n_c + self.n_d))
        self.flg = 1 + self.n_c + self.n_d; self.n_qubits = self.flg + 1
        check_qubit_budget(self.n_qubits)
        self.lam_max_est, self.bound_method = spectral_bound(A, spectral, rng)
        Nc = 2 ** self.n_c
        self.t = 2 * np.pi * (Nc // 2 - 1) / (Nc * self.lam_max_est)
        self.Delta = 2 * np.pi / (Nc * self.t)                  # clock resolution
        # reciprocal constant C: "resolution" -> C = Delta (no knowledge of lambda_min assumed; worst postselection);
        # "kappa_known" -> C = k_min*Delta with k_min = floor(|lambda|_min/Delta) (HHL's standard assumption that kappa is known;
        # here |lambda|_min is taken from the exact spectrum; this is an assumption of the method and timed as t_spectral_assumption)
        t_sa = time.perf_counter()
        if c_mode == "kappa_known":
            lam_min_abs = float(np.abs(np.linalg.eigvalsh(A)).min())
            self.k_min = int(max(1, min(Nc // 2 - 1, math.floor(lam_min_abs / self.Delta))))
        else:
            self.k_min = 1
        self.t_spectral_assumption = time.perf_counter() - t_sa
        self.C = self.k_min * self.Delta
        U = sla.expm(1j * A * self.t)
        self.U_pow = [U]
        for j in range(1, self.n_c): self.U_pow.append(self.U_pow[-1] @ self.U_pow[-1])
        self.U_pow_dag = [u.conj().T for u in self.U_pow]
        self.xp = DEVICE_XP if CONFIG["quantum"].get("device", "cpu") == "gpu" else np      # simulator backend
        self.U_pow_x = [self.xp.asarray(u) for u in self.U_pow]; self.U_pow_dag_x = [self.xp.asarray(u) for u in self.U_pow_dag]
        # rotation table: clock value k -> signed integer k_s -> theta
        self.thetas = {}
        for k in range(1, Nc):
            ks = k if k < Nc // 2 else k - Nc
            if ks == 0: continue
            amp = np.sign(ks) * min(1.0, self.k_min / abs(ks))      # C / lambda_tilde, clipped to 1 for |lambda| below the assumed minimum
            self.thetas[k] = 2 * np.arcsin(amp)
        self.t_prep = time.perf_counter() - t0
        # diagnostics from the exact spectrum, not used by the circuit
        t1 = time.perf_counter()
        ev = np.linalg.eigvalsh(A); self.eig = ev
        self.kappa = float(np.abs(ev).max() / max(np.abs(ev).min(), 1e-300))
        self.lam_max_true = float(np.abs(ev).max()); self.lam_min_abs = float(np.abs(ev).min())
        self.n_eig_below_resolution = int(np.sum(np.abs(ev) < self.Delta / 2))
        self.n_eig_unresolved_10pct = int(np.sum(np.abs(ev) < 5 * self.Delta))   # eigenvalues with >10% relative quantisation error
        self.t_diag = time.perf_counter() - t1

    def reference(self, b):
        r = np.ones(self.N) / np.sqrt(self.N)
        Ar = self.A @ r
        s = np.linalg.norm(b) / max(np.linalg.norm(Ar), 1e-300)
        return r * s, Ar * s

    def run(self, b, shots=0, sampling_repeats=1, shots_list=None, x_exact=None, eff_shots=0):
        """Run the circuit for rhs b. Returns dict with ideal-statevector and finite-shot reconstructions and diagnostics."""
        b = np.asarray(b, dtype=float); nb = np.linalg.norm(b)
        out = {"n_qubits": self.n_qubits, "n_clock": self.n_c, "n_data": self.n_d, "N": self.N, "kappa": self.kappa,
               "lam_max_est": self.lam_max_est, "lam_max_true": self.lam_max_true, "lam_min_abs": self.lam_min_abs,
               "clock_resolution": self.Delta, "reciprocal_mode": self.c_mode, "k_min": self.k_min, "C_over_lam_max": self.C / self.lam_max_est,
               "n_eig_below_resolution": self.n_eig_below_resolution,
               "n_eig_unresolved_10pct": self.n_eig_unresolved_10pct, "t_prep": self.t_prep, "t_diag": self.t_diag,
               "spectral_bound_method": self.bound_method, "rhs_norm": nb}
        if nb == 0:
            out.update({"x_ideal": np.zeros(self.N), "x_shots": np.zeros(self.N), "status": "zero_rhs"}); return out
        t0 = time.perf_counter()
        r_ref, Ar = self.reference(b)
        v = np.concatenate([b, Ar]); vnorm = np.linalg.norm(v); v = v / vnorm
        sim = MiniSim(self.n_qubits, xp=self.xp)
        sim.prepare(v.astype(complex), self.dat + [self.flg])
        for q in self.clk: sim.h(q)
        for j, q in enumerate(self.clk): sim.cu(self.U_pow_x[j], q, self.dat, name="c-U^2^j")
        sim.iqft(self.clk)
        Nc = 2 ** self.n_c
        th_arr = np.zeros(Nc)
        for k, th in self.thetas.items(): th_arr[k] = th
        sim.uniform_ry(th_arr, self.clk, self.anc)               # all 2^n_c - 1 reciprocal rotations, vectorised
        sim.qft(self.clk)
        for j in reversed(range(self.n_c)): sim.cu(self.U_pow_dag_x[j], self.clk[j], self.dat, name="c-U^-2^j")
        for q in self.clk: sim.h(q)
        out["t_circuit_sim"] = time.perf_counter() - t0
        out["gate_counts"] = sim.gate_summary(); out["n_gates_highlevel"] = len(sim.gates); out["depth_highlevel"] = sim.depth
        # ideal statevector diagnostics (not available on hardware)
        t1 = time.perf_counter()
        p_anc = sim.probs([self.anc]); out["p_postselect_exact"] = float(p_anc[1])
        p_anc_clk = sim.probs([self.anc] + self.clk).reshape(-1)     # index = anc + 2*clockvalue
        out["p_leak_exact"] = float(p_anc_clk[1::2][1:].sum())      # anc=1 and clock != 0
        fixed = {self.anc: 1}; fixed.update({q: 0 for q in self.clk})
        amp = sim.amplitudes(fixed)                                  # remaining qubits: dat (low) then flg (high)
        x_id, r_id = amp[:self.N], amp[self.N:]
        c = np.vdot(r_ref, r_id) / np.vdot(r_ref, r_ref)
        x_ideal = (x_id / c) if abs(c) > 0 else np.full(self.N, np.nan)
        out["ideal_imag_fraction"] = float(np.linalg.norm(x_ideal.imag) / max(np.linalg.norm(x_ideal), 1e-300))
        out["x_ideal"] = x_ideal.real
        out["scale_from_postselection"] = float(np.sqrt(max(out["p_postselect_exact"] * vnorm ** 2 / self.C ** 2 - np.dot(r_ref, r_ref), 0.0)))
        # finite-shot readout: Z setting and X setting on the flag
        meas = [self.anc, self.flg] + self.dat                       # index = anc + 2*flg + 4*i
        pZ = sim.probs(meas)
        sim.h(self.flg); pX = sim.probs(meas)
        out["pZ"], out["pX"] = pZ, pX
        out["t_readout_probs"] = time.perf_counter() - t1
        out["r_ref"] = r_ref; out["v_norm"] = vnorm; out["b"] = b
        out["x_exact"] = np.linalg.solve(self.A, b) if x_exact is None else x_exact      # classical reference, diagnostic only
        if shots:
            recs = reconstruct_from_shots(self, out, shots, sampling_repeats, self.rng)
            out["x_shots"] = recs[0]["x_rec"]; out["shot_records"] = recs
        if eff_shots:
            recs = reconstruct_aa_emulated(self, out, eff_shots, sampling_repeats, self.rng)
            out["x_aa"] = recs[0]["x_rec"]; out["aa_records"] = recs
        return out

def reconstruct_from_counts(sys_, out, cZ, cX, denom, rep=0, mode="shots"):
    """Rebuild a signed, scaled vector from Z- and X-setting counts (shape [N, flg, anc]); denom = shots per setting."""
    N = sys_.N; r_ref = out["r_ref"]; x_exact = out["x_exact"]; nx = max(np.linalg.norm(x_exact), 1e-300); t0 = time.perf_counter()
    z0, z1 = cZ[:, 0, 1].astype(float), cZ[:, 1, 1].astype(float); n0, n1 = z0.sum(), z1.sum()
    post_rate = (n0 + n1) / denom
    if n0 == 0 or n1 == 0:
        return {"repeat": rep, "readout": mode, "shots_per_setting": denom, "postselect_rate": post_rate, "status": "postselection_failed", "x_rec": np.full(N, np.nan),
                "rel_err": np.nan, "rel_err_ls": np.nan, "rel_resid": np.nan, "t_reconstruct": time.perf_counter() - t0}
    norm_x = np.linalg.norm(r_ref) * np.sqrt(n0 / n1); mag = norm_x * np.sqrt(z0 / n0)
    sign = np.sign(cX[:, 0, 1].astype(float) - cX[:, 1, 1].astype(float)); sign[sign == 0] = 1.0
    x_rec = sign * mag; b = out["b"]; Ax = sys_.A @ x_rec
    alpha = float(Ax @ b) / max(float(Ax @ Ax), 1e-300); x_ls = alpha * x_rec
    n_sign_err = int(np.sum((np.sign(x_rec) != np.sign(x_exact)) & (np.abs(x_exact) > 1e-3 * nx)))
    return {"repeat": rep, "readout": mode, "shots_per_setting": denom, "postselect_rate": post_rate, "status": "ok", "postselected_shots": int(n0 + n1), "x_rec": x_rec, "x_rec_ls": x_ls,
            "rel_err": float(np.linalg.norm(x_rec - x_exact) / nx), "rel_err_ls": float(np.linalg.norm(x_ls - x_exact) / nx), "rel_resid": float(relnorm(Ax - b, b)),
            "scale_rel_err": float(abs(norm_x - nx) / nx), "n_sign_errors": n_sign_err, "t_reconstruct": time.perf_counter() - t0}

def reconstruct_from_shots(sys_, out, shots, repeats, rng):
    """Raw finite-shot readout (hardware-like): sample the full Born distribution, postselect anc = 1."""
    recs = []
    for rep in range(repeats):
        cZ = sample_counts(out["pZ"], shots, rng).reshape(sys_.N, 2, 2); cX = sample_counts(out["pX"], shots, rng).reshape(sys_.N, 2, 2)
        recs.append(reconstruct_from_counts(sys_, out, cZ, cX, shots, rep, "shots"))
    return recs

def reconstruct_aa_emulated(sys_, out, eff_shots, repeats, rng):
    """Readout with emulated amplitude amplification: samples are drawn from the distribution conditioned on anc = 1 (what ideal AA
    would deliver), so eff_shots is the number of accepted samples per setting. The AA circuit is not simulated; its cost is reported as
    aa_overhead = 1/sqrt(P(anc=1)) circuit repetitions per accepted sample (raw-shot equivalent without AA: eff_shots / P(anc=1))."""
    recs = []; p_post = out["p_postselect_exact"]
    for rep in range(repeats):
        pZ = out["pZ"].reshape(sys_.N, 2, 2).copy(); pZ[:, :, 0] = 0.0; pX = out["pX"].reshape(sys_.N, 2, 2).copy(); pX[:, :, 0] = 0.0
        if pZ.sum() <= 0 or pX.sum() <= 0:
            recs.append({"repeat": rep, "readout": "aa_shots", "shots_per_setting": eff_shots, "status": "postselection_failed", "x_rec": np.full(sys_.N, np.nan), "rel_err": np.nan, "rel_err_ls": np.nan, "rel_resid": np.nan, "t_reconstruct": 0.0}); continue
        cZ = sample_counts(pZ.reshape(-1), eff_shots, rng).reshape(sys_.N, 2, 2); cX = sample_counts(pX.reshape(-1), eff_shots, rng).reshape(sys_.N, 2, 2)
        rec = reconstruct_from_counts(sys_, out, cZ, cX, eff_shots, rep, "aa_shots")
        rec.update({"aa_overhead_per_sample": 1.0 / math.sqrt(max(p_post, 1e-300)), "raw_shots_equivalent_no_AA": eff_shots / max(p_post, 1e-300), "postselect_rate": p_post})
        recs.append(rec)
    return recs

def regularize_kkt(K, asm, delta):
    """Primal-dual regularisation K + delta*J, J = diag(+I_n, -I_{p+m}) (same sign convention as CVXOPT's kktreg)."""
    if delta == 0: return K
    N = K.shape[0]; J = np.ones(N); J[asm.n:] = -1.0
    return K + delta * np.diag(J)

class HHLKind:
    """KKT-solver kind 'hhl' inside the protocol of KKTSolveEngine.
    target = (optionally regularised) K -> (optionally Ruiz-scaled) -> padded to 2^n -> HHL circuit -> readout (ideal / raw shots / AA-emulated).
    The engine verifies the direction on the original system and, if the test fails, calls base_solve again on the true residual
    (iterative refinement with the quantum solver as preconditioner), then falls back classically if allowed."""
    def __init__(self, spec, K, asm, W, eng):
        Kd = K.toarray() if sp.issparse(K) else np.asarray(K, dtype=float)
        self.reg = float(spec.get("reg", 0.0))
        self.scaled = spec.get("scaling", "ruiz") == "ruiz"
        self.d = ruiz_equilibrate(Kd) if self.scaled else None
        Ks = apply_scaling(Kd, self.d) if self.scaled else Kd
        Ks = regularize_kkt(Ks, asm, self.reg)            # on the scaled system (unit row norms), so kappa <= ~(1+delta)/delta
        self.rng = rng_for("hhl", asm.inst["id"], spec["name"])
        lam, _ = spectral_bound(Ks, CONFIG["quantum"]["spectral_bound"], self.rng)
        self.Ap, _, self.N0 = pad_to_pow2(Ks, np.zeros(Ks.shape[0]), lam)
        self.sys = HHLSystem(self.Ap, spec["n_clock"], self.rng, spectral=CONFIG["quantum"]["spectral_bound"], c_mode=spec.get("reciprocal", "kappa_known"))
        self.recon = spec.get("recon", "statevector"); self.shots = int(spec.get("shots", 0)); self.eff_shots = int(spec.get("eff_shots", 0))
        self.qlog = []
    def base_solve(self, r, ctx):
        rs = self.d * r if self.scaled else r
        rp = np.zeros(self.Ap.shape[0]); rp[:self.N0] = rs
        out = self.sys.run(rp, shots=(self.shots if self.recon == "shots" else 0), eff_shots=(self.eff_shots if self.recon == "aa_shots" else 0), sampling_repeats=1)
        if self.recon == "shots": up, rec = out["x_shots"], out["shot_records"][0]
        elif self.recon == "aa_shots": up, rec = out["x_aa"], out["aa_records"][0]
        else: up, rec = out["x_ideal"], None
        u = up[:self.N0]
        if self.scaled: u = self.d * u
        xe = out["x_exact"]; nx = max(np.linalg.norm(xe), 1e-300)
        info = {"q_n_qubits": out["n_qubits"], "q_n_clock": out["n_clock"], "q_kappa_target": out["kappa"], "q_reg": self.reg, "q_scaling": "ruiz" if self.scaled else "none",
                "q_reciprocal_mode": out["reciprocal_mode"], "q_k_min": out["k_min"], "q_p_postselect": out.get("p_postselect_exact", np.nan), "q_p_leak": out.get("p_leak_exact", np.nan),
                "q_rel_err_ideal_target": float(np.linalg.norm(out["x_ideal"] - xe) / nx) if "x_ideal" in out else np.nan,
                "q_rel_err_used_target": float(np.linalg.norm(up - xe) / nx) if np.all(np.isfinite(up)) else np.inf, "q_recon": self.recon, "q_shots": self.shots, "q_eff_shots": self.eff_shots,
                "q_t_prep": out["t_prep"], "q_t_circuit_sim": out.get("t_circuit_sim", np.nan), "q_t_readout": out.get("t_readout_probs", np.nan) + (rec["t_reconstruct"] if rec else 0.0),
                "q_gates": out.get("n_gates_highlevel", np.nan), "q_depth": out.get("depth_highlevel", np.nan), "q_n_eig_unresolved_10pct": out["n_eig_unresolved_10pct"]}
        if rec is not None:
            info.update({"q_postselect_rate_empirical": rec.get("postselect_rate", np.nan), "q_shot_status": rec["status"], "q_sign_errors": rec.get("n_sign_errors", np.nan), "q_aa_overhead": rec.get("aa_overhead_per_sample", np.nan)})
        self.qlog.append(info)
        return u, info
SOLVER_KINDS["hhl"] = HHLKind

# quick functional test on a 4x4 symmetric indefinite system
_A = np.array([[3.0, 0.5, 0, 0.2], [0.5, -1.0, 0.3, 0], [0, 0.3, 2.0, 0.1], [0.2, 0, 0.1, -2.5]]); _b = np.array([1.0, -0.5, 0.3, 0.8])
_sys = HHLSystem(_A, 6, rng_for("hhl_selftest"))
_out = _sys.run(_b, shots=20000, sampling_repeats=2, eff_shots=2000)
_xe = np.linalg.solve(_A, _b)
print(f"HHL self-test (N=4, n_clock=6, kappa={_sys.kappa:.1f}): rel_err ideal={np.linalg.norm(_out['x_ideal']-_xe)/np.linalg.norm(_xe):.3e}, "
      f"shots(2x20000)={_out['shot_records'][0]['rel_err']:.3e}, AA-emulated(2x2000 accepted)={_out['aa_records'][0]['rel_err']:.3e}, P(anc=1)={_out['p_postselect_exact']:.3f}, leak={_out['p_leak_exact']:.2e}, "
      f"qubits={_out['n_qubits']}, gates={_out['n_gates_highlevel']}, depth={_out['depth_highlevel']}, sim {_out['t_circuit_sim']*1e3:.0f} ms")
assert np.linalg.norm(_out["x_ideal"] - _xe) / np.linalg.norm(_xe) < 0.2, "HHL ideal reconstruction is off; check the qubit-order conventions"
