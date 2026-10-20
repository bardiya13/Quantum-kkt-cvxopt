# Polynomial linear solver: p(A) ~ A^-1 as a linear combination of unitaries over a qubitised walk (Childs, Kothari, Somma 2017).
# The error decreases exponentially in the degree d = O(kappa log(kappa/eps)), so precision costs log(1/eps) instead of the
# 1/eps of phase estimation.
#
# Simulated at the statevector level, with the same oracle model as the HHL cell:
#   * block encoding U_A = [[A, S], [S, -A]], S = sqrt(I - A^2) (one ancilla, built densely);
#   * walk W = (2 Pi - I) U_A with Pi W^k Pi = T_k(A) (checked below);
#   * LCU: PREP on an index register, SELECT = sum_j |j><j| (x) W^(2j+1), PREP^dagger, post-selection on index = 0 and
#     ancilla = 0. SELECT only controls on the index register, so the post-selected output is accumulated term by term;
#     an explicit index-register statevector is compared with this on a small case;
#   * readout as for HHL (reference branch on a flag qubit, Z and X settings).
# Assumed: ||A|| and kappa known, block encoding and state preparation free. Not simulated: gate synthesis of U_A and PREP,
# amplitude amplification, noise. Reported: walk steps (block-encoding queries), qubits, ||c||_1, post-selection probability.
from scipy.stats import binom as _binom

def cks_coefficients(kappa, eps):
    b = int(math.ceil(kappa ** 2 * math.log(kappa / eps))); j0 = int(math.ceil(math.sqrt(b * math.log(4 * b / eps))))
    j = np.arange(j0 + 1); return 4.0 * (-1.0) ** j * _binom.sf(b + j, 2 * b, 0.5), b

def optimal_inverse_coefficients(kappa, eps):
    """Minimax-optimal odd polynomial for 1/x on [-1,-a] U [a,1], a = 1/kappa (Sunderhauf, Nemeth, Walayat, Patterson, Berntson,
    arXiv:2507.15537, Theorem 1; stable recurrence Eqs. (9)-(12), Chebyshev coefficients by exact interpolation at degree d = 2n-1).
    Uniform error eps_d = (1-a)^n / (a (1+a)^(n-1)).  Returns (coefficients of T_1, T_3, ..., degree, measured max |p(x) - 1/x| on the target set)."""
    from scipy.fft import dct
    a = 1.0 / kappa; n = max(1, int(math.ceil((math.log(1 / eps) + math.log(1 / a) + math.log(1 + a)) / (math.log(1 + a) - math.log(1 - a)))))
    al = (1 + a) / (2 * (1 - a)); d = 2 * n - 1; xs = np.cos(np.pi * (np.arange(d + 1) + 0.5) / (d + 1)); y = (2 * xs ** 2 - (1 + a * a)) / (1 - a * a)
    L1 = (y + (1 - a) / (1 + a)) / al; L2 = (y * y + (1 - a) / (2 * (1 + a)) * y - 0.5) / al ** 2
    Lm, Ln = (L1, L1) if n == 1 else (L1, L2)
    for _ in range(max(n - 2, 0)): Lm, Ln = Ln, y * Ln / al - Lm / (4 * al * al)
    pv = (1 - (-1) ** n * (1 + a) ** 2 / (4 * a) * Ln) / xs                       # d + 1 is even, so x = 0 is never a node
    ck = dct(pv, type=2) / (d + 1); c = ck[1::2].copy()
    ng = 3000 if d <= 50000 else 600                                                # check grid inside [a, 1], denser near a where the error peaks
    xg = np.unique(np.concatenate([np.linspace(a, min(3 * a, 1.0), ng // 2), np.geomspace(a, 1.0, ng // 2)])); acc = c[0] * xg; tp, tc = np.ones_like(xg), xg.copy()
    for j in range(1, len(c)):                                                      # evaluate sum c_j T_{2j+1}(x) on a grid with the three-term recurrence
        tp = 2 * xg * tc - tp; tc = 2 * xg * tp - tc; acc += c[j] * tc
    return c, d, float(np.max(np.abs(acc - 1 / xg)))

def polynomial_degree(kappa, eps, poly):
    """Degree implied by (kappa, eps) without generating coefficients (used for the simulation budget)."""
    if poly == "optimal":
        a = 1.0 / kappa; return 2 * max(1, int(math.ceil((math.log(1 / eps) + math.log(1 / a) + math.log(1 + a)) / (math.log(1 + a) - math.log(1 - a))))) - 1
    b = int(math.ceil(kappa ** 2 * math.log(kappa / eps))); return 2 * int(math.ceil(math.sqrt(b * math.log(4 * b / eps)))) + 1

OPTIMAL_POLY_MAX_DEGREE = 60000      # the recurrence costs O(d^2) (~10 s at 60 000); above this the CKS expansion is used (labelled)
def inverse_polynomial(kappa, eps, poly="optimal"):
    """Coefficients of the inversion polynomial. 'optimal' falls back to the CKS expansion (labelled) if its double-precision
    evaluation misses the requested accuracy (seen only for kappa >= 1000 with eps <= 1e-6) or its degree exceeds OPTIMAL_POLY_MAX_DEGREE."""
    if poly == "optimal":
        if polynomial_degree(kappa, eps, "optimal") > OPTIMAL_POLY_MAX_DEGREE:
            c, _ = cks_coefficients(kappa, eps); return c, "cks (fallback: optimal-polynomial degree above the generation cap)", float("nan")
        c, d, err = optimal_inverse_coefficients(kappa, eps)
        if err <= 3.0 * eps: return c, "optimal", err
        c, _ = cks_coefficients(kappa, eps); return c, "cks (fallback: optimal polynomial lost accuracy in double precision)", float("nan")
    c, _ = cks_coefficients(kappa, eps); return c, "cks", float("nan")

def block_encode(A):
    """Hermitian unitary block encoding of a Hermitian A with ||A|| <= 1."""
    lam, V = np.linalg.eigh(A); lam = np.clip(lam, -1.0, 1.0); S = (V * np.sqrt(1.0 - lam ** 2)) @ V.T
    return np.block([[A, S], [S, -A]])

def block_encoding_costs(A):
    """Normalisation factors alpha of two standard block encodings of A (the walk degree scales with kappa_eff = kappa * alpha / ||A||_2; the simulation uses the ideal
    alpha = ||A||_2).  (1) sparse-access encoding: alpha = s * max|a_ij|, s = max nonzeros per row;  (2) LCU of Pauli strings: alpha = sum_P |c_P|, plus the number of
    Pauli terms (Ozbaygin, Kocuk and Moran report SDP matrices ~98 % dense in the Pauli basis)."""
    N = A.shape[0]; q = int(round(math.log2(N))); nz = np.abs(A) > 1e-14 * np.abs(A).max(); out = {"alpha_ideal": float(np.linalg.norm(A, 2)), "alpha_sparse": float(nz.sum(axis=1).max() * np.abs(A).max())}
    T = A.astype(complex).reshape([2] * (2 * q))                       # axes: row bits, column bits; one (row bit, column bit) pair -> one Pauli axis per step
    for k in range(q):
        T = np.moveaxis(T, [0, q - k], [-2, -1]); a, b_, c, d_ = T[..., 0, 0], T[..., 0, 1], T[..., 1, 0], T[..., 1, 1]
        T = np.stack([(a + d_) / 2, (b_ + c) / 2, 1j * (b_ - c) / 2, (a - d_) / 2], axis=-1)
    coef = np.abs(T).ravel(); out["alpha_pauli"] = float(coef.sum()); out["pauli_terms"] = int((coef > 1e-12 * coef.max()).sum()); out["pauli_terms_possible"] = int(N * (N + 1) // 2)       # a real symmetric matrix has at most N(N+1)/2 Pauli strings (those with an even number of Y)
    return out

class CKSSystem:
    def __init__(self, A, eps, rng, kappa_margin=1.05, max_degree=None, poly=None):
        t0 = time.perf_counter(); self.A = np.asarray(A, dtype=float); self.N = self.A.shape[0]; self.n_d = int(round(math.log2(self.N))); self.rng = rng
        ev = np.linalg.eigvalsh(self.A); self.t_spectral_assumption = time.perf_counter() - t0          # assumed known, as in HHL 'kappa_known'
        self.alpha = float(np.abs(ev).max()) * (1 + 1e-9); self.kappa = float(np.abs(ev).max() / np.abs(ev).min()); self.eps = float(eps)
        poly = poly or CONFIG["quantum"].get("cks", {}).get("poly", "optimal")
        # simulation budget: degree of the cheapest admissible polynomial, before any coefficients are generated
        d_est = polynomial_degree(self.kappa * kappa_margin, self.eps, "cks" if (poly == "cks" or polynomial_degree(self.kappa * kappa_margin, self.eps, "optimal") > OPTIMAL_POLY_MAX_DEGREE) else "optimal")
        if max_degree and d_est > max_degree:
            self.c = np.zeros(1); self.poly = poly + " (not generated: degree cap)"; self.poly_err = float("nan"); self.j0 = (d_est - 1) // 2; self.degree = d_est; self.l1 = float("nan"); self.over_budget = True
            self.n_index = max(1, int(math.ceil(math.log2(self.j0 + 1)))); self.n_qubits = self.n_d + 1 + 1 + self.n_index + 1; self.t_prep = time.perf_counter() - t0; self.W = None; self.walk_check_diff = None; return
        self.c, self.poly, self.poly_err = inverse_polynomial(self.kappa * kappa_margin, self.eps, poly); self.j0 = len(self.c) - 1; self.degree = 2 * self.j0 + 1; self.l1 = float(np.abs(self.c).sum())
        self.over_budget = bool(max_degree) and self.degree > max_degree          # the run is then reported as over budget, not replaced
        U = block_encode(self.A / self.alpha); N = self.N; Wm = U.copy(); Wm[N:, :] *= -1.0; self.W = Wm              # W = (2Pi - I) U
        self.n_index = max(1, int(math.ceil(math.log2(self.j0 + 1)))); self.n_qubits = self.n_d + 1 + 1 + self.n_index + 1       # data + flag + block-encoding ancilla + index + success flag
        # The circuit output for any right-hand side is p(A/alpha) b. The walk is simulated on the first solve and compared with
        # the spectral form V p(Lambda) V' of the same polynomial; later solves with this matrix reuse p(lambda_i). This only makes
        # the simulation cheaper; the reported resources are those of the circuit.
        lam_, self._V = np.linalg.eigh(self.A / self.alpha); tp, tc = np.ones_like(lam_), lam_.copy(); pl = self.c[0] * tc
        for j in range(1, self.j0 + 1):
            tp = 2 * lam_ * tc - tp; tc = 2 * lam_ * tp - tc; pl += self.c[j] * tc
        self._plam = pl; self.walk_check_diff = None
        self.t_prep = time.perf_counter() - t0
    def apply_poly_spectral(self, V): return self._V @ (self._plam[:, None] * (self._V.T @ V))
    def apply_poly(self, V):
        """Columns of V (N x k) -> sum_j c_j T_{2j+1}(A/alpha) V, accumulated from the ancilla-0 block of W^(2j+1) (walk-level statevector simulation)."""
        N = self.N; st = np.vstack([V, np.zeros_like(V)]); st = self.W @ st; acc = self.c[0] * st[:N]
        W2 = self.W @ self.W
        for j in range(1, self.j0 + 1):
            st = W2 @ st; acc += self.c[j] * st[:N]
        return acc
    def reference(self, b):
        r = np.ones(self.N) / np.sqrt(self.N); Ar = self.A @ r; s = np.linalg.norm(b) / max(np.linalg.norm(Ar), 1e-300); return r * s, Ar * s
    def run(self, b, shots=0, sampling_repeats=1, x_exact=None, eff_shots=0):
        b = np.asarray(b, dtype=float); nb = np.linalg.norm(b); N = self.N
        out = {"n_qubits": self.n_qubits, "n_index": self.n_index, "N": N, "kappa": self.kappa, "degree": self.degree, "walk_steps": self.degree, "l1_norm": self.l1, "walk_check_diff": self.walk_check_diff, "eps_target": self.eps,
               "t_prep": self.t_prep, "t_spectral_assumption": self.t_spectral_assumption, "rhs_norm": nb}
        if nb == 0: out.update({"x_ideal": np.zeros(N), "status": "zero_rhs"}); return out
        if self.over_budget:
            nanv = np.full(N, np.nan); out.update({"x_ideal": nanv, "x_shots": nanv, "x_aa": nanv, "status": "degree_cap", "x_exact": np.linalg.solve(self.A, b), "t_circuit_sim": 0.0,
                                                    "shot_records": [{"status": "degree_cap", "x_rec": nanv}], "aa_records": [{"status": "degree_cap", "x_rec": nanv}]}); return out
        t0 = time.perf_counter(); r_ref, Ar = self.reference(b); v = np.concatenate([b, Ar]); vnorm = np.linalg.norm(v)
        Vin = np.stack([b, Ar], axis=1) / vnorm
        if self.walk_check_diff is None:                                   # first solve: walk-level simulation and check of the spectral form
            Yw = self.apply_poly(Vin); Ys = self.apply_poly_spectral(Vin); self.walk_check_diff = float(np.max(np.abs(Yw - Ys)) / max(np.max(np.abs(Yw)), 1e-300))
            assert self.walk_check_diff < 1e-7, f"spectral form of the polynomial disagrees with the walk simulation ({self.walk_check_diff:.1e})"
            Y = Yw / self.l1
        else: Y = self.apply_poly_spectral(Vin) / self.l1                       # success-branch amplitudes (flag = 0 / 1 columns); LCU normalisation 1/||c||_1
        out["t_circuit_sim"] = time.perf_counter() - t0; y0, y1 = Y[:, 0], Y[:, 1]; p_s = float(y0 @ y0 + y1 @ y1); out["p_postselect_exact"] = p_s
        cfac = np.vdot(r_ref, y1) / np.vdot(r_ref, r_ref); out["x_ideal"] = (y0 / cfac) if abs(cfac) > 0 else np.full(N, np.nan)
        pZ = np.zeros((N, 2, 2)); pZ[:, 0, 1] = y0 ** 2; pZ[:, 1, 1] = y1 ** 2; pZ[0, 0, 0] += max(1.0 - p_s, 0.0)
        yp, ym = (y0 + y1) / np.sqrt(2), (y0 - y1) / np.sqrt(2); pX = np.zeros((N, 2, 2)); pX[:, 0, 1] = yp ** 2; pX[:, 1, 1] = ym ** 2; pX[0, 0, 0] += max(1.0 - p_s, 0.0)
        out["pZ"], out["pX"] = pZ.reshape(-1), pX.reshape(-1); out["r_ref"] = r_ref; out["v_norm"] = vnorm; out["b"] = b
        out["x_exact"] = np.linalg.solve(self.A, b) if x_exact is None else x_exact               # classical reference, diagnostic only
        if shots: recs = reconstruct_from_shots(self, out, shots, sampling_repeats, self.rng); out["x_shots"] = recs[0]["x_rec"]; out["shot_records"] = recs
        if eff_shots: recs = reconstruct_aa_emulated(self, out, eff_shots, sampling_repeats, self.rng); out["x_aa"] = recs[0]["x_rec"]; out["aa_records"] = recs
        return out

class CKSKind:
    """KKT-solver kind 'cks': Ruiz-scaled KKT matrix -> padded to 2^n -> CKS/LCU walk simulation -> readout; same protocol and accounting as 'hhl'."""
    def __init__(self, spec, K, asm, W, eng):
        Kd = K.toarray() if sp.issparse(K) else np.asarray(K, dtype=float)
        self.scaled = spec.get("scaling", "ruiz") == "ruiz"; self.d = ruiz_equilibrate(Kd) if self.scaled else None
        Ks = apply_scaling(Kd, self.d) if self.scaled else Kd; self.rng = rng_for("cks", asm.inst["id"], spec["name"])
        lam, _ = spectral_bound(Ks, CONFIG["quantum"]["spectral_bound"], self.rng); self.Ap, _, self.N0 = pad_to_pow2(Ks, np.zeros(Ks.shape[0]), lam)
        self.sys = CKSSystem(self.Ap, float(spec.get("eps", 1e-10)), self.rng, max_degree=CONFIG["quantum"].get("cks", {}).get("max_degree"), poly=spec.get("poly"))
        self.recon = spec.get("recon", "statevector"); self.shots = int(spec.get("shots", 0)); self.eff_shots = int(spec.get("eff_shots", 0))
    def base_solve(self, r, ctx):
        rs = self.d * r if self.scaled else r; rp = np.zeros(self.Ap.shape[0]); rp[:self.N0] = rs
        out = self.sys.run(rp, shots=(self.shots if self.recon == "shots" else 0), eff_shots=(self.eff_shots if self.recon == "aa_shots" else 0))
        if self.recon == "shots": up, rec = out["x_shots"], out["shot_records"][0]
        elif self.recon == "aa_shots": up, rec = out["x_aa"], out["aa_records"][0]
        else: up, rec = out["x_ideal"], None
        u = up[:self.N0]; u = self.d * u if self.scaled else u; xe = out["x_exact"]; nx = max(np.linalg.norm(xe), 1e-300)
        info = {"q_solver": "lcu_walk", "q_poly": self.sys.poly, "q_eps_circuit": self.sys.eps, "q_n_qubits": out["n_qubits"], "q_n_index": out["n_index"], "q_kappa_target": out["kappa"], "q_degree": out["degree"], "q_l1_norm": out["l1_norm"],
                "q_p_postselect": out.get("p_postselect_exact", np.nan), "q_walk_check_diff": self.sys.walk_check_diff, "q_status": out.get("status", "ok"), "q_rel_err_ideal_target": float(np.linalg.norm(out["x_ideal"] - xe) / nx) if np.all(np.isfinite(out["x_ideal"])) else np.inf,
                "q_rel_err_used_target": float(np.linalg.norm(up - xe) / nx) if np.all(np.isfinite(up)) else np.inf, "q_recon": self.recon, "q_shots": self.shots, "q_eff_shots": self.eff_shots,
                "q_t_prep": out["t_prep"], "q_t_circuit_sim": out.get("t_circuit_sim", np.nan), "q_scaling": "ruiz" if self.scaled else "none"}
        if rec is not None: info.update({"q_shot_status": rec["status"], "q_sign_errors": rec.get("n_sign_errors", np.nan), "q_aa_overhead": rec.get("aa_overhead_per_sample", np.nan)})
        return u, info
SOLVER_KINDS["cks"] = CKSKind


# Polynomial solver for an SPD target (null-space system). Shifted Chebyshev expansion of 1/x on [a, 1], a = 1/kappa, in
# y = (2x - (1+a))/(1-a): c_0 = sqrt(kappa), c_k = 2 sqrt(kappa) (-1)^k rho^k, rho = (sqrt(kappa)-1)/(sqrt(kappa)+1), so the degree
# is O(sqrt(kappa) log(kappa/eps)) against O(kappa log(1/eps)) for the odd polynomial of the indefinite augmented system.
# The walk acts on a block encoding of A~ = (2A/alpha - (1+a)I)/(1-a), with Pi W^k Pi = T_k(A~) for every k. The LCU
# normalisation ~ (3+a)/(1-a) of building that encoding from one of A is reported, not simulated.
def polynomial_degree_spd(kappa, eps):
    """Smallest degree whose Chebyshev tail bound  sqrt(kappa)(sqrt(kappa)+1) rho^(d+1)  is <= eps (closed form, no coefficients generated)."""
    rho = (math.sqrt(kappa) - 1.0) / (math.sqrt(kappa) + 1.0)
    if rho <= 0: return 1
    return max(1, int(math.ceil(math.log(eps / (math.sqrt(kappa) * (math.sqrt(kappa) + 1.0))) / math.log(rho))) - 1)

def spd_inverse_coefficients(kappa, eps):
    """Coefficients c_0..c_d of the shifted Chebyshev expansion of 1/x on [1/kappa, 1] truncated at the degree of polynomial_degree_spd, and the
    measured max |p(x) - 1/x| on a grid inside [1/kappa, 1] (denser near 1/kappa, where the error peaks)."""
    a = 1.0 / kappa; sk = math.sqrt(kappa); rho = (sk - 1.0) / (sk + 1.0); d = polynomial_degree_spd(kappa, eps)
    k = np.arange(d + 1); c = 2.0 * sk * (-1.0) ** k * rho ** k; c[0] = sk
    xg = np.unique(np.concatenate([np.linspace(a, min(3 * a, 1.0), 1500), np.geomspace(a, 1.0, 1500)])); y = (2 * xg - (1 + a)) / (1 - a)
    tp, tc = np.ones_like(y), y.copy(); acc = c[0] * tp + (c[1] * tc if d >= 1 else 0.0)
    for j in range(2, d + 1):
        tp, tc = tc, 2 * y * tc - tp; acc += c[j] * tc
    return c, d, float(np.max(np.abs(acc - 1.0 / xg)))

class SPDPolySystem(CKSSystem):
    """Same simulation protocol, readout and accounting as CKSSystem, for an SPD matrix with the shifted Chebyshev polynomial (all degrees, not only odd)."""
    def __init__(self, A, eps, rng, kappa_margin=1.05, max_degree=None, poly=None):
        t0 = time.perf_counter(); self.A = np.asarray(A, dtype=float); self.N = self.A.shape[0]; self.n_d = int(round(math.log2(self.N))); self.rng = rng
        ev = np.linalg.eigvalsh(self.A); self.t_spectral_assumption = time.perf_counter() - t0
        assert ev.min() > 0, "SPD polynomial solver needs a positive definite target"
        self.alpha = float(ev.max()) * (1 + 1e-9); self.kappa = float(ev.max() / ev.min()); self.eps = float(eps); self.poly = "chebyshev_spd"
        kap = self.kappa * kappa_margin; self.a = 1.0 / kap; d_est = polynomial_degree_spd(kap, self.eps)
        self.be_shift_inflation = (3.0 + self.a) / (1.0 - self.a)          # reported, not simulated
        if max_degree and d_est > max_degree:
            self.c = np.zeros(1); self.poly = "chebyshev_spd (not generated: degree cap)"; self.poly_err = float("nan"); self.j0 = d_est; self.degree = d_est; self.l1 = float("nan"); self.over_budget = True
            self.n_index = max(1, int(math.ceil(math.log2(d_est + 1)))); self.n_qubits = self.n_d + 1 + 1 + self.n_index + 1; self.t_prep = time.perf_counter() - t0; self.W = None; self.walk_check_diff = None; return
        self.c, self.degree, self.poly_err = spd_inverse_coefficients(kap, self.eps); self.j0 = self.degree; self.l1 = float(np.abs(self.c).sum())
        self.over_budget = bool(max_degree) and self.degree > max_degree
        At = (2.0 * self.A / self.alpha - (1.0 + self.a) * np.eye(self.N)) / (1.0 - self.a)
        U = block_encode(At); N = self.N; Wm = U.copy(); Wm[N:, :] *= -1.0; self.W = Wm
        self.n_index = max(1, int(math.ceil(math.log2(self.degree + 1)))); self.n_qubits = self.n_d + 1 + 1 + self.n_index + 1
        lam_, self._V = np.linalg.eigh(At); tp, tc = np.ones_like(lam_), lam_.copy(); pl = self.c[0] * tp + (self.c[1] * tc if self.degree >= 1 else 0.0)
        for j in range(2, self.degree + 1):
            tp, tc = tc, 2 * lam_ * tc - tp; pl += self.c[j] * tc
        self._plam = pl; self.walk_check_diff = None; self.t_prep = time.perf_counter() - t0
    def apply_poly(self, V):
        """Columns of V -> sum_k c_k T_k(A~) V accumulated from the ancilla-0 block of W^k (walk-level simulation, one walk step per term)."""
        N = self.N; st = np.vstack([V, np.zeros_like(V)]); acc = self.c[0] * V.copy()
        for k in range(1, self.degree + 1):
            st = self.W @ st; acc += self.c[k] * st[:N]
        return acc

class OSSPolyKind:
    """KKT-solver kind 'lcu_oss': the null-space SPD system D Z'SZ D (Jacobi-scaled, padded with lambda_max I to 2^n) is solved by the SPD
    polynomial solver; the direction is completed classically as in OSSKind (x_p, u_z, u_y) and verified on the full KKT system. Also logged:
    kappa and odd-polynomial degree that the Ruiz-scaled augmented system would need at the same eps."""
    def __init__(self, spec, K, asm, W, eng):
        self.cls = OSSKind(dict(spec, max_n=10 ** 9), K, asm, W, eng); M = self.cls.matrix(); self.rng = rng_for("lcu_oss", asm.inst["id"], spec["name"])
        lam_max = float(np.linalg.eigvalsh(M).max()); self.Mp, _, self.N0 = pad_to_pow2(M, np.zeros(M.shape[0]), lam_max)
        self.sys = SPDPolySystem(self.Mp, float(spec.get("eps", 1e-10)), self.rng, max_degree=CONFIG["quantum"].get("cks", {}).get("max_degree"))
        self.recon = spec.get("recon", "statevector"); self.shots = int(spec.get("shots", 0)); self.eff_shots = int(spec.get("eff_shots", 0))
        Kd = K.toarray() if sp.issparse(K) else np.asarray(K, dtype=float); Ks = apply_scaling(Kd, ruiz_equilibrate(Kd)); self.kappa_aug = float(np.linalg.cond(Ks))
        self.deg_aug = polynomial_degree(self.kappa_aug * 1.05, self.sys.eps, "optimal"); self.qubits_aug = int(math.ceil(math.log2(Kd.shape[0]))) + 3 + max(1, int(math.ceil(math.log2((self.deg_aug + 1) // 2 + 1))))
        self.Kd = Kd; self._lu = None
    def base_solve(self, r, ctx):
        xp_, b = self.cls.reduced_rhs(r); rp = np.zeros(self.Mp.shape[0]); rp[:self.N0] = b
        out = self.sys.run(rp, shots=(self.shots if self.recon == "shots" else 0), eff_shots=(self.eff_shots if self.recon == "aa_shots" else 0))
        if self.recon == "shots": vp, rec = out["x_shots"], out["shot_records"][0]
        elif self.recon == "aa_shots": vp, rec = out["x_aa"], out["aa_records"][0]
        else: vp, rec = out["x_ideal"], None
        v = vp[:self.N0]; u = self.cls.full(v, r, xp_) if np.all(np.isfinite(v)) else np.full(r.shape, np.nan)
        if self._lu is None: self._lu = sla.lu_factor(self.Kd, check_finite=False)
        ue = sla.lu_solve(self._lu, r, check_finite=False); nu = max(np.linalg.norm(ue), 1e-300); xe = out["x_exact"]; nx = max(np.linalg.norm(xe), 1e-300)
        info = {"q_solver": "lcu_oss_spd", "q_target": "null_space_spd", "q_poly": self.sys.poly, "q_eps_circuit": self.sys.eps, "q_n_qubits": out["n_qubits"], "q_n_index": out["n_index"], "q_kappa_target": out["kappa"], "q_degree": out["degree"], "q_l1_norm": out["l1_norm"],
                "q_p_postselect": out.get("p_postselect_exact", np.nan), "q_walk_check_diff": self.sys.walk_check_diff, "q_status": out.get("status", "ok"), "q_dim_reduced": self.N0, "q_N_padded": self.Mp.shape[0],
                "q_rel_err_ideal_target": float(np.linalg.norm(out["x_ideal"] - xe) / nx) if np.all(np.isfinite(out["x_ideal"])) else np.inf, "q_rel_err_used_target": float(np.linalg.norm(vp - xe) / nx) if np.all(np.isfinite(vp)) else np.inf,
                "q_rel_err_used_full": float(np.linalg.norm(u - ue) / nu) if np.all(np.isfinite(u)) else np.inf, "q_recon": self.recon, "q_shots": self.shots, "q_eff_shots": self.eff_shots,
                "q_t_prep": out["t_prep"], "q_t_circuit_sim": out.get("t_circuit_sim", np.nan), "q_scaling": "jacobi", "q_kappa_augmented": self.kappa_aug, "q_degree_augmented_equiv": self.deg_aug, "q_n_qubits_augmented_equiv": self.qubits_aug,
                "q_be_shift_inflation": self.sys.be_shift_inflation, "kappa_reduced": self.cls.kappa_reduced, "reduced_dim": self.N0}
        if rec is not None: info.update({"q_shot_status": rec["status"], "q_sign_errors": rec.get("n_sign_errors", np.nan), "q_aa_overhead": rec.get("aa_overhead_per_sample", np.nan)})
        return u, info
SOLVER_KINDS["lcu_oss"] = OSSPolyKind

# self-tests: (1) qubitisation identity, (2) accuracy, (3) explicit index-register statevector against the accumulated walk output
_cs = CKSSystem(_A, 1e-10, rng_for("cks_selftest"), poly="cks"); _N = _cs.N; _Wk = np.eye(2 * _N); _T0, _T1 = np.eye(_N), _A / _cs.alpha
for _k in range(1, 6):
    _Wk = _cs.W @ _Wk
    if _k > 1: _T0, _T1 = _T1, 2 * (_A / _cs.alpha) @ _T1 - _T0
    assert np.allclose(_Wk[:_N, :_N], _T1, atol=1e-12), "qubitized walk does not block-encode T_k(A)"
_co = _cs.run(_b, shots=20000, eff_shots=2000)
print(f"CKS self-test (N=4, kappa={_cs.kappa:.1f}, eps=1e-10): degree {_cs.degree}, {_cs.n_index} index qubits, ||c||_1={_cs.l1:.1f}, P(success)={_co['p_postselect_exact']:.2e}; "
      f"rel_err ideal={np.linalg.norm(_co['x_ideal'] - _xe) / np.linalg.norm(_xe):.2e} (HHL n_clock=6 above: {np.linalg.norm(_out['x_ideal'] - _xe) / np.linalg.norm(_xe):.2e}), "
      f"shots(2x20000)={_co['shot_records'][0]['rel_err']:.2e}, AA-emulated(2x2000)={_co['aa_records'][0]['rel_err']:.2e}")
assert np.linalg.norm(_co["x_ideal"] - _xe) / np.linalg.norm(_xe) < 1e-8
# explicit index-register statevector on a low-degree instance of the same construction (eps = 0.3, few terms)
_cg = CKSSystem(_A, 0.3, rng_for("cks_selftest2"), poly="cks"); _ni = _cg.n_index; _J = 2 ** _ni
_cpad = np.zeros(_J); _cpad[:_cg.j0 + 1] = _cg.c; _prepR = np.sqrt(np.abs(_cpad) / _cg.l1) * np.sign(np.where(_cpad == 0, 1.0, _cpad)); _prepL = np.sqrt(np.abs(_cpad) / _cg.l1)
_bn = _b / np.linalg.norm(_b); _P = _cg.W @ _cg.W
_state = np.kron(_prepR, np.kron([1.0, 0.0], _bn)).reshape(_J, 2 * _N)                      # explicit statevector of (index) x (ancilla, data)
_state = (_cg.W @ _state.T).T                                                             # W on every index branch
for _bit in range(_ni):                                                                   # SELECT = W * prod_bits controlled-(W^2)^(2^bit)
    for _jv in range(_J):
        if (_jv >> _bit) & 1: _state[_jv] = _P @ _state[_jv]
    _P = _P @ _P
_post = (_prepL[:, None] * _state).sum(axis=0)[:_N]                                       # PREP_L^dagger, post-select index = 0 and ancilla = 0
_walk = _cg.apply_poly((_bn)[:, None])[:, 0] / _cg.l1
print(f"  explicit index-register statevector ({_ni} index qubits, degree {_cg.degree}) vs walk-level accumulation: max |diff| = {np.max(np.abs(_post - _walk)):.1e}")
assert np.max(np.abs(_post - _walk)) < 1e-12

# optimal polynomial against the CKS expansion, on the same 4x4 system and over a (kappa, eps) grid
_co2 = CKSSystem(_A, 1e-10, rng_for("opt_selftest"), poly="optimal"); _oo = _co2.run(_b)
print(f"optimal polynomial (N=4, eps=1e-10): degree {_co2.degree} (CKS {_cs.degree}), ||c||_1 {_co2.l1:.1f} (CKS {_cs.l1:.1f}), P(success) {_oo['p_postselect_exact']:.2e} (CKS {_co['p_postselect_exact']:.2e}), "
      f"rel_err ideal {np.linalg.norm(_oo['x_ideal'] - _xe) / np.linalg.norm(_xe):.2e}; polynomial used: {_co2.poly}")
assert np.linalg.norm(_oo["x_ideal"] - _xe) / np.linalg.norm(_xe) < 1e-8
POLY_TABLE = []
for _kp in [5, 20, 100, 1000]:
    for _ep in [3e-1, 1e-2, 1e-6]:
        _c0, _ = cks_coefficients(_kp, _ep); _c1, _lab, _er = inverse_polynomial(_kp, _ep, "optimal")
        POLY_TABLE.append({"kappa": _kp, "eps": _ep, "degree_cks": 2 * len(_c0) - 1, "degree_optimal": 2 * len(_c1) - 1, "l1_cks": float(np.abs(_c0).sum()), "l1_optimal": float(np.abs(_c1).sum()), "optimal_measured_err": _er, "used": _lab})
POLY_TABLE = pd.DataFrame(POLY_TABLE); POLY_TABLE["degree_ratio"] = POLY_TABLE.degree_cks / POLY_TABLE.degree_optimal
POLY_TABLE["cost_ratio_degree_over_psuccess"] = POLY_TABLE.degree_ratio * (POLY_TABLE.l1_cks / POLY_TABLE.l1_optimal) ** 2
# degree and 1-norm of the SPD polynomial at the same kappa and eps: the effect of the target system alone
POLY_TABLE["degree_spd"] = [polynomial_degree_spd(k_, e_) for k_, e_ in zip(POLY_TABLE.kappa, POLY_TABLE.eps)]
POLY_TABLE["l1_spd"] = [float(np.abs(spd_inverse_coefficients(k_, e_)[0]).sum()) for k_, e_ in zip(POLY_TABLE.kappa, POLY_TABLE.eps)]
POLY_TABLE["spd_measured_err"] = [spd_inverse_coefficients(k_, e_)[2] for k_, e_ in zip(POLY_TABLE.kappa, POLY_TABLE.eps)]
POLY_TABLE["degree_ratio_optimal_over_spd"] = POLY_TABLE.degree_optimal / POLY_TABLE.degree_spd
print(POLY_TABLE.round(3).to_string(index=False))
# self-test of the SPD polynomial solver on a 4x4 SPD system (walk vs spectral form, accuracy, readout)
_As = _A @ _A + 0.5 * np.eye(4); _xs = np.linalg.solve(_As, _b); _ss = SPDPolySystem(_As, 1e-10, rng_for("spd_selftest")); _so = _ss.run(_b, shots=20000, eff_shots=2000)
print(f"SPD polynomial self-test (N=4, kappa={_ss.kappa:.1f}, eps=1e-10): degree {_ss.degree} (odd optimal polynomial at this kappa: {polynomial_degree(_ss.kappa * 1.05, 1e-10, 'optimal')}), ||c||_1={_ss.l1:.1f}, "
      f"P(success)={_so['p_postselect_exact']:.2e}, walk-vs-spectral diff {_ss.walk_check_diff:.1e}, rel_err ideal={np.linalg.norm(_so['x_ideal'] - _xs) / np.linalg.norm(_xs):.2e}, AA-emulated(2x2000)={_so['aa_records'][0]['rel_err']:.2e}; measured polynomial error {_ss.poly_err:.1e}")
assert np.linalg.norm(_so["x_ideal"] - _xs) / np.linalg.norm(_xs) < 1e-8 and _ss.walk_check_diff < 1e-7
