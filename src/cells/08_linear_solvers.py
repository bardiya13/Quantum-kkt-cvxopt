# KKT solvers and the solve protocol: acceptance test on the original system, refinement, classical fallback.
_KW_TOL_MINRES = "rtol" if "rtol" in inspect.signature(spla.minres).parameters else "tol"
_KW_TOL_CG = "rtol" if "rtol" in inspect.signature(spla.cg).parameters else "tol"

class ExactFactor:
    def __init__(self, K):
        self.sparse = sp.issparse(K)
        self.lu = spla.splu(K.tocsc()) if self.sparse else sla.lu_factor(K, check_finite=False)
    def solve(self, r):
        return self.lu.solve(r) if self.sparse else sla.lu_solve(self.lu, r, check_finite=False)

def norm2_estimate(K, iters=None, rng=None):
    """||K||_2 by power iteration (K symmetric): iters mat-vecs, cheap relative to a solve."""
    iters = CONFIG["inexact"]["norm_est_iters"] if iters is None else iters
    rng = np.random.default_rng(0) if rng is None else rng
    v = rng.standard_normal(K.shape[0]); v /= np.linalg.norm(v); lam = 0.0
    for _ in range(iters):
        w = K @ v; lam = float(np.linalg.norm(w))
        if lam == 0: return 0.0
        v = w / lam
    return lam

class _Converged(Exception):
    pass

def krylov_solve(Kop, r, kind, maxiter, stop_fn, check_every):
    """MINRES / CG with an explicit stopping test stop_fn(xk) -> bool evaluated every `check_every` iterations.
    SciPy's own test is disabled (rtol=1e-30) so the criterion is exactly the one requested; the checking cost is timed (t_check)."""
    cnt = {"it": 0, "t_check": 0.0}; box = {"u": None}
    def cb(xk):
        cnt["it"] += 1
        if cnt["it"] % check_every == 0:
            t0 = time.perf_counter(); ok = stop_fn(xk); cnt["t_check"] += time.perf_counter() - t0
            if ok:
                box["u"] = xk.copy(); raise _Converged()
    try:
        if kind == "minres":
            u, info = spla.minres(Kop, r, maxiter=maxiter, callback=cb, **{_KW_TOL_MINRES: 1e-30})
        else:
            u, info = spla.cg(Kop, r, maxiter=maxiter, callback=cb, **{_KW_TOL_CG: 1e-30})
        info = int(info)
    except _Converged:
        u, info = box["u"], 0
    return u, info, cnt["it"], cnt["t_check"]

SOLVER_KINDS = {}

class ExactKind:
    def __init__(self, spec, K, asm, W, eng): self.f = ExactFactor(K)
    def base_solve(self, r, ctx): return self.f.solve(r), {}
SOLVER_KINDS["exact"] = ExactKind

class NoiseKind:
    """Exact solve plus a random perturbation of prescribed relative size (sensitivity experiment, not a quantum solver).
    model 'iso'  : u += delta*||u|| * g/||g||
    model 'block': u_b += delta*||u_b|| * g_b/||g_b|| per block"""
    def __init__(self, spec, K, asm, W, eng):
        self.f = ExactFactor(K); self.delta = spec["delta"]; self.model = spec.get("noise_model", "iso")
        self.rng = rng_for("noise", spec["name"], asm.inst["id"]); self.blocks = eng.block_slices
    def base_solve(self, r, ctx):
        u = self.f.solve(r)
        if self.model == "iso":
            g = self.rng.standard_normal(u.shape); g /= max(np.linalg.norm(g), 1e-300); u = u + self.delta * np.linalg.norm(u) * g
        else:
            u = u.copy()
            for sl in self.blocks:
                if sl.stop - sl.start == 0: continue
                g = self.rng.standard_normal(sl.stop - sl.start); g /= max(np.linalg.norm(g), 1e-300)
                u[sl] += self.delta * np.linalg.norm(u[sl]) * g
        return u, {"injected_rel_noise": self.delta, "noise_model": self.model}
SOLVER_KINDS["noise"] = NoiseKind

class MinresKind:
    def __init__(self, spec, K, asm, W, eng):
        self.scaled = spec.get("scaling", "ruiz") == "ruiz"
        self.d = ruiz_equilibrate(K) if self.scaled else None
        self.Ks = apply_scaling(K, self.d) if self.scaled else K
        self.maxiter = int(min(CONFIG["inexact"]["minres_maxiter_cap"], CONFIG["inexact"]["minres_maxiter_factor"] * K.shape[0]))
        self.eng = eng
    def base_solve(self, r, ctx):
        rs = self.d * r if self.scaled else r
        to_full = (lambda xs: self.d * xs) if self.scaled else (lambda xs: xs)
        stop = lambda xk: self.eng.criterion_holds(to_full(xk), r, ctx)
        us, info, its, tc = krylov_solve(self.Ks, rs, "minres", self.maxiter, stop, CONFIG["inexact"]["check_every"])
        return to_full(us), {"krylov_iters": its, "krylov_info": info, "t_check": tc}
SOLVER_KINDS["minres"] = MinresKind

class CGReducedKind:
    """CG on the reduced SPD system S = P + G'W^{-1}W^{-T}G (p = 0 only), Ruiz-scaled, expanded to the full system for verification."""
    def __init__(self, spec, K, asm, W, eng):
        self.S, self.rhs_S, self.expand = reduced_spd(asm, W)
        self.scaled = spec.get("scaling", "ruiz") == "ruiz"
        self.d = ruiz_equilibrate(self.S) if self.scaled else None
        self.Ss = apply_scaling(self.S, self.d) if self.scaled else self.S
        self.maxiter = int(min(CONFIG["inexact"]["minres_maxiter_cap"], CONFIG["inexact"]["minres_maxiter_factor"] * self.S.shape[0]))
        self.kappa_S = cond_exact(self.S) if spec.get("_compute_cond") else float("nan"); self.eng = eng
    def base_solve(self, r, ctx):
        rS = self.rhs_S(r); rSs = self.d * rS if self.scaled else rS
        to_full = (lambda xs: self.expand(self.d * xs if self.scaled else xs, r))
        stop = lambda xk: self.eng.criterion_holds(to_full(xk), r, ctx)
        us, info, its, tc = krylov_solve(self.Ss, rSs, "cg", self.maxiter, stop, CONFIG["inexact"]["check_every"])
        return to_full(us), {"krylov_iters": its, "krylov_info": info, "t_check": tc, "kappa_reduced": self.kappa_S}
SOLVER_KINDS["cg_reduced"] = CGReducedKind

def tolerance_for(spec, ctx):
    rule = spec.get("tol_rule", "fixed")
    if rule == "fixed": return float(spec.get("tol", 1e-8))
    if rule == "adaptive_nt":
        c = spec; return float(min(c["tol_max"], max(c["tol_min"], c["eta"] * ctx["nt_proxy"])))
    raise ValueError(rule)

class KKTSolveEngine:
    """One object per kktsolver(W) call: holds K, its norm estimate, the block layout and the acceptance criterion."""
    def __init__(self, spec, K, asm, W):
        self.spec, self.K, self.asm, self.W = spec, K, asm, W
        self.kind = spec["kind"]; self.criterion = spec.get("criterion", "backward")
        n, p, mz = asm.n, asm.p, asm.mz
        self.block_slices = [slice(0, n), slice(n, n + p), slice(n + p, n + p + mz)]
        inst = asm.inst
        floor = float(spec.get("block_abs_floor", CONFIG["inexact"]["block_abs_floor"]))
        self.block_floor = [floor * max(1.0, np.linalg.norm(inst["q"])), floor * max(1.0, np.linalg.norm(inst["b"]) if inst.get("b") is not None and p > 0 else 1.0),
                            floor * max(1.0, np.linalg.norm(inst["h"]))]
        self.nK = norm2_estimate(K)
        self.accept_factor = spec.get("accept_factor", CONFIG["inexact"]["accept_factor"])
        self.max_refine = spec.get("max_refine", CONFIG["inexact"]["max_refine"])
        self.fallback = spec.get("fallback", CONFIG["inexact"]["fallback"])
        if self.kind == "exact": self.max_refine, self.fallback = 0, False
        self.impl = SOLVER_KINDS[self.kind](spec, K, asm, W, self)
        if self.kind == "exact_reduced": self.max_refine, self.fallback = 0, False
        self._exact = self.impl.f if isinstance(self.impl, (ExactKind, NoiseKind)) else None
        self._last_measure = None
    def exact(self):
        if self._exact is None:
            # the fallback factors the reduced quasi-definite system with CHOLMOD, as CVXOPT does; the LU of the full KKT
            # matrix fills in badly on the sparse families (fallback_reduced=False restores it)
            if self.asm.l_only and CONFIG["inexact"].get("fallback_reduced", True):
                try: self._exact = _ReducedFallback(self.K, self.asm, self.W, self)
                except Exception: self._exact = ExactFactor(self.K)
            else: self._exact = ExactFactor(self.K)
        return self._exact
    # acceptance tests, always on the original system
    def measure(self, u, r):
        e = self.K @ u - r
        ne, nr, nu = np.linalg.norm(e), np.linalg.norm(r), np.linalg.norm(u)
        relres = ne / nr if nr > 0 else ne
        backward = ne / (self.nK * nu + nr) if (self.nK * nu + nr) > 0 else ne
        ratios = []
        for sl, fl in zip(self.block_slices, self.block_floor):
            if sl.stop - sl.start == 0: ratios.append(0.0); continue
            ratios.append(float(np.linalg.norm(e[sl]) / max(np.linalg.norm(r[sl]), fl)))
        return {"relres": float(relres), "backward_err": float(backward), "block_ratio_x": ratios[0], "block_ratio_y": ratios[1], "block_ratio_z": ratios[2],
                "block_ratio_max": float(max(ratios)), "abs_resid": float(ne)}
    def threshold(self, ctx):
        if self.criterion == "block": return float(self.spec.get("eta", 0.1))
        if self.criterion in ("relres_floor", "combined"): return 1.0
        return float(self.accept_factor * tolerance_for(self.spec, ctx))
    def criterion_value(self, m, ctx=None):
        if self.criterion == "relres_floor":
            # relres, or a backward error already at the level of a direct solve
            tol = self.accept_factor * tolerance_for(self.spec, ctx or {}); return float(min(m["relres"] / tol, m["backward_err"] / self.spec.get("be_floor", 1e-14)))
        if self.criterion == "combined":
            tol = self.accept_factor * tolerance_for(self.spec, ctx or {}); return float(max(m["block_ratio_max"] / self.spec.get("eta", 0.1), m["backward_err"] / tol))
        return {"relres": m["relres"], "backward": m["backward_err"], "block": m["block_ratio_max"]}[self.criterion]
    def criterion_holds(self, u, r, ctx):
        if not np.all(np.isfinite(u)): return False
        m = self.measure(u, r); self._last_measure = m
        return self.criterion_value(m, ctx) <= self.threshold(ctx)
    # base solve -> verify -> refine -> fallback
    def solve(self, r, ctx):
        rtol = tolerance_for(self.spec, ctx) if self.criterion not in ("block",) else float(self.spec.get("eta", 0.1))
        accept = self.threshold(ctx)
        if self.kind == "noise": accept = float("inf")
        if self.kind == "exact_reduced": accept = 1e-6
        info_all = {}
        with Timer() as t_solve:
            u, info = self.impl.base_solve(r, ctx)
        info_all.update(info)
        with Timer() as t_verify:
            finite = bool(np.all(np.isfinite(u)))
            m0 = self.measure(u, r) if finite else {"relres": np.inf, "backward_err": np.inf, "block_ratio_x": np.inf, "block_ratio_y": np.inf, "block_ratio_z": np.inf, "block_ratio_max": np.inf, "abs_resid": np.inf}
            val0 = self.criterion_value(m0, ctx)
        m, val, n_ref, t_ref = m0, val0, 0, Timer()
        if finite and val > accept and self.max_refine > 0:
            with t_ref:
                while val > accept and n_ref < self.max_refine:
                    du, info2 = self.impl.base_solve(r - self.K @ u, dict(ctx, is_refine=True))
                    for k2, v2 in info2.items():
                        if k2 in ("krylov_iters", "t_check"): info_all[k2] = info_all.get(k2, 0) + v2
                    if not np.all(np.isfinite(du)): break
                    u = u + du; n_ref += 1; m = self.measure(u, r); val = self.criterion_value(m, ctx)
        used_fallback, t_fb, solve_failed = False, Timer(), False
        if (not np.all(np.isfinite(u))) or val > accept:
            if self.fallback:
                with t_fb:
                    u = self.exact().solve(r); used_fallback = True; info_all["fallback_backend"] = getattr(self._exact, "backend", "lu_full")
                    if np.all(np.isfinite(u)): m = self.measure(u, r); val = self.criterion_value(m, ctx)
                    else: val = np.inf
        if not np.all(np.isfinite(u)): solve_failed = True      # only non-finite directions abort a run (CVXOPT behaviour)
        log = {"solve_failed": solve_failed, "criterion": self.criterion, "target_rtol": rtol, "accept_rtol": accept, "crit_initial": val0, "crit_final": val,
               "rel_resid_initial": m0["relres"], "rel_resid_final": m["relres"], "backward_err_initial": m0["backward_err"], "backward_err_final": m["backward_err"],
               "block_ratio_max_initial": m0["block_ratio_max"], "block_ratio_max_final": m["block_ratio_max"],
               "block_ratio_x": m["block_ratio_x"], "block_ratio_y": m["block_ratio_y"], "block_ratio_z": m["block_ratio_z"], "abs_resid_final": m["abs_resid"],
               "n_refine": n_ref, "fallback": used_fallback, "accepted": bool(val <= accept), "method_used": "classical_fallback" if used_fallback else self.kind,
               "t_solve": t_solve.t - info_all.get("t_check", 0.0), "t_verify": t_verify.t, "t_refine": t_ref.t, "t_fallback": t_fb.t, "t_check": info_all.get("t_check", 0.0),
               "rhs_norm": float(np.linalg.norm(r)), "rhs_norm_x": float(np.linalg.norm(r[self.block_slices[0]])), "rhs_norm_y": float(np.linalg.norm(r[self.block_slices[1]])),
               "rhs_norm_z": float(np.linalg.norm(r[self.block_slices[2]])), "kkt_dim": int(self.K.shape[0]), "nK_est": float(self.nK)}
        log.update(info_all)
        return u, log



# MINRES on NumPy or CuPy arrays, LSQR on the square-root system, reduced sparse direct solve.
def minres_base_kernel(matvec, b, xp, x0=None, maxiter=1000, stop_fn=None, check_every=8):
    """Paige-Saunders MINRES written against the array module xp (numpy or cupy).
    Stops when stop_fn(x) is True (checked every `check_every` iterations and when the recurrence residual estimate is tiny).
    Returns (x, info, iterations, t_check); info = 0 converged by stop_fn, 1 maxiter, 2 breakdown/exact."""
    t_check = 0.0
    x = xp.zeros_like(b) if x0 is None else x0.copy()
    r = b if x0 is None else b - matvec(x)
    beta1 = float(xp.linalg.norm(r))
    def check(xk):
        nonlocal t_check
        t0 = time.perf_counter(); ok = bool(stop_fn(xk)) if stop_fn is not None else False; t_check += time.perf_counter() - t0
        return ok
    if beta1 == 0.0: return x, 0, 0, t_check
    if x0 is not None and check(x): return x, 0, 0, t_check
    v_old = xp.zeros_like(b); v = r / beta1; beta = beta1
    w_old = xp.zeros_like(b); w = xp.zeros_like(b)
    eta = beta1; g0 = g1 = 1.0; s0 = s1 = 0.0; bnorm = float(xp.linalg.norm(b)) or 1.0
    for it in range(1, maxiter + 1):
        Av = matvec(v); alpha = float(xp.vdot(v, Av).real)
        v_new = Av - alpha * v - beta * v_old
        beta_new = float(xp.linalg.norm(v_new))
        delta = g1 * alpha - g0 * s1 * beta
        rho1 = math.hypot(delta, beta_new); rho2 = s1 * alpha + g0 * g1 * beta; rho3 = s0 * beta
        if rho1 == 0.0: return x, 2, it, t_check
        g_new = delta / rho1; s_new = beta_new / rho1
        w_new = (v - rho3 * w_old - rho2 * w) / rho1
        x = x + (g_new * eta) * w_new
        eta = -s_new * eta
        v_old, v = v, (v_new / beta_new if beta_new > 0 else v_new)
        w_old, w = w, w_new; beta = beta_new; g0, g1 = g1, g_new; s0, s1 = s1, s_new
        if (it % check_every == 0) or abs(eta) <= 1e-15 * bnorm or beta_new == 0.0:
            if check(x): return x, 0, it, t_check
            if beta_new == 0.0: return x, 2, it, t_check
    return x, 1, maxiter, t_check

class MinresBaseKind:
    """Ruiz-scaled MINRES with the backend-agnostic kernel; backend 'cpu' (numpy) or 'gpu' (cupy). Optional warm start:
    the combined-direction solve of an IPM iteration starts from the affine direction computed with the same matrix."""
    def __init__(self, spec, K, asm, W, eng):
        self.gpu = spec.get("backend", "cpu") == "gpu"
        if self.gpu and not GPU_OK: raise RuntimeError("GPU backend requested but no usable GPU")
        self.xp = DEVICE_XP if self.gpu else np
        self.eng = eng; self.ws = bool(spec.get("warm_start", False)); self.last = None
        d = ruiz_equilibrate(K) if spec.get("scaling", "ruiz") == "ruiz" else np.ones(K.shape[0])
        Ks = apply_scaling(K, d)
        t0 = time.perf_counter()
        self.K = to_device(K) if self.gpu else K; self.Ks = to_device(Ks) if self.gpu else Ks; self.d = self.xp.asarray(d)
        self.t_transfer = time.perf_counter() - t0
        self.maxiter = int(min(CONFIG["inexact"]["minres_maxiter_cap"], CONFIG["inexact"]["minres_maxiter_factor"] * K.shape[0]))
        self.floors = eng.block_floor; self.slices = eng.block_slices
    def _holds(self, u, r, ctx):
        xp = self.xp; e = self.K @ u - r
        ne, nr, nu = float(xp.linalg.norm(e)), float(xp.linalg.norm(r)), float(xp.linalg.norm(u))
        if not np.isfinite(ne): return False
        ratios = [0.0 if sl.stop == sl.start else float(xp.linalg.norm(e[sl])) / max(float(xp.linalg.norm(r[sl])), fl) for sl, fl in zip(self.slices, self.floors)]
        m = {"relres": ne / nr if nr > 0 else ne, "backward_err": ne / (self.eng.nK * nu + nr) if (self.eng.nK * nu + nr) > 0 else ne, "block_ratio_max": max(ratios)}
        return self.eng.criterion_value(m, ctx) <= self.eng.threshold(ctx)
    def base_solve(self, r, ctx):
        xp = self.xp; r_d = xp.asarray(r); rs = self.d * r_d
        warm = self.ws and self.last is not None and not ctx.get("is_refine"); x0 = self.last if warm else None
        # Warm start as a correction solve K delta = r - K u0, with the acceptance test applied to that system. Testing u0 + delta
        # against r is unsafe: the affine direction u0 can pass the block test for the combined right-hand side at iteration 0,
        # the corrector term is then dropped and the IPM diverges (seen on 2 of 35 instances).
        r_t = (r_d - self.K @ x0) if warm else r_d
        xs, info, its, tc = minres_base_kernel(lambda v: self.Ks @ v, self.d * r_t, xp, x0=None, maxiter=self.maxiter,
                                      stop_fn=lambda xk: self._holds(self.d * xk, r_t, ctx), check_every=CONFIG["inexact"]["check_every"])
        u_d = self.d * xs + (x0 if warm else 0.0)
        if not ctx.get("is_refine"): self.last = u_d
        if self.gpu: dev_sync()
        return to_host(u_d) if self.gpu else u_d, {"krylov_iters": its, "krylov_info": info, "t_check": tc, "warm_started": bool(warm), "backend": "gpu" if self.gpu else "cpu", "t_transfer": self.t_transfer}
SOLVER_KINDS["minres_base"] = MinresBaseKind

class LSQRSqrtKind:
    """Square-root target system for P > 0 and no equalities: S = P + Ghat'Ghat = B'B with B = [L'; Ghat], P = L L'.
    LSQR on B sees kappa(B) = sqrt(kappa(S))."""
    def __init__(self, spec, K, asm, W, eng):
        assert asm.l_only and asm.p == 0
        inst = asm.inst; n = asm.n
        if "_Lchol" not in inst:
            Pd = inst["P"].toarray() if sp.issparse(inst["P"]) else inst["P"]; inst["_Lchol"] = np.linalg.cholesky(Pd)
        self.L = inst["_Lchol"]; di = to_np(W["di"]); G = inst["G"]
        self.Gh = (sp.diags(di) @ G).toarray() if sp.issparse(G) else di[:, None] * G
        B = np.vstack([self.L.T, self.Gh]); self.c = 1.0 / np.maximum(np.linalg.norm(B, axis=0), 1e-300); self.Bs = B * self.c[None, :]
        self.n = n; self.maxiter = int(min(CONFIG["inexact"]["minres_maxiter_cap"], 4 * n)); self.tol = float(spec.get("lsqr_tol", 1e-13))
    def base_solve(self, r, ctx):
        n = self.n; bx, bz = r[:n], r[n:]
        t = np.concatenate([sla.solve_triangular(self.L, bx, lower=True), bz])
        y, istop, itn = spla.lsqr(self.Bs, t, atol=self.tol, btol=self.tol, iter_lim=self.maxiter)[:3]
        ux = self.c * y
        return np.concatenate([ux, self.Gh @ ux - bz]), {"krylov_iters": int(itn), "krylov_info": int(istop)}
SOLVER_KINDS["lsqr_sqrt"] = LSQRSqrtKind

class ReducedCholFactor:
    """Sparse factorisation of the reduced quasi-definite system M = [[S, A'], [A, 0]] (S symmetric positive definite) with
    CHOLMOD (shipped with CVXOPT) and a dense p x p Schur complement C = A S^-1 A'. At n = 3 200 this takes 0.44 s per
    factorisation against 4.7-7.1 s for SuperLU. Every solve applies one step of iterative refinement on M. Uses SuperLU
    (symmetric-mode ordering) if CHOLMOD reports that S is not positive definite."""
    def __init__(self, S, A, M):
        from cvxopt import cholmod as _chol, spmatrix as _spm
        self._chol = _chol; self.M = M; self.n = S.shape[0]; self.p = 0 if A is None else A.shape[0]; self.backend = "cholmod"
        try:
            C = sp.tril(S, format="coo"); Sc = _spm(C.data.tolist(), C.row.tolist(), C.col.tolist(), S.shape)
            self.F = _chol.symbolic(Sc); _chol.numeric(Sc, self.F)
            if self.p:
                self.A = A.tocsr() if sp.issparse(A) else sp.csr_matrix(A)
                Z = matrix(np.asfortranarray(self.A.T.toarray())); _chol.solve(self.F, Z); self.SiAt = np.array(Z)
                Cs = self.A @ self.SiAt; self.Cf = sla.cho_factor(0.5 * (Cs + Cs.T), check_finite=False)
        except Exception as e:
            self.backend = f"superlu ({type(e).__name__})"; self.lu = spla.splu(sp.csc_matrix(M), permc_spec="MMD_AT_PLUS_A")
    def _S_solve(self, v):
        B = matrix(np.ascontiguousarray(v, dtype=float).reshape(-1, 1)); self._chol.solve(self.F, B); return np.array(B).ravel()
    def _solve_once(self, b):
        if self.backend != "cholmod": return self.lu.solve(b)
        n, p = self.n, self.p
        if not p: return self._S_solve(b)
        t = self._S_solve(b[:n]); uy = sla.cho_solve(self.Cf, self.A @ t - b[n:], check_finite=False)
        return np.concatenate([t - self.SiAt @ uy, uy])
    def solve(self, b):
        x = self._solve_once(b); return x + self._solve_once(b - self.M @ x)

def reduced_factor(S, A, M, sparse):
    return ReducedCholFactor(S, A, M) if (sparse and CONFIG["inexact"].get("use_cholmod", True)) else ExactFactor(M)

class ReducedExactKind:
    """Sparse direct reference: factor the reduced quasi-definite system [[S, A'], [A, 0]], S = P + Ghat'Ghat, instead of the
    full KKT matrix, whose sparse LU fills in badly (970 s per run at KKT dimension 9 920)."""
    def __init__(self, spec, K, asm, W, eng):
        assert asm.l_only
        inst = asm.inst; n, p = asm.n, asm.p; di = to_np(W["di"]); G = inst["G"]; P = inst["P"]
        if asm.sparse:
            self.Gh = (sp.diags(di) @ G).tocsr(); S = (P + self.Gh.T @ self.Gh)
            M = sp.bmat([[S, inst["A"].T], [inst["A"], None]], format="csc") if p else S.tocsc()
        else:
            Gd = G.toarray() if sp.issparse(G) else G; self.Gh = di[:, None] * Gd; S = (P.toarray() if sp.issparse(P) else P) + self.Gh.T @ self.Gh
            A = inst["A"].toarray() if (p and sp.issparse(inst["A"])) else inst["A"]
            M = np.block([[S, A.T], [A, np.zeros((p, p))]]) if p else S
        self.f = reduced_factor(S, inst["A"] if p else None, M, asm.sparse); self.n, self.p = n, p
    def base_solve(self, r, ctx):
        n, p = self.n, self.p; bx, by, bz = r[:n], r[n:n + p], r[n + p:]
        sol = self.f.solve(np.concatenate([bx + self.Gh.T @ bz, by])); ux = sol[:n]
        return np.concatenate([ux, sol[n:], self.Gh @ ux - bz]), {}
SOLVER_KINDS["exact_reduced"] = ReducedExactKind

class _ReducedFallback:
    """ReducedExactKind behind a .solve(r) interface: the classical fallback of every inexact method."""
    def __init__(self, K, asm, W, eng):
        self.impl = ReducedExactKind({"name": "fallback_reduced", "kind": "exact_reduced"}, K, asm, W, eng); self.backend = "reduced_" + str(getattr(self.impl.f, "backend", "lu"))
    def solve(self, r): return self.impl.base_solve(r, {})[0]


# Null-space target system. u_x = x_p + Z v with A x_p = r_y and Z an orthonormal basis of null(A);
# (Z'SZ) v = Z'(r_x + Gh' r_z - S x_p) with S = P + Gh'Gh SPD; u_z = Gh u_x - r_z; u_y = (AA')^-1 A (r_x - P u_x - Gh' u_z).
# The equality block is exact and the acceptance test is applied to the full KKT system. Z is dense (QR of A', once per
# instance), so for these methods the iteration counts and kappa are the quantities of interest, not the time.
def oss_basis(inst):
    if "_oss" not in inst:
        n, p = inst["n"], inst["p"]
        if p == 0: inst["_oss"] = {"Z": None, "Ad": None, "AAt": None}
        else:
            Ad = inst["A"].toarray() if sp.issparse(inst["A"]) else np.asarray(inst["A"], dtype=float); Q, _ = np.linalg.qr(Ad.T, mode="complete")
            inst["_oss"] = {"Z": np.ascontiguousarray(Q[:, p:]), "Ad": Ad, "AAt": sla.cho_factor(Ad @ Ad.T, check_finite=False)}
    return inst["_oss"]

class OSSKind:
    """KKT-solver kind 'oss_cg': CG on the Jacobi-scaled null-space SPD system, combined acceptance on the full KKT system."""
    def __init__(self, spec, K, asm, W, eng):
        assert asm.l_only, "null-space solver: linear cone only"
        inst = asm.inst; n, p = asm.n, asm.p
        if n > int(spec.get("max_n", CONFIG["inexact"].get("oss_max_n", 3200))): raise RuntimeError(f"oss: n = {n} above oss_max_n (dense null-space basis)")
        di = to_np(W["di"]); G = inst["G"]; P = inst["P"]
        self.Gh = (sp.diags(di) @ G).tocsr() if sp.issparse(G) else di[:, None] * np.asarray(G)
        self.S = (P + self.Gh.T @ self.Gh).tocsr() if sp.issparse(P) else (np.asarray(P) + self.Gh.T @ self.Gh)
        self.P = P; bs = oss_basis(inst); self.Z, self.Ad, self.AAt = bs["Z"], bs["Ad"], bs["AAt"]; self.n, self.p, self.eng = n, p, eng; self.dim = n - p
        SZ = self.S @ self.Z if p else None
        diag = np.einsum("ij,ij->j", self.Z, SZ) if p else np.asarray(self.S.diagonal() if sp.issparse(self.S) else np.diag(self.S)).ravel()
        self.dj = 1.0 / np.sqrt(np.maximum(diag, 1e-300))                           # Jacobi scaling of the reduced system
        self.maxiter = int(min(spec.get("minres_maxiter_cap", CONFIG["inexact"]["minres_maxiter_cap"]), CONFIG["inexact"]["minres_maxiter_factor"] * K.shape[0]))    # same cap as the augmented-system methods
        self.kappa_reduced = float("nan"); self.reduced_dim = self.dim
    def matrix(self):
        """Explicit Jacobi-scaled reduced matrix D Z'SZ D (dense; used for kappa and by the quantum OSS solver on small instances)."""
        M = (self.Z.T @ (self.S @ self.Z)) if self.p else (self.S.toarray() if sp.issparse(self.S) else np.asarray(self.S))
        return self.dj[:, None] * M * self.dj[None, :]
    def compute_kappa(self):
        if self.dim <= CONFIG["cond_limit"]: self.kappa_reduced = float(np.linalg.cond(self.matrix()))
        return self.kappa_reduced
    def matvec(self, v):
        w = self.dj * v; x = self.Z @ w if self.p else w; y = self.S @ x
        return self.dj * (self.Z.T @ y if self.p else y)
    def reduced_rhs(self, r):
        n, p = self.n, self.p; rx, ry, rz = r[:n], r[n:n + p], r[n + p:]
        xp_ = self.Ad.T @ sla.cho_solve(self.AAt, ry, check_finite=False) if p else np.zeros(n)   # particular solution of A x = r_y
        b = rx + self.Gh.T @ rz - (self.S @ xp_ if p else 0.0)
        return xp_, self.dj * (self.Z.T @ b if p else b)
    def full(self, v, r, xp_):
        n, p = self.n, self.p; rx, rz = r[:n], r[n + p:]; w = self.dj * v; ux = xp_ + (self.Z @ w if p else w); uz = self.Gh @ ux - rz
        uy = sla.cho_solve(self.AAt, self.Ad @ (rx - self.P @ ux - self.Gh.T @ uz), check_finite=False) if p else np.zeros(0)
        return np.concatenate([ux, uy, uz])
    def base_solve(self, r, ctx):
        xp_, b = self.reduced_rhs(r); op = spla.LinearOperator((self.dim, self.dim), matvec=self.matvec, dtype=float)
        stop = lambda vk: self.eng.criterion_holds(self.full(vk, r, xp_), r, ctx)
        v, info, its, tc = krylov_solve(op, b, "cg", self.maxiter, stop, CONFIG["inexact"]["check_every"])
        return self.full(v, r, xp_), {"krylov_iters": its, "krylov_info": info, "t_check": tc, "kappa_reduced": self.kappa_reduced, "reduced_dim": self.dim}
SOLVER_KINDS["oss_cg"] = OSSKind

class StaleLUKind:
    """Lazy refactorisation. The reduced system M_k = [[P + G'W_k^-2 G, A'], [A, 0]] is factorised only when needed; in
    between, the factors of an earlier IPM iteration are a right preconditioner for GMRES on the current M_k
    (M_k P^-1 y = b, u = P^-1 y, so the monitored residual is the true one). The matrix is refactorised when GMRES misses
    its tolerance within `stale_maxiter` iterations (the solve is then finished with the new factors) or when the previous
    iteration needed more than `stale_refresh_iters` iterations. Every direction still has to pass the acceptance test on
    the full KKT system. The state lives on the per-run assembler object."""
    def __init__(self, spec, K, asm, W, eng):
        assert asm.l_only
        inst = asm.inst; n, p = asm.n, asm.p; di = to_np(W["di"]); G = inst["G"]; P = inst["P"]
        if asm.sparse:
            self.Gh = (sp.diags(di) @ G).tocsr(); S = (P + self.Gh.T @ self.Gh)
            self.M = sp.bmat([[S, inst["A"].T], [inst["A"], None]], format="csc") if p else S.tocsc()
        else:
            Gd = G.toarray() if sp.issparse(G) else G; self.Gh = di[:, None] * Gd; S = (P.toarray() if sp.issparse(P) else P) + self.Gh.T @ self.Gh
            A = inst["A"].toarray() if (p and sp.issparse(inst["A"])) else inst["A"]
            self.M = np.block([[S, A.T], [A, np.zeros((p, p))]]) if p else S
        self.n, self.p, self.asm = n, p, asm; self.S = S; self.A = inst["A"] if p else None
        self.maxit = int(spec.get("stale_maxiter", 40)); self.refresh_at = int(spec.get("stale_refresh_iters", 15)); self.rtol = float(spec.get("stale_rtol", 1e-12))
        st = getattr(asm, "_stale", None)
        self.did_factor = st is None or st.get("refresh", True)
        if self.did_factor: self._factor_now()
        self.age = asm._stale["age"]; asm._stale["age"] += 1
    def _factor_now(self):
        self.asm._stale = {"f": reduced_factor(self.S, self.A, self.M, self.asm.sparse), "age": 0, "refresh": False}; self.fresh = True
    def base_solve(self, r, ctx):
        n, p = self.n, self.p; bx, by, bz = r[:n], r[n:n + p], r[n + p:]
        b = np.concatenate([bx + self.Gh.T @ bz, by]); f = self.asm._stale["f"]; its = 0; refactored = False
        if getattr(self, "fresh", False):
            sol = f.solve(b)
        else:
            cnt = [0]
            def _cb(_): cnt[0] += 1
            psolve = getattr(f, "_solve_once", f.solve)          # preconditioner application: no refinement step needed
            op = spla.LinearOperator(self.M.shape, matvec=lambda v: self.M @ psolve(v), dtype=float)
            y, info = spla.gmres(op, b, rtol=self.rtol, atol=0.0, restart=self.maxit, maxiter=1, callback=_cb, callback_type="pr_norm")
            its = cnt[0]; sol = psolve(y)
            bad = (info != 0) or (not np.all(np.isfinite(sol))) or np.linalg.norm(self.M @ sol - b) > 1e3 * self.rtol * max(np.linalg.norm(b), 1e-300)
            if bad:
                self._factor_now(); sol = self.asm._stale["f"].solve(b); refactored = True
            elif its > self.refresh_at:
                self.asm._stale["refresh"] = True                 # next IPM iteration starts with a fresh factorisation
        ux = sol[:n]
        return np.concatenate([ux, sol[n:], self.Gh @ ux - bz]), {"krylov_iters": its, "refactored_now": refactored, "stale_age": 0 if getattr(self, "fresh", False) else self.age}
SOLVER_KINDS["gmres_stalelu"] = StaleLUKind


# Latency-optimised MINRES: one host synchronisation per iteration, fused vector updates (cupy.fuse, checked against the
# plain code), explicit checks spaced as max(check_every, it // 16) (at most ~6 % extra iterations), one synchronisation per
# check, and a device-resident sparsity pattern (only K.data and the Ruiz vector are uploaded per factorisation).
_FUSED_CACHE = {}
def _lanczos_step_plain(Av, v, v_old, alpha, beta): return Av - alpha * v - beta * v_old
def _wx_step_plain(v, w_old, w, x, rho3, rho2, inv_rho1, ge):
    w_new = (v - rho3 * w_old - rho2 * w) * inv_rho1
    return w_new, x + ge * w_new
def fused_ops(xp):
    key = getattr(xp, "__name__", str(xp))
    if key in _FUSED_CACHE: return _FUSED_CACHE[key]
    out = (_lanczos_step_plain, _wx_step_plain, "plain")
    if xp is not np and CONFIG["inexact"].get("gpu_fuse", True) and hasattr(xp, "fuse"):
        try:
            f1, f2 = xp.fuse(_lanczos_step_plain), xp.fuse(_wx_step_plain)
            a, b_, c, x_ = (xp.asarray(np.random.default_rng(k_).standard_normal(257)) for k_ in range(4)); al = xp.vdot(a, b_)
            t1, t2 = f1(a, b_, c, al, 0.3), _lanczos_step_plain(a, b_, c, al, 0.3)
            u1, u2 = f2(a, b_, c, x_, 0.1, 0.2, 0.7, 1.3), _wx_step_plain(a, b_, c, x_, 0.1, 0.2, 0.7, 1.3)
            err = max(float(xp.max(xp.abs(t1 - t2))), float(xp.max(xp.abs(u1[0] - u2[0]))), float(xp.max(xp.abs(u1[1] - u2[1]))))
            if not err <= 1e-12: raise ValueError(f"fused kernels differ from plain code by {err:.1e}")
            out = (f1, f2, "cupy.fuse")
        except Exception as e:
            out = (_lanczos_step_plain, _wx_step_plain, f"plain (fuse unavailable: {type(e).__name__}: {e})"[:120])
    _FUSED_CACHE[key] = out
    return out

def minres_fused_kernel(matvec, b, xp, maxiter=1000, stop_fn=None, check_every=8, adaptive_div=16):
    """Paige-Saunders MINRES with the same arithmetic as minres_base_kernel, arranged for few host synchronisations.
    Returns (x, info, iterations, t_check, n_checks); info = 0 converged by stop_fn, 1 maxiter, 2 breakdown."""
    lanczos_step, wx_step, _ = fused_ops(xp); t_check = 0.0; n_checks = 0
    beta1 = float(xp.linalg.norm(b)); x = xp.zeros_like(b)
    if beta1 == 0.0: return x, 0, 0, t_check, n_checks
    v_old = xp.zeros_like(b); v = b * (1.0 / beta1); beta = beta1; w_old = xp.zeros_like(b); w = xp.zeros_like(b)
    eta = beta1; g0 = g1 = 1.0; s0 = s1 = 0.0; next_check = check_every
    for it in range(1, maxiter + 1):
        Av = matvec(v); alpha_d = xp.vdot(v, Av)
        v_new = lanczos_step(Av, v, v_old, alpha_d, beta)
        if xp is np: alpha, beta_new = float(alpha_d), float(np.linalg.norm(v_new))
        else:
            ab = xp.stack((alpha_d, xp.linalg.norm(v_new))); ab = ab.get() if hasattr(ab, "get") else ab  # the only synchronisation of the iteration
            alpha, beta_new = float(ab[0]), float(ab[1])
        delta = g1 * alpha - g0 * s1 * beta
        rho1 = math.hypot(delta, beta_new); rho2 = s1 * alpha + g0 * g1 * beta; rho3 = s0 * beta
        if rho1 == 0.0: return x, 2, it, t_check, n_checks
        g_new = delta / rho1; s_new = beta_new / rho1
        w_new, x = wx_step(v, w_old, w, x, rho3, rho2, 1.0 / rho1, g_new * eta)
        eta = -s_new * eta
        v_old, v = v, (v_new * (1.0 / beta_new) if beta_new > 0 else v_new)
        w_old, w = w, w_new; beta = beta_new; g0, g1 = g1, g_new; s0, s1 = s1, s_new
        if it >= next_check or abs(eta) <= 1e-15 * beta1 or beta_new == 0.0:
            t0 = time.perf_counter(); ok = bool(stop_fn(x)) if stop_fn is not None else False; t_check += time.perf_counter() - t0; n_checks += 1
            if ok: return x, 0, it, t_check, n_checks
            if beta_new == 0.0: return x, 2, it, t_check, n_checks
            next_check = it + max(check_every, it // max(int(adaptive_div), 1))
    return x, 1, maxiter, t_check, n_checks

class MinresFusedKind:
    """Ruiz-scaled MINRES with the latency-optimised kernel.
    backend = 'cpu' | 'gpu' | 'auto' ('auto' uses the GPU from CONFIG['inexact']['gpu_auto_min_dim'] on, else the CPU)."""
    def __init__(self, spec, K, asm, W, eng):
        be = spec.get("backend", "cpu"); N = K.shape[0]
        if be == "auto": be = "gpu" if (GPU_OK and N >= CONFIG["inexact"]["gpu_auto_min_dim"]) else "cpu"
        self.gpu = be == "gpu"
        if self.gpu and not GPU_OK: raise RuntimeError("GPU backend requested but no usable GPU")
        self.xp = DEVICE_XP if self.gpu else np; xp = self.xp; self.eng = eng
        d = ruiz_equilibrate(K) if spec.get("scaling", "ruiz") == "ruiz" else np.ones(N)
        t0 = time.perf_counter()
        if sp.issparse(K):
            Kc = K.tocsr(); Kc.sort_indices()
            if self.gpu:
                pat = getattr(asm, "_dev_pattern", None)
                if pat is None or pat["nnz"] != Kc.nnz or not (np.array_equal(pat["indptr_h"], Kc.indptr) and np.array_equal(pat["indices_h"], Kc.indices)):
                    ip = Kc.indptr.astype(np.int32); ix = Kc.indices.astype(np.int32)
                    pat = {"nnz": Kc.nnz, "indptr_h": Kc.indptr.copy(), "indices_h": Kc.indices.copy(), "indptr": xp.asarray(ip), "indices": xp.asarray(ix),
                           "rows": xp.asarray(np.repeat(np.arange(N, dtype=np.int32), np.diff(Kc.indptr))), "uploads": 0}
                    asm._dev_pattern = pat
                pat["uploads"] += 1; data_d = xp.asarray(Kc.data); d_d = xp.asarray(d)
                self.K = DEVICE_SPARSE.csr_matrix((data_d, pat["indices"], pat["indptr"]), shape=Kc.shape)
                self.Ks = DEVICE_SPARSE.csr_matrix((data_d * d_d[pat["rows"]] * d_d[pat["indices"]], pat["indices"], pat["indptr"]), shape=Kc.shape)
                self.d = d_d
            else:
                rows = np.repeat(np.arange(N), np.diff(Kc.indptr))
                self.K = Kc; self.Ks = sp.csr_matrix((Kc.data * d[rows] * d[Kc.indices], Kc.indices, Kc.indptr), shape=Kc.shape); self.d = d
        else:
            Ks = d[:, None] * K * d[None, :]
            self.K = xp.asarray(K) if self.gpu else K; self.Ks = xp.asarray(Ks) if self.gpu else Ks; self.d = xp.asarray(d)
        if self.gpu: dev_sync()
        self.t_transfer = time.perf_counter() - t0
        self.maxiter = int(min(spec.get("minres_maxiter_cap", CONFIG["inexact"]["minres_maxiter_cap"]), CONFIG["inexact"]["minres_maxiter_factor"] * N))
        self.floors = eng.block_floor; self.slices = [sl for sl in eng.block_slices if sl.stop > sl.start]; self.fl = [fl for sl, fl in zip(eng.block_slices, eng.block_floor) if sl.stop > sl.start]
        self.fuse_mode = fused_ops(xp)[2]
    def _holds(self, u, r, ctx):
        xp = self.xp; e = self.K @ u - r; nrm = xp.linalg.norm
        vals = xp.stack([nrm(e), nrm(r), nrm(u)] + [nrm(e[sl]) for sl in self.slices] + [nrm(r[sl]) for sl in self.slices])
        vals = np.asarray(vals.get() if hasattr(vals, "get") else vals, dtype=float)                       # one synchronisation per check
        ne, nr, nu = vals[:3]; nb = len(self.slices)
        if not np.isfinite(ne): return False
        ratios = [vals[3 + j] / max(vals[3 + nb + j], self.fl[j]) for j in range(nb)]
        den = self.eng.nK * nu + nr
        m = {"relres": ne / nr if nr > 0 else ne, "backward_err": ne / den if den > 0 else ne, "block_ratio_max": max(ratios) if ratios else 0.0}
        return self.eng.criterion_value(m, ctx) <= self.eng.threshold(ctx)
    def base_solve(self, r, ctx):
        xp = self.xp; r_d = xp.asarray(r)
        xs, info, its, tc, nchk = minres_fused_kernel(lambda v: self.Ks @ v, self.d * r_d, xp, maxiter=self.maxiter, stop_fn=lambda xk: self._holds(self.d * xk, r_d, ctx),
                                            check_every=CONFIG["inexact"]["check_every"], adaptive_div=CONFIG["inexact"]["check_adaptive_div"])
        u_d = self.d * xs
        if self.gpu: dev_sync()
        return (to_host(u_d) if self.gpu else u_d), {"krylov_iters": its, "krylov_info": info, "t_check": tc, "n_checks": nchk, "backend": "gpu" if self.gpu else "cpu",
                                                     "kernel": "fused", "fuse_mode": self.fuse_mode, "t_transfer": self.t_transfer}
SOLVER_KINDS["minres_fused"] = MinresFusedKind

# self-test of the MINRES kernel against a dense solve on a symmetric indefinite system
_r = rng_for("minres_selftest"); _Q, _ = np.linalg.qr(_r.standard_normal((60, 60))); _ev = np.concatenate([np.linspace(1, 50, 40), -np.linspace(1, 30, 20)])
_Am = (_Q * _ev) @ _Q.T; _bv = _r.standard_normal(60); _xe = np.linalg.solve(_Am, _bv)
_xm, _info, _its, _ = minres_base_kernel(lambda v: _Am @ v, _bv, np, maxiter=400, stop_fn=lambda x: np.linalg.norm(_Am @ x - _bv) <= 1e-11 * np.linalg.norm(_bv), check_every=4)
assert np.linalg.norm(_xm - _xe) <= 1e-8 * np.linalg.norm(_xe), "minres_base_kernel failed its self-test"
_xw, _, _its_w, _ = minres_base_kernel(lambda v: _Am @ v, _bv, np, x0=_xe * (1 + 1e-3), maxiter=400, stop_fn=lambda x: np.linalg.norm(_Am @ x - _bv) <= 1e-11 * np.linalg.norm(_bv), check_every=4)
print(f"minres_base_kernel self-test passed: {_its} iterations cold, {_its_w} warm-started; rel. error {np.linalg.norm(_xm - _xe) / np.linalg.norm(_xe):.1e}")
if GPU_OK:
    _xg, _, _itg, _ = minres_base_kernel(lambda v: DEVICE_XP.asarray(_Am) @ v, DEVICE_XP.asarray(_bv), DEVICE_XP, maxiter=400, stop_fn=lambda x: float(DEVICE_XP.linalg.norm(DEVICE_XP.asarray(_Am) @ x - DEVICE_XP.asarray(_bv))) <= 1e-11 * np.linalg.norm(_bv), check_every=4)
    assert np.linalg.norm(to_host(_xg) - _xe) <= 1e-8 * np.linalg.norm(_xe); print(f"  device MINRES matches ({_itg} iterations)")
_x5, _i5, _it5, _, _nc5 = minres_fused_kernel(lambda v: _Am @ v, _bv, np, maxiter=400, stop_fn=lambda x: np.linalg.norm(_Am @ x - _bv) <= 1e-11 * np.linalg.norm(_bv), check_every=4)
assert np.linalg.norm(_x5 - _xe) <= 1e-8 * np.linalg.norm(_xe), "minres_fused_kernel failed its self-test"
print(f"minres_fused_kernel self-test passed: {_it5} iterations, {_nc5} explicit checks (minres_base_kernel: {_its} iterations, {_its // 4} checks)")
if GPU_OK:
    _Ad, _bd = DEVICE_XP.asarray(_Am), DEVICE_XP.asarray(_bv)
    _xg5, _, _itg5, _, _ = minres_fused_kernel(lambda v: _Ad @ v, _bd, DEVICE_XP, maxiter=400, stop_fn=lambda x: float(DEVICE_XP.linalg.norm(_Ad @ x - _bd)) <= 1e-11 * np.linalg.norm(_bv), check_every=4)
    assert np.linalg.norm(to_host(_xg5) - _xe) <= 1e-8 * np.linalg.norm(_xe), "device minres_fused_kernel failed its self-test"
    GPU_STATUS["fuse_mode"] = fused_ops(DEVICE_XP)[2]; print(f"  device minres_fused_kernel matches ({_itg5} iterations); vector kernels: {GPU_STATUS['fuse_mode']}")
print("solver kinds:", list(SOLVER_KINDS))
