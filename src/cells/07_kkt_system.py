# KKT assembly (same layout as cvxopt.misc.kkt_ldl), condition numbers, Ruiz scaling, reduced SPD system.
class KKTAssembler:
    """Symmetric scaled KKT matrix  K~ = [P A' G'W^{-1}; A 0 0; W^{-T}G 0 -I]  and its rhs/solution packing."""
    def __init__(self, inst):
        self.inst = inst
        self.n, self.p, self.dims = inst["n"], inst["p"], inst["dims"]
        self.l_only = not (self.dims["q"] or self.dims["s"])
        self.sparse = bool(inst.get("sparse", False)) and self.l_only
        self.mz = self.dims["l"] + sum(self.dims["q"]) + sum(k * (k + 1) // 2 for k in self.dims["s"])
        self.dim = self.n + self.p + self.mz
        self.P = inst["P"]; self.G = inst["G"]; self.A = inst["A"] if self.p > 0 else None
        if not self.l_only:
            cd = cvx_data(inst)
            self.Gc = matrix(cd["G"]) if isinstance(cd["G"], spmatrix) else cd["G"]   # dense cvxopt copy for column scaling

    def build(self, W):
        n, p, N = self.n, self.p, self.dim
        if self.l_only:
            di = to_np(W["di"])
            if self.sparse:
                Gs = (sp.diags(di) @ self.G).tocsr()
                P = self.P if self.P is not None else sp.csr_matrix((n, n))
                blocks = [[P, self.A.T if p else None, Gs.T],
                          [self.A if p else None, None, None],
                          [Gs, None, -sp.identity(self.mz, format="csr")]]
                if p == 0:
                    blocks = [[P, Gs.T], [Gs, -sp.identity(self.mz, format="csr")]]
                return sp.bmat(blocks, format="csc")
            K = np.zeros((N, N))
            if self.P is not None:
                K[:n, :n] = self.P.toarray() if sp.issparse(self.P) else self.P
            if p:
                A = self.A.toarray() if sp.issparse(self.A) else self.A
                K[n:n + p, :n] = A; K[:n, n:n + p] = A.T
            G = self.G.toarray() if sp.issparse(self.G) else self.G
            Gs = di[:, None] * G
            K[n + p:, :n] = Gs; K[:n, n + p:] = Gs.T
            K[n + p:, n + p:] = -np.eye(self.mz)
            return K
        # generic cone path (used for SDP): identical construction to misc.kkt_ldl
        K = np.zeros((N, N))
        if self.P is not None:
            K[:n, :n] = self.P.toarray() if sp.issparse(self.P) else self.P
        if p:
            A = self.A.toarray() if sp.issparse(self.A) else self.A
            K[n:n + p, :n] = A; K[:n, n:n + p] = A.T
        gp = matrix(0.0, (self.mz, 1))
        for k in range(n):
            g = self.Gc[:, k]
            misc.scale(g, W, trans="T", inverse="I")
            misc.pack(g, gp, self.dims)
            col = to_np(gp)
            K[n + p:, k] = col; K[k, n + p:] = col
        K[n + p:, n + p:] = -np.eye(self.mz)
        return K

    def rhs(self, x, y, z, W):
        parts = [to_np(x)]
        if self.p: parts.append(to_np(y))
        if self.l_only:
            parts.append(to_np(W["di"]) * to_np(z))
        else:
            zc = matrix(z); misc.scale(zc, W, trans="T", inverse="I")
            zp = matrix(0.0, (self.mz, 1)); misc.pack(zc, zp, self.dims)
            parts.append(to_np(zp))
        return np.concatenate(parts)

    def write_back(self, u, x, y, z):
        n, p = self.n, self.p
        write_back(x, u[:n])
        if p: write_back(y, u[n:n + p])
        if self.l_only:
            write_back(z, u[n + p:])
        else:
            misc.unpack(to_cvx(u[n + p:]), z, self.dims)

def nt_spread(W):
    """Heuristic proxy for the duality measure mu from the NT scaling: min/max of the scaling factors.
    For the linear cone d_i = sqrt(s_i/z_i): active constraints give d_i ~ sqrt(mu), inactive ones d_i ~ 1/sqrt(mu),
    so min(d)/max(d) ~ mu (up to constants). For SDP blocks the singular values of r_k are used."""
    vals = [to_np(W["d"])] if W["d"].size[0] > 0 else []
    for r in W.get("r", []):
        s = np.linalg.svd(np.array(r), compute_uv=False); vals.append(s ** 2)
    for bta in W.get("beta", []):
        vals.append(np.array([bta]))
    if not vals: return 1.0
    v = np.concatenate(vals)
    return float(v.min() / v.max()) if v.max() > 0 else 1.0

def cond_exact(K, limit=None):
    """2-norm condition number (SVD) for dimension <= limit; NaN otherwise."""
    limit = CONFIG["cond_limit"] if limit is None else limit
    N = K.shape[0]
    if N > limit: return float("nan")
    Kd = K.toarray() if sp.issparse(K) else K
    try:
        return float(np.linalg.cond(Kd))
    except Exception:
        return float("nan")

def ruiz_equilibrate(K, iters=None):
    """Symmetric Ruiz equilibration: returns d such that D K D has rows/cols of ~unit inf-norm."""
    iters = CONFIG["inexact"]["ruiz_iters"] if iters is None else iters
    N = K.shape[0]; d = np.ones(N)
    if sp.issparse(K):
        # sweeps on the CSR arrays; forming D K D with two sparse products per sweep is much slower at large N
        Kc = K.tocsr(); ip = Kc.indptr; cnt = np.diff(ip); a = np.abs(Kc.data); rows = np.repeat(np.arange(N), cnt); cols = Kc.indices; ne = cnt > 0
        for _ in range(iters):
            v = a * d[rows] * d[cols]; rn = np.ones(N)
            if v.size: rn[ne] = np.maximum.reduceat(v, ip[:-1][ne])
            rn[rn == 0] = 1.0; d = d / np.sqrt(rn)
        return d
    for _ in range(iters):
        Ks = apply_scaling(K, d)
        if sp.issparse(Ks):
            rn = np.asarray(abs(Ks).max(axis=1).todense()).ravel()
        else:
            rn = np.abs(Ks).max(axis=1)
        rn[rn == 0] = 1.0
        d = d / np.sqrt(rn)
    return d

def apply_scaling(K, d):
    if sp.issparse(K):
        return (sp.diags(d) @ K @ sp.diags(d)).tocsc()
    return d[:, None] * K * d[None, :]

def reduced_spd(asm, W):
    """For p = 0 and the linear cone: S = P + G' W^{-1} W^{-T} G (SPD) plus maps between the full and reduced systems."""
    assert asm.l_only and asm.p == 0
    di = to_np(W["di"])
    if asm.sparse:
        Gs = (sp.diags(di) @ asm.G).tocsr()
        S = (asm.P + Gs.T @ Gs).tocsc()
    else:
        G = asm.G.toarray() if sp.issparse(asm.G) else asm.G
        Gs = di[:, None] * G
        P = asm.P.toarray() if sp.issparse(asm.P) else asm.P
        S = P + Gs.T @ Gs
    n = asm.n
    def rhs_S(r):                       # r = [bx; di*bz]
        return r[:n] + Gs.T @ r[n:]
    def expand(ux, r):                  # w = W^{-T} G ux - W^{-T} bz
        return np.concatenate([ux, Gs @ ux - r[n:]])
    return S, rhs_S, expand

# quick self-test of the assembler against cvxopt's own LDL solver on a tiny instance
_inst = INSTANCES["tiny_B"]; _asm = KKTAssembler(_inst)
_W = {"d": matrix(np.array([0.5, 2.0, 1.0, 3.0])), "di": matrix(np.array([2.0, 0.5, 1.0, 1 / 3])), "v": [], "beta": [], "r": [], "rti": []}
_K = _asm.build(_W); _cd = cvx_data(_inst)
_ref = misc.kkt_ldl(_cd["G"], _inst["dims"], _cd["A"])(_W, _cd["P"])
_x, _y, _z = to_cvx(np.arange(1, 4.0)), to_cvx(np.array([0.7])), to_cvx(np.array([1.0, -2.0, 0.5, 0.25]))
_r = _asm.rhs(_x, _y, _z, _W); _u = np.linalg.solve(_K, _r)
_ref(_x, _y, _z)
_uref = np.concatenate([to_np(_x), to_np(_y), to_np(_z)])
print("assembler vs cvxopt.misc.kkt_ldl: max |diff| =", float(np.abs(_u - _uref).max()), " (must be ~1e-15)")
assert np.allclose(_u, _uref, atol=1e-10)
