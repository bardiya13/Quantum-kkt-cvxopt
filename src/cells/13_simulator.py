# MiniSim: gate-level statevector simulator, little-endian qubit order as in Qiskit.
class ResourceLimit(Exception):
    pass

H_GATE = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
X_GATE = np.array([[0, 1], [1, 0]], dtype=complex)
SWAP_GATE = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)
def RY(theta): c, s = np.cos(theta / 2), np.sin(theta / 2); return np.array([[c, -s], [s, c]], dtype=complex)
def PHASE(phi): return np.array([[1, 0], [0, np.exp(1j * phi)]], dtype=complex)

def check_qubit_budget(n_qubits):
    q = CONFIG["quantum"]
    if n_qubits > q["max_qubits"] or 16 * 2 ** n_qubits > q["max_state_bytes"]:
        raise ResourceLimit(f"{n_qubits} qubits exceeds the configured limit (max_qubits={q['max_qubits']}, "
                            f"max_state_bytes={q['max_state_bytes']:.0f})")

class MiniSim:
    """Statevector simulator. Basis index i = sum_q bit_q(i) 2^q  (qubit q <-> bit q). Gates are applied one by one and logged."""
    def __init__(self, n_qubits, xp=None):
        check_qubit_budget(n_qubits)
        self.n = n_qubits; self.xp = np if xp is None else xp          # numpy or cupy
        self.psi = self.xp.zeros(2 ** n_qubits, dtype=complex); self.psi[0] = 1.0
        self.gates = []; self.counts = {}; self._last = np.zeros(n_qubits, dtype=int); self.depth = 0
    def _log(self, name, qubits):
        self.gates.append((name, tuple(qubits))); self.counts[name] = self.counts.get(name, 0) + 1
        d = 1 + max(self._last[q] for q in qubits)
        for q in qubits: self._last[q] = d
        self.depth = max(self.depth, d)
    def apply(self, U, targets, controls=(), ctrl_values=None, name="U"):
        n = self.n; controls = list(controls); targets = list(targets)
        psi = self.psi.reshape([2] * n)
        idx = [slice(None)] * n
        if controls:
            ctrl_values = [1] * len(controls) if ctrl_values is None else list(ctrl_values)
            for c, v in zip(controls, ctrl_values): idx[n - 1 - c] = int(v)
        sub = psi[tuple(idx)]
        cset = set(controls)
        remaining = [q for q in range(n - 1, -1, -1) if q not in cset]
        pos = {q: i for i, q in enumerate(remaining)}
        k = len(targets)
        src = [pos[t] for t in reversed(targets)]
        dst = list(range(sub.ndim - k, sub.ndim))
        xp = self.xp; U = xp.asarray(U, dtype=complex)
        moved = xp.moveaxis(sub, src, dst)
        shp = moved.shape
        out = moved.reshape(-1, 2 ** k) @ U.T
        moved[...] = out.reshape(shp)
        self._log(name, controls + targets)
    # gates
    def h(self, q): self.apply(H_GATE, [q], name="h")
    def x(self, q): self.apply(X_GATE, [q], name="x")
    def ry(self, theta, q): self.apply(RY(theta), [q], name="ry")
    def cp(self, phi, c, t): self.apply(PHASE(phi), [t], controls=[c], name="cp")
    def swap(self, a, b): self.apply(SWAP_GATE, [a, b], name="swap")
    def cu(self, U, c, targets, name="c-U"): self.apply(U, targets, controls=[c], name=name)
    def mcry(self, theta, controls, ctrl_values, target): self.apply(RY(theta), [target], controls=controls, ctrl_values=ctrl_values, name="mcry")
    def uniform_ry(self, thetas_by_value, controls, target):
        """All 2^c multi-controlled RY rotations of the exact reciprocal as one vectorised operation.
        thetas_by_value[k] is the angle applied when the control register holds the little-endian value k (0 = identity).
        Mathematically identical to the loop of mcry gates; logged as one mcry per non-zero angle for gate counting."""
        xp = self.xp; n = self.n; c = len(controls); th = np.asarray(thetas_by_value, dtype=float)
        R = np.zeros((2 ** c, 2, 2), dtype=complex); cs, sn = np.cos(th / 2), np.sin(th / 2)
        R[:, 0, 0] = cs; R[:, 0, 1] = -sn; R[:, 1, 0] = sn; R[:, 1, 1] = cs
        psi = self.psi.reshape([2] * n)
        src = [n - 1 - q for q in reversed(controls)] + [n - 1 - target]          # control MSB first ... control LSB, then target
        dst = list(range(n - c - 1, n))
        moved = xp.moveaxis(psi, src, dst); shp = moved.shape
        flat = moved.reshape(-1, 2 ** c, 2)
        out = xp.einsum("kji,bki->bkj", xp.asarray(R), flat)
        moved[...] = out.reshape(shp)
        for k in np.nonzero(th)[0]: self._log("mcry", list(controls) + [target])
    def prepare(self, vec, targets):
        """Amplitude encoding of a normalised vector on a register that is in |0...0>: one unitary whose first column is vec."""
        N = 2 ** len(targets); v = np.asarray(vec, dtype=complex)
        assert v.shape == (N,) and abs(np.linalg.norm(v) - 1) < 1e-9
        M = np.eye(N, dtype=complex); M[:, 0] = v
        Q, R = np.linalg.qr(M)
        Q[:, 0] *= (R[0, 0] / abs(R[0, 0])) if abs(R[0, 0]) > 0 else 1.0      # make the first column exactly v (up to sign fix)
        if np.linalg.norm(Q[:, 0] - v) > 1e-8: Q[:, 0] = v                   # (numerically safe: v is unit norm and orthogonal to the rest)
        self.apply(Q, targets, name="state_prep")
    def qft(self, qs):
        """Standard QFT: |x> -> 1/sqrt(N) sum_k exp(2 pi i x k / N)|k> with little-endian integers on qs."""
        n = len(qs)
        for i in range(n - 1, -1, -1):
            self.h(qs[i])
            for j in range(i - 1, -1, -1):
                self.cp(np.pi / 2 ** (i - j), qs[j], qs[i])
        for i in range(n // 2): self.swap(qs[i], qs[n - 1 - i])
    def iqft(self, qs):
        n = len(qs)
        for i in range(n // 2): self.swap(qs[i], qs[n - 1 - i])
        for i in range(n):
            for j in range(i):
                self.cp(-np.pi / 2 ** (i - j), qs[j], qs[i])
            self.h(qs[i])
    # measurement
    def probs(self, qubits):
        """Marginal Born probabilities of the listed qubits, indexed little-endian in the given order (qubits[0] = LSB)."""
        p = (self.xp.abs(self.psi) ** 2).reshape([2] * self.n)
        others = tuple(self.n - 1 - q for q in range(self.n) if q not in qubits)
        if others: p = p.sum(axis=others)
        p = to_host(p)
        kept_desc = [q for q in range(self.n - 1, -1, -1) if q in qubits]
        perm = [kept_desc.index(q) for q in reversed(qubits)]
        return np.transpose(p, perm).reshape(-1)
    def amplitudes(self, fixed):
        """Sub-vector of amplitudes with some qubits fixed: fixed = {qubit: value}; remaining qubits little-endian."""
        psi = self.psi.reshape([2] * self.n)
        idx = [slice(None)] * self.n
        for q, v in fixed.items(): idx[self.n - 1 - q] = int(v)
        return to_host(psi[tuple(idx)].reshape(-1))
    def gate_summary(self):
        return dict(self.counts)

def sample_counts(p, shots, rng):
    p = np.clip(p, 0, None); p = p / p.sum()
    return rng.multinomial(int(shots), p)

# self-tests: QFT against the DFT matrix, controlled gates, marginals, state preparation
def _dft_matrix(n):
    N = 2 ** n; w = np.exp(2j * np.pi / N)
    return np.array([[w ** (x * k) for k in range(N)] for x in range(N)]) / np.sqrt(N)
_rng = rng_for("minisim_selftest")
for _n in (2, 3, 4):
    _v = _rng.standard_normal(2 ** _n) + 1j * _rng.standard_normal(2 ** _n); _v /= np.linalg.norm(_v)
    _s = MiniSim(_n); _s.psi[:] = _v; _s.qft(list(range(_n)))
    assert np.allclose(_s.psi, _dft_matrix(_n).T @ _v) or np.allclose(_s.psi, _dft_matrix(_n) @ _v), "QFT convention mismatch"
    _s.iqft(list(range(_n)))
    assert np.allclose(_s.psi, _v), "iQFT is not the inverse of QFT"
_s = MiniSim(3); _s.x(2); _s.cu(X_GATE, 2, [0]); assert abs(_s.psi[0b101]) == 1.0, "controlled gate ordering"
_s = MiniSim(3); _s.h(0); _s.h(2); _p = _s.probs([2, 0]); assert np.allclose(_p, 0.25), "marginals"
_s = MiniSim(2); _s.prepare(np.array([0.6, 0.0, 0.8, 0.0]), [0, 1]); assert np.allclose(_s.psi, [0.6, 0, 0.8, 0]), "state prep"
# vectorised exact-reciprocal rotation == loop of multi-controlled RY gates
_th = _rng.uniform(-1, 1, 8); _th[0] = 0.0
_v = _rng.standard_normal(32) + 1j * _rng.standard_normal(32); _v /= np.linalg.norm(_v)
_a = MiniSim(5); _a.psi[:] = _v; _b = MiniSim(5); _b.psi[:] = _v
for _k in range(1, 8): _a.mcry(_th[_k], [1, 2, 3], [(_k >> j) & 1 for j in range(3)], 0)
_b.uniform_ry(_th, [1, 2, 3], 0)
assert np.allclose(_a.psi, _b.psi), "uniform_ry differs from the mcry loop"; assert _a.counts["mcry"] == _b.counts["mcry"]
if DEVICE_XP is not np:
    _g = MiniSim(5, xp=DEVICE_XP); _g.psi[:] = DEVICE_XP.asarray(_v); _g.h(2); _g.cu(X_GATE, 1, [4]); _g.uniform_ry(_th, [1, 2, 3], 0)
    _c = MiniSim(5); _c.psi[:] = _v; _c.h(2); _c.cu(X_GATE, 1, [4]); _c.uniform_ry(_th, [1, 2, 3], 0)
    assert np.allclose(to_host(_g.psi), _c.psi), "device simulator differs from CPU simulator"
    print("device (GPU) simulator matches the CPU simulator")
print("MiniSim self-tests passed (QFT = DFT matrix, controlled-U ordering, marginals, state preparation)")
