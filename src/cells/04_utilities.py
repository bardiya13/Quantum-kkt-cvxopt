# Utilities: seeded RNG, timing, CVXOPT/NumPy conversion, result tables with periodic flushing to results/partial/.
GLOBAL_SEED = CONFIG["global_seed"]

def rng_for(*keys):
    """Deterministic RNG derived from the global seed and any hashable keys."""
    h = hashlib.sha256(("|".join(map(str, keys))).encode()).digest()
    return np.random.default_rng([GLOBAL_SEED, int.from_bytes(h[:4], "little")])

class Timer:
    def __init__(self): self.t = 0.0
    def __enter__(self): self._t0 = time.perf_counter(); return self
    def __exit__(self, *a): self.t += time.perf_counter() - self._t0

def to_np(m):
    """cvxopt matrix -> numpy (column vector -> 1-D)."""
    a = np.array(m, dtype=float)
    return a.ravel() if a.ndim == 2 and a.shape[1] == 1 else a

def to_cvx(a):
    a = np.asarray(a, dtype=float)
    if a.ndim == 1: a = a.reshape(-1, 1)
    return matrix(np.ascontiguousarray(a), tc="d")

def to_cvx_sparse(a):
    c = sp.coo_matrix(a)
    return spmatrix(c.data.astype(float).tolist(), c.row.astype(int).tolist(), c.col.astype(int).tolist(), c.shape, tc="d")

def cvx_matrix_from(a, sparse):
    return to_cvx_sparse(a) if sparse else to_cvx(np.asarray(a.todense()) if sp.issparse(a) else a)

def write_back(cvx_vec, values):
    """Overwrite a cvxopt column matrix in place with numpy values."""
    v = np.ascontiguousarray(np.asarray(values, dtype=float).reshape(-1, 1))
    cvx_vec[:] = matrix(v, tc="d")

def relnorm(a, b):
    nb = np.linalg.norm(b)
    return float(np.linalg.norm(a) / nb) if nb > 0 else float(np.linalg.norm(a))

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return float("nan")

class Records:
    """All exported tables live here; every append can be flushed to results/partial/*.csv."""
    TABLES = ["benchmark_results", "iteration_logs", "linear_solve_logs", "quantum_results", "failures", "experiment_status"]
    def __init__(self):
        for t in self.TABLES: setattr(self, t, [])
        self._last_flush = time.time()
    def add(self, table, rec):
        getattr(self, table).append(rec)
    def status(self, experiment, status, reason="", **extra):
        rec = {"experiment": experiment, "status": status, "reason": reason, "timestamp": time.strftime("%H:%M:%S")}
        rec.update(extra)
        # replace an earlier status line for the same experiment
        self.experiment_status = [r for r in self.experiment_status if r["experiment"] != experiment] + [rec]
        print(f"[status] {experiment}: {status}" + (f" ({reason})" if reason else ""))
    def failure(self, run_id, stage, reason, **extra):
        rec = {"run_id": run_id, "stage": stage, "reason": str(reason)[:500]}
        rec.update(extra); self.failures.append(rec)
    def df(self, table):
        return pd.DataFrame(getattr(self, table))
    def flush(self, final=False, min_interval=20.0):
        if not final and time.time() - self._last_flush < min_interval: return
        self._last_flush = time.time()
        outdir = RESULTS_DIR if final else PARTIAL_DIR
        for t in self.TABLES:
            rows = getattr(self, t)
            if rows or final:
                pd.DataFrame(rows).to_csv(os.path.join(outdir, f"{t}.csv"), index=False)

REC = Records()

import threading
try:
    import psutil as _psutil
except Exception:
    _psutil = None
class PeakMemory:
    """Samples the resident set size of the process (and the CuPy pool when a GPU is used) from a daemon thread during a coneqp call.
    The RSS is process-wide and memory is not returned to the OS between runs, so the increase over the start value is what the
    method needed on top of what was already held."""
    def __init__(self, interval=None):
        c = CONFIG.get("memory_sampling", {}); self.interval = float(c.get("interval_s", 0.05) if interval is None else interval); self.enabled = bool(c.get("enabled", True)) and _psutil is not None
        self.proc = _psutil.Process() if self.enabled else None; self.pool = None
        if self.enabled and GPU_OK and DEVICE_XP is not np:
            try: self.pool = DEVICE_XP.get_default_memory_pool()
            except Exception: self.pool = None
        self.rss0 = self.rss_peak = self.gpu0 = self.gpu_peak = float("nan"); self.n = 0; self._stop = threading.Event(); self._thr = None
    def _sample(self):
        try:
            r = float(self.proc.memory_info().rss); self.rss_peak = max(self.rss_peak, r) if np.isfinite(self.rss_peak) else r
        except Exception: pass
        if self.pool is not None:
            try:
                g = float(self.pool.used_bytes()); self.gpu_peak = max(self.gpu_peak, g) if np.isfinite(self.gpu_peak) else g
            except Exception: self.pool = None
        self.n += 1
    def _loop(self):
        while not self._stop.wait(self.interval): self._sample()
    def __enter__(self):
        if not self.enabled: return self
        self._sample(); self.rss0 = self.rss_peak; self.gpu0 = self.gpu_peak
        self._thr = threading.Thread(target=self._loop, daemon=True); self._thr.start(); return self
    def __exit__(self, *a):
        if not self.enabled: return
        self._stop.set(); self._thr.join(timeout=2.0); self._sample()
    def fields(self):
        mb = 1.0 / 2 ** 20
        return {"mem_rss_start_mb": self.rss0 * mb, "mem_rss_peak_mb": self.rss_peak * mb, "mem_rss_increase_mb": max(0.0, (self.rss_peak - self.rss0) * mb) if np.isfinite(self.rss_peak) else float("nan"),
                "mem_gpu_pool_start_mb": self.gpu0 * mb, "mem_gpu_pool_peak_mb": self.gpu_peak * mb, "mem_gpu_pool_increase_mb": max(0.0, (self.gpu_peak - self.gpu0) * mb) if np.isfinite(self.gpu_peak) else float("nan"),
                "mem_samples": self.n}
print("utilities ready; partial results dir:", PARTIAL_DIR, "| memory sampling:", "psutil" if _psutil is not None else "unavailable (psutil missing)")
