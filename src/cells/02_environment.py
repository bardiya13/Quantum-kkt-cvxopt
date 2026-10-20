# Environment: install missing packages before numpy is imported, record versions and hardware.
import importlib, subprocess, sys, os, json, time, platform

ENV = {"python": sys.version, "platform": platform.platform(), "notebook_started": time.strftime("%Y-%m-%d %H:%M:%S"), "code_version": CODE_VERSION,
       "install_log": [], "kaggle": os.path.isdir("/kaggle/working"), "run_profile": CONFIG["run_profile"]}

def _have(mod):
    try:
        importlib.import_module(mod); return True
    except Exception:
        return False

def _pip_install(spec):
    """Install missing packages only (never upgrades what is present). Returns (ok, message)."""
    cmd = [sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check"] + spec.split()
    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        ok = (r.returncode == 0)
        msg = (r.stdout + r.stderr)[-2000:]
    except Exception as e:
        ok, msg = False, repr(e)
    ENV["install_log"].append({"spec": spec, "ok": ok, "seconds": round(time.time() - t0, 1), "tail": msg[-500:]})
    return ok, msg

# --- required packages ---
REQUIRED = {"numpy": "numpy", "scipy": "scipy", "pandas": "pandas", "matplotlib": "matplotlib", "cvxopt": "cvxopt>=1.3,<1.4"}
for mod, spec in REQUIRED.items():
    if not _have(mod):
        print(f"[env] {mod} missing -> pip install {spec}  (requires Internet)")
        ok, msg = _pip_install(spec)
        print("      ->", "installed" if ok else "FAILED: " + msg[-300:])
missing = [m for m in REQUIRED if not _have(m)]
if missing:
    raise RuntimeError(f"Required packages could not be imported/installed: {missing}. "
                       "On Kaggle: Settings -> Internet -> ON, then re-run this cell.")

# --- optional Qiskit (cross-check only) ---
QISKIT_STATUS = {"available": False, "reason": "", "version": None, "aer": False}
if CONFIG["quantum"]["qiskit_crosscheck"]:
    if not _have("qiskit") and CONFIG["quantum"]["install_qiskit_if_missing"]:
        print("[env] qiskit missing -> optional pip install (Internet needed; failure is non-fatal)")
        ok, msg = _pip_install(CONFIG["quantum"]["qiskit_pins"])
        print("      ->", "installed" if ok else "not installed (" + msg[-200:].strip() + ")")
    try:
        import qiskit
        QISKIT_STATUS.update({"available": True, "version": qiskit.__version__})
        try:
            import qiskit_aer; QISKIT_STATUS["aer"] = True; QISKIT_STATUS["aer_version"] = qiskit_aer.__version__
        except Exception as e:
            QISKIT_STATUS["aer"] = False; QISKIT_STATUS["aer_reason"] = repr(e)[:200]
    except Exception as e:
        QISKIT_STATUS.update({"available": False, "reason": f"import failed: {e!r}"[:300]})
else:
    QISKIT_STATUS["reason"] = "disabled in CONFIG"

# --- imports used everywhere ---
import numpy as np, scipy, scipy.linalg as sla, scipy.sparse as sp, scipy.sparse.linalg as spla
import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cvxopt
from cvxopt import matrix, spmatrix, solvers, misc
import warnings, io, contextlib, re, math, hashlib, shutil, zipfile, traceback, inspect
warnings.filterwarnings("ignore", category=DeprecationWarning)

# --- compute device: CuPy if requested and usable, otherwise NumPy ---
GPU_STATUS = {"requested": CONFIG["device"], "available": False, "reason": "", "name": None}
DEVICE_XP = np; DEVICE_SPARSE = None
if CONFIG["device"] == "gpu":
    if os.environ.get("QKKT_FAKE_GPU") == "1":            # NumPy stand-in to test the GPU code paths without a GPU
        class _FakeSparse:
            csr_matrix = staticmethod(lambda a, shape=None: sp.csr_matrix(a, shape=shape))
        GPU_STATUS.update({"available": True, "name": "FAKE (numpy stand-in, logic test only)"}); DEVICE_SPARSE = _FakeSparse
    else:
        try:
            try:
                import cupy
            except ImportError:                                # Kaggle GPU images ship CuPy; otherwise try one install
                if shutil.which("nvidia-smi"):
                    print("[env] cupy missing -> pip install cupy-cuda12x"); print("   ", _pip_install("cupy-cuda12x"))
            import cupy, cupyx.scipy.sparse as _cusp
            if cupy.cuda.runtime.getDeviceCount() < 1: raise RuntimeError("no CUDA device")
            _t = cupy.arange(8, dtype=float); assert float((_t @ _t)) == 140.0
            DEVICE_XP = cupy; DEVICE_SPARSE = _cusp
            GPU_STATUS.update({"available": True, "name": cupy.cuda.runtime.getDeviceProperties(0)["name"].decode(), "cupy": cupy.__version__,
                               "memory_gb": round(cupy.cuda.runtime.memGetInfo()[1] / 2**30, 2)})
        except Exception as e:
            GPU_STATUS["reason"] = f"{type(e).__name__}: {e}"[:300]
    if not GPU_STATUS["available"]:
        print("[env] GPU requested but not usable ->", GPU_STATUS["reason"], "-> every GPU method is reported as skipped; CPU methods run normally")
GPU_OK = GPU_STATUS["available"]
CONFIG["quantum"]["device"] = "gpu" if (GPU_OK and DEVICE_XP is not np) else "cpu"
def to_device(a):
    """numpy / scipy.sparse -> device array (cupy / cupyx sparse); identity on CPU."""
    if not GPU_OK: return a
    if sp.issparse(a): return DEVICE_SPARSE.csr_matrix(a.tocsr())
    return DEVICE_XP.asarray(a)
def to_host(a):
    """cupy -> numpy (no-op for numpy)."""
    return a.get() if hasattr(a, "get") else np.asarray(a)
def dev_sync():
    if GPU_OK and DEVICE_XP is not np: DEVICE_XP.cuda.Stream.null.synchronize()

# --- hardware ---
def _hardware():
    hw = {"cpu_count": os.cpu_count(), "machine": platform.machine(), "processor": platform.processor()}
    try:
        import psutil
        hw["memory_total_gb"] = round(psutil.virtual_memory().total / 2**30, 2)
        hw["memory_available_gb"] = round(psutil.virtual_memory().available / 2**30, 2)
    except Exception:
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        hw["memory_total_gb"] = round(int(line.split()[1]) / 2**20, 2)
        except Exception:
            pass
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if "model name" in line:
                    hw["cpu_model"] = line.split(":", 1)[1].strip(); break
    except Exception:
        pass
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        hw["gpu_present"] = r.stdout.strip() if r.returncode == 0 else "none detected"
    except Exception:
        hw["gpu_present"] = "none detected"
    return hw

ENV["hardware"] = _hardware()
ENV["packages"] = {"numpy": np.__version__, "scipy": scipy.__version__, "pandas": pd.__version__,
                   "matplotlib": matplotlib.__version__, "cvxopt": cvxopt.__version__,
                   "qiskit": QISKIT_STATUS.get("version"), "qiskit_aer": QISKIT_STATUS.get("aer_version")}
ENV["qiskit_status"] = QISKIT_STATUS; ENV["gpu_status"] = GPU_STATUS
ENV["seed"] = CONFIG["global_seed"]
ENV["blas"] = {k: v for k, v in np.__config__.CONFIG.get("Build Dependencies", {}).items()} if hasattr(np.__config__, "CONFIG") else {}
# thread count used by BLAS (affects timing reproducibility)
ENV["thread_env"] = {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
with open(os.path.join(RESULTS_DIR, "environment.json"), "w") as f:
    json.dump(ENV, f, indent=2, default=str)

print(json.dumps({k: ENV[k] for k in ("python", "platform", "hardware", "packages", "qiskit_status", "gpu_status", "seed")}, indent=1, default=str))
print("Internet needed for installation:", "NO (everything already available)" if not ENV["install_log"] else "YES (see install_log above)")
