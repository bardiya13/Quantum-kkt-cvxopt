## 2. Environment

* Required: `numpy`, `scipy`, `pandas`, `matplotlib` and `cvxopt` (1.3). Kaggle images ship all of them; a missing package is installed with `pip` before `numpy` is imported, so no kernel restart is needed. Installed packages are never upgraded.
* Optional: `qiskit` and `qiskit-aer`, used only to cross-check the statevector simulator and to obtain transpiled gate counts. If Qiskit cannot be installed, the cross-check is reported as `skipped`.
* GPU: with `device = "gpu"` the MINRES kernels and the simulator run on CuPy. The cell checks that a CUDA device is usable; if not, every GPU method is reported as `skipped` and the CPU methods run unchanged. Setting `QKKT_FAKE_GPU=1` replaces CuPy by NumPy to test the GPU code paths on a machine without a GPU (no GPU timing is meaningful in that mode).

Package versions, hardware, thread settings and the seed are written to `results/environment.json`.
