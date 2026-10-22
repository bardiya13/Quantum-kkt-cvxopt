# Certified classical and quantum KKT solves inside CVXOPT

Code, derived data and manuscript sources for the paper

> **Certified Classical and Quantum Linear Solves inside a Production Interior-Point Method: Equilibration, Low-Precision Refinement, and Conditions for a Quantum Advantage.** Authors, 2026.

The KKT factorisation of `cvxopt.solvers.coneqp` is replaced, through its `kktsolver` callback, by classical Krylov solvers (equilibrated MINRES on CPU and GPU, CG on reduced and null-space systems) and by simulated quantum linear solvers (phase-estimation HHL and a polynomial solver based on a qubitised walk). Every replacement runs under the same acceptance test, the same classical-fallback protocol and the same accounting, so that classical and quantum methods are compared on equal terms. All quantum results are classical simulations of ideal circuits with oracle block encoding; no end-to-end quantum speed-up is claimed.

## Contents

```
notebooks/   quantum_kkt_cvxopt_cpu.ipynb, quantum_kkt_cvxopt_gpu.ipynb   (built from src/cells)
src/
  cells/             notebook source, one file per cell (NN_name.md, NN_name.py)
  build_notebook.py  assembles the two notebooks, or a single script for local runs
scripts/
  bundles.py                  reading the result bundles written by the notebook
  extract_tables_main.py      bundles -> results/tables (session M, replication sessions B-E)
  extract_tables_sessionA.py  bundles -> results/tables (session A)
  mm_kappa_diag.py            condition numbers on eight Maros-Meszaros QPs (Table 15, Fig. 7(b))
  make_tex_tables.py          results/tables -> paper/tex (LaTeX table bodies)
  make_figures.py             results/tables -> paper/fig
results/tables/  the derived tables behind every table and figure of the paper (see results/README.md)
paper/           paper.tex, refs.bib, generated tables and figures, paper.pdf
```

## Running the notebook

The notebook is written for Kaggle. Upload one of the two builds, switch Internet access on (the Maros-Mészáros and SDPLIB files are downloaded at run time, and Qiskit is installed for the simulator cross-check) and run all cells. The builds differ only in the default device.

| build | Kaggle accelerator | run time, profile `medium` |
|---|---|---|
| `quantum_kkt_cvxopt_cpu.ipynb` | none | 4.7 h |
| `quantum_kkt_cvxopt_gpu.ipynb` | GPU T4 | 5.8 h |

The problem sizes are set by `RUN_PROFILE` in the configuration cell, or by the environment variable `QKKT_PROFILE`: `smoke` (every code path on tiny instances), `small` (default), `medium` (the profile of the paper) and `large`. Results are written to `/kaggle/working/results/` and packed into `results_bundle_<cpu|gpu>_<profile>.zip`. Without Internet access, the experiments that need downloaded data are marked `skipped` in `experiment_status.csv` and the rest of the notebook runs.

To run outside Kaggle, build a script from the cells and run it:

```
pip install -r requirements.txt
python src/build_notebook.py --script run.py
QKKT_PROFILE=smoke python run.py
```

Output then goes to `./kaggle_working_local/results/`. The `smoke` profile takes about 8 minutes on a 4-core CPU. For the GPU code path install CuPy (`cupy-cuda12x`) and set `QKKT_DEVICE=gpu` or build with `--device gpu`.

## Reproducing the tables and figures of the paper

The paper rests on six notebook runs (sessions). Session M is the main session; A supplies the acceptance-test ablation, the spectrum study, the experiments on captured KKT systems and SDPLIB; B-E are replications (paper, Table 3 and Appendix C).

| session | build | profile | date |
|---|---|---|---|
| M | GPU | medium | 26 Sep 2026 |
| A | CPU | medium | 21 Sep 2026 |
| B | GPU | medium | 21 Sep 2026 |
| C | GPU | small | 24 Sep 2026 |
| D | CPU | small | 26 Sep 2026 |
| E | CPU | medium | 27 Sep 2026 |

Sessions A-C ran with an earlier fallback factorisation (LU of the augmented matrix) and GPU threshold; the differences are listed in Table 3 of the paper.

The result bundles (about 150 MB) are not stored in this repository. The tables derived from them are in `results/tables/`, so the last two steps below work without the bundles.

```
# 1. condition-number diagnostic on eight Maros-Meszaros QPs (about 6 minutes on a 4-core CPU)
python scripts/mm_kappa_diag.py

# 2. bundles -> results/tables
python scripts/extract_tables_main.py --main M.zip --repB B.zip --repC C.zip --repD D.zip --repE E.zip \
       --mmdiag results/tables/mm_kappa_diag.csv
python scripts/extract_tables_sessionA.py --sessionA A.zip --sessionB B.zip

# 3. LaTeX table bodies and figures
python scripts/make_tex_tables.py
python scripts/make_figures.py

# 4. the paper
cd paper && latexmk -pdf paper.tex      # or: tectonic paper.tex
```

Steps 1-3 regenerate the committed tables, LaTeX bodies and figures exactly. The paper builds with pdfLaTeX and BibTeX (TeX Live 2026, as on Overleaf) and with [Tectonic](https://tectonic-typesetting.github.io).

The bundles were written before some methods were renamed. `scripts/bundles.py` maps the old names on reading (for example `minresx5` to `minres_fused`), so all derived tables use the names of the current code.

## Changing the notebook

Edit the cells in `src/cells/` and rebuild with `python src/build_notebook.py`; do not edit the `.ipynb` files directly. Cells are ordered by their numeric prefix, with the markdown cell before the code cell of the same prefix. Kaggle failed to save a 660 KB version of this notebook, while versions up to 520 KB were saved without problems; the current builds are 395 KiB.

## Citation

```bibtex
@article{quantumkkt2026,
  title  = {Certified Classical and Quantum Linear Solves inside a Production Interior-Point Method:
            Equilibration, Low-Precision Refinement, and Conditions for a Quantum Advantage},
  author = {Authors},
  year   = {2026}
}
```

## License

The code is released under the MIT License (see `LICENSE`). The manuscript in `paper/` is copyright of the authors and is not covered by this license..
