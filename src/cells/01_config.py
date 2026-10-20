# Configuration: the only cell that normally needs editing.
import os, sys, json, time, platform, hashlib

CODE_VERSION = "1.0.0"
DEFAULT_DEVICE = "cpu"          # "cpu" | "gpu"; without a usable GPU every GPU method is reported as skipped
RUN_PROFILE = os.environ.get("QKKT_PROFILE", "small")     # "smoke" | "small" | "medium" | "large"

CONFIG = {
    "code_version": CODE_VERSION, "run_profile": RUN_PROFILE,
    "device": os.environ.get("QKKT_DEVICE", DEFAULT_DEVICE),
    "global_seed": 20260911,
    "output_root": "/kaggle/working" if os.path.isdir("/kaggle/working") else os.path.abspath("./kaggle_working_local"),

    # CVXOPT algorithm parameters (CVXOPT defaults, identical for every method)
    "cvxopt": {"maxiters": 100, "abstol": 1e-7, "reltol": 1e-6, "feastol": 1e-7, "refinement_qp": 0, "refinement_sdp": 1},

    # acceptance of the final iterate, recomputed independently of the solver's own report
    "accept": {"pres": 1e-6, "dres": 1e-6, "relgap": 1e-5, "cone_viol": 1e-8, "obj_rel_err": 1e-5},

    # synthetic benchmark families
    "families": {
        "dense_kappa": {"structure": "dense", "n_list": [10, 20, 40], "m_ratio": 2.0, "p_ratio": 0.2,
                        "kappa_list": [1e0, 1e3, 1e6], "seeds": [0, 1, 2], "active_frac": 0.3, "density": 1.0},
        "dense_noeq": {"structure": "dense", "n_list": [10, 40], "m_ratio": 2.0, "p_ratio": 0.0,
                       "kappa_list": [1e0, 1e6], "seeds": [0, 1], "active_frac": 0.3, "density": 1.0},
        # random 5 % density: G'G fills in completely, so this family is kept small
        "sparse_scale": {"structure": "random_sparse", "n_list": [25, 50, 100, 200], "m_ratio": 2.0, "p_ratio": 0.1,
                         "kappa_list": [1e3], "seeds": [0], "active_frac": 0.3, "density": 0.05},
        # banded P, 4 nonzeros per row of G and A in random columns: P + G'W^-2 G is a random sparse graph whose
        # Cholesky factor fills in almost completely (high-fill family); `seeds_large` applies from `seeds_large_from_n` on
        "sparse_banded": {"structure": "banded", "n_list": [100, 200, 400, 800, 1600, 3200], "m_ratio": 2.0, "p_ratio": 0.1,
                          "kappa_list": [1e3], "seeds": [0, 1, 2], "seeds_large": [0], "seeds_large_from_n": 1600, "active_frac": 0.3, "density": 0.0, "nnz_per_row": 4, "bandwidth": 3},
        # same sizes, but the nonzeros of each row of G and A lie in a window of 13 neighbouring columns: P + G'W^-2 G is
        # banded and sparse Cholesky is O(n) (low-fill control family, where a direct solver should win)
        "sparse_lowfill": {"structure": "banded_local", "n_list": [100, 200, 400, 800, 1600, 3200], "m_ratio": 2.0, "p_ratio": 0.1,
                           "kappa_list": [1e3], "seeds": [0, 1], "seeds_large": [0], "seeds_large_from_n": 1600, "active_frac": 0.3, "density": 0.0, "nnz_per_row": 4, "bandwidth": 3},
    },
    "repeats": 1,
    "inexact": {
        # acceptance tests for one KKT solve (section 3.4): "relres" = ||Ku - r|| / ||r||; "backward" = normwise backward error
        # ||Ku - r|| / (||K||_2 ||u|| + ||r||); "block" = per-block forcing ||e_b|| <= eta * max(||r_b||, floor_b); "combined" = block AND backward
        "be_tols": [1e-6, 1e-8, 1e-10, 1e-12],
        "be_floor": 1e-14,                  # backward error of a direct solve; used by "relres_floor"
        "full_ablation": False,             # True: full criterion x scaling x tolerance grid in E2; False: essential method set
        "warm_start": True,                 # combined-direction solve starts from the affine direction (minres_base_*_ws_*)
        "switch_policy": True,              # inexact solver until its first rejected direction, exact solver afterwards (*_switch_*)
        "block_etas": [0.5, 0.1, 0.01],
        "relres_tol": 1e-6,                 # tolerance of the relative-residual test
        "accept_factor": 10.0,              # "relres" / "backward": accept if value <= accept_factor * tolerance
        "block_abs_floor": 1e-9,            # absolute floor per block, times max(1, ||q||), max(1, ||b||), max(1, ||h||)
        "max_refine": 2, "fallback": True,
        "check_adaptive_div": 16,           # fused kernel: next explicit check after max(check_every, it // 16) iterations
        "gpu_fuse": True,                   # fuse the vector recurrences with cupy.fuse (self-tested; plain code otherwise)
        "gpu_auto_min_dim": 20000,          # `*_auto` methods use the GPU from this KKT dimension on
        "oss_max_n": 3200,                  # null-space methods form a dense n x (n - p) basis of null(A): only up to this n
        "fallback_reduced": True,           # fallback factors the reduced quasi-definite system (CHOLMOD); False: LU of the full KKT matrix
        "check_every": 8,                   # Krylov iterations between explicit acceptance checks
        "norm_est_iters": 20,               # power iterations for ||K||_2 (once per factorisation)
        "adaptive_nt": {"eta": 1e-2, "tol_min": 1e-10, "tol_max": 1e-2},
        "noise_iso": [1e-8, 1e-7, 1e-6],    # isotropic error model: delta * ||u|| spread over all components
        "noise_block": [1e-1, 1e-2, 1e-3],  # block-relative error model: delta * ||u_b|| per block
        "minres_maxiter_cap": 20000, "minres_maxiter_factor": 2, "ruiz_iters": 6,
    },
    "cond_limit": 600,
    "memory_sampling": {"enabled": True, "interval_s": 0.05},     # peak RSS (and CuPy pool) sampled by a thread during every coneqp call
    # Maros-Meszaros QPs: .mat files of github.com/qpsolvers/maros_meszaros_qpbenchmark, min 1/2 x'Px + q'x + r  s.t.  l <= Ax <= u
    "maros_meszaros": {
        "enabled": True, "search_dirs": ["/kaggle/input", "./maros_meszaros", "./data/maros_meszaros"],
        "download_if_missing": True, "download_url": "https://raw.githubusercontent.com/qpsolvers/maros_meszaros_qpbenchmark/main/data/{name}.mat",
        # KKT dimension after conversion: 91 ... 1 260 in the default profile.  Left out because the unmodified coneqp run
        # ends with status 'unknown': QSHARE2B, QRECIPE, QPCBOEI2, QFORPLAN, QCAPRI, QSCFXM1-3, QGFRDXPN.
        "instances": ["QAFIRO", "DUAL1", "CVXQP1_S", "QPCBLEND", "QADLITTL", "DUALC1", "QSCAGR7", "PRIMALC1", "QISRAEL", "QSC205", "PRIMAL1", "QSCTAP1", "KSIP"],
        "methods": ["cvxopt_default", "custom_exact_reduced", "gmres_stalelu_comb", "minres_fused_comb_cpu", "minres_fused_comb_switch_cpu", "minres_fused_comb_gpu_mm"],
        "budget_s": 120, "max_kkt_dim": 12000,
    },
    # E4b: IPM tolerance level tau x direction error; acceptance thresholds scale with tau (tau = 1e-7 reproduces the defaults)
    "tolerance_sweep": {"levels": [1e-4, 1e-6, 1e-8], "noise_iso": [1e-8, 1e-7, 1e-6, 1e-5], "noise_block": [1e-1, 1e-2, 1e-3], "extra_methods": ["cvxopt_default", "minres_fused_comb_cpu"],
                        "families": ["dense_kappa", "dense_noeq"], "n_max": 40, "max_instances": 30},
    "capture": {"max_dim": 160, "instances": ["tiny_A", "tiny_B", "tiny_C", "dense_kappa_n20_k6_s0", "dense_kappa_n40_k6_s0"]},   # KKT systems saved from the reference run

    # quantum linear solvers (statevector simulation)
    "quantum": {
        "enabled": True, "max_qubits": 17, "max_state_bytes": 64 * 2**20,
        "n_clock_list": [4, 6, 8, 10], "shots_list": [1000, 10000, 100000],
        "eff_shots_list": [100, 1000, 10000],   # accepted samples per setting with amplitude-amplification-emulated readout
        "sampling_repeats": 3, "scalings": ["none", "ruiz"], "reciprocal_modes": ["resolution", "kappa_known"],
        "ir_rounds": 4,                         # refinement rounds with HHL on the true residual (Q1)
        # Q4: regularisation delta added to the Ruiz-scaled system of ill-conditioned captured systems; corrector = LU of the
        # regularised system (classical diagnostic) or HHL with ideal readout (padded dimension <= quantum_max_dim)
        "reg_experiment": {"deltas": [0.0, 1e-4, 1e-2, 1e-1], "ir_rounds": 6, "min_kappa_scaled": 50.0, "classical_max_dim": 512, "quantum_max_dim": 32, "quantum_n_clock": [6, 8]},
        "spectral_bound": "power", "power_iters": 40, "power_safety": 1.15, "reference": "uniform", "captured_iters": "auto",
        "integrated": {
            "instances": ["tiny_A", "tiny_B"],
            "runs": [   # combined acceptance test unless 'criterion' overrides it
                {"name": "hhl_noscale_res_nc8_fb",     "recon": "statevector", "n_clock": 8, "scaling": "none", "reciprocal": "resolution", "reg": 0.0, "fallback": True},
                {"name": "hhl_scaled_kk_nc8_fb",       "recon": "statevector", "n_clock": 8, "scaling": "ruiz", "reciprocal": "kappa_known", "reg": 0.0, "fallback": True},
                {"name": "hhl_scaled_kk_nc8_nofb",     "recon": "statevector", "n_clock": 8, "scaling": "ruiz", "reciprocal": "kappa_known", "reg": 0.0, "fallback": False},
                {"name": "hhl_scaled_kk_nc8_aa1000_fb", "recon": "aa_shots", "eff_shots": 1000, "n_clock": 8, "scaling": "ruiz", "reciprocal": "kappa_known", "reg": 0.0, "fallback": True},
                {"name": "hhl_scaled_kk_nc12_nofb",    "recon": "statevector", "n_clock": 12, "scaling": "ruiz", "reciprocal": "kappa_known", "reg": 0.0, "fallback": False},
                {"name": "hhl_scaled_kk_nc12_switch",  "recon": "statevector", "n_clock": 12, "scaling": "ruiz", "reciprocal": "kappa_known", "reg": 0.0, "fallback": True, "switch_policy": True},
                {"name": "hhl_scaled_kk_nc8_blockonly_nofb", "recon": "statevector", "n_clock": 8, "scaling": "ruiz", "reciprocal": "kappa_known", "reg": 0.0, "fallback": False, "criterion": "block", "instances_only": ["tiny_B"]},   # block test alone (ablation)
            ],
            "accept_criterion": "combined", "accept_eta": 0.1, "accept_be_tol": 1e-8, "max_refine": 4, "maxiters": 40,
        },
        # polynomial solver (qubitised walk + LCU), section 15
        "cks": {"poly": "optimal",                       # "optimal" (Sunderhauf et al.) | "cks" (Childs-Kothari-Somma expansion)
                "eps_list": [3e-1, 1e-2, 1e-6, 1e-10], "max_degree": 400000,
                # Q5b grid: circuit precision x accepted samples per refinement round (both measurement settings together)
                "ir_eps_list": [3e-1, 1e-2, 1e-6], "ir_samples_per_round": [200, 500, 1000, 4000, 20000], "ir_rounds": 120, "ir_max_systems": 9,
                # instances whose first KKT matrix already needs more walk steps than max_degree (e.g. dense_noeq_n10_k6) are not used
                "integrated_extra_instances": ["dense_kappa_n10_k0_s0", "dense_kappa_n10_k3_s0", "dense_noeq_n10_k0_s0", "dense_kappa_n10_k0_s1", "dense_kappa_n10_k3_s1"],
                "integrated_be_tol": 1e-9,                 # tau of the combined test in the quantum runs (classical comb methods: 1e-10)
                "integrated_max_refine": 60, "hhl_fair_extra_instances": ["dense_kappa_n10_k0_s0"], "hhl_fair_extra_runs": ["hhl_nc10_ideal_ir60_nofb"], "hhl_fair_maxiters": 20,
                "integrated_runs": [{"name": "lcu_opt_eps1e-10_ideal_nofb", "eps": 1e-10, "recon": "statevector"},                                 # high-precision circuit, ideal readout
                                    {"name": "lcu_opt_eps0.3_ideal_ir_nofb", "eps": 3e-1, "recon": "statevector"},                                  # low-precision circuit + refinement
                                    {"name": "lcu_opt_eps0.3_aa500_ir_nofb", "eps": 3e-1, "recon": "aa_shots", "eff_shots": 250},                   # eps = 0.3, 500 samples per round
                                    {"name": "lcu_cks_eps1e-6_aa2e4_ir_nofb", "eps": 1e-6, "recon": "aa_shots", "eff_shots": 10000, "poly": "cks"}],  # CKS expansion, high precision
                # HHL with the same refinement budget as the polynomial solver
                "hhl_fair_runs": [{"name": "hhl_nc8_ideal_ir60_nofb", "n_clock": 8, "recon": "statevector"}, {"name": "hhl_nc10_ideal_ir60_nofb", "n_clock": 10, "recon": "statevector"},
                                  {"name": "hhl_nc10_aa500_ir60_nofb", "n_clock": 10, "recon": "aa_shots", "eff_shots": 250}],
                # Q5e: finite-shot runs over sampling seeds x accepted samples per refinement round (both settings together)
                "shot_sweep": {"eps": 3e-1, "samples_per_round": [100, 250, 1000, 4000], "seeds": [0, 1, 2, 3, 4], "max_instances": 8},
                # Q5c/Q5d: SPD polynomial on the null-space target Z'SZ instead of the indefinite augmented system
                "oss_runs": [{"name": "lcu_oss_eps0.3_ideal_ir_nofb", "eps": 3e-1, "recon": "statevector"},
                             {"name": "lcu_oss_eps0.3_aa500_ir_nofb", "eps": 3e-1, "recon": "aa_shots", "eff_shots": 250}]},
        "qiskit_crosscheck": True, "qiskit_max_clock": 6, "qiskit_max_qubits": 12,
        "install_qiskit_if_missing": True, "qiskit_pins": "qiskit>=1.2,<3 qiskit-aer>=0.14",
    },

    # SDPLIB (optional, conelp)
    "sdplib": {
        "enabled": True,
        "search_dirs": ["/kaggle/input", "./sdplib", "./data/sdplib"],
        "download_if_missing": True, "download_url": "https://raw.githubusercontent.com/vsdp/SDPLIB/master/data/{name}.dat-s",
        "instances": ["truss1", "truss4", "hinf2", "control1", "truss3", "truss2", "qap5"],   # unmodified conelp fails on hinf1/3/10-15
        "max_kkt_dim_custom": 900, "max_kkt_dim_inexact": 200, "quantum_instances": ["truss1"],
    },
    # Per-method budgets.  A method that exceeds max_run_seconds on an instance is skipped for larger instances of the same
    # family only.  With predictive_budget, it is skipped at the next size when its extrapolated time (power law through its
    # last two sizes, exponent clipped to [1, 3.5]; 2.0 with one point) exceeds budget_margin * max_run_seconds.  A running
    # coneqp call cannot be interrupted safely, so budgets act between runs; partial results are flushed after every run.
    # The LU reference is not run above exact_max_dim (fill-in makes it O(N^2.5)).
    "limits": {"max_dense_kkt_dim": 3000, "max_run_seconds": 180, "exact_max_dim": 5000,
               "predictive_budget": True, "budget_margin": 1.5},
}

if RUN_PROFILE == "smoke":
    CONFIG["families"] = {
        "dense_kappa": {"structure": "dense", "n_list": [10], "m_ratio": 2.0, "p_ratio": 0.2, "kappa_list": [1e0, 1e6], "seeds": [0], "active_frac": 0.3, "density": 1.0},
        "dense_noeq": {"structure": "dense", "n_list": [10], "m_ratio": 2.0, "p_ratio": 0.0, "kappa_list": [1e6], "seeds": [0], "active_frac": 0.3, "density": 1.0},
        "sparse_scale": {"structure": "random_sparse", "n_list": [25], "m_ratio": 2.0, "p_ratio": 0.1, "kappa_list": [1e3], "seeds": [0], "active_frac": 0.3, "density": 0.05},
        "sparse_banded": {"structure": "banded", "n_list": [100, 200], "m_ratio": 2.0, "p_ratio": 0.1, "kappa_list": [1e3], "seeds": [0], "active_frac": 0.3, "density": 0.0, "nnz_per_row": 4, "bandwidth": 3},
        "sparse_lowfill": {"structure": "banded_local", "n_list": [100, 200], "m_ratio": 2.0, "p_ratio": 0.1, "kappa_list": [1e3], "seeds": [0], "active_frac": 0.3, "density": 0.0, "nnz_per_row": 4, "bandwidth": 3},
    }
    CONFIG["inexact"].update({"be_tols": [1e-8, 1e-12], "block_etas": [0.1], "noise_iso": [1e-6], "noise_block": [1e-2]})
    CONFIG["quantum"].update({"n_clock_list": [4, 6], "shots_list": [1000, 10000], "eff_shots_list": [100, 1000], "sampling_repeats": 2, "ir_rounds": 2, "qiskit_max_clock": 4})
    CONFIG["quantum"]["reg_experiment"].update({"deltas": [0.0, 1e-2], "ir_rounds": 3, "quantum_n_clock": [6]})
    CONFIG["quantum"]["integrated"]["instances"] = ["tiny_A", "tiny_B"]
    CONFIG["quantum"]["integrated"]["runs"] = [r for r in CONFIG["quantum"]["integrated"]["runs"] if r["name"] in ("hhl_scaled_kk_nc8_fb", "hhl_scaled_kk_nc8_nofb", "hhl_scaled_kk_nc12_switch")]
    for r in CONFIG["quantum"]["integrated"]["runs"]: r["n_clock"] = 6 if r["n_clock"] == 8 else 8
    CONFIG["quantum"]["integrated"]["maxiters"] = 20
    CONFIG["quantum"]["cks"].update({"eps_list": [3e-1, 1e-6], "ir_eps_list": [3e-1, 1e-6], "ir_samples_per_round": [500, 4000], "ir_rounds": 60, "ir_max_systems": 3, "max_degree": 30000, "integrated_extra_instances": [], "hhl_fair_extra_instances": [], "integrated_max_refine": 60})
    CONFIG["quantum"]["cks"]["shot_sweep"].update({"samples_per_round": [250, 1000], "seeds": [0, 1], "max_instances": 2})
    CONFIG["families"]["sparse_banded"]["seeds"] = [0, 1]; CONFIG["families"]["sparse_lowfill"]["seeds"] = [0]
    CONFIG["maros_meszaros"]["instances"] = ["QAFIRO", "CVXQP1_S"]; CONFIG["maros_meszaros"]["methods"] = ["cvxopt_default", "custom_exact_reduced", "minres_fused_comb_cpu", "minres_fused_comb_switch_cpu", "minres_fused_comb_gpu"]
    CONFIG["tolerance_sweep"].update({"levels": [1e-4, 1e-8], "noise_iso": [1e-7, 1e-5], "noise_block": [1e-2], "max_instances": 4})
    CONFIG["quantum"]["cks"]["hhl_fair_runs"] = [{"name": "hhl_nc8_ideal_ir60_nofb", "n_clock": 8, "recon": "statevector"}]
    CONFIG["capture"]["instances"] = ["tiny_A", "tiny_B", "tiny_C", "dense_kappa_n10_k6_s0"]
    CONFIG["sdplib"]["instances"] = ["truss1", "hinf2"]
elif RUN_PROFILE == "medium":
    CONFIG["families"]["dense_kappa"]["n_list"] = [10, 20, 40, 80]; CONFIG["families"]["dense_kappa"]["seeds"] = [0, 1, 2, 3, 4]
    CONFIG["families"]["sparse_banded"].update({"n_list": [100, 200, 400, 800, 1600, 3200, 6400], "seeds": [0, 1, 2, 3, 4], "seeds_large": [0, 1], "seeds_large_from_n": 6400})
    CONFIG["families"]["sparse_lowfill"].update({"n_list": [100, 200, 400, 800, 1600, 3200, 6400], "seeds": [0, 1, 2], "seeds_large": [0], "seeds_large_from_n": 6400})
    CONFIG["maros_meszaros"]["instances"] += ["DUAL2", "CVXQP2_S", "CVXQP3_S", "QSHARE1B", "VALUES", "QBRANDY", "QBEACONF", "PRIMAL2", "QE226", "QBORE3D", "QGROW7", "QSCORPIO", "QBANDM", "QPCSTAIR",
                                              "QPCBOEI1", "QSCAGR25", "PRIMAL4", "QSCSD1", "QETAMACR", "GOULDQP3", "QSTANDAT", "QSCSD6", "LASER", "QSEBA", "CVXQP2_M", "QSHIP04S", "CVXQP1_M", "CVXQP3_M",
                                              "MOSARQP2", "QSHELL", "Q25FV47", "QSCTAP2", "MOSARQP1", "QSCTAP3", "CONT-050"]      # 48 instances, KKT dimension up to 10 192
    CONFIG["maros_meszaros"]["budget_s"] = 420      # MINRES needs up to ~350 s on QSHELL and Q25FV47; a lower budget would skip larger instances
    CONFIG["repeats"] = 2; CONFIG["inexact"]["full_ablation"] = True
    CONFIG["quantum"].update({"n_clock_list": [4, 6, 8, 10], "shots_list": [1000, 10000, 100000, 1000000], "eff_shots_list": [100, 1000, 10000, 100000], "max_qubits": 17})
    CONFIG["quantum"]["reg_experiment"].update({"quantum_n_clock": [6, 8, 10]})
    CONFIG["quantum"]["integrated"]["instances"] = ["tiny_A", "tiny_B", "tiny_C"]
    CONFIG["quantum"]["integrated"]["runs"] += [{"name": "hhl_scaled_kk_nc10_nofb", "recon": "statevector", "n_clock": 10, "scaling": "ruiz", "reciprocal": "kappa_known", "reg": 0.0, "fallback": False}]
    CONFIG["quantum"]["integrated"]["maxiters"] = 60
    CONFIG["sdplib"]["instances"] += ["control2", "hinf1", "theta1"]; CONFIG["sdplib"]["max_kkt_dim_inexact"] = 350
    CONFIG["cond_limit"] = 1200; CONFIG["limits"]["max_run_seconds"] = 300; CONFIG["limits"]["exact_max_dim"] = 5000
elif RUN_PROFILE == "large":
    CONFIG["families"]["dense_kappa"]["n_list"] = [10, 20, 40, 80, 160]; CONFIG["families"]["dense_kappa"]["seeds"] = [0, 1, 2, 3, 4]
    CONFIG["families"]["sparse_banded"].update({"n_list": [100, 200, 400, 800, 1600, 3200, 6400, 12800], "seeds": [0, 1, 2, 3, 4], "seeds_large": [0, 1, 2], "seeds_large_from_n": 6400})
    CONFIG["families"]["sparse_lowfill"].update({"n_list": [100, 200, 400, 800, 1600, 3200, 6400, 12800], "seeds": [0, 1, 2], "seeds_large": [0], "seeds_large_from_n": 6400})
    CONFIG["maros_meszaros"]["instances"] += ["DUAL2", "CVXQP2_S", "CVXQP3_S", "QSHARE1B", "VALUES", "QBRANDY", "QBEACONF", "PRIMAL2", "QE226", "QBORE3D", "QGROW7", "QSCORPIO", "QBANDM", "QPCSTAIR",
                                              "QPCBOEI1", "QSCAGR25", "PRIMAL4", "QSCSD1", "QETAMACR", "GOULDQP3", "QSTANDAT", "QSCSD6", "LASER", "QSEBA", "CVXQP2_M", "QSHIP04S", "CVXQP1_M", "CVXQP3_M",
                                              "MOSARQP2", "QSHELL", "Q25FV47", "QSCTAP2", "MOSARQP1", "QSCTAP3", "CONT-050", "QSHIP08S", "AUG3DCQP", "STCQP2", "QSHIP08L"]
    CONFIG["maros_meszaros"].update({"budget_s": 600, "max_kkt_dim": 20000})
    CONFIG["repeats"] = 3; CONFIG["inexact"]["full_ablation"] = True
    CONFIG["quantum"].update({"n_clock_list": [4, 6, 8, 10, 12], "shots_list": [1000, 10000, 100000, 1000000], "eff_shots_list": [100, 1000, 10000, 100000], "max_qubits": 18})
    CONFIG["quantum"]["integrated"]["instances"] = ["tiny_A", "tiny_B", "tiny_C"]
    CONFIG["quantum"]["integrated"]["runs"] += [{"name": "hhl_scaled_kk_nc10_nofb", "recon": "statevector", "n_clock": 10, "scaling": "ruiz", "reciprocal": "kappa_known", "reg": 0.0, "fallback": False},
                                                {"name": "hhl_scaled_kk_nc12_nofb", "recon": "statevector", "n_clock": 12, "scaling": "ruiz", "reciprocal": "kappa_known", "reg": 0.0, "fallback": False}]
    CONFIG["quantum"]["reg_experiment"].update({"quantum_n_clock": [6, 8, 10, 12], "classical_max_dim": 1024})
    CONFIG["quantum"]["integrated"]["maxiters"] = 60
    CONFIG["sdplib"]["instances"] += ["control2", "control3", "hinf1", "hinf3", "theta1"]; CONFIG["sdplib"]["max_kkt_dim_inexact"] = 500
    CONFIG["cond_limit"] = 2000; CONFIG["limits"]["max_run_seconds"] = 900; CONFIG["limits"]["exact_max_dim"] = 10000

if CONFIG["device"] == "gpu" and RUN_PROFILE != "smoke":
    # GPU build: extend the sparse families to sizes where a GPU can matter (KKT dimension up to ~79 000)
    CONFIG["families"]["sparse_banded"]["n_list"] = sorted(set(CONFIG["families"]["sparse_banded"]["n_list"]) | {6400, 12800, 25600})
    CONFIG["families"]["sparse_lowfill"]["n_list"] = sorted(set(CONFIG["families"]["sparse_lowfill"]["n_list"]) | {6400, 12800, 25600})
    CONFIG["limits"]["max_run_seconds"] = max(CONFIG["limits"]["max_run_seconds"], 300)
    CONFIG["quantum"]["max_qubits"] = max(CONFIG["quantum"]["max_qubits"], 20); CONFIG["quantum"]["max_state_bytes"] = 512 * 2**20
CONFIG_HASH = hashlib.sha256(json.dumps(CONFIG, sort_keys=True, default=str).encode()).hexdigest()[:10]
CONFIG["config_hash"] = CONFIG_HASH
RESULTS_DIR = os.path.join(CONFIG["output_root"], "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
PARTIAL_DIR = os.path.join(RESULTS_DIR, "partial")
NPZ_DIR = os.path.join(RESULTS_DIR, "instances_npz")
for d in (RESULTS_DIR, PLOTS_DIR, PARTIAL_DIR, NPZ_DIR):
    os.makedirs(d, exist_ok=True)
NOTEBOOK_T0 = time.time()
with open(os.path.join(RESULTS_DIR, "config.json"), "w") as f:
    json.dump(CONFIG, f, indent=2, default=str)
print(f"version={CODE_VERSION} profile={RUN_PROFILE} device={CONFIG['device']} config_hash={CONFIG_HASH} output_root={CONFIG['output_root']}")
