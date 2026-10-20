"""Reading result bundles written by the notebook (a .zip file or an unpacked directory)."""
import os
import tempfile
import zipfile

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLES = os.path.join(ROOT, "results", "tables")

# The archived bundles of the paper were written before some methods were renamed; their names are mapped on reading.
LEGACY_NAMES = [
    ("minres_v1_ruiz_relres_1e-06", "minres_relres_1e-06"),
    ("minres_v1_relres_1e-06", "minres_relres_1e-06_noscale"),
    ("hhl_v1_nc8_fb", "hhl_noscale_res_nc8_fb"),
    ("minresx5", "minres_fused"),
    ("minresx", "minres_base"),
]
_NAME_COLUMNS = ("method", "method_kind", "method_used", "run_id", "inner_solver", "solver")


def root_of(bundle):
    """Directory that holds benchmark_results.csv inside a bundle; a .zip file is extracted to a temporary directory."""
    d = bundle
    if not os.path.isdir(bundle):
        d = tempfile.mkdtemp(prefix="bundle_")
        with zipfile.ZipFile(bundle) as z:
            z.extractall(d)
    for r, _, files in os.walk(d):
        if "benchmark_results.csv" in files and "partial" not in r:
            return r
    raise SystemExit(f"no benchmark_results.csv in {bundle}")


def rename_legacy(df):
    for c in _NAME_COLUMNS:
        if c in df and (pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c])):
            ok = df[c].notna()
            v = df.loc[ok, c].astype(str)
            for old, new in LEGACY_NAMES:
                v = v.str.replace(old, new, regex=False)
            df.loc[ok, c] = v
    return df


def read_table(root, name):
    """One CSV of a bundle, with legacy method names mapped; an empty DataFrame if the file is missing or empty."""
    p = os.path.join(root, name)
    if not (os.path.isfile(p) and os.path.getsize(p) > 2):
        return pd.DataFrame()
    return rename_legacy(pd.read_csv(p, low_memory=False))
