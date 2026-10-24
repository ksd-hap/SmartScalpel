"""Load and validate the REIMS breast-tissue dataset (Sci Rep 2020, 41598_2020_77102_MOESM3_ESM.xlsx)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = ROOT / "data" / "41598_2020_77102_MOESM3_ESM.xlsx"
SHEET = "MAM_REIMS_A_S_T.data"

# Raw label code in the file -> human-readable class name.
LABEL_NAMES = {"T": "tumour", "F": "stroma", "A": "adipose"}
# Fixed class order used everywhere (model classes_, score dicts, confusion matrix).
CLASS_CODES = ["T", "F", "A"]
CLASS_NAMES = [LABEL_NAMES[c] for c in CLASS_CODES]

EXPECTED = {"n_spectra": 689, "n_patients": 72, "n_features": 8000}


@dataclass
class Dataset:
    X: np.ndarray            # (n_spectra, n_features) float64
    y: np.ndarray            # (n_spectra,) class codes 'T'/'F'/'A'
    patients: np.ndarray     # (n_spectra,) patient id strings
    mz: np.ndarray           # (n_features,) ordered m/z values (or channel index if headers are not numeric)
    report: dict = field(default_factory=dict)


def _read_sheet(path: Path) -> pd.DataFrame:
    cache = path.with_suffix(".cache.pkl")
    if cache.exists() and cache.stat().st_mtime >= path.stat().st_mtime:
        return pd.read_pickle(cache)
    df = pd.read_excel(path, sheet_name=SHEET, engine="openpyxl")
    try:
        df.to_pickle(cache)  # reading 5M Excel cells is slow; cache next to the file
    except OSError:
        pass
    return df


def load_dataset(path: str | Path | None = None, expected: dict | None = None, strict: bool = True) -> Dataset:
    """Read, validate and return the dataset.

    Column 0 = tissue label, column 1 = patient id, remaining columns = intensities.
    `expected` overrides the published counts (used by tests with small synthetic files).
    With strict=True, a mismatch with the published counts raises ValueError; otherwise it is only reported.
    """
    path = Path(path) if path else DEFAULT_PATH
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    exp = {**EXPECTED, **(expected or {})}
    df = _read_sheet(path)
    problems: list[str] = []

    if df.shape[1] < 3:
        raise ValueError(f"Expected >= 3 columns (label, patient, features), got {df.shape[1]}")

    label_col, patient_col = df.columns[0], df.columns[1]
    y = df.iloc[:, 0].astype(str).str.strip().to_numpy()
    patients = df.iloc[:, 1].astype(str).str.strip().to_numpy()

    bad_labels = sorted(set(y) - set(CLASS_CODES))
    if bad_labels:
        raise ValueError(f"Unexpected label values {bad_labels}; allowed {CLASS_CODES}")

    feat = df.iloc[:, 2:]
    X = feat.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    n_nan = int(np.isnan(X).sum())
    n_inf = int(np.isinf(X).sum())
    # Non-numeric cells become NaN via coerce; count cells that were non-null originally but failed.
    n_invalid_text = int((np.isnan(X) & feat.notna().to_numpy()).sum())
    if n_nan or n_inf:
        raise ValueError(f"Invalid spectral values: {n_nan} NaN/missing ({n_invalid_text} non-numeric), {n_inf} inf")

    mz = pd.to_numeric(pd.Series(feat.columns), errors="coerce").to_numpy(dtype=np.float64)
    mz_from_headers = not np.isnan(mz).any()
    if not mz_from_headers:
        mz = np.arange(X.shape[1], dtype=np.float64)
        problems.append("Feature headers are not all numeric; using channel index instead of m/z.")
    elif not np.all(np.diff(mz) > 0):
        problems.append("m/z headers are not strictly increasing.")

    report = {
        "path": str(path),
        "sheet": SHEET,
        "label_column": str(label_col),
        "patient_column": str(patient_col),
        "n_spectra": int(X.shape[0]),
        "n_patients": int(len(set(patients))),
        "n_features": int(X.shape[1]),
        "class_counts": {LABEL_NAMES[c]: int((y == c).sum()) for c in CLASS_CODES},
        "mz_from_headers": bool(mz_from_headers),
        "mz_range": [float(mz.min()), float(mz.max())],
        "n_missing": n_nan,
        "n_inf": n_inf,
        "intensity_min": float(X.min()),
        "intensity_max": float(X.max()),
        "problems": problems,
    }
    for key in ("n_spectra", "n_patients", "n_features"):
        if report[key] != exp[key]:
            problems.append(f"{key}: found {report[key]}, expected {exp[key]}")
    if strict and any("found" in p for p in problems):
        raise ValueError("Dataset does not match published structure: " + "; ".join(problems))
    return Dataset(X=X, y=y, patients=patients, mz=mz, report=report)


if __name__ == "__main__":
    import json
    import sys

    ds = load_dataset(sys.argv[1] if len(sys.argv) > 1 else None, strict=False)
    print(json.dumps(ds.report, indent=2))
