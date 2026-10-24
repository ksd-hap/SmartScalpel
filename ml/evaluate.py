"""Metrics for the tissue classifier. `python -m ml.evaluate` re-computes the held-out metrics from the saved model."""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, recall_score

from ml.scoring import scaled_proba
from ml.load_data import CLASS_CODES, CLASS_NAMES, LABEL_NAMES, ROOT, load_dataset

MODEL_PATH = ROOT / "models" / "tissue_model.joblib"
REPORT_PATH = ROOT / "reports" / "evaluation.json"


def measure_latency(pipe, X: np.ndarray, n: int = 200, T: float = 1.0) -> dict:
    """Single-spectrum predict_proba latency in milliseconds (wall clock, this machine)."""
    n = min(n, len(X))
    times = []
    for i in range(n):
        t = time.perf_counter()
        scaled_proba(pipe, X[i : i + 1], T)
        times.append((time.perf_counter() - t) * 1000)
    return {"n_calls": n, "median_ms": float(np.median(times)), "p95_ms": float(np.percentile(times, 95))}


def compute_metrics(pipe, X, y, patients, threshold: float, with_latency: bool = True, T: float = 1.0) -> dict:
    """All metrics on a held-out set. y holds class codes ('T','F','A')."""
    proba = scaled_proba(pipe, X, T)
    classes = list(pipe.classes_)
    pred = np.array(classes)[proba.argmax(1)]
    accepted = proba.max(1) >= threshold
    recalls = recall_score(y, pred, labels=CLASS_CODES, average=None, zero_division=0)
    per_class = {LABEL_NAMES[c]: float(r) for c, r in zip(CLASS_CODES, recalls)}
    out = {
        "n_spectra": int(len(y)),
        "n_independent_patients": int(len(set(patients))),
        "class_counts": {LABEL_NAMES[c]: int((y == c).sum()) for c in CLASS_CODES},
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=CLASS_CODES, average="macro", zero_division=0)),
        "per_class_recall": per_class,
        "tumour_recall": per_class["tumour"],
        "confusion_matrix": {
            "labels_rows_true_cols_predicted": CLASS_NAMES,
            "matrix": confusion_matrix(y, pred, labels=CLASS_CODES).tolist(),
        },
        "rejection": {
            "threshold": float(threshold),
            "fraction_rejected": float(1 - accepted.mean()),
            "accuracy_on_accepted": float(accuracy_score(y[accepted], pred[accepted])) if accepted.any() else None,
        },
    }
    if with_latency:
        out["latency_single_spectrum"] = measure_latency(pipe, X, T=T)
    return out


def main() -> dict:
    art = joblib.load(MODEL_PATH)
    ds = load_dataset(strict=False)
    mask = np.isin(ds.patients, art["test_patients"])
    metrics = compute_metrics(art["pipeline"], ds.X[mask], ds.y[mask], ds.patients[mask], art["reject_threshold"], T=art["temperature"])
    saved = art["evaluation"]
    for k in ("accuracy", "balanced_accuracy", "macro_f1", "tumour_recall"):
        if abs(saved[k] - metrics[k]) > 1e-9:
            raise AssertionError(f"{k} not reproducible: saved {saved[k]} vs recomputed {metrics[k]}")
    print(json.dumps({k: v for k, v in metrics.items() if k != "confusion_matrix"}, indent=2))
    print("Reproduced saved held-out metrics exactly.")
    return metrics


if __name__ == "__main__":
    main()
