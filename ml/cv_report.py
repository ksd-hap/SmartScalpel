"""Nested patient-grouped CV over ALL patients: a more stable estimate than the single 18-patient hold-out.

Outer folds: StratifiedGroupKFold (no patient in both train and test). Inside each outer-train set the model/hyper-parameters
are chosen by an inner GroupKFold grid search, exactly as in ml/train.py. Results are merged into reports/evaluation.json
under "nested_grouped_cv"; the saved model is NOT affected.
"""
import json

import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedGroupKFold

from ml.load_data import CLASS_CODES, LABEL_NAMES, load_dataset
from ml.evaluate import REPORT_PATH
from ml.train import SEED, candidates


def main(n_outer=6):
    ds = load_dataset(strict=False)
    outer = StratifiedGroupKFold(n_splits=n_outer, shuffle=True, random_state=SEED)
    pred = np.empty(len(ds.y), dtype=ds.y.dtype)
    folds, chosen = [], []
    for k, (tr, te) in enumerate(outer.split(ds.X, ds.y, ds.patients)):
        assert not set(ds.patients[tr]) & set(ds.patients[te])
        best = None
        for name, (pipe, grid) in candidates().items():
            gs = GridSearchCV(pipe, grid, cv=GroupKFold(5), scoring="balanced_accuracy", n_jobs=-1)
            gs.fit(ds.X[tr], ds.y[tr], groups=ds.patients[tr])
            if best is None or gs.best_score_ > best[1] + 1e-9:
                best = (name, gs.best_score_, gs.best_estimator_)
        pred[te] = best[2].predict(ds.X[te])
        chosen.append(best[0])
        folds.append({"fold": k, "n_test_patients": len(set(ds.patients[te])), "n_test_spectra": int(len(te)),
                      "selected_model": best[0],
                      "accuracy": float(accuracy_score(ds.y[te], pred[te])),
                      "balanced_accuracy": float(balanced_accuracy_score(ds.y[te], pred[te]))})
        print(folds[-1])
    rec = recall_score(ds.y, pred, labels=CLASS_CODES, average=None)
    res = {
        "n_outer_folds": n_outer, "n_patients": int(len(set(ds.patients))), "n_spectra": int(len(ds.y)),
        "pooled_accuracy": float(accuracy_score(ds.y, pred)),
        "pooled_balanced_accuracy": float(balanced_accuracy_score(ds.y, pred)),
        "pooled_macro_f1": float(f1_score(ds.y, pred, average="macro")),
        "per_class_recall": {LABEL_NAMES[c]: float(r) for c, r in zip(CLASS_CODES, rec)},
        "fold_balanced_accuracy_mean_std": [float(np.mean([f["balanced_accuracy"] for f in folds])),
                                            float(np.std([f["balanced_accuracy"] for f in folds]))],
        "models_selected_per_fold": chosen, "folds": folds,
        "note": "Every spectrum is predicted by a model that never saw that patient; selection repeated inside each fold.",
    }
    rep = json.loads(REPORT_PATH.read_text())
    rep["nested_grouped_cv"] = res
    REPORT_PATH.write_text(json.dumps(rep, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "folds"}, indent=2))


if __name__ == "__main__":
    main()
