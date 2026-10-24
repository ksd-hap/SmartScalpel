"""Train PCA+LDA (baseline), PCA+SVM and PCA+LogReg with patient-grouped CV, save the best pipeline.

Protocol (no leakage):
  1. Hold out ~25% of PATIENTS (GroupShuffleSplit) as the final test set. Never used for any choice.
  2. On the training patients only: GroupKFold grid search per candidate; model chosen by inner CV balanced accuracy.
  3. Rejection threshold chosen from out-of-fold (GroupKFold) scores on training patients only.
  4. Final pipeline is fitted on training patients only, so the test patients stay genuinely unseen.
  5. Test metrics are computed once and reported; they are not used to change anything.
"""
from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ml.evaluate import MODEL_PATH, REPORT_PATH, compute_metrics
from ml.scoring import fit_temperature, scaled_proba
from ml.load_data import CLASS_CODES, LABEL_NAMES, ROOT, Dataset, load_dataset

SEED = 42
DEMO_PATH = ROOT / "models" / "demo_samples.npz"
N_DEMO_PER_CLASS = 5


def candidates() -> dict:
    def pipe(clf):
        return Pipeline([("scale", StandardScaler()), ("pca", PCA(random_state=SEED)), ("clf", clf)])

    comps = [10, 20, 30, 50]
    return {  # first entry is the baseline and wins ties
        "pca-lda": (pipe(LinearDiscriminantAnalysis()), {"pca__n_components": comps}),
        "pca-svm": (pipe(CalibratedClassifierCV(SVC(kernel="rbf", random_state=SEED), ensemble=False, cv=3)),
                    {"pca__n_components": comps, "clf__estimator__C": [1, 10]}),
        "pca-logreg": (pipe(LogisticRegression(max_iter=2000)),
                       {"pca__n_components": comps, "clf__C": [0.01, 0.1, 1]}),
    }


def split_patients(ds: Dataset, test_size: float = 0.25, min_per_class: int = 5):
    """Patient-grouped split; deterministic; ensures every class appears in train and test."""
    for seed in range(SEED, SEED + 200):
        tr, te = next(GroupShuffleSplit(1, test_size=test_size, random_state=seed).split(ds.X, ds.y, ds.patients))
        if all((ds.y[te] == c).sum() >= min_per_class and (ds.y[tr] == c).sum() >= min_per_class for c in CLASS_CODES):
            assert not set(ds.patients[tr]) & set(ds.patients[te])
            return tr, te, seed
    raise RuntimeError("Could not find a split with all classes in train and test")


def scaled_proba_from(p, T):
    z = np.log(np.clip(p, 1e-300, 1.0)) / T
    z -= z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def choose_threshold(oof_proba, y, classes, target_acc=0.95, min_coverage=0.70):
    """Lowest threshold whose accepted-subset accuracy >= target while keeping coverage >= min_coverage."""
    pred = np.array(classes)[oof_proba.argmax(1)]
    conf = oof_proba.max(1)
    for t in np.round(np.arange(0.40, 0.99, 0.01), 2):
        acc_mask = conf >= t
        if acc_mask.mean() < min_coverage:
            break
        if (pred[acc_mask] == y[acc_mask]).mean() >= target_acc:
            return float(t), "target met"
    return 0.5, f"target accuracy {target_acc} at coverage>={min_coverage} not reachable; fell back to 0.5"


def train(ds: Dataset, n_demo: int = N_DEMO_PER_CLASS, save: bool = True, n_splits: int = 5) -> dict:
    tr, te, split_seed = split_patients(ds)
    Xtr, ytr, gtr = ds.X[tr], ds.y[tr], ds.patients[tr]
    n_splits = min(n_splits, len(set(gtr)))
    cv = GroupKFold(n_splits=n_splits)

    comparison, best = {}, None
    for name, (pipe, grid) in candidates().items():
        gs = GridSearchCV(pipe, grid, cv=cv, scoring="balanced_accuracy", n_jobs=-1)
        gs.fit(Xtr, ytr, groups=gtr)
        comparison[name] = {"cv_balanced_accuracy": float(gs.best_score_), "best_params": gs.best_params_}
        print(f"{name}: grouped-CV balanced acc {gs.best_score_:.3f} {gs.best_params_}")
        if best is None or gs.best_score_ > best[1] + 1e-9:
            best = (name, gs.best_score_, gs.best_estimator_)
    name, _, pipe = best
    classes = list(pipe.classes_)

    oof = cross_val_predict(pipe, Xtr, ytr, groups=gtr, cv=cv, method="predict_proba")
    T = fit_temperature(oof, ytr, classes)
    print(f"temperature (fit on out-of-fold train scores): {T:.1f}")
    oof = scaled_proba_from(oof, T)
    threshold, thr_note = choose_threshold(oof, ytr, classes)

    metrics = compute_metrics(pipe, ds.X[te], ds.y[te], ds.patients[te], threshold, T=T)
    print(json.dumps({k: v for k, v in metrics.items() if k != "confusion_matrix"}, indent=2))

    version = f"{name}-v1"
    artifact = {
        "pipeline": pipe,
        "model_version": version,
        "n_features": int(ds.X.shape[1]),
        "mz": ds.mz,
        "label_mapping": LABEL_NAMES,
        "class_codes": list(pipe.classes_),
        "reject_threshold": threshold,
        "temperature": T,
        "score_type": "temperature-scaled model score (T fit on out-of-fold training data); not a clinical probability",
        "evaluation": metrics,
        "train_patients": sorted(set(gtr)),
        "test_patients": sorted(set(ds.patients[te])),
        "sklearn_version": sklearn.__version__,
        "disclaimer": "Research simulation on published ex-vivo REIMS spectra. Not clinically validated.",
    }
    report = {
        "model_version": version,
        "selected_model": name,
        "model_comparison_train_grouped_cv": comparison,
        "split": {"method": "GroupShuffleSplit by patient", "seed": split_seed,
                  "n_train_patients": len(set(gtr)), "n_train_spectra": int(len(tr)),
                  "n_test_patients": metrics["n_independent_patients"], "n_test_spectra": int(len(te))},
        "temperature": T,
        "rejection_threshold_selection": {"source": "out-of-fold scores on training patients only", "note": thr_note,
                                          "threshold": threshold},
        "heldout_test": metrics,
        "dataset": ds.report,
        "notes": ["Scores are temperature-scaled model scores, not clinical probabilities.",
                  "Not a clinical diagnostic tool; trained on ex-vivo REIMS spectra only."],
    }

    # Demo samples: random (seeded) held-out spectra per class, NOT filtered by whether the model gets them right.
    rng = np.random.default_rng(SEED)
    idx = []
    for c in CLASS_CODES:
        pool = te[ds.y[te] == c]
        idx.extend(rng.choice(pool, size=min(n_demo, len(pool)), replace=False).tolist())
    idx = np.array(idx)
    demo_pred = pipe.predict(ds.X[idx])
    report["demo_samples"] = {"n": int(len(idx)), "model_correct_on_demo": int((demo_pred == ds.y[idx]).sum()),
                              "selection": "seeded random per class from held-out test patients; not filtered by correctness"}

    if save:
        MODEL_PATH.parent.mkdir(exist_ok=True)
        REPORT_PATH.parent.mkdir(exist_ok=True)
        joblib.dump(artifact, MODEL_PATH)
        np.savez(DEMO_PATH, X=ds.X[idx], region_type=np.array([LABEL_NAMES[c] for c in ds.y[idx]]),
                 patient=ds.patients[idx])
        REPORT_PATH.write_text(json.dumps(report, indent=2))
        print(f"Saved {MODEL_PATH}, {DEMO_PATH}, {REPORT_PATH}")
    return {"artifact": artifact, "report": report, "test_idx": te, "train_idx": tr}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    args = ap.parse_args()
    train(load_dataset(args.data, strict=False))
