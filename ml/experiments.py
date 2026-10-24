"""Compare more PCA components / class weighting / shrinkage with the SAME nested patient-grouped folds as ml/cv_report.py."""
import json
import time

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ml.load_data import CLASS_CODES, LABEL_NAMES, ROOT, load_dataset
from ml.train import SEED

OLD, NEW = [10, 20, 30, 50], [30, 50, 75, 100, 150]


def P(clf):
    return Pipeline([("scale", StandardScaler()), ("pca", PCA(random_state=SEED)), ("clf", clf)])


FAMILIES = {
    "lda_baseline (10-50 comps)": (P(LDA()), {"pca__n_components": OLD}),
    "lda_more_comps (30-150)": (P(LDA()), {"pca__n_components": NEW}),
    "lda_uniform_priors": (P(LDA(priors=[1 / 3] * 3)), {"pca__n_components": NEW}),
    "lda_shrinkage": (P(LDA(solver="lsqr", shrinkage="auto")), {"pca__n_components": NEW}),
    "logreg_balanced": (P(LogisticRegression(max_iter=3000, class_weight="balanced")),
                        {"pca__n_components": NEW, "clf__C": [0.01, 0.1, 1]}),
    "svm_balanced": (P(SVC(kernel="rbf", class_weight="balanced")),
                     {"pca__n_components": NEW, "clf__C": [1, 10, 100]}),
}


def main(n_outer=6):
    ds = load_dataset(strict=False)
    outer = list(StratifiedGroupKFold(n_splits=n_outer, shuffle=True, random_state=SEED).split(ds.X, ds.y, ds.patients))
    results = {}
    for name, (pipe, grid) in FAMILIES.items():
        t = time.time()
        pred = np.empty(len(ds.y), dtype=ds.y.dtype)
        picks = []
        for tr, te in outer:
            gs = GridSearchCV(pipe, grid, cv=GroupKFold(5), scoring="balanced_accuracy", n_jobs=-1)
            gs.fit(ds.X[tr], ds.y[tr], groups=ds.patients[tr])
            pred[te] = gs.predict(ds.X[te])
            picks.append(gs.best_params_)
        rec = recall_score(ds.y, pred, labels=CLASS_CODES, average=None)
        results[name] = {"accuracy": float(accuracy_score(ds.y, pred)),
                         "balanced_accuracy": float(balanced_accuracy_score(ds.y, pred)),
                         "recall": {LABEL_NAMES[c]: float(r) for c, r in zip(CLASS_CODES, rec)},
                         "chosen_params_per_fold": picks}
        r = results[name]
        print(f"{name:28s} acc {r['accuracy']:.3f} bal {r['balanced_accuracy']:.3f} "
              f"T {r['recall']['tumour']:.3f} S {r['recall']['stroma']:.3f} A {r['recall']['adipose']:.3f} ({time.time()-t:.0f}s)", flush=True)
    (ROOT / "reports" / "experiments.json").write_text(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
