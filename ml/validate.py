"""One-command validation. Run: py -m ml.validate
Prints dataset checks, reproduced hold-out metrics, baselines, a shuffled-label control, and the saved nested-CV result."""
import json

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from ml.evaluate import MODEL_PATH, REPORT_PATH, compute_metrics
from ml.load_data import CLASS_CODES, CLASS_NAMES, load_dataset


def main():
    art = joblib.load(MODEL_PATH)
    ds = load_dataset()  # strict: must match 689 / 72 / 8000
    r = ds.report
    print(f"[1] Dataset OK: {r['n_spectra']} spectra, {r['n_patients']} patients, {r['n_features']} features, "
          f"m/z {r['mz_range'][0]}-{r['mz_range'][1]}, missing={r['n_missing']}, classes={r['class_counts']}")

    tr_p, te_p = set(art["train_patients"]), set(art["test_patients"])
    assert not tr_p & te_p, "PATIENT LEAKAGE"
    print(f"[2] No leakage: {len(tr_p)} train patients, {len(te_p)} test patients, overlap = 0")

    te = np.isin(ds.patients, art["test_patients"])
    m = compute_metrics(art["pipeline"], ds.X[te], ds.y[te], ds.patients[te], art["reject_threshold"], T=art["temperature"])
    saved = art["evaluation"]
    for k in ("accuracy", "balanced_accuracy", "macro_f1", "tumour_recall"):
        assert abs(saved[k] - m[k]) < 1e-9, k
    print(f"[3] Held-out ({m['n_spectra']} spectra, {m['n_independent_patients']} unseen patients), reproduced exactly:")
    print(f"    accuracy {m['accuracy']:.3f} | balanced acc {m['balanced_accuracy']:.3f} | macro F1 {m['macro_f1']:.3f}")
    print("    recall: " + ", ".join(f"{k} {v:.3f}" for k, v in m["per_class_recall"].items()))
    print("    confusion matrix (rows = true, cols = predicted):", CLASS_NAMES)
    for name, row in zip(CLASS_NAMES, m["confusion_matrix"]["matrix"]):
        print(f"      {name:8s} {row}")
    rj = m["rejection"]
    print(f"    rejection @ {rj['threshold']}: {rj['fraction_rejected']*100:.1f}% rejected, accuracy on accepted {rj['accuracy_on_accepted']:.3f}")
    lat = m["latency_single_spectrum"]
    print(f"    latency per spectrum: median {lat['median_ms']:.2f} ms, p95 {lat['p95_ms']:.2f} ms")

    maj = max(CLASS_CODES, key=lambda c: (ds.y[te] == c).sum())
    print(f"[4] Baseline: always predicting '{maj}' gives accuracy {(ds.y[te] == maj).mean():.3f} (chance ~0.33)")

    # Shuffled-label control: same pipeline, labels randomly permuted, patient-grouped CV. Should land near chance.
    rng = np.random.default_rng(0)
    pipe = clone(art["pipeline"])
    ys = rng.permutation(ds.y)
    pred = cross_val_predict(pipe, ds.X, ys, groups=ds.patients, cv=GroupKFold(5))
    print(f"[5] Shuffled-label control (grouped CV): accuracy {accuracy_score(ys, pred):.3f}, balanced {balanced_accuracy_score(ys, pred):.3f} "
          "-> near chance means the real result is not a pipeline artefact")

    n = json.loads(REPORT_PATH.read_text()).get("nested_grouped_cv")
    if n:
        print(f"[6] Nested patient-grouped CV, all {n['n_patients']} patients ({n['n_outer_folds']} folds), saved in report:")
        print(f"    accuracy {n['pooled_accuracy']:.3f} | balanced {n['pooled_balanced_accuracy']:.3f} | macro F1 {n['pooled_macro_f1']:.3f}")
        print("    recall: " + ", ".join(f"{k} {v:.3f}" for k, v in n["per_class_recall"].items())
              + f" | fold spread +/-{n['fold_balanced_accuracy_mean_std'][1]:.3f}  (re-run: py -m ml.cv_report, ~1.5 min)")
    print(f"Model: {art['model_version']} | sklearn {art['sklearn_version']} | simulation only, not clinically validated")


if __name__ == "__main__":
    main()
