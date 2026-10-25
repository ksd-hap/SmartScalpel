"""Build viz/smartscalpel_visual.html: a self-contained (offline) visual built from the real data and saved model.
Run: py viz/build_viz.py     (needs data/ and models/)"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ml.load_data import LABEL_NAMES, load_dataset  # noqa: E402
from ml.scoring import scaled_proba  # noqa: E402

art = joblib.load(ROOT / "models" / "tissue_model.joblib")
ds = load_dataset()
pipe, T, thr = art["pipeline"], art["temperature"], art["reject_threshold"]
names = [LABEL_NAMES[c] for c in pipe.classes_]            # model class order
order = ["tumour", "stroma", "adipose"]

def pool(X, k):
    return X.reshape(X.shape[0], -1, k).mean(2)

# 1) class mean spectra, 400 bins (2 Da each), shared global scale
means = {n: pool(ds.X[ds.y == c], 20).mean(0) for c, n in LABEL_NAMES.items()}
gmax = max(m.max() for m in means.values())
means = {n: np.round(m / gmax, 4).tolist() for n, m in means.items()}

# 2) 2-D map: the two discriminant directions (LDA fitted on TRAINING patients only), shown for all spectra
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as _LDA
F = pipe.named_steps["pca"].transform(pipe.named_steps["scale"].transform(ds.X))
tr_mask = np.isin(ds.patients, art["train_patients"])
Z = _LDA(solver="svd", n_components=2).fit(F[tr_mask], ds.y[tr_mask]).transform(F)
lo, hi = np.percentile(Z, 2, axis=0), np.percentile(Z, 98, axis=0)
Zn = np.clip((Z - lo) / (hi - lo) * 0.9 + 0.05, 0.0, 1.0)

test_mask = np.isin(ds.patients, art["test_patients"])
test_idx = np.where(test_mask)[0]
P = scaled_proba(pipe, ds.X[test_idx], T)
pred = [names[i] for i in P.argmax(1)]
pat_ids = {p: i + 1 for i, p in enumerate(sorted(set(ds.patients[test_idx])))}

sp = np.sqrt(pool(ds.X[test_idx], 40))                      # 200 bins
sp = np.clip(sp / np.percentile(sp, 99.5), 0, 1)
test = []
for k, i in enumerate(test_idx):
    test.append({"p": pat_ids[ds.patients[i]], "c": LABEL_NAMES[ds.y[i]], "pred": pred[k],
                 "s": [round(float(P[k][names.index(n)]), 4) for n in order],
                 "ok": bool(P[k].max() >= thr), "spec": (sp[k] * 255).round().astype(int).tolist(), "pt": int(i)})
pts = [{"x": round(float(Zn[i, 0]), 4), "y": round(float(Zn[i, 1]), 4), "c": LABEL_NAMES[ds.y[i]], "t": int(test_mask[i])}
       for i in range(len(ds.y))]
rep = json.loads((ROOT / "reports" / "evaluation.json").read_text())
n = rep["nested_grouped_cv"]; h = rep["heldout_test"]
stats = {"nested_acc": n["pooled_accuracy"], "nested_tumour": n["per_class_recall"]["tumour"],
         "heldout_acc": h["accuracy"], "n_spectra": len(ds.y), "n_patients": len(set(ds.patients)),
         "n_test": len(test), "n_test_patients": len(pat_ids), "threshold": thr, "version": art["model_version"],
         "paper_acc": 0.923, "map": "LDA discriminants fitted on training patients"}
data = {"means": means, "pts": pts, "test": test, "stats": stats}
tpl = (ROOT / "viz" / "template.html").read_text(encoding="utf-8")
out = ROOT / "viz" / "smartscalpel_visual.html"
out.write_text(tpl.replace("/*DATA*/null", json.dumps(data, separators=(",", ":"))), encoding="utf-8")
print(f"wrote {out} ({out.stat().st_size/1024:.0f} KB), {len(test)} held-out spectra, "
      f"{sum(t['c']!=t['pred'] for t in test)} model errors shown")
