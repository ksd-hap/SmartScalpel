"""Standard result graphs (PNG) from the saved model and reports. Run: py viz/make_graphs.py  -> graphs/"""
import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ml.load_data import load_dataset, LABEL_NAMES  # noqa: E402
from ml.scoring import scaled_proba  # noqa: E402

OUT = ROOT / "graphs"; OUT.mkdir(exist_ok=True)
INK, MUTE, GRID = "#1b2233", "#6b7390", "#e4e7ef"
CLS = {"tumour": "#d6293e", "stroma": "#3b5bdb", "adipose": "#12a06b"}
SER = ["#26304f", "#9db0e8"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": MUTE,
                     "xtick.color": MUTE, "ytick.color": MUTE, "text.color": INK, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True})

rep = json.loads((ROOT / "reports" / "evaluation.json").read_text())
exp = json.loads((ROOT / "reports" / "experiments.json").read_text())
H, N = rep["heldout_test"], rep["nested_grouped_cv"]
art = joblib.load(ROOT / "models" / "tissue_model.joblib")
ds = load_dataset()
names = ["tumour", "stroma", "adipose"]

def save(fig, name):
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight", facecolor="white"); plt.close(fig)

# 1 confusion matrix (held-out)
cm = np.array(H["confusion_matrix"]["matrix"]); pct = cm / cm.sum(1, keepdims=True)
fig, ax = plt.subplots(figsize=(5.6, 4.8)); ax.grid(False)
ax.imshow(pct, cmap="Blues", vmin=0, vmax=1)
for i in range(3):
    for j in range(3):
        ax.text(j, i, f"{cm[i,j]}\n{pct[i,j]*100:.0f}%", ha="center", va="center", fontsize=13,
                color="white" if pct[i, j] > .55 else INK, fontweight="bold" if i == j else "normal")
ax.set_xticks(range(3), names); ax.set_yticks(range(3), names)
ax.set_xlabel("Model prediction"); ax.set_ylabel("True tissue")
ax.set_title(f"Held-out patients: {H['accuracy']*100:.1f}% correct\n(160 spectra, 18 unseen patients)", loc="left", fontsize=12.5)
for s in ax.spines.values(): s.set_visible(False)
save(fig, "1_confusion_matrix.png")

# 2 recall per class, two evaluations
fig, ax = plt.subplots(figsize=(7.2, 4.4)); x = np.arange(3); w = .36
a = [N["per_class_recall"][n] * 100 for n in names]; b = [H["per_class_recall"][n] * 100 for n in names]
for k, (vals, lab) in enumerate([(a, "Nested CV, all 72 patients"), (b, "Hold-out, 18 patients")]):
    bars = ax.bar(x + (k - .5) * w, vals, w - .04, color=SER[k], label=lab)
    for r, v in zip(bars, vals): ax.text(r.get_x() + r.get_width() / 2, v + 1, f"{v:.0f}%", ha="center", fontsize=10.5)
ax.set_xticks(x, names); ax.set_ylim(0, 108); ax.set_ylabel("Recall (%)")
ax.set_title("Tumour is recalled most reliably across all 72 patients", loc="left", fontsize=12.5)
ax.legend(frameon=False, loc="lower center", ncol=2, bbox_to_anchor=(.5, -.22)); ax.grid(axis="x", visible=False)
save(fig, "2_recall_by_class.png")

# 3 model comparison (nested CV)
items = sorted(exp.items(), key=lambda kv: kv[1]["balanced_accuracy"])
fig, ax = plt.subplots(figsize=(7.4, 4.0))
vals = [v["balanced_accuracy"] * 100 for _, v in items]
cols = [SER[0] if "lda_shrinkage" in k else SER[1] for k, _ in items]
bars = ax.barh([k.replace("_", " ").capitalize() for k, _ in items], vals, color=cols, height=.62)
for r, v in zip(bars, vals): ax.text(v + .3, r.get_y() + r.get_height() / 2, f"{v:.1f}%", va="center", fontsize=10.5)
ax.set_xlim(80, 94); ax.set_xlabel("Balanced accuracy, nested patient-grouped CV (%)"); ax.grid(axis="y", visible=False)
ax.set_title("Simple regularised LDA matched or beat heavier models", loc="left", fontsize=12.5)
save(fig, "3_model_comparison.png")

# 4 confidence vs accuracy (held-out)
te = np.isin(ds.patients, art["test_patients"])
P = scaled_proba(art["pipeline"], ds.X[te], art["temperature"]); conf = P.max(1)
ok = np.array(art["pipeline"].classes_)[P.argmax(1)] == ds.y[te]
edges = [0, .6, .77, .9, .99, 1.01]; labs = ["<0.60", "0.60-0.77", "0.77-0.90", "0.90-0.99", ">=0.99"]
accs, ns = [], []
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (conf >= lo) & (conf < hi); ns.append(int(m.sum())); accs.append(ok[m].mean() * 100 if m.any() else 0)
fig, ax = plt.subplots(figsize=(7.2, 4.3))
bars = ax.bar(labs, accs, color=SER[0], width=.62)
for r, v, n in zip(bars, accs, ns): ax.text(r.get_x() + r.get_width() / 2, v + 1.5, f"{v:.0f}%\n(n={n})", ha="center", fontsize=10)
ax.axvline(1.5, color=CLS["tumour"], lw=1.2, ls="--"); ax.text(1.55, 6, "reject threshold 0.77", color=CLS["tumour"], fontsize=9.5)
ax.set_ylim(0, 118); ax.set_ylabel("Actual accuracy (%)"); ax.set_xlabel("Model's top score")
ax.set_title("When the model is unsure, it really is less accurate", loc="left", fontsize=12.5); ax.grid(axis="x", visible=False)
save(fig, "4_confidence_vs_accuracy.png")

# 5 per-fold nested CV accuracy
fa = [f["accuracy"] * 100 for f in N["folds"]]
fig, ax = plt.subplots(figsize=(7.2, 4.2))
bars = ax.bar([f"Fold {i+1}" for i in range(len(fa))], fa, color=SER[1], width=.6)
for r, v in zip(bars, fa): ax.text(r.get_x() + r.get_width() / 2, 71, f"{v:.1f}%", ha="center", va="bottom", fontsize=11, color=INK)
ax.set_xlim(-.6, 7.4)
ax.axhline(N["pooled_accuracy"] * 100, color=SER[0], lw=1.8); ax.text(5.55, N["pooled_accuracy"] * 100 - .5, f"mean {N['pooled_accuracy']*100:.1f}%", va="top", color=SER[0], fontsize=10)
ax.axhline(92.3, color=MUTE, lw=1.2, ls="--"); ax.text(5.55, 92.3 + .4, "original study 92.3%", color=MUTE, fontsize=10)
ax.set_ylim(70, 100); ax.set_ylabel("Accuracy (%)"); ax.grid(axis="x", visible=False)
ax.set_title("Each fold tests patients the model never saw", loc="left", fontsize=12.5)
save(fig, "5_nested_cv_folds.png")

# 6 class mean spectra
fig, axs = plt.subplots(3, 1, figsize=(8.4, 6.2), sharex=True)
k = 10; mz = ds.mz.reshape(-1, k).mean(1); gmax = 0
means = {}
for c, n in LABEL_NAMES.items():
    means[n] = ds.X[ds.y == c].mean(0).reshape(-1, k).mean(1); gmax = max(gmax, means[n].max())
for ax, n in zip(axs, names):
    ax.fill_between(mz, means[n] / gmax, color=CLS[n], alpha=.18); ax.plot(mz, means[n] / gmax, color=CLS[n], lw=1.3)
    ax.text(.995, .82, n, transform=ax.transAxes, ha="right", fontsize=12, fontweight="bold", color=INK)
    ax.set_ylim(0, 1.05); ax.set_yticks([]); ax.grid(axis="x", visible=False)
axs[-1].set_xlabel("m/z (mass-to-charge)"); axs[0].set_title("Each tissue has a different average lipid spectrum", loc="left", fontsize=12.5)
for ax in axs: ax.axvspan(800, 1000, color="#000", alpha=.025)
fig.tight_layout(); save(fig, "6_average_spectra.png")
print("wrote", sorted(p.name for p in OUT.glob("*.png")))
