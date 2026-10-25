"""Export a tidy hand-off package of the held-out test patients (never used for training or model selection).
Run: py -m ml.export_csv  ->  exports/teammate_package/
"""
import joblib
import numpy as np
import pandas as pd

from app.ml_service import LED_COLOURS
from ml.evaluate import MODEL_PATH
from ml.load_data import LABEL_NAMES, ROOT, load_dataset
from ml.scoring import scaled_proba

OUT = ROOT / "exports" / "teammate_package"

README = """SmartScalpel - held-out test package (simulation only, not clinical)

All spectra here come from 18 patients the model never saw in training or model selection.

spectra_input.csv            sample_id + 8000 intensity columns (header = m/z). This is the ONLY model input. No labels.
answers.csv                  sample_id, patient, true_label. For checking results only - never feed to the model.
model_predictions.csv        Saved model output per sample: predicted_class, three scores, accepted, LED state/colour, correct.
demo_samples.csv             10 curated demo spectra (tumour and adipose only; stroma is not simulated) (region_type = the virtual region the simulation selects them for;
                             kind = typical / uncertain). Same 8000 columns.

Use: py -c "import pandas as pd; from app.ml_service import predict_tissue; d=pd.read_csv('spectra_input.csv'); print(predict_tissue(d.iloc[0,1:].to_numpy()))"
(Run from the repo root; the first column is sample_id, so drop it.)
LED mapping: tumour=red, adipose=green, stroma=off, uncertain (rejected)=off.
"""


def main():
    art = joblib.load(MODEL_PATH)
    ds = load_dataset()
    OUT.mkdir(parents=True, exist_ok=True)  # overwrite files in place (rmtree fails on OneDrive-locked folders)
    cols = [f"{m:.2f}" for m in ds.mz]
    te = np.where(np.isin(ds.patients, art["test_patients"]))[0]
    true = [LABEL_NAMES[c] for c in ds.y[te]]

    pd.concat([pd.DataFrame({"sample_id": te}), pd.DataFrame(ds.X[te], columns=cols)], axis=1).to_csv(
        OUT / "spectra_input.csv", index=False, float_format="%.10g")
    pd.DataFrame({"sample_id": te, "patient": ds.patients[te], "true_label": true}).to_csv(OUT / "answers.csv", index=False)

    p = scaled_proba(art["pipeline"], ds.X[te], art["temperature"])
    names = [LABEL_NAMES[c] for c in art["pipeline"].classes_]
    pred = [names[i] for i in p.argmax(1)]
    acc = p.max(1) >= art["reject_threshold"]
    state = [pr if a else "uncertain" for pr, a in zip(pred, acc)]
    out = pd.DataFrame({"sample_id": te, "true_label": true, "predicted_class": pred})
    for j, n in enumerate(names):
        out[f"score_{n}"] = p[:, j].round(4)
    out["accepted"] = acc
    out["led_state"] = state
    out["led_colour"] = [LED_COLOURS[s][0] for s in state]
    out["correct"] = out.true_label == out.predicted_class
    out.to_csv(OUT / "model_predictions.csv", index=False)

    d = np.load(ROOT / "models" / "demo_samples.npz", allow_pickle=True)
    keep = np.isin(d["region_type"], ["tumour", "adipose"])  # stroma is not simulated in the demo
    dm = pd.DataFrame({"demo_index": range(int(keep.sum())), "region_type": d["region_type"][keep], "kind": d["kind"][keep]})
    pd.concat([dm, pd.DataFrame(d["X"][keep], columns=cols)], axis=1).to_csv(OUT / "demo_samples.csv", index=False, float_format="%.10g")
    (OUT / "README.txt").write_text(README)
    print(f"{len(te)} held-out spectra, {len(set(ds.patients[te]))} patients -> {OUT}")


if __name__ == "__main__":
    main()
