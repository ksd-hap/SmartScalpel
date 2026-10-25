"""Run: py -m pytest tests -v
Synthetic tests always run. Tests marked `real` need the real dataset in data/ and a trained model."""
import numpy as np
import pandas as pd
import pytest

from app import ml_service
from ml import load_data, train
from ml.evaluate import compute_metrics

NF = 60
SMALL = {"n_spectra": 240, "n_patients": 24, "n_features": NF}
real = pytest.mark.skipif(not load_data.DEFAULT_PATH.exists(), reason="real dataset not present")


def make_xlsx(path, seed=0, bad_label=False):
    rng = np.random.default_rng(seed)
    rows, centres = [], {c: rng.normal(0, 1, NF) * 2 for c in "TFA"}
    for p in range(24):
        for c in "TFA":
            for _ in range(3 if c != "A" else 4):
                rows.append([c, f"P{p:02d}", *(centres[c] + rng.normal(0, 1, NF))])
    cols = ["Label", "Patient"] + [str(100 + i * 0.5) for i in range(NF)]
    df = pd.DataFrame(rows[:240] if len(rows) > 240 else rows, columns=cols)
    if bad_label:
        df.iloc[0, 0] = "X"
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        df.to_excel(w, sheet_name=load_data.SHEET, index=False)
    return df


@pytest.fixture(scope="module")
def ds(tmp_path_factory):
    p = tmp_path_factory.mktemp("d") / "syn.xlsx"
    make_xlsx(p)
    return load_data.load_dataset(p, expected={"n_spectra": 240, "n_patients": 24, "n_features": NF})


def test_loader_schema(ds):
    assert ds.X.shape == (240, NF) and ds.mz.shape == (NF,) and ds.report["mz_from_headers"]
    assert set(ds.y) == {"T", "F", "A"} and np.all(np.diff(ds.mz) > 0)


def test_loader_rejects_bad_label(tmp_path):
    p = tmp_path / "bad.xlsx"
    make_xlsx(p, bad_label=True)
    with pytest.raises(ValueError, match="Unexpected label"):
        load_data.load_dataset(p, expected=SMALL)


def test_loader_rejects_wrong_structure(tmp_path):
    p = tmp_path / "s.xlsx"
    make_xlsx(p)
    with pytest.raises(ValueError, match="published structure"):
        load_data.load_dataset(p)  # default expects 689/72/8000


def test_class_mapping():
    assert load_data.LABEL_NAMES == {"T": "tumour", "F": "stroma", "A": "adipose"}


def test_no_patient_leakage(ds):
    tr, te, _ = train.split_patients(ds)
    assert not set(ds.patients[tr]) & set(ds.patients[te])
    assert set(ds.y[te]) == {"T", "F", "A"}


@pytest.fixture(scope="module")
def trained(ds):
    return train.train(ds, n_demo=2, save=False)


def test_metrics_reproducible_and_no_invented_scores(ds, trained):
    art, te = trained["artifact"], trained["test_idx"]
    m = compute_metrics(art["pipeline"], ds.X[te], ds.y[te], ds.patients[te], art["reject_threshold"], with_latency=False)
    for k in ("accuracy", "balanced_accuracy", "macro_f1", "tumour_recall"):
        assert m[k] == pytest.approx(art["evaluation"][k])
    proba = art["pipeline"].predict_proba(ds.X[te])
    assert np.allclose(proba.sum(1), 1) and (proba >= 0).all()


def test_service_contract(ds, trained, monkeypatch, tmp_path):
    import joblib
    joblib.dump(trained["artifact"], tmp_path / "m.joblib")
    monkeypatch.setattr(ml_service, "_model", None)
    ml_service.load_model(tmp_path / "m.joblib")  # reload from disk, no retraining
    x = ds.X[0]
    r1, r2 = ml_service.predict_tissue(x), ml_service.predict_tissue(x.tolist())
    assert r1 == r2  # deterministic
    assert set(r1) >= {"predicted_class", "scores", "accepted", "model_version"}
    assert set(r1["scores"]) == {"tumour", "stroma", "adipose"}
    assert sum(r1["scores"].values()) == pytest.approx(1)
    assert r1["predicted_class"] == max(r1["scores"], key=r1["scores"].get)
    for bad in (x[:-1], np.append(x, 1.0), x.reshape(1, -1), np.full(NF, np.nan)):
        with pytest.raises(ValueError):
            ml_service.predict_tissue(bad)


def test_demo_samples_have_no_label_leak():
    import inspect
    assert "label" not in inspect.signature(ml_service.predict_tissue).parameters
    with pytest.raises(ValueError):
        ml_service.get_demo_spectrum("bone")
    with pytest.raises(ValueError):
        ml_service.get_demo_spectrum("stroma")  # not used in the demo


@real
def test_real_dataset_structure():
    ds = load_data.load_dataset()
    assert ds.X.shape == (689, 8000) and ds.report["n_patients"] == 72
    assert ds.report["problems"] == [] or all("found" not in p for p in ds.report["problems"])


@real
def test_real_saved_model_roundtrip():
    ml_service._model = None
    m = ml_service.load_model()
    assert m["n_features"] == 8000
    for region in ml_service.DEMO_REGION_TYPES:
        s = ml_service.get_demo_spectrum(region, index=0)
        out = ml_service.predict_tissue(s["spectrum"])
        assert out == ml_service.predict_tissue(s["spectrum"])


def test_led_state_mapping():
    base = {"scores": {}, "model_version": "x"}
    assert ml_service.led_state({**base, "predicted_class": "tumour", "accepted": True})["rgb"] == [255, 0, 0]
    assert ml_service.led_state({**base, "predicted_class": "adipose", "accepted": True})["rgb"] == [0, 255, 0]
    st = ml_service.led_state({**base, "predicted_class": "stroma", "accepted": True})
    assert st["led_state"] == "stroma" and st["rgb"] == [0, 0, 0]
    u = ml_service.led_state({**base, "predicted_class": "tumour", "accepted": False})
    assert u["led_state"] == "uncertain" and u["rgb"] == [0, 0, 0] and set(u) == {"led_state", "colour", "rgb"}
