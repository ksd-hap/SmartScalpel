"""Inference service for SmartScalpel. SIMULATION ONLY - not a clinical diagnostic tool.

    from app.ml_service import load_model, predict_tissue, get_demo_spectrum
"""
from __future__ import annotations

import random
from pathlib import Path

import joblib
import numpy as np

from ml.scoring import scaled_proba

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "tissue_model.joblib"
DEMO_PATH = ROOT / "models" / "demo_samples.npz"
REGION_TYPES = ("tumour", "stroma", "adipose")      # classes the model predicts
DEMO_REGION_TYPES = ("tumour", "adipose")            # virtual regions used in the demo (stroma deliberately not simulated)

# LED mapping agreed by the team: two colours only. Stroma and rejected (low-confidence) results leave the LED off.
LED_COLOURS = {
    "tumour": ("red", (255, 0, 0)),
    "adipose": ("green", (0, 255, 0)),
    "stroma": ("off", (0, 0, 0)),
    "uncertain": ("off", (0, 0, 0)),
}

_model: dict | None = None
_demo: dict | None = None


def load_model(path: str | Path | None = None) -> dict:
    """Load the saved fitted pipeline artifact (cached). Raises FileNotFoundError if not trained yet."""
    global _model
    if path is not None or _model is None:
        p = Path(path) if path else MODEL_PATH
        if not p.exists():
            raise FileNotFoundError(f"Model not found at {p}. Run: py -m ml.train")
        _model = joblib.load(p)
    return _model


def predict_tissue(spectrum) -> dict:
    """Classify exactly one spectrum (1-D, length == model['n_features'], finite numbers).

    Raises ValueError for wrong size/shape or non-finite values.
    `scores` are temperature-scaled classifier scores, not clinical probabilities.
    """
    m = load_model()
    x = np.asarray(spectrum, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"Expected one 1-D spectrum, got shape {x.shape}")
    if x.shape[0] != m["n_features"]:
        raise ValueError(f"Expected {m['n_features']} features, got {x.shape[0]}")
    if not np.isfinite(x).all():
        raise ValueError("Spectrum contains NaN or infinite values")

    proba = scaled_proba(m["pipeline"], x.reshape(1, -1), m["temperature"])[0]
    names = [m["label_mapping"][c] for c in m["pipeline"].classes_]
    scores = {n: float(p) for n, p in zip(names, proba)}
    best = max(scores, key=scores.get)
    return {
        "predicted_class": best,
        "scores": {n: scores[n] for n in REGION_TYPES},
        "accepted": bool(scores[best] >= m["reject_threshold"]),
        "model_version": m["model_version"],
        "score_type": "temperature_scaled_model_score",
    }


def led_state(result: dict) -> dict:
    """Map a predict_tissue() result to the LED state sent to the hardware. Contains no scores or model details."""
    state = result["predicted_class"] if result["accepted"] else "uncertain"
    name, rgb = LED_COLOURS[state]
    return {"led_state": state, "colour": name, "rgb": list(rgb)}


def _load_demo() -> dict:
    global _demo
    if _demo is None:
        if not DEMO_PATH.exists():
            raise FileNotFoundError(f"Demo samples not found at {DEMO_PATH}. Run: py -m ml.train")
        d = np.load(DEMO_PATH, allow_pickle=False)
        _demo = {"X": d["X"], "region": [str(r) for r in d["region_type"]], "kind": [str(k) for k in d["kind"]]}
    return _demo


def get_demo_spectrum(region_type: str, index: int | None = None, seed: int | None = None, kind: str | None = "typical") -> dict:
    """Return a real held-out spectrum for a virtual region ('tumour'|'stroma'|'adipose').

    The region type only selects WHICH recorded spectrum to show; the true label is not returned and is
    never given to the classifier. `kind`: "typical" (default, clean examples), "uncertain" (shows rejection) or None for all.
    `index` picks a specific sample (modulo count); otherwise random.
    Returns {"sample_id": int, "spectrum": [8000 floats]}.
    """
    if region_type not in DEMO_REGION_TYPES:
        raise ValueError(f"region_type must be one of {DEMO_REGION_TYPES} (stroma is not used in the demo), got {region_type!r}")
    d = _load_demo()
    ids = [i for i, r in enumerate(d["region"]) if r == region_type and (kind is None or d["kind"][i] == kind)]
    if not ids:
        raise LookupError(f"No demo samples for {region_type}")
    pick = ids[index % len(ids)] if index is not None else random.Random(seed).choice(ids)
    return {"sample_id": int(pick), "spectrum": d["X"][pick].tolist()}


if __name__ == "__main__":
    for r in DEMO_REGION_TYPES:
        s = get_demo_spectrum(r, seed=0)
        print(r, "->", predict_tissue(s["spectrum"]))
