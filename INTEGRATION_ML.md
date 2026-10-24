# SmartScalpel ML – integration notes (Person 1)

SIMULATION ONLY. Trained on published ex-vivo REIMS spectra (Sci Rep 2020). Not clinically validated. The Apple Pencil does
not sense tissue, and the model does not locate tumours; it classifies one recorded spectrum at a time.

## For Person 4 (backend)
- Packages: `py -m pip install -r requirements-ml.txt` (numpy, pandas, scikit-learn 1.9.1, scipy, joblib, openpyxl, pytest).
  Use the same scikit-learn version as the one that trained the model (stored in the artifact as `sklearn_version`).
- Artefacts (commit these): `models/tissue_model.joblib`, `models/demo_samples.npz`, `reports/evaluation.json`.
- Needs the `ml/` package importable (`ml/scoring.py`) - run the backend from the repo root.
- Load/predict:
  ```python
  from app.ml_service import load_model, predict_tissue, get_demo_spectrum
  load_model()                                   # once at startup
  s = get_demo_spectrum("tumour")                # {"sample_id": int, "spectrum": [8000 floats]}
  result = predict_tissue(s["spectrum"])
  ```
- Input: exactly ONE spectrum, 1-D, **8000** finite floats (m/z 200.05-999.95). Wrong size/NaN raises `ValueError` -> return HTTP 422.
- Output:
  ```json
  {"predicted_class": "tumour|stroma|adipose",
   "scores": {"tumour": 0.0, "stroma": 0.0, "adipose": 0.0},
   "accepted": true, "model_version": "pca-lda-v1",
   "score_type": "temperature_scaled_model_score"}
  ```
  `scores` are temperature-scaled model scores (sum to 1), NOT clinical probabilities. `accepted=false` means the model is too
  unsure (top score < 0.82): show "uncertain" and tell Person 3 to use the neutral/off colour.
- Latency: median ~0.5 ms, p95 ~0.5 ms per spectrum on the dev laptop (measured in `reports/evaluation.json`).
- Demo spectra: 5 per region type (tumour/stroma/adipose), from held-out patients the model never trained on.
  `get_demo_spectrum(region, index=None)`; `index` picks one deterministically, otherwise random. The true label is never returned.

## For Person 2 (iPad/frontend)
Do not call the model directly. Ask the backend for a region's spectrum (or have the backend do it), then show `predicted_class`
and, if you wish, `scores`. Treat `accepted: false` as "uncertain". Suggested wording: "Model classification of a recorded
spectrum (simulation)", not "cancer detected".

## For Person 3 (LED)
Receives only the LED state from the backend, never spectra or scores. Suggested mapping is the team's choice; the ML side
only provides `predicted_class` and `accepted`.

## Expected performance (quote these, with the caveats)
- Best estimate: nested patient-grouped 6-fold CV over all 72 patients (`py -m ml.cv_report`): accuracy **88.7%**, balanced
  accuracy 88.5%, macro F1 0.885; recall tumour 91.5%, stroma 83.7%, adipose 90.2%. Fold-to-fold balanced accuracy spread: +/-4 points.
- The saved model was also checked on one fixed hold-out of 18 patients (160 spectra): accuracy 92.5%, tumour recall 96.9%.
  That split was somewhat favourable; do not quote it as the headline number.
- Both are on ex-vivo research spectra from one published dataset. Not clinically validated.
