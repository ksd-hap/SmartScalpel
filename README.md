# SmartScalpel Live

An AI-powered **surgical simulation** built for a healthcare AI hackathon. On an iPad, a user moves an Apple Pencil across virtual
breast tissue. For each region the backend retrieves a **real, recorded REIMS mass spectrum**, a trained model classifies it as
**tumour, stroma or adipose**, and an RGB LED changes colour to match.

> **Simulation only. Not a clinical or diagnostic tool.** The pencil does not sense tissue and the LED never measures anything.
> The model classifies pre-recorded *ex-vivo* spectra from a published dataset. It does not find tumour locations and is not
> clinically validated.

## How it works

```
Apple Pencil on iPad  ->  (x, y)  ->  backend maps position to a virtual region (tumour / stroma / adipose zone)
   ->  real held-out REIMS spectrum for that region  ->  ML model classifies it independently
   ->  result shown on iPad  +  LED colour sent to the Arduino
```

The region type only chooses *which recorded spectrum* to show. The model never receives the true label.

## ML results

Dataset: Sci Rep 2020 ([article](https://www.nature.com/articles/s41598-020-77102-1)), 689 spectra from 72 patients, 8,000 m/z features.
Model: standardise -> PCA (100 components) -> shrinkage LDA, with an uncertainty-rejection threshold (`pca-lda-v2`).

| Evaluation | Accuracy | Tumour recall | Stroma recall | Adipose recall |
|---|---|---|---|---|
| **Nested patient-grouped CV, all 72 patients** (best estimate) | **91.1%** | 95.1% | 86.6% | 91.4% |
| Fixed hold-out, 18 unseen patients (160 spectra) | 91.2% | 93.8% | 93.6% | 85.4% |

- No patient appears in both training and test data (checked by tests).
- Scores are temperature-scaled model scores, **not clinical probabilities**. Results below the 0.77 threshold are marked *uncertain*.
- A shuffled-label control gives chance-level accuracy (33.8%), so the result is not a pipeline artefact.
- Latency is about 1 ms per spectrum. Fold-to-fold spread is about +/-2 points; 72 patients is a small sample.
- Demo spectra are curated (typical, correctly classified, plus a few deliberately uncertain ones) and do not represent overall accuracy.

Full numbers: `reports/evaluation.json`. Reproduce them with `py -m ml.validate`.

## LED output (proposed mapping)

| Model result | LED |
|---|---|
| tumour | red (255, 0, 0) |
| stroma | blue (0, 0, 255) |
| adipose | yellow (255, 180, 0) |
| uncertain (low score) | white |

The hardware receives only the LED state, never spectra or scores. An Arduino Mega has no Wi-Fi, so it connects to the laptop by USB serial.

## Quick start

```bash
git clone https://github.com/ksd-hap/SmartScalpel.git
cd SmartScalpel
git checkout feature/ml-engine
py -m pip install -r requirements-ml.txt
py -c "from app.ml_service import *; r = predict_tissue(get_demo_spectrum('tumour')['spectrum']); print(r, led_state(r))"
```

No dataset or retraining is needed; the trained model is in `models/`. Use scikit-learn 1.9.1 (installed by the requirements file).

```python
from app.ml_service import load_model, predict_tissue, get_demo_spectrum, led_state

load_model()                                  # once at startup
s = get_demo_spectrum("stroma")               # {"sample_id": ..., "spectrum": [8000 floats]}
result = predict_tissue(s["spectrum"])        # exactly one 1-D spectrum, 8000 finite floats
# {"predicted_class": "stroma", "scores": {...}, "accepted": True, "model_version": "pca-lda-v2", ...}
led = led_state(result)                       # {"led_state": "stroma", "colour": "blue", "rgb": [0, 0, 255]}
```

Wrong input size or NaN values raise `ValueError`. Per-role integration notes are in [`INTEGRATION_ML.md`](INTEGRATION_ML.md).

## Repository layout

| Path | Purpose |
|---|---|
| `ml/load_data.py` | Load and validate the Excel dataset |
| `ml/train.py` | Patient-grouped training, model comparison, saves model and demo samples |
| `ml/evaluate.py`, `ml/cv_report.py`, `ml/validate.py` | Metrics, nested CV, one-command validation |
| `ml/experiments.py`, `ml/export_csv.py`, `ml/scoring.py` | Variant comparison, CSV export, score scaling |
| `app/ml_service.py` | Inference API used by the backend |
| `models/` | Trained pipeline and demo spectra |
| `reports/` | Evaluation and experiment results |
| `tests/` | Automated tests |

## Retraining and testing

The raw dataset is not in the repo (75 MB). Download `41598_2020_77102_MOESM3_ESM.xlsx` (supplementary file 3) from the article into `data/`, then:

```bash
py -m ml.train          # retrain, overwrite models/ and reports/
py -m ml.validate       # print all validation numbers
py -m pytest tests -v   # run tests
```

## Team

Person 1: machine learning and AI. Person 2: iPad simulation frontend. Person 3: LED hardware. Person 4: Python backend and API.
