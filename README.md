# SmartScalpel Live

Healthcare AI hackathon project: iPad virtual breast-tissue simulation + Apple Pencil -> Python backend -> REIMS spectrum
classifier -> RGB LED (Arduino Mega). **Simulation only; not a clinical tool.** The pencil does not sense tissue.

## ML engine (`ml/`, `app/ml_service.py`)
Setup: `py -m pip install -r requirements-ml.txt`

Dataset: download `41598_2020_77102_MOESM3_ESM.xlsx` from https://www.nature.com/articles/s41598-020-77102-1 (supplementary
file 3) into `data/` (git-ignored).

```
py -m ml.train      # train + save models/ and reports/
py -m ml.evaluate   # reproduce held-out metrics
py -m pytest tests -v
```
Integration details for teammates: see `INTEGRATION_ML.md`. Trained artefacts are committed so nobody has to retrain.
