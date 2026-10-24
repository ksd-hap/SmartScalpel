"""Temperature-scaled model scores. T is fitted on out-of-fold TRAINING predictions only."""
import numpy as np
from scipy.optimize import minimize_scalar


def scaled_proba(pipe, X, T=1.0):
    logp = np.log(np.clip(pipe.predict_proba(X), 1e-300, 1.0)) / T
    logp -= logp.max(1, keepdims=True)
    p = np.exp(logp)
    return p / p.sum(1, keepdims=True)


def fit_temperature(oof_proba, y, classes):
    """Minimise NLL of true class over T in [1, 1e4] using out-of-fold probabilities."""
    idx = np.array([classes.index(c) for c in y])
    logp = np.log(np.clip(oof_proba, 1e-300, 1.0))

    def nll(logT):
        z = logp / np.exp(logT)
        z -= z.max(1, keepdims=True)
        lp = z - np.log(np.exp(z).sum(1, keepdims=True))
        return -lp[np.arange(len(y)), idx].mean()

    r = minimize_scalar(nll, bounds=(0, np.log(1e4)), method="bounded")
    return float(np.exp(r.x))
