"""CV protocol shared by every comparison table in the thesis.

10 independent StratifiedKFold(5) splits, averaged out-of-fold
probabilities — NOT RepeatedStratifiedKFold, which cross_val_predict
refuses (each document would appear in 10 test folds, which is not a true
partition). AUC is the primary metric, reported with a bootstrap 95% CI.
paired_bootstrap() is the decisive test for "does A beat B": use it instead
of comparing marginal CIs, which overstates significance for a paired
design where both models are scored on the same documents.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

N_REPEATS_DEFAULT = 10
N_SPLITS_DEFAULT = 5


def cv_proba(model, X, y, seed: int, n_repeats: int = N_REPEATS_DEFAULT,
             n_splits: int = N_SPLITS_DEFAULT) -> np.ndarray:
    """Out-of-fold probabilities, averaged over n_repeats independent
    n_splits-fold splits."""
    acc = np.zeros(len(y), dtype=float)
    for r in range(n_repeats):
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
        acc += cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    return acc / n_repeats


def boot_ci(y_true, p, n: int = 2000, seed: int = 42):
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n):
        i = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[i])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[i], p[i]))
    return np.percentile(aucs, [2.5, 97.5])


def evaluate(name: str, X, y, seed: int, model=None) -> dict:
    """Returns a result dict including the raw out-of-fold probabilities
    under "_proba" — keep them if you plan to run paired_bootstrap() against
    another arm, so both contrasts use the exact same fold assignment."""
    model = model or make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")
    )
    p = cv_proba(model, X, y, seed=seed)
    lo, hi = boot_ci(y, p, seed=seed)
    return {
        "model": name,
        # 2-D matrix -> column count; raw-text input to a pipeline with a
        # vectoriser (1-D) -> "text"
        "n_feat": (X.shape[1] if getattr(X, "ndim", None) == 2 else "text"),
        "AUC": roc_auc_score(y, p),
        "CI_low": lo, "CI_high": hi,
        "macroF1": f1_score(y, p > 0.5, average="macro"),
        "balAcc": balanced_accuracy_score(y, p > 0.5),
        "_proba": p,
    }


def paired_bootstrap(p_a: np.ndarray, p_b: np.ndarray, y, n: int = 2000, seed: int = 42):
    """Paired bootstrap contrast on out-of-fold probabilities from the same
    fold assignment: mean dAUC (a - b), its 95% CI, and P(dAUC <= 0).

    Returns (mean_delta, (ci_low, ci_high), p_delta_leq_0).
    """
    rng = np.random.default_rng(seed)
    n_obs = len(y)
    deltas = []
    for _ in range(n):
        i = rng.integers(0, n_obs, n_obs)
        if len(np.unique(y[i])) < 2:
            continue
        deltas.append(roc_auc_score(y[i], p_a[i]) - roc_auc_score(y[i], p_b[i]))
    deltas = np.array(deltas)
    mean_delta = float(np.mean(deltas))
    ci = tuple(np.percentile(deltas, [2.5, 97.5]))
    p_leq_0 = float(np.mean(deltas <= 0))
    return mean_delta, ci, p_leq_0
