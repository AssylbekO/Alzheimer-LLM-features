"""Cross-model feature agreement: Spearman rank correlation, ICC(2,1)
absolute agreement, and exact-match rate.

Spearman alone can hide a systematic level shift between two models — they
can rank documents identically while disagreeing sharply on the actual
count (this is exactly what happened with n_entities: 122b returns ~0,
35b returns ~9, so Spearman collapses to ~0 even though the direction of
every other feature agreed). ICC(2,1) — two-way random effects, single
measures, absolute agreement (Shrout & Fleiss, 1979) — is included
specifically to catch that.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def icc_2_1(a: np.ndarray, b: np.ndarray) -> float:
    """ICC(2,1): two-way random effects, single measures, absolute
    agreement. a, b are paired measurements on the same subjects (documents)
    from two 'raters' (two models)."""
    ratings = np.column_stack([a, b]).astype(float)
    n, k = ratings.shape  # n subjects, k=2 raters
    grand_mean = ratings.mean()

    row_means = ratings.mean(axis=1)
    col_means = ratings.mean(axis=0)

    ss_total = ((ratings - grand_mean) ** 2).sum()
    ss_rows = k * ((row_means - grand_mean) ** 2).sum()
    ss_cols = n * ((col_means - grand_mean) ** 2).sum()
    ss_error = ss_total - ss_rows - ss_cols

    ms_rows = ss_rows / (n - 1)
    ms_cols = ss_cols / (k - 1)
    ms_error = ss_error / ((n - 1) * (k - 1)) if (n - 1) * (k - 1) else np.nan

    if not np.isfinite(ms_error):
        return np.nan
    denom = ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n
    if denom == 0:
        return np.nan
    return (ms_rows - ms_error) / denom


def stability_table(dfA: pd.DataFrame, dfB: pd.DataFrame, features: list,
                     key: str, stability_min_rho: float,
                     count_features: list | None = None) -> pd.DataFrame:
    """dfA, dfB: one row per document, one column per feature. Returns one
    row per feature indexed by feature name:
    spearman, pearson, icc21, exact_match, medA, medB, mean_abs_diff, and a
    boolean `stable` flag at spearman >= stability_min_rho.

    `exact_match` is the isclose-agreement rate for integer count features
    and NaN for rates/ratios (comparing per-100-word rates for exact
    equality is meaningless). Pass `count_features` to mark which columns
    are counts; if omitted, exact_match is computed for every feature.

    A feature with zero variance in both runs gets stable=False rather than
    a divide-by-zero NaN treated as True — a constant feature is not
    "stable", it is uninformative.
    """
    counts = set(features if count_features is None else count_features)
    m = dfA[[key] + features].merge(dfB[[key] + features], on=key, suffixes=("_a", "_b"))
    rows = []
    for f in features:
        a = m[f + "_a"].astype(float).values
        b = m[f + "_b"].astype(float).values
        sa, sb = np.std(a), np.std(b)
        both_vary = sa > 0 and sb > 0
        has_variance = sa > 0 or sb > 0
        rho = stats.spearmanr(a, b).correlation if both_vary else np.nan
        pear = float(np.corrcoef(a, b)[0, 1]) if both_vary else np.nan
        icc = icc_2_1(a, b) if has_variance else np.nan
        rows.append({
            "feature": f,
            "spearman": rho,
            "pearson": pear,
            "icc21": icc,
            "exact_match": float(np.mean(np.isclose(a, b))) if f in counts else np.nan,
            "medA": float(np.median(a)),
            "medB": float(np.median(b)),
            "mean_abs_diff": float(np.mean(np.abs(a - b))),
            "stable": bool(pd.notna(rho) and rho >= stability_min_rho),
        })
    return pd.DataFrame(rows).set_index("feature")
