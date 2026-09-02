"""Raw LLM JSON -> feature matrix.

Every count becomes a per-100-word rate, plus two derived ratios the V1
diagnostic pointed at directly: the balance of specific vs. generic verbs,
and how much of the scene (water / land / sky) the speaker refers to.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

LIST_FIELDS = {
    "n_entities":      "named_entities",
    "n_specific_verb": "specific_action_verbs",
    "n_generic_verb":  "generic_verbs",
    "n_locative":      "locative_expressions",
    "n_hedge":         "hedge_spans",
    "n_deictic":       "deictic_spans",
    "n_metacomment":   "metacomment_spans",
    "n_diminutive":    "diminutive_or_affective_forms",
    "n_quantity":      "quantity_expressions",
}
INT_FIELDS = {"n_proposition": "complete_propositions", "n_selfcorrect": "self_corrections"}

EVIDENCE_FIELDS = list(LIST_FIELDS.values())
REQUIRED_FIELDS = EVIDENCE_FIELDS + [
    "complete_propositions", "regions_referenced",
    "repeated_content_lemmas", "self_corrections",
]


def to_row(obj: dict) -> dict:
    r = {}
    for out, key in LIST_FIELDS.items():
        v = obj.get(key) or []
        r[out] = len(v) if isinstance(v, list) else 0
    for out, key in INT_FIELDS.items():
        v = obj.get(key, 0)
        r[out] = int(v) if isinstance(v, (int, float)) else 0
    reg = obj.get("regions_referenced") or []
    r["n_region"] = len(set(reg)) if isinstance(reg, list) else 0
    rep = obj.get("repeated_content_lemmas") or []
    r["n_repeated_lemma"] = sum(1 for x in rep if isinstance(x, dict) and x.get("lemma"))
    r["repeat_mass"] = sum(
        int(x.get("count", 0)) for x in rep
        if isinstance(x, dict) and str(x.get("count", "")).isdigit()
    )
    return r


def groundedness(obj: dict, text: str) -> float:
    """Share of quoted spans that literally occur in the transcript. NaN if
    the document produced no spans at all. Fabricated evidence means
    fabricated counts — this is the gate that decides whether anything
    downstream is trustworthy."""
    low, hit, tot = text.lower(), 0, 0
    for f in EVIDENCE_FIELDS:
        for s in obj.get(f, []) or []:
            if isinstance(s, str) and s.strip():
                tot += 1
                hit += s.strip().lower() in low
    return hit / tot if tot else np.nan


def missing_fields(obj: dict) -> list:
    return [f for f in REQUIRED_FIELDS if f not in obj]


def build_feature_matrix(docs: pd.DataFrame, ok: dict):
    """`ok` maps filename -> parsed response dict (successes only).
    Returns (df, COUNTS, RATES, RATIOS)."""
    feat = pd.DataFrame([{"file": f, **to_row(o)} for f, o in ok.items()])
    df = docs.merge(feat, on="file", how="inner")

    counts = [c for c in feat.columns if c != "file"]
    for c in counts:
        df[c + "_r100"] = 100 * df[c] / df["n_word"].clip(lower=1)

    df["specific_verb_ratio"] = df.n_specific_verb / (df.n_specific_verb + df.n_generic_verb).clip(lower=1)
    df["region_breadth"] = df.n_region / 3.0

    rates = [c + "_r100" for c in counts]
    ratios = ["specific_verb_ratio", "region_breadth"]
    return df, counts, rates, ratios


def degeneracy_gate(lab: pd.DataFrame, counts: list, zero_share_max: float) -> pd.DataFrame:
    """Per-feature zero_share and length-confound r^2 (vs n_word, for both
    the raw count and the rate). `degenerate` is True if zero_share exceeds
    zero_share_max on the labelled set — that is the ONLY automatic drop
    criterion; a feature can have a high r2_len_rate and still be kept, but
    it should be flagged 'LENGTH PROXY' when reporting the table."""
    def r2_vs_length(col):
        x, y = lab["n_word"], lab[col]
        if y.std() == 0:
            return np.nan
        return np.corrcoef(x, y)[0, 1] ** 2

    qc = pd.DataFrame({
        "zero_share":   [(lab[c] == 0).mean() for c in counts],
        "r2_len_count": [r2_vs_length(c) for c in counts],
        "r2_len_rate":  [r2_vs_length(c + "_r100") for c in counts],
    }, index=counts).round(3)
    qc["degenerate"] = qc.zero_share > zero_share_max
    return qc
