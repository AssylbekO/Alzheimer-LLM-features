"""Dataset discovery and loading.

Nothing in here reads test/ except load_test(), and load_test() exists as a
deliberately separate, obviously-named function so a notebook that only
imports load_labelled_and_unlabelled can never accidentally touch the
sealed test set.
"""
from __future__ import annotations

import re
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def find_dataset(extra_candidates: Optional[list] = None) -> Path:
    """Locate fileDataset/, unzipping fileDataset.zip if that is all we have.

    Add machine-specific paths (e.g. a local OneDrive sync folder) via
    extra_candidates=[Path(...)] from the calling notebook rather than
    editing this function — that keeps machine-specific paths out of the
    shared package.
    """
    candidates = [
        Path("fileDataset"), Path("../fileDataset"), Path("data/fileDataset"),
        Path.home() / "fileDataset",
        *(extra_candidates or []),
    ]
    for c in candidates:
        if (c / "train").is_dir():
            return c.resolve()
    for z in [Path("fileDataset.zip"), Path("../fileDataset.zip")]:
        if z.exists():
            zipfile.ZipFile(z).extractall(z.parent)
            if (z.parent / "fileDataset" / "train").is_dir():
                return (z.parent / "fileDataset").resolve()
    raise FileNotFoundError(
        "fileDataset/ not found. Put the folder (or fileDataset.zip) next to "
        "this notebook, or pass its location via extra_candidates=[Path(...)]."
    )


def read_split(folder: Path, label) -> list:
    rows = []
    for f in sorted(folder.glob("*.txt")):
        rows.append({
            "file": f.name,
            "label": label,
            "text": f.read_text(encoding="utf-8", errors="replace").strip(),
        })
    return rows


_WORD = re.compile(r"\w+", re.UNICODE)


def surface(t: str) -> dict:
    """Cheap surface statistics: the confound every LLM feature is measured
    against (per-100-word rates), and also the 'surface stats' comparison
    arm in the classifier tables.

    Formulas are byte-faithful to notebooks 08/09 so that training-time and
    Step-B test-time surface stats can never diverge on a transcript quirk:
    n_word is a whitespace token count floored at 1 (NOT \\w+), n_type is a
    distinct-\\w+ count on the lowercased text, n_sent is the number of
    [.!?]+ runs floored at 1.
    """
    n_word = max(1, len(t.split()))
    n_type = len(set(_WORD.findall(t.lower())))
    n_sent = max(1, len(re.findall(r"[.!?]+", t)))
    return {
        "n_char": len(t),
        "n_word": n_word,
        "n_type": n_type,
        "ttr": n_type / n_word,
        "n_sent": n_sent,
        "mlu": n_word / n_sent,
        "n_comma": t.count(","),
    }


SURFACE_COLS = ["n_word", "n_type", "ttr", "n_sent", "mlu", "n_comma", "n_char"]


def load_labelled_and_unlabelled(data_root: Path) -> pd.DataFrame:
    """241 labelled train/ + 86 unlabelled overview/ = 327 docs. test/ is
    never referenced in this function — the two asserts below are the
    guardrail, not a formality; they exist to fail loudly if the folder
    layout ever changes."""
    rows = read_split(data_root / "train" / "negative", 0)
    rows += read_split(data_root / "train" / "positive", 1)
    rows += read_split(data_root / "overview", None)
    docs = pd.DataFrame(rows)

    assert not any("test" in str(p) for p in [data_root / "train", data_root / "overview"])
    n_lab = len(docs[docs.label.notna()])
    n_unlab = len(docs[docs.label.isna()])
    assert n_lab == 241, f"expected 241 labelled, got {n_lab}"
    assert n_unlab == 86, f"expected 86 unlabelled, got {n_unlab}"

    docs = pd.concat([docs, docs.text.apply(lambda t: pd.Series(surface(t)))], axis=1)
    return docs


def load_test(data_root: Path) -> pd.DataFrame:
    """The 61 sealed test transcripts. Call this ONLY from the one-shot test
    evaluation notebook, and only after the pipeline is frozen. Do not
    import this function into any notebook used before that point.

    NOTE: assumes test/negative and test/positive mirror the train/ layout.
    Verify this against your actual fileDataset/test/ structure before
    running — adjust the two read_split() calls below if it differs.
    """
    rows = read_split(data_root / "test" / "negative", 0)
    rows += read_split(data_root / "test" / "positive", 1)
    docs = pd.DataFrame(rows)
    assert len(docs) == 61, f"expected 61 test docs, got {len(docs)}"

    docs = pd.concat([docs, docs.text.apply(lambda t: pd.Series(surface(t)))], axis=1)
    return docs
