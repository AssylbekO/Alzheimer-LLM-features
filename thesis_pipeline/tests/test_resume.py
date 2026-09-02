"""Verify: crash mid-extraction -> valid JSONL -> resume skips completed
docs and repairs a truncated last line rather than corrupting it.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from thesis_pipeline.extraction import load_raw, seal_last_line


def _make_docs() -> pd.DataFrame:
    return pd.DataFrame({
        "file": [f"d{i}.txt" for i in range(20)],
        "label": [0] * 10 + [1] * 10,
        "text": ["x"] * 20,
    })


def _run(raw_path: Path, docs: pd.DataFrame, crash_at: int = None):
    seal_last_line(raw_path)
    done = load_raw(raw_path)
    todo = docs[~docs.file.isin(done)]
    n = 0
    with raw_path.open("a", encoding="utf-8") as fh:
        for i, (_, r) in enumerate(todo.iterrows(), 1):
            if crash_at and i == crash_at:
                fh.write('{"file": "PARTIAL_')          # simulate a kill mid-write
                fh.flush()
                raise KeyboardInterrupt("simulated crash")
            fh.write(json.dumps({"file": r.file, "response": {"ok": 1}, "error": None}) + "\n")
            fh.flush()
            n += 1
    return len(done), n


def test_clean_run_extracts_everything(tmp_path):
    raw = tmp_path / "raw.jsonl"
    _run(raw, _make_docs())
    assert len(load_raw(raw)) == 20


def test_crash_then_resume_recovers_all_documents(tmp_path):
    raw = tmp_path / "raw.jsonl"
    docs = _make_docs()

    with pytest.raises(KeyboardInterrupt):
        _run(raw, docs, crash_at=8)
    # 7 valid records were flushed before the crash, plus one truncated
    # line with no trailing newline that load_raw() must skip, not parse.
    assert len(load_raw(raw)) == 7

    d0, n0 = _run(raw, docs)
    assert d0 == 7            # resumed knowing about the 7 already-done docs
    assert n0 == 13            # wrote exactly the remaining 13
    assert len(load_raw(raw)) == 20   # nothing lost, nothing duplicated
