"""Resumable JSONL extraction.

Each response is appended to the raw JSONL file and flushed immediately, so
a crash costs at most the one document being written when it happened, not
the run so far. seal_last_line() guards the one way this can still corrupt
data: a crash that leaves a line without a trailing newline, onto which the
next append would otherwise be silently concatenated.
"""
from __future__ import annotations

import datetime
import json
import time
from pathlib import Path
from typing import Optional

import pandas as pd


def load_raw(raw_path: Path) -> dict:
    """Read the JSONL, skipping any line left half-written by a crash."""
    out = {}
    if raw_path.exists():
        for line in raw_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    rec = json.loads(line)
                    out[rec["file"]] = rec
                except Exception:
                    pass
    return out


def seal_last_line(raw_path: Path) -> None:
    """If a crash left the file without a trailing newline, the next append
    would be concatenated onto that broken line and silently destroy a
    second record. Terminate it first."""
    if raw_path.exists() and raw_path.stat().st_size:
        with raw_path.open("rb+") as fh:
            fh.seek(-1, 2)
            if fh.read(1) != b"\n":
                fh.write(b"\n")


def extract_one(provider, prompt: str, text: str) -> dict:
    return provider.text_features([text], prompt=prompt)[0]


def run_resumable_extraction(docs: pd.DataFrame, provider, prompt: str, model: str,
                              prompt_sha: str, raw_path: Path,
                              progress_every: int = 10) -> dict:
    """Extract every row in `docs` not already present in raw_path, appending
    as it goes. Safe to interrupt (Ctrl-C, kernel crash, Colab disconnect)
    and rerun — already-done documents are skipped."""
    seal_last_line(raw_path)
    done = load_raw(raw_path)
    todo = docs[~docs.file.isin(done)]
    print(f"already done: {len(done)}   remaining: {len(todo)}")

    t0 = time.time()
    with raw_path.open("a", encoding="utf-8") as fh:
        for i, (_, r) in enumerate(todo.iterrows(), 1):
            rec = {
                "file": r.file,
                "label": (None if pd.isna(r.label) else int(r.label)),
                "model": model,
                "prompt_sha": prompt_sha,
            }
            try:
                rec["response"] = extract_one(provider, prompt, r.text)
                rec["error"] = None
            except Exception as e:
                rec["response"] = None
                rec["error"] = f"{type(e).__name__}: {e}"[:300]
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()   # survive a kernel crash
            if i % progress_every == 0 or i == len(todo):
                el = time.time() - t0
                remaining = (el / i) * (len(todo) - i) if i else 0.0
                print(f"  {i}/{len(todo)}  {el/60:.1f} min elapsed, "
                      f"~{remaining/60:.1f} min left", flush=True)

    return load_raw(raw_path)


def write_run_metadata(meta_path: Path, *, model: str, base_url: str, temperature: float,
                        max_tokens: int, prompt_sha: str, prompt_chars: int,
                        n_documents_sent: int, n_succeeded: int,
                        n_labelled_for_modelling: int = 241,
                        n_unlabelled_extracted: int = 86,
                        test_set_used: bool = False,
                        extra: Optional[dict] = None) -> None:
    payload = {
        "model": model, "base_url": base_url, "temperature": temperature,
        "max_tokens": max_tokens, "streaming": True,
        "prompt_version": "v2", "prompt_sha256_16": prompt_sha, "prompt_chars": prompt_chars,
        "n_documents_sent": n_documents_sent, "n_succeeded": n_succeeded,
        "n_failed": n_documents_sent - n_succeeded,
        "n_labelled_for_modelling": n_labelled_for_modelling,
        "n_unlabelled_extracted": n_unlabelled_extracted,
        "test_set_used": test_set_used,
        "run_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }
    if extra:
        payload.update(extra)
    meta_path.write_text(json.dumps(payload, indent=2))
