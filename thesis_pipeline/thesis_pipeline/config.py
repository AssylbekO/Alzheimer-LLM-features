"""Central configuration for the V2 LLM-feature pipeline.

Single source of truth for endpoint settings, model names, gate thresholds
and output paths. Every notebook imports from here instead of redefining
these constants — that is the whole point of this module: a threshold or
prompt-hash edit happens in exactly one place, not once per notebook.

NOTE on ZERO_SHARE_MAX: notebook 08's QC block used 0.85 as the "degenerate"
cutoff; notebook 09 tightened it to 0.70 and that is the value the frozen
pipeline was actually built with. This module uses 0.70 as canonical. If
that tightening was not a deliberate decision, revisit it — but pick one
value and let both notebooks import it, rather than letting them silently
disagree.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# .env is looked up from the current working directory upward, which covers
# "run from repo root" and "run from a notebooks/ subfolder".
for _candidate in (Path(".env"), Path("../.env"), Path("../../.env")):
    if _candidate.exists():
        load_dotenv(_candidate)
        break

BASE_URL = "https://litellm.vse.cz/v1"
API_KEY = os.environ.get("LOCAL_OPENAI_API_KEY", "")
if not API_KEY:
    raise RuntimeError(
        "LOCAL_OPENAI_API_KEY is not set. Put it in a .env file at the repo "
        "root (LOCAL_OPENAI_API_KEY=sk-...) and add .env to .gitignore. "
        "Never hardcode the key in a notebook or in this file."
    )

MODEL_A = "qwen3.5:122b"   # original V2 run (notebook 08)
MODEL_B = "qwen3.6-35b"    # second model, stability check (notebook 09)

TEMPERATURE = 0.0
MAX_TOKENS = 4096          # V2 returns long lists; 2048 truncates on verbose docs
TIMEOUT_S = 600

N_CALIB = 10                # docs used to sanity-check the prompt before a full run
SEED = 42

# --- automatic gate thresholds, asserted against / printed by every notebook ----
ZERO_SHARE_MAX = 0.70        # degeneracy gate: drop a count if zero_share exceeds this
STABILITY_MIN_RHO = 0.60     # "stable" if Spearman rho(model A, model B) >= this
EXPECTED_PROMPT_SHA = "f52aae5d939d620e"   # must match prompt.PROMPT_SHA


@dataclass(frozen=True)
class Paths:
    data: Path
    out: Path
    frozen_dir: Path
    raw_a: Path
    raw_b: Path
    meta_a: Path
    meta_b: Path

    @classmethod
    def under(cls, data_root: Path) -> "Paths":
        out = data_root / "outputs"
        out.mkdir(parents=True, exist_ok=True)
        frozen = out / "frozen_pipeline_v2"
        frozen.mkdir(parents=True, exist_ok=True)
        return cls(
            data=data_root,
            out=out,
            frozen_dir=frozen,
            raw_a=out / "v2_raw.jsonl",
            raw_b=out / "v2b_raw.jsonl",
            meta_a=out / "run_metadata.json",
            meta_b=out / "v2b_run_metadata.json",
        )
