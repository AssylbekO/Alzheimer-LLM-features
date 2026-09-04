"""Shared code for the V2 LLM-feature thesis notebooks (08-11).

One flat module, deliberately: a reader should be able to open this file and
read it top to bottom in a few minutes. Only three things live here, and only
because keeping them identical across notebooks is correctness-critical:

  1. config constants + the frozen V2 prompt (one edit point for anything that
     would change the numbers),
  2. StreamingLocalProvider (non-trivial; a second hand-copy already drifted
     once),
  3. the CV / bootstrap evaluation core (evaluate / boot_ci / paired_bootstrap
     and the 10x5 out-of-fold averaging).

Everything else that used to be in a package (surface stats, to_row, the
degeneracy gate, the stability table, the resumable-extraction loop, the
freeze/consistency checks) is now short enough to live inline in the one or two
notebook cells that use it -- see notebooks 08/09/10/11.

Run the notebooks from this folder (`notebooks_preliminary/`) so `import helpers`
resolves.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import zipfile
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

# ===========================================================================
# 1. Configuration
# ===========================================================================

# .env lives at the repo root, one level up from notebooks_preliminary/.
for _candidate in (Path(".env"), Path("../.env"), Path("../../.env")):
    if _candidate.exists():
        load_dotenv(_candidate)
        break

BASE_URL = "https://litellm.vse.cz/v1"          # university LiteLLM proxy (OpenAI-compatible)
API_KEY = os.environ.get("LOCAL_OPENAI_API_KEY", "")
if not API_KEY:
    raise RuntimeError(
        "LOCAL_OPENAI_API_KEY is not set. Put it in a .env file at the repo root "
        "(LOCAL_OPENAI_API_KEY=sk-...) and keep .env out of git. Never hardcode "
        "the key in a notebook or in this file."
    )

MODEL_A = "qwen3.5:122b"    # primary: all frozen features are built from this model's run
MODEL_B = "qwen3.6-35b"     # stability check only; never used to build the frozen set

TEMPERATURE = 0.0
MAX_TOKENS = 4096           # V2 returns long lists; 2048 truncates on verbose docs
TIMEOUT_S = 600

N_CALIB = 10               # docs used to sanity-check the prompt before a full extraction
SEED = 42

# Automatic, mechanical feature-set gates (applied from values on disk, never
# hand-picked after seeing downstream results).
ZERO_SHARE_MAX = 0.70       # degeneracy gate: drop a count if zero_share exceeds this
STABILITY_MIN_RHO = 0.60    # "stable" if Spearman rho(model A, model B) >= this

EXPECTED_PROMPT_SHA = "f52aae5d939d620e"   # must equal PROMPT_SHA below

# CV protocol (see evaluate / cv_proba)
N_REPEATS_DEFAULT = 10
N_SPLITS_DEFAULT = 5


# ===========================================================================
# 2. The frozen V2 extraction prompt
# ===========================================================================
# Label-blind and domain-blind: the model counts observable linguistic events
# and quotes evidence, and is never told (even by naming what NOT to infer)
# that this is a cognitive-screening study. FROZEN for the rest of the project
# -- do not reword. Every notebook that extracts or compares V2 runs imports
# V2_PROMPT from here and re-asserts the hash at the top (belt-and-suspenders
# against a stale import).

V2_PROMPT = r"""You are annotating a transcript of spontaneous Czech speech. A person was shown a drawing of a
lakeshore scene and asked to describe it aloud. The text is an automatic transcription.

Your job is to COUNT observable linguistic events and QUOTE the evidence for each count.
You are an annotator, not an evaluator.

RULES
- Quote evidence verbatim from the transcript, in Czech. Never translate or paraphrase.
- If a category has no instances, return an empty list. Empty is a valid answer.
- Never infer anything about the speaker: not their health, ability, intelligence, age,
  education or state of mind. Describe the language, never the person.
- Do not compare this speaker to anyone else or to any norm.
- Count occurrences, not impressions. Two hedges in one sentence are two entries.
- Return only JSON.

CATEGORIES
1. named_entities - distinct objects, creatures or people explicitly named. Lemmatise and list
   each distinct entity exactly once (a set, not a list of mentions).
2. specific_action_verbs - verbs naming a particular manner of action. List every occurrence.
3. generic_verbs - verbs of bare existence, possession, location or unspecified movement.
4. complete_propositions - integer count of clauses with an explicit subject and predicate
   plus at least one further argument or adjunct.
5. locative_expressions - phrases placing something somewhere. List every occurrence.
6. regions_referenced - distinct regions among "water", "land", "sky".
7. hedge_spans - expressions of uncertainty. List every occurrence.
8. deictic_spans - references such as "there", "that thing" that substitute for naming.
9. metacomment_spans - remarks about the speaker's own describing, remembering, or the task.
10. repeated_content_lemmas - content words used more than once, with counts.
11. self_corrections - integer count.
12. diminutive_or_affective_forms - diminutive or affectionate noun forms.
13. quantity_expressions - numerals or quantifiers applied to things in the scene.

Return exactly this JSON and nothing else:
{
  "named_entities": [],
  "specific_action_verbs": [],
  "generic_verbs": [],
  "complete_propositions": 0,
  "locative_expressions": [],
  "regions_referenced": [],
  "hedge_spans": [],
  "deictic_spans": [],
  "metacomment_spans": [],
  "repeated_content_lemmas": [{"lemma": "", "count": 0}],
  "self_corrections": 0,
  "diminutive_or_affective_forms": [],
  "quantity_expressions": []
}"""

PROMPT_SHA = hashlib.sha256(V2_PROMPT.encode()).hexdigest()[:16]

_BANNED_TERMS = [
    "dementia", "alzheimer", "cognitive", "impair", "patient", "diagnos",
    "control group", "demence", "pacient",
]


def assert_domain_blind() -> None:
    """The prompt must never leak the research domain -- not even by naming
    what NOT to infer, since that still tells the model what the study is about."""
    low = V2_PROMPT.lower()
    for term in _BANNED_TERMS:
        assert term not in low, f"prompt leaks domain term: {term!r}"


def assert_prompt_hash(expected: str) -> None:
    """Call at the top of every notebook that compares two extraction runs or
    loads a frozen pipeline. A mismatch means the prompt text changed since the
    runs / artifact were produced, and the comparison is no longer valid."""
    assert PROMPT_SHA == expected, (
        f"prompt hash mismatch: got {PROMPT_SHA}, expected {expected}. The prompt "
        "text has changed since the runs being compared were produced -- do not "
        "proceed until this is resolved."
    )


# ===========================================================================
# 3. Dataset locator
# ===========================================================================

def find_dataset(extra_candidates: list | None = None) -> Path:
    """Locate fileDataset/, unzipping fileDataset.zip if that is all we have.
    Pass machine-specific locations via extra_candidates=[Path(...)] from the
    notebook rather than editing this list."""
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
        "fileDataset/ not found. Put the folder (or fileDataset.zip) next to the "
        "notebook, or pass its location via extra_candidates=[Path(...)]."
    )


# ===========================================================================
# 4. Provider -- StreamingLocalProvider
# ===========================================================================
# The raw llm-feature-gen LocalProvider breaks three ways against this endpoint
# on the 122B model:
#   1. a non-streamed 90s+ generation looks idle to the reverse proxy -> 504;
#      streaming keeps bytes flowing.
#   2. Qwen3.x reasons internally and can spend the whole token budget on
#      <think> content -> empty body; extra_body={"think": False} plus a regex
#      strip as backup.
#   3. the base library retries rate limits only, not timeouts / 5xx / empty /
#      unparseable -> this subclass retries all of those with backoff.

import openai
from openai import BadRequestError
from llm_feature_gen.providers.local_provider import LocalProvider

_THINK_TAG = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


class StreamingLocalProvider(LocalProvider):
    """LocalProvider + streaming + think-off + retries on transient failures."""

    def __init__(self, *a, stream: bool = True, **kw):
        super().__init__(*a, **kw)
        self.stream = stream
        self.client = openai.OpenAI(
            base_url=self.base_url, api_key=self.api_key, timeout=TIMEOUT_S
        )
        self._think_supported = True   # flipped off if the endpoint rejects the field

    def _raw_call(self, model, messages, json_mode):
        kw = dict(model=model, messages=messages,
                  temperature=self.temperature, max_tokens=self.max_tokens)
        if json_mode:
            kw["response_format"] = {"type": "json_object"}
        if self._think_supported:
            kw["extra_body"] = {"think": False}

        if not self.stream:
            return self.client.chat.completions.create(**kw).choices[0].message.content or ""

        parts = []
        for ev in self.client.chat.completions.create(stream=True, **kw):
            if ev.choices and ev.choices[0].delta and ev.choices[0].delta.content:
                parts.append(ev.choices[0].delta.content)
        return "".join(parts)

    def _chat_json(self, deployment_name, system_prompt, user_content, json_mode=False):
        if json_mode and "JSON" not in system_prompt:
            system_prompt += " Respond in strict JSON format."
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}]

        last, backoff = None, 3
        for attempt in range(self.max_retries):
            try:
                text = self._raw_call(deployment_name, messages, json_mode)
                text = _THINK_TAG.sub("", text).strip()
                if not text:
                    raise ValueError("empty completion")
                try:
                    return json.loads(text)
                except Exception:
                    got = self._extract_json(text)      # handles ```json fences
                    if got:
                        return {"features": got} if isinstance(got, list) else got
                    raise ValueError(f"unparseable: {text[:200]}")

            except BadRequestError as e:
                msg = str(e)
                if self._think_supported and "think" in msg:
                    self._think_supported = False       # endpoint rejects the flag
                    continue
                if json_mode and "json_object" in msg:
                    json_mode = False
                    continue
                raise

            except Exception as e:                      # timeout, 5xx, empty, unparseable
                last = e
                if attempt < self.max_retries - 1:
                    time.sleep(backoff)
                    backoff *= 2
                    continue
        raise RuntimeError(f"failed after {self.max_retries} attempts: {last}")


def make_provider(model: str, max_retries: int = 4, stream: bool = True) -> StreamingLocalProvider:
    """Ready-to-use provider for `model` (MODEL_A / MODEL_B, or any model on the endpoint)."""
    return StreamingLocalProvider(
        base_url=BASE_URL, api_key=API_KEY, default_text_model=model,
        temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
        max_retries=max_retries, stream=stream,
    )


# ===========================================================================
# 5. Evaluation core -- CV, bootstrap CI, paired bootstrap
# ===========================================================================
# 10 independent StratifiedKFold(5) splits, out-of-fold probabilities averaged.
# NOT RepeatedStratifiedKFold: cross_val_predict needs a true partition, and a
# repeated splitter would place each document in 10 test folds.

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def cv_proba(model, X, y, seed: int = SEED, n_repeats: int = N_REPEATS_DEFAULT,
             n_splits: int = N_SPLITS_DEFAULT) -> np.ndarray:
    """Out-of-fold P(positive), averaged over n_repeats independent n_splits-fold splits."""
    acc = np.zeros(len(y), dtype=float)
    for r in range(n_repeats):
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
        acc += cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    return acc / n_repeats


def boot_ci(y_true, p, n: int = 2000, seed: int = SEED):
    """Percentile bootstrap 95% CI for AUC (n resamples of the subjects)."""
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n):
        i = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[i])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[i], p[i]))
    return np.percentile(aucs, [2.5, 97.5])


def evaluate(name: str, X, y, seed: int = SEED, model=None) -> dict:
    """CV-evaluate one representation. Returns a result dict; the raw out-of-fold
    probabilities are under "_proba" -- keep them if you will run
    paired_bootstrap() against another arm (both must share the fold assignment,
    i.e. the same seed)."""
    model = model or make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")
    )
    p = cv_proba(model, X, y, seed=seed)
    lo, hi = boot_ci(y, p, seed=seed)
    return {
        "model": name,
        # 2-D matrix -> column count; raw text handed to a pipeline with a
        # vectoriser (1-D) -> "text"
        "n_feat": (X.shape[1] if getattr(X, "ndim", None) == 2 else "text"),
        "AUC": roc_auc_score(y, p),
        "CI_low": lo, "CI_high": hi,
        "macroF1": f1_score(y, p > 0.5, average="macro"),
        "balAcc": balanced_accuracy_score(y, p > 0.5),
        "_proba": p,
    }


def paired_bootstrap(p_a: np.ndarray, p_b: np.ndarray, y, n: int = 2000, seed: int = SEED):
    """Paired bootstrap on out-of-fold probabilities from the SAME fold
    assignment: mean dAUC (a - b), its 95% CI, and P(dAUC <= 0). Use this for
    model comparisons -- never eyeball whether two marginal CIs overlap.

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
    return float(np.mean(deltas)), tuple(np.percentile(deltas, [2.5, 97.5])), float(np.mean(deltas <= 0))
