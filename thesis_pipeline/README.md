# thesis_pipeline

Shared code for the DigiDiaDem V2 LLM-feature extraction thesis. Extracted
out of notebooks 08 and 09, where every module below was previously pasted
twice (and would otherwise be pasted a third and fourth time into notebooks
10 and 11).

## Install

From the repo root, with the `fileDataset` folder and a `.env` file (see
below) already in place:

```bash
pip install -e ./thesis_pipeline
```

This also works unchanged in Colab: clone the repo, then run the same
command in a cell.

## `.env`

Create a `.env` file at the repo root (never commit it — add it to
`.gitignore`):

```
LOCAL_OPENAI_API_KEY=sk-...
```

`thesis_pipeline.config` loads this automatically and raises immediately if
it's missing, instead of letting a notebook run with a hardcoded key.

## Layout

| module | what it holds | why it was duplicated before |
|---|---|---|
| `config.py` | endpoint, model names, gate thresholds, output paths | pasted per-notebook; thresholds could silently drift (see the ZERO_SHARE_MAX note in the file) |
| `prompt.py` | `V2_PROMPT`, `PROMPT_SHA`, `assert_prompt_hash()` | pasted verbatim in notebooks 08 and 09 |
| `provider.py` | `StreamingLocalProvider` | notebook 09 literally says "identical StreamingLocalProvider as notebook 08" |
| `data.py` | `find_dataset()`, `read_split()`, `surface()`, dataset loaders | pasted per-notebook |
| `extraction.py` | resumable JSONL extraction, `seal_last_line()` | pasted per-notebook |
| `features.py` | `to_row()`, feature matrix builder, degeneracy gate | pasted per-notebook, rebuilt for both model A and model B in 09 |
| `stability.py` | Spearman / ICC(2,1) / exact-match agreement | new in 09, will be reused by any future stability check |
| `evaluation.py` | `cv_proba()`, `boot_ci()`, `evaluate()`, `paired_bootstrap()` | reused in 08, 09, and 09b |
| `freeze.py` | write/reload/assert-consistent for the frozen pipeline spec | used at the end of extraction and twice in the freeze block |
| `plotting.py` | the two comparison figures, thesis palette | reused across notebooks that report AUC comparisons |

## Tests

```bash
pip install -e "./thesis_pipeline[dev]"
pytest thesis_pipeline/tests -v
```

`test_provider_faults.py` never touches the network — it's your existing
fault-injection harness (504s, timeouts, empty bodies, `<think>` tags,
```json``` fences, an endpoint that rejects `extra_body={"think": False}`),
converted to pytest. `test_resume.py` is your existing crash-mid-write test:
verifies a simulated crash at document 8 leaves exactly 7 valid records,
and that resuming recovers all 20 with none lost or duplicated.

## What did NOT move in here

The domain-blindness assertion, the `assert len(docs) == 241 / 86` hard
counts, and the printed QC/degeneracy tables stay in the notebooks, not in
this package — they're meant to be visible evidence in the executed
notebook output, not hidden behind a function call. Import the *building
blocks* from here; keep the *assertions and printed audit trail* in the
notebook that a reader will actually open.
