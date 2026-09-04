## 0. Code-level calibration — read this first, applies to every session

This is a Master's thesis preliminary research project, not a PhD-level
research software package and not production code. Write code at a
strong graduate/student research level:

- Prioritize methodological correctness, reproducibility, and readability.
- Keep the implementation straightforward and easy for a Master's student
  and supervisor to inspect.
- Use normal pandas / NumPy / scikit-learn workflows and a small number
  of helper functions where useful.
- Prefer explicit, readable notebook cells over deeply abstracted
  architectures.
- Keep rigorous experimental design where it affects validity: train/test
  separation, leakage prevention, frozen prompts/features, paired
  evaluation, reproducible seeds, and correct CV.
- Do not over-engineer error handling, abstractions, configuration
  systems, logging, testing frameworks, or package architecture unless
  they are genuinely necessary.
- Do not introduce sophisticated statistical methodology just because it
  is possible.
- Do not implement additional models, experiments, or analyses that are
  not required by the thesis research questions.
- Do not refactor existing working notebooks into a production-style
  system.
- A few clear TODOs for later refinement are preferable to speculative
  implementation.
- Comments should explain methodological decisions, not every line of
  Python.

The target is: "Rigorous experimental methodology, simple and
understandable implementation." These notebooks should look like
credible Master's thesis research notebooks that a supervisor can open
and discuss, not publication-ready research infrastructure. §11 and §12
below are a concrete case study of this rule being violated and how to
correct it — treat them as the worked example, not a one-off fix.

# Project context for coding-agent notebook rewrite

Paste this whole file as system/project context (e.g. as `CLAUDE.md` /
`AGENTS.md` in the repo root, or as the first message to a fresh agent
session) before asking it to rewrite any notebook. It exists so you can give
short, specific prompts ("rewrite notebook 08 cleanly using thesis_pipeline")
without re-explaining the study, the data, or the guardrails every time.

Two sections below are marked **[FILL IN]** — confirm those with what's
actually in your repo before handing this to the agent; everything else is
verified against the executed notebooks and real run outputs.

---

## 1. What this project is

Master's thesis: use an LLM to extract interpretable linguistic features
from Czech picture-description transcripts (DigiDiaDem dataset), and test
whether traditional ML models on those features can compete with —
or at least meaningfully inform — black-box text-embedding classifiers,
while remaining clinically interpretable. Supervised by Dr. Kliegr and
Dr. Vita (FIS VŠE).

Source paper: Šmídl et al., "The DigiDiaDem Speech-Cognitive Dataset:
Initial Experiments on Detecting Cognitive Impairments From Speech,"
IEEE Access (2026).

**Research questions the thesis must answer (verbatim from the
supervisor):**

1. Can traditional ML classifiers on LLM-generated features match or
   approach embedding-based classifiers?
2. **Sensitivity on the model** — is there a difference between LLMs used
   for feature extraction?
3. **Sensitivity on the prompt** — is there a difference between the
   `llm-feature-gen` library's built-in default prompt and a specially
   prepared prompt?
4. Can action rules / association rules over LLM-generated features augment
   the dataset for better downstream classification? (not yet started)
5. Do the resulting features transfer to English (Cookie Theft picture
   task)? (not yet started)

## 2. The data — hard rules, not suggestions

Source: pickled DigiDiaDem export. Fields used: `kobar_kategorizace_definitivni`
(diagnosis; only 0/2/3 kept, 1 excluded; binarized 0→negative, {2,3}→positive),
`splits`, `task4_Complex scene description_obraz u jezera_recognized` (the
transcript text).

| split | n | usage |
|---|---|---|
| `train` (`fileDataset/train/{positive,negative}/*.txt`) | 241 (171 neg / 70 pos), labelled | supervised feature design, model selection, CV |
| `overview` / `train.extra` (`fileDataset/overview/*.txt`) | 86, **unlabelled** | prompt calibration and exploratory extraction ONLY — never fed to a classifier |
| `test` (`fileDataset/test/`) | 61, labelled | **sealed**. Loaded in exactly one notebook, exactly once, after everything upstream is frozen |

**Non-negotiable rule for any rewritten notebook:** never open `test/` except
in the designated one-shot test-evaluation notebook. Every notebook that
touches `train`/`overview` should assert the labelled/unlabelled counts
(241 / 86) and assert that no path under `test/` was read — this is existing
practice, keep it, don't strip it as "unnecessary."

Dataset root is located via `thesis_pipeline.data.find_dataset()` — don't
hardcode a path; it already handles multiple candidate locations and
unzipping `fileDataset.zip` if needed.

## 3. The V2 extraction prompt — frozen, do not edit

The prompt asks the LLM to **count observable linguistic events and quote
evidence**, not render categorical judgements (that was the V1 design,
whose 3-level encoding destroyed most of the signal — CV AUC 0.687,
indistinguishable from transcript length alone at 0.692). It is
**label-blind** (never mentions hidden classes exist) and **domain-blind**
(never mentions cognition, health, or screening, even as something not to
infer).

13 categories: `named_entities`, `specific_action_verbs`, `generic_verbs`,
`complete_propositions`, `locative_expressions`, `regions_referenced`,
`hedge_spans`, `deictic_spans`, `metacomment_spans`,
`repeated_content_lemmas`, `self_corrections`,
`diminutive_or_affective_forms`, `quantity_expressions`.
`named_entities` and `regions_referenced` are sets (each distinct item
once); everything else lists every occurrence.

**Prompt SHA256 (first 16 hex chars): `f52aae5d939d620e`.** This prompt is
frozen through the rest of this project — through the second-model
stability check, feature freezing, the one-shot test evaluation, and the
thesis write-up. Do not reword it, even to fix a known issue (see §7). The
canonical copy lives in `thesis_pipeline.prompt.V2_PROMPT`; every notebook
that extracts or compares V2 features must import it from there (not
paste it) and call `assert_prompt_hash(EXPECTED_PROMPT_SHA)` +
`assert_domain_blind()` at the top — this is a deliberate belt-and-suspenders
check against stale imports, keep it even though the text is single-sourced.

## 4. LLM endpoint and provider

- Endpoint: `https://litellm.vse.cz/v1` (university LiteLLM proxy,
  OpenAI-compatible `/v1`).
- Models: `qwen3.5:122b` ("model A", primary — all frozen features are
  built from this model's extractions) and `qwen3.6-35b` ("model B",
  used only for the stability check, never for building the frozen feature
  set). `temperature=0.0`, `max_tokens=4096`, `timeout=600s`.
- API key: `LOCAL_OPENAI_API_KEY`, loaded from `.env` by
  `thesis_pipeline.config`, which **raises immediately if it's missing**.
  Never hardcode a key in a notebook. If you ever see one, the key must be
  treated as compromised and rotated, not just deleted from the file.
- `thesis_pipeline.provider.StreamingLocalProvider` (subclass of
  `llm_feature_gen.providers.local_provider.LocalProvider`) exists because
  the raw library breaks three ways against this endpoint: (1) a
  non-streamed 90s+ generation looks idle to the reverse proxy → 504, fixed
  by streaming; (2) Qwen3.x reasons internally before answering and can
  burn the whole token budget on `<think>` content → fixed with
  `extra_body={"think": False}` plus a regex strip as backup; (3) the base
  library only retries rate limits, not timeouts/5xx/empty bodies/unparseable
  JSON → this subclass retries all of those with exponential backoff. Reuse
  this class; don't reimplement retry/streaming logic per notebook.

## 5. Shared code — `helpers.py`, not a package (revised, see §11)

**This section is superseded by §11 — read §11 first.** The original DRY
refactor went too far: a formally pip-installable multi-module package is
more engineering than a Master's thesis notebook set needs, and it hurt
readability (a reader has to open eight files to follow one notebook cell).
§11 replaces this with a single flat `helpers.py`. The list below still
describes *what logic needs to be shared* — that hasn't changed — just not
*how* it should be packaged:

- `thesis_pipeline.config` — `BASE_URL`, `MODEL_A`, `MODEL_B`,
  `TEMPERATURE`, `MAX_TOKENS`, `SEED`, `N_CALIB`, `EXPECTED_PROMPT_SHA`,
  gate thresholds (`ZERO_SHARE_MAX=0.70`, `STABILITY_MIN_RHO=0.60`), and a
  `Paths` helper (`config.Paths.under(Path("fileDataset"))` → `.data`,
  `.out`, `.raw_a`, `.meta_a`, etc.). One source of truth for everything
  that would change the numbers if edited.
- `thesis_pipeline.prompt` — `V2_PROMPT`, `PROMPT_SHA`,
  `assert_prompt_hash()`, `assert_domain_blind()`.
- `thesis_pipeline.provider` — `StreamingLocalProvider`.
- `thesis_pipeline.data` — `find_dataset()`, `read_split()`, surface-stats
  (`n_char`, `n_word`, `n_type`, `ttr`, `n_sent`, `mlu`, `n_comma`).
- `thesis_pipeline.extraction` — resumable JSONL extraction: `load_raw()`
  (skips crash-truncated lines), `seal_last_line()` (guards against a
  crash mid-write corrupting the next append on resume), the
  append-with-flush loop, `run_metadata.json` writer.
- `thesis_pipeline.features` — `to_row()` (raw LLM JSON → count columns),
  the `LIST_FIELDS`/`INT_FIELDS` mapping, rate (`_r100`) and ratio
  (`specific_verb_ratio`, `region_breadth`) computation, the degeneracy
  gate.
- `thesis_pipeline.stability` — Spearman ρ, ICC(2,1) (two-way, absolute
  agreement), exact-match rate between two extraction runs.
- `thesis_pipeline.evaluation` — `cv_proba()` (10× independent
  `StratifiedKFold(5)`, out-of-fold probabilities averaged — **not**
  `RepeatedStratifiedKFold`, which is not a valid partition for
  `cross_val_predict`), `evaluate()`, `boot_ci()` (2000× bootstrap),
  `paired_bootstrap()` (matched per-document prediction contrasts —
  use this, never compare two models by eyeballing overlapping marginal
  CIs).
- `thesis_pipeline.freeze` — write/read `frozen_spec.json`, plus the
  reload-and-assert consistency check pattern (reload the artifact and the
  spec fresh, assert the feature list matches `pipeline.n_features_in_`,
  assert the prompt hash, assert `test_set_used: false`).

If a rewritten notebook needs something this package doesn't have yet, add
it to the package and import it — don't inline a one-off copy "just for
this notebook."

## 6. Statistical protocol — keep these exact choices

- Primary metric: **AUC**, with 95% bootstrap CI (2000 resamples). Accuracy
  is reported but explicitly noted as uninformative here (class split is
  171/70, so predicting all-negative scores 0.71 with zero signal).
- CV: 10 independent `StratifiedKFold(n_splits=5, shuffle=True)` with a
  different `random_state` per repeat, out-of-fold probabilities averaged
  across the 10 runs. This exists specifically to avoid the
  `RepeatedStratifiedKFold` + `cross_val_predict` incompatibility (the
  latter requires a true partition).
- Model comparisons use **paired bootstrap contrasts** on matched
  per-document prediction pairs (ΔAUC, 95% CI, P(Δ≤0)) — never compare two
  models by whether their marginal CIs happen to overlap.
- Classifier set is capped and fixed: **ElasticNet-regularized logistic
  regression is primary** (nested CV tuning of `C`/`l1_ratio` inside each
  fold — necessary because EPV is only ~2–6), with plain L2 logistic
  regression and a calibrated linear SVM as robustness checks only. Do not
  add more classifiers (random forest, gradient boosting, kernel SVM,
  etc.) — with n≈241 and this few events per variable, more classifiers
  means more multiple-comparisons risk for no real gain, not a stronger
  result.
- Feature-set gates are threshold-based and mechanical, applied from values
  written to disk (`v2_stability.csv`), never hand-picked after seeing
  downstream results: degeneracy = `zero_share > 0.70` on the labelled set
  in either extraction run → auto-drop; stability = `Spearman ρ(run A,
  run B) ≥ 0.60` → "stable" flag, carried into a `V2_STABLE` sensitivity
  arm.

## 7. Current results — what's already established, don't re-derive these from scratch

- **Extraction quality:** 327/327 succeeded with both `qwen3.5:122b` and
  `qwen3.6-35b`, groundedness ≈0.98 (quoted spans verified to literally
  appear in the transcript).
- **Stability (A vs. B):** median Spearman ρ (counts) = 0.68; 13/16
  count+ratio features stable at ρ≥0.60. **16/16 features agree on
  direction** across both models — the qualitative profile (less
  specificity/propositional density, more hedging/deixis/metacommentary in
  the positive class) is model-independent even where exact counts aren't.
  `n_entities` is degenerate (auto-dropped) and, with `n_generic_verb` and
  `specific_verb_ratio`, unstable (auto-flagged, known prompt-ambiguity
  issue — see below, don't fix it now).
- **Frozen primary pipeline:** `V2_STABLE + surface` (13 stable V2 rates +
  7 surface stats, 20 features total), ElasticNet, CV AUC **0.806** vs.
  0.743 for surface stats alone. Paired ΔAUC +0.063, 95% CI
  **[+0.004, +0.120]**, P(ΔAUC≤0)=0.019 — real but boundary-significant,
  report it that way, don't oversell it.
- **Full comparison table (CV AUC):** majority class 0.50, transcript
  length alone 0.691, V1 categorical 0.693, surface stats 0.743–0.749,
  V2 rates 0.768, V2 stable-only 0.776–0.806 (frozen primary), TF-IDF
  char 3–5gram ceiling 0.867.
- **Embedding baseline (separate track, notebook 12):** RobeCzech-base
  0.836 AUC / 0.733 F1 beats the original FERNET-C5-RoBERTa baseline
  (0.744 AUC / 0.606 F1). CZERT-B 0.822 AUC. Small-E-Czech collapsed to
  majority-class-only predictions despite 0.782 AUC — don't treat its
  reported accuracy as meaningful without recalibrating its threshold.
- **Known, deliberately unfixed issue:** `named_entities`/`specific_verb`
  extraction conflates proper nouns with identified common-noun referents
  in the prompt wording — this is why `n_entities` is degenerate. This is
  documented as a limitation, not something to patch, because the prompt
  is frozen (§3).
- **Sealed test-set result (notebook 10, DONE — cannot be rerun):** frozen
  primary AUC **0.807**, 95% CI [0.677, 0.916] (n=61) vs. surface baseline
  0.796. Paired ΔAUC +0.011, CI [-0.103, +0.124], P(Δ≤0)=0.421 — same
  direction as the CV result but not independently significant at n=61
  (expected — report both numbers together, not the test one alone). No
  overfit signal: CV AUC 0.8058 vs. test AUC 0.8070, gap +0.001.
- **RQ3 answer (notebook 11, DONE):** V2 (custom prompt) vs. V1 (confirmed
  to be `llm-feature-gen`'s built-in discovery+generation prompt pipeline
  — **still needs a final check against `my_prompt.txt`** to confirm V1's
  generation step used the library's unmodified default template and not
  a hand-edited one). Result: V2 significantly beats V1, ΔAUC +0.113, CI
  [+0.031, +0.194], P=0.004. V2 also beats surface (P=0.014); V1 does not
  (P=0.889) — consistent with V1's already-known weakness elsewhere.
- **Both core empirical questions (RQ1 test result, RQ3 prompt
  sensitivity) are now answered.** What remains is largely the `01`–`07`
  (or `1`–`6` — confirm the actual count, it's been referred to both ways)
  notebook review, the code-level cleanup in §11/§12, and the thesis
  write-up itself.

## 8. Target notebook structure

```
0X_data_preparation                      [FILL IN — confirm this exists / what it's numbered]
0X..0X_v1_feature_extraction_evaluation  [FILL IN — how many stages, what numbers]
08_v2_extraction                         DONE — full 327-doc extraction (model A), QC gates,
                                          first V2 vs V1 vs length vs TF-IDF comparison
09_v2_stability                          DONE — model-B stability check, gates, corrected
                                          frozen pipeline, decisive paired bootstrap
10_v2_test_evaluation                    DONE — sealed, cannot rerun (test_set_used: true).
                                          Result: frozen primary AUC 0.807 [0.677,0.916] vs
                                          surface 0.796; paired dAUC +0.011, CI [-0.103,+0.124],
                                          P=0.421 (not significant alone, n=61) — directionally
                                          consistent with the CV result (dAUC +0.064, P=0.016,
                                          n=241). No overfit signal (CV 0.8058 vs test 0.8070).
11_v2_prompt_sensitivity                 DONE — RQ3 answer: V2 custom vs V1 (confirmed built-in
                                          llm-feature-gen discovery+generation prompt — verify
                                          this against my_prompt.txt before finalizing). V2 beats
                                          V1: dAUC +0.113, CI [+0.031,+0.194], P=0.004. V2 beats
                                          surface (P=0.014); V1 does not (P=0.889).
12_embedding_baseline_comparison         DONE — rename from
                                          Comparison_CZ_bin_classification_multi_encoder_colab
```

Excluded from the final numbered sequence (dev/test scratch, not thesis
content): connectivity smoke-tests, fault-injection/crash-resume test
scripts (these belong in `tests/` as `pytest`, not as notebooks with
"outputs"), any `testrun*`/duplicate draft files.

**[FILL IN]** — before asking the agent to rewrite anything, confirm what's
actually in the `01`–`07` range (or however your V1/data-prep notebooks are
numbered today) so the rewrite doesn't collide with or accidentally drop
real content. One line per notebook is enough: number → what it does →
still needed or superseded by 08/09.

## 9. Engineering and integrity conventions to preserve in the rewrite

- **Format:** jupytext `py:percent` cells (`# %%` / `# %% [markdown]`),
  each block numbered and titled ("Block N — ...") with a short markdown
  cell before the code explaining *why*, not just what. Keep this — it's
  what makes the notebooks readable as a methods section, not just a
  script dump.
- **Audit trail over brevity.** Printed QC tables, threshold assertions,
  the domain-blind check, `seal_last_line()`, and the reload-and-assert
  consistency checks all look like verbose scaffolding but are each
  answering a specific "how do you know X didn't happen" question a
  thesis committee could ask. Don't strip these for tidiness — if
  anything, make them more visible (group under a clearly labeled cell),
  not less.
- **DRY, but not at the cost of the runtime prompt-hash check** — see §3.
  Single-source config/thresholds/prompt text via the package; keep the
  redundant runtime assertion.
- **Never hardcode secrets.** `.env` only, loaded via `thesis_pipeline.config`,
  which fails loudly if missing.
- **Resumability matters.** Any long-running extraction loop must be
  crash-safe (append-with-flush, skip-already-done, seal-last-line-before-resume)
  — re-extracting 327 documents from scratch after a Colab disconnect is
  the exact failure mode this pattern exists to prevent.
- **Disclose AI-assisted coding** in the thesis methods/acknowledgments —
  this is expected and normal, especially given the supervisor's own
  reference material is about an LLM-feature-generation Python package.
  Keep the notebook version history (rather than squashing into one
  "final" commit) as evidence of genuine iterative development.

## 10. What NOT to do in a rewrite

- Don't touch `fileDataset/test/` outside notebook 10.
- Don't reword, retranslate, or "improve" `V2_PROMPT` — frozen (§3).
- Don't add classifiers beyond ElasticNet (primary) + L2 LR + linear SVM
  (robustness checks).
- Don't hand-pick which features survive the stability/degeneracy gates —
  they're threshold-mechanical, applied from `v2_stability.csv`.
- Don't fit anything (including a "just curious" run) on the 61 test cases
  before notebook 10 is the one deliberately doing so, and don't re-run
  notebook 10 more than once after it has produced a result.
- **Never execute notebook 10 again, for any reason — including testing
  the rewrite.** It is sealed (`test_set_used: true`); Block 0's
  `assert_consistent()` will correctly fail if it's run. Do not "fix" that
  failure, and do not reset `test_set_used` to `false` to test the flow —
  that would unseal the test set, which is not recoverable. Any code
  cleanup in notebook 10 (helpers.py flatten, explicit asserts per §11)
  must be **cosmetic-only, verified by reading the code, not by running
  it.** Leave its existing output cells exactly as they are — Jupyter
  shows stale output if code is edited without rerunning, which is fine
  and expected here; do not clear or regenerate them. If there's any
  doubt whether an edit changes behavior, don't make that edit.
- For notebooks 08/09/11 (still re-runnable, not sealed), after applying
  the §11/§12 cleanup, re-run each one and confirm the headline numbers
  reproduce (08/09's frozen_primary_cv_auc 0.8058; 11's V2-vs-V1 ΔAUC
  +0.113) before treating the rewrite as done. A cleanup that silently
  changes a result is a bug, not a simplification.

## 11. Notebooks 08–11 were over-engineered — correct this in the rewrite

Reviewing the executed notebooks 08–11 found they'd drifted past "Master's
thesis notebook" into "small production ML pipeline": a formally
pip-installable `thesis_pipeline` package with 8 submodules and ~18–20
named functions/classes, a `Paths` path-registry class, and a persisted-
state locking mechanism (`assert_consistent()` bundling artifact-reload +
feature-count check + prompt-hash check + `test_set_used` check into one
opaque call). Individually each piece was a reasonable response to a real
problem encountered along the way, but the overall result requires a
reader to open multiple files to follow one notebook cell — the opposite
of "obviously correct and simple" (§ code-level calibration above). This
did not affect correctness or the results already obtained (10 is sealed
and stands regardless) — it's a readability/defensibility rewrite, not a
re-run.

**Target shape for the rewrite:**

- Replace the `thesis_pipeline` package with a single flat `helpers.py`
  (or similarly plain module) sitting next to the notebooks. No
  `pyproject.toml`, no `pip install -e`, no submodules — a reader should
  be able to open one file and read it top to bottom in a few minutes.
- Keep in the shared file only what's substantial *and* correctness-
  critical to keep identical across notebooks: `StreamingLocalProvider`
  (non-trivial, and duplicating it risks silent drift — this already
  happened once), and the core evaluation functions (`evaluate`,
  `boot_ci`, `paired_bootstrap`, the CV-averaging logic). These are worth
  one canonical implementation.
- Inline everything else directly into the notebook cells that use it
  instead of importing it: `groundedness()`, `missing_fields()`,
  `seal_last_line()`, the surface-stats formula, `to_row()`/feature-matrix
  building. Each is short (<15 lines); duplicating a short function costs
  far less readability than hiding it behind an import.
- Replace `Paths.under(...)` with plain top-level path variables per
  notebook (`DATA = Path(...)`, `OUT = Path(...)`, etc.) — more lines,
  instantly readable without opening a class definition.
- Replace `assert_consistent()` with 4–5 explicit `assert` lines visible
  directly in the notebook cell. Keep the underlying guarantee (don't
  silently re-open the sealed test set, don't run under a drifted prompt)
  — just don't hide the mechanism behind one opaque function name.
- Local, short, single-purpose helpers defined *inside* a notebook (e.g.
  a `run()` wrapper around evaluate+store+print, a classifier-factory
  function) are fine as-is and don't need to change — that's the right
  level of local abstraction, not a smell.
- Never print the same result table in two different layouts in the same
  notebook (this happened in 09/11's comparison output) — pick one.

## 12. Comparison-table reporting — report fewer arms, not more code

The full comparison tables (14–16 rows mixing baselines, encoding-design
ablation arms, and 3-classifier robustness repeats all together) are hard
to read and mix three different purposes into one table. Restructure into:

- **Main results table — 5 rows, one classifier (ElasticNet) each:**
  Transcript length only → Surface stats → V1 categorical → **V2 frozen
  primary (stable rates + surface)** → TF-IDF char 3–5gram. This is the
  whole story: confound floor, real competitor, prior approach, your
  result, reference ceiling.
- **Robustness table — separate, 6 rows:** surface stats and the frozen
  primary only, each under ElasticNet/L2/SVM, captioned explicitly as a
  classifier-sensitivity check, not additional competing arms.
- **Drop from any table, keep only as one sentence in the methods
  narrative:** `V2 rates` (no surface), `V2 counts+rates`, `V2
  stable-only` (no surface). These were genuinely useful while deciding
  the encoding (counts vs. rates vs. both) — that decision belongs in
  prose ("we compared X, Y, Z; rates were preferred because raw counts
  still embed transcript length"), not as surviving table rows once the
  encoding choice is frozen.
- **TF-IDF is a reference ceiling, not a competing interpretable model.**
  Keep it (it's the empirical anchor for the interpretability-vs-
  performance trade-off argument), but always caption it as such and set
  it visually apart (a separator row, a distinct plot color — 08's plot
  already does this with `C_REF`) rather than letting it sit in the
  ranked list as if it's something the interpretable models should beat.
- **Known numbering trap:** notebook 08's original exploratory comparison
  reports "V2 rates + surface [enet], n_feat=22, AUC=0.805" — this is the
  *pre-stability-gate* feature set (15 un-gated rates + 7 surface), not
  the actual frozen primary (13 stability-gated rates + 7 surface = 20
  features, CV AUC 0.8058, confirmed in notebooks 09/10). The two numbers
  are close enough to be confused. Label 08's table explicitly as
  exploratory/superseded and point to 09/10 for the number that counts.