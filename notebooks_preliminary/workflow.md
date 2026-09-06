# V2 feature-stability & test-evaluation workflow — review

**Scope.** Notebooks `09_v2_stability.ipynb` (stability check + freeze) and
`10_v2_test_evaluation.ipynb` (one-shot sealed test), with `08_v2_extraction.ipynb` as
upstream context. Shared code lives in `notebooks_preliminary/helpers.py`. All paths below are
relative to `notebooks_preliminary/`.

**Two research questions**
1. **Q1 — stability.** Are the V2 LLM-extracted count features a stable property of the
   transcripts, or an artefact of one model? (Re-extract with a second LLM, correlate.)
2. **Q2 — added value.** Do V2 features add predictive information *beyond* cheap surface
   statistics (word count, type count, TTR, MLU, …), on held-out data?

**One-line answers (from the runs on disk):**
- Q1: *Partially stable.* Countable, well-defined events transfer across models (Spearman ρ
  0.76–0.90); the specific-vs-generic-verb judgment does not (ρ 0.42–0.44); `named_entities`
  is unusable (ρ ≈ 0). Median ρ over counts = **0.68**; 13/16 count+ratio features "stable"
  (ρ ≥ 0.60).
- Q2: *No, not on held-out data.* In-sample CV shows a small ElasticNet-only gain
  (ΔAUC ≈ +0.063, P ≈ 0.016) that is not reproduced by L2/SVM and **collapses to ΔAUC = +0.011
  (95% CI −0.103…+0.124, P = 0.42) on the sealed 61-case test set.** The frozen interpretable
  model does reach **test AUC 0.807**, but SHAP shows that performance is carried by the
  surface statistic `n_type` (lexical diversity), not the LLM features.

---

## 1. Pipeline at a glance

```
08_v2_extraction     qwen3.5:122b  → v2_raw.jsonl (327 docs)          [run A, "MODEL_A"]
      │
09_v2_stability      qwen3.6-35b   → v2b_raw.jsonl (327 docs)         [run B, "MODEL_B"]
      │  ├─ Block 7  stability: Spearman ρ / ICC(2,1) / exact-match, run A vs run B
      │  ├─ Block 8  automatic degeneracy gate (zero_share > 0.70) + stability flag (ρ ≥ 0.60)
      │  ├─ Block 9  classifier comparison (ElasticNet primary; L2 + Linear-SVM robustness)
      │  ├─ Block 9b corrected frozen set = V2_STABLE + SURFACE (drops 2 unstable feats)
      │  ├─ Block 10 freeze → frozen_pipeline_v2/ (3 pipelines + frozen_spec.json)
      │  ├─ Block 10b reload-and-assert consistency check
      │  └─ Block 11 SHAP explanation of the frozen model
      │
10_v2_test_evaluation qwen3.5:122b → v2_test_raw.jsonl (61 docs)      [sealed test, one-shot]
         └─ apply frozen pipelines → test AUC + paired bootstrap → seal frozen_spec.json
```

The **61-case test set is loaded exactly once**, in notebook 10 Block 1. `frozen_spec.json`
carries `test_set_used: true` and notebook 10's Block 0 refuses to run again once it is set.

---

## 2. Artefacts produced

| File (`fileDataset/outputs/`) | Produced by | Contents |
|---|---|---|
| `v2b_raw.jsonl` | 09 Block 5 | 327 raw MODEL_B responses (qwen3.6-35b), 327/327 OK |
| `v2b_run_metadata.json` | 09 Block 5 | model, prompt SHA, counts, `test_set_used: false` |
| `v2_stability.csv` | 09 Block 7 | per-feature ρ, Pearson, ICC(2,1), exact-match, medA/medB, `stable` |
| `v2_stability_comparison.csv` | 09 Block 9 | 13 model×representation rows: AUC, bootstrap CI, macroF1, balAcc |
| `frozen_pipeline_v2/frozen_spec.json` | 09 Block 10 → 10 Block 8 | full spec: models, prompt SHA, gates, feature sets, CV protocol, classifiers, stability summary, CV paired bootstrap, **and the test results + seal** |
| `frozen_pipeline_v2/pipe_v2_stable_surface.joblib` | 09 Block 10 | frozen primary: `V2_STABLE + SURFACE` (20 cols), ElasticNet, fit on 241 |
| `frozen_pipeline_v2/pipe_surface.joblib` | 09 Block 10 | frozen baseline: 7 surface stats, L2 logreg |
| `frozen_pipeline_v2/pipe_tfidf_char35.joblib` | 09 Block 10 | reference ceiling: char 3–5-gram TF-IDF + L2 logreg |
| `frozen_pipeline_v2/pipe_v2_rates_surface_deprecated.joblib` | 09 Block 10 | audit copy of the *superseded* un-gated primary |
| `shap_feature_importance.csv` | 09 Block 11 | mean \|SHAP\| per feature (log-odds) |
| `shap_importance_bar.png`, `shap_summary_beeswarm.png` | 09 Block 11 | SHAP plots for the frozen model |
| `v2_test_raw.jsonl` | 10 Block 3 | 61 raw MODEL_A responses on the test set |
| `v2_test_run_metadata.json` | 10 Block 3 | extraction metadata for the test run |

---

## 3. Notebook 09 — step by step

Config: `RUN_LLM=False` on the committed run (both JSONLs cached; analysis-only re-run).
`SEED=42`, CV = 10 repeats × stratified 5-fold, bootstrap CI = 2000 subject resamples.
Gate thresholds from `helpers.py`: `ZERO_SHARE_MAX=0.70`, `STABILITY_MIN_RHO=0.60`.

| Block | What it does | Key output |
|---|---|---|
| 0 config | paths, gate thresholds, `RUN_LLM` | `MODEL_A=qwen3.5:122b`, `MODEL_B=qwen3.6-35b` |
| 1 prompt | imports `V2_PROMPT` from `helpers`, re-asserts SHA + domain-blind | `prompt sha256[:16] = f52aae5d939d620e` — matches run A |
| 2 data | reads `train/negative`+`train/positive`+`overview`, computes 7 surface stats; asserts (241 labelled, 86 unlabelled); `test/` never read | `327 docs → 241 labelled (171 neg / 70 pos) + 86 unlabelled` |
| 3 provider | `helpers.make_provider(MODEL_B)` — StreamingLocalProvider (stream + `think:false` + retries) | provider ready, qwen3.6-35b |
| 4 calibration | groundedness / missing-field probe on 5 unlabelled docs (skipped, `RUN_LLM=False`) | — |
| 5 extraction | resumable JSONL append; writes `v2b_run_metadata.json` | `MODEL_B extracted: 327/327 ok, 0 failed` |
| 6 feature matrices | `to_row()` mapping (identical to nb 08) → 14 counts, +14 per-100-word rates, +2 scale-free ratios (`specific_verb_ratio`, `region_breadth`); builds `dfA` (122b) and `dfB` (35b); aligns labelled rows | `241` labelled docs present in both runs |
| 7 stability | Spearman ρ (headline), ICC(2,1) (absolute agreement — catches level shifts), exact-match (integer counts), medA/medB; `stable = ρ ≥ 0.60` | see §4 |
| 8 degeneracy gate | `zero_share > 0.70` in *either* run → drop count + its rate. Builds `V2_RATES` (15), `V2_ALL` (28), `V2_STABLE` (13), `FROZEN_PRIMARY_FEATURES = V2_STABLE + SURFACE` (20). In-cell asserts re-derive `V2_STABLE` from the on-disk CSV and confirm it drops exactly `n_generic_verb_r100` + `specific_verb_ratio` | `dropped by degeneracy gate: ['n_entities']` |
| 9 comparison | ElasticNet `LogisticRegressionCV` (l1_ratio & C tuned by inner CV per fold), plus L2 logreg + calibrated Linear-SVM robustness arms; V1 (leak-free OHE) and TF-IDF (leak-free, vectoriser inside pipeline) as reference; writes `v2_stability_comparison.csv` | see §5 |
| 9b corrected frozen set | reproducibility guard: re-derives `V2 stable-only` AUC on the same folds, asserts it (and the frozen primary, a superset) land inside the reported CI band; runs the decisive paired bootstrap | `frozen primary CV AUC 0.8058 in band → OK` |
| 10 freeze | fits `surface` / `v2_stable_surface` / `tfidf_char35` on all 241 run-A docs, `joblib.dump`s them, writes `frozen_spec.json`. Renames any prior un-gated primary to `*_deprecated.joblib` (never deletes). Preserves an existing test seal if notebook 10 already ran | `froze surface / v2_stable_surface / tfidf_char35`; "existing test-set seal preserved" |
| 10b consistency | reloads spec + artefact, asserts feature count == `n_features_in_`, feature list == `V2_STABLE+SURFACE`, unstable feats absent, prompt SHA == `f52aae5d939d620e`, seal state valid | `ALL CHECKS PASSED` |
| 11 SHAP | `shap.LinearExplainer` (exact for a linear model) on the frozen pipeline, same 241 docs; global mean \|SHAP\|, beeswarm, and a coef check for true ElasticNet sparsity | see §6 |
| 12 verdict | prints the Q1 / Q2 summary | see §7 |

---

## 4. Q1 results — feature stability (`v2_stability.csv`, n = 327 docs compared)

**Rank agreement (Spearman ρ), raw counts:**

| tier | features (ρ) |
|---|---|
| strong ρ ≥ 0.78 | n_proposition **0.90**, n_locative **0.88**, n_quantity **0.88**, n_hedge 0.81, n_metacomment 0.79 |
| moderate 0.60–0.78 | n_specific_verb 0.76, repeat_mass 0.70, n_repeated_lemma 0.67, n_deictic 0.66, n_selfcorrect 0.63, n_diminutive 0.62, n_region 0.61, region_breadth 0.61 |
| **weak ρ < 0.50** | **n_generic_verb 0.42, specific_verb_ratio 0.44, n_entities −0.05** |

- **median Spearman over counts = 0.68**; **13 / 16 count+ratio features "stable"** (ρ ≥ 0.60).
- **ICC(2,1) tracks ρ** for the stable set (0.61–0.88) — when the two models agree on ranking
  they also agree on the *level*, not just the order. Exception: `n_entities` ICC ≈ 0.
- **Exact integer-count agreement is much lower than rank agreement** — n_specific_verb 0.30,
  n_proposition 0.46, repeat_mass 0.21. The models rarely emit the same number but rank
  documents alike ⇒ fine for a rank/AUC model, not usable as absolute norms.
- **Systematic level shifts (median A → median B):** `n_entities` **0 → 9** (122b almost never
  returned entities; 35b does — opposite behaviour, hence ρ ≈ 0), n_generic_verb 1 → 2,
  n_metacomment 1 → 0, n_deictic 2 → 1, n_diminutive 3 → 2. High-signal features barely move
  (specific_verb 11→10, proposition 13→12, locative 7→7).

**Degeneracy gate (Block 8, labelled docs):** `n_entities` zero-share = **0.946 in run A**,
**0.386 in run B** → degenerate under the "either run" rule → dropped (count + `_r100`).
No other feature exceeds 0.70 in either run. `n_generic_verb` (zero-share 0.18 / 0.15) is *not*
degenerate but *is* unstable (ρ 0.42) → kept in `V2_RATES`, excluded from `V2_STABLE`.

**Feature sets after gating:** `COUNTS` 14 · `RATES` 14 · `RATIOS` 2 →
`V2_RATES` = 13 kept rates + 2 ratios (15) · `V2_STABLE` = 12 stable rates + `region_breadth`
(13) · **`FROZEN_PRIMARY_FEATURES` = `V2_STABLE` + 7 surface (20)**. `V2_STABLE` drops
`n_generic_verb_r100` (ρ 0.34) and `specific_verb_ratio` (ρ 0.44) relative to `V2_RATES`.

**Supplementary cross-run check (run separately, not in the notebook):** all **16/16** rate
features agree on the *sign* of the class association between the two models; **6/16** are
BH-significant in *both* runs (n_specific_verb_r100, n_locative_r100, n_hedge_r100,
n_deictic_r100, n_metacomment_r100, region_breadth). `specific_verb_ratio` is BH-significant
only under 122b — consistent with its low ρ.

**Q1 verdict.** Moderately stable. The countable, well-defined events transfer cleanly; the
specific/generic-verb distinction (including `specific_verb_ratio`, a top length-independent
signal in the earlier univariate analysis) is model-dependent and was correctly excluded from
the frozen set. `n_entities` is unusable and was gated out automatically.

---

## 5. Q2 results — classifier comparison (`v2_stability_comparison.csv`, run-A features, 10×5 CV)

**Main table — ElasticNet (TF-IDF = reference ceiling, not a competitor):**

| representation | n_feat | AUC | 95% CI |
|---|---|---|---|
| Transcript length | 1 | 0.694 | 0.618–0.766 |
| Surface stats | 7 | 0.743 | 0.673–0.803 |
| V1 categorical | 10 | 0.693 | 0.617–0.766 |
| **V2 stable + surface (frozen primary)** | **20** | **0.806** | 0.739–0.867 |
| TF-IDF char 3–5-gram (ceiling) | text | 0.867 | 0.816–0.912 |

**Robustness — frozen primary vs surface under 3 classifiers:**

| representation | clf | AUC |
|---|---|---|
| Surface stats | enet / l2 / svm | 0.743 / 0.742 / 0.749 |
| V2 stable + surface | enet | **0.806** |
| V2 stable + surface | l2 | 0.789 |
| V2 stable + surface | svm | 0.787 |

**Encoding ablation (in the CSV, not tabled):** V2 rates 0.768 · V2 counts+rates 0.795 ·
V2 stable-only 0.775 · V2 rates+surface 0.805. Raw counts beat rates only because counts still
embed transcript length.

**Paired bootstrap (Block 9b / Block 22 — the decisive contrasts, same folds/seed):**

| contrast | ΔAUC | 95% CI | P(Δ≤0) |
|---|---|---|---|
| **V2 stable+surface (frozen primary) vs surface** | **+0.063** | **+0.004 … +0.120** | **0.016–0.019 ✳** |
| V2 rates vs surface alone | +0.025 | −0.042 … +0.089 | 0.224 |
| V2 stable-only (no surface) vs surface | +0.033 | −0.035 … +0.097 | 0.161 |
| [L2] frozen primary vs surface | +0.047 | −0.016 … +0.110 | 0.075 |
| [SVM] frozen primary vs surface | +0.038 | −0.023 … +0.098 | 0.107 |
| V2 rates vs transcript length | +0.074 | −0.011 … +0.161 | 0.045 |
| V2 rates vs V1 categorical | +0.075 | −0.006 … +0.157 | 0.036 |
| TF-IDF vs V2 rates | +0.100 | +0.036 … +0.166 | 0.001 ✳ |

**Reading:** the "V2 adds beyond surface" claim holds **only** for the ElasticNet frozen-primary
arm, and marginally (CI lower bound +0.004). Under L2 and SVM the same contrast's CI crosses
zero. V2 remains ~0.06 below the TF-IDF ceiling (paired P = 0.001).

---

## 6. SHAP — what the frozen model actually uses (`shap_feature_importance.csv`)

ElasticNet chose **C = 0.268, l1_ratio = 0.90** (very sparse): **11 of 20 features non-zero.**

| feature | mean \|SHAP\| (log-odds) | coef | direction |
|---|---|---|---|
| **`n_type`** (surface — unique word types) | **0.712** | −0.92 | fewer types → positive |
| `n_deictic_r100` | 0.432 | +0.54 | more deixis → positive |
| `n_specific_verb_r100` | 0.354 | −0.47 | fewer specific verbs → positive |
| **`n_char`** (surface — length) | 0.308 | −0.39 | shorter → positive |
| `n_proposition_r100` | 0.177 | −0.25 | fewer propositions → positive |
| `n_selfcorrect_r100` | 0.159 | +0.21 | more self-corrections → positive |
| `n_locative_r100` | 0.154 | −0.20 | fewer locatives → positive |
| `n_repeated_lemma_r100` | 0.102 | −0.13 | — |
| `mlu`, `n_hedge_r100`, `n_diminutive_r100` | ≤ 0.04 | ~0 | negligible |
| zeroed (9) | 0 | 0 | n_metacomment_r100, n_quantity_r100, n_region_r100, repeat_mass_r100, region_breadth, n_word, ttr, n_sent, n_comma |

**The single most important feature is a surface statistic** (`n_type`, ~1.6× the next), and
`n_char` is #4. The surviving LLM features are exactly the "empty-speech" markers — reduced
specificity/propositional density/locative precision, more deixis and self-correction — and
are the ones that were BH-significant in *both* extraction runs. So the model is using the
stable V2 features, but its biggest single lever is lexical diversity, which needs no LLM.

---

## 7. Notebook 10 — sealed one-shot test evaluation

Loads `pipe_v2_stable_surface.joblib` + `pipe_surface.joblib` and applies them to the 61 test
transcripts. Extraction model = `qwen3.5:122b` (MODEL_A — the run the pipeline was trained on).

| Block | What it does | Output |
|---|---|---|
| 0 | reload `frozen_spec.json` + artefact; assert feature count, prompt SHA, `test_set_used is False` | "safe to proceed" |
| 1 | **the only `fileDataset/test/` read in the project** — 61 docs (42 neg / 19 pos), surface stats via the identical `surface()` formula | median 65 words (train 69) |
| 2–3 | provider (MODEL_A); resumable extraction → `v2_test_raw.jsonl` | 61/61 succeeded, 0 failed |
| 4 | QC — failure rate / groundedness / missing fields | **groundedness mean 0.990, median 1.000, 0 below 0.5, 0 missing fields** (matches train QC 0.98) |
| 5 | build test feature matrix, assert all 20 frozen columns present | 61 × 30 LLM features, all frozen cols present |
| 6 | `predict_proba` (pure forward pass, no fitting) + bootstrap CI | see below |
| 7 | paired bootstrap primary vs surface | see below |
| 8 | write `test_results` + `test_set_used: true` + `test_set_used_utc` into `frozen_spec.json` | seal written 2026-09-04 |
| 9 | verdict | CI crosses 0; CV↔test gap +0.001 |

**FINAL TEST-SET RESULT (n = 61, one-shot):**

| model | test AUC | 95% CI | balAcc | macroF1 |
|---|---|---|---|---|
| **V2 stable + surface (frozen primary)** | **0.807** | 0.677 – 0.916 | 0.709 | 0.702 |
| Surface stats (frozen baseline) | 0.796 | 0.659 – 0.909 | 0.666 | 0.627 |

**Paired bootstrap (primary vs surface, 2000 resamples): ΔAUC = +0.011, 95% CI
[−0.103, +0.124], P(Δ≤0) = 0.42.**

- CV primary AUC 0.806 → test primary AUC 0.807 (**gap +0.001**) → no overfit; the frozen
  procedure generalised exactly as predicted.
- The in-sample ElasticNet edge over surface (+0.063, P ≈ 0.016) **does not hold out of
  sample** (+0.011, P = 0.42). With n = 61 the test paired test is underpowered — this is
  "not demonstrated", not a proven null — but it is the honest final number and it is
  consistent with the L2/SVM CV arms, which never reached significance either.

**External context (from earlier, separate work — not re-run here):** TF-IDF char 3–5-gram
test AUC ≈ 0.862; RoBERTa test AUC 0.742; V1 categorical test accuracy ≈ majority (0.69).
The frozen interpretable model's 0.807 sits above RoBERTa and below TF-IDF.

---

## 8. Consolidated verdict

| question | answer | evidence |
|---|---|---|
| Are V2 features stable across LLMs? | **Moderately.** Countable events yes (ρ 0.76–0.90); verb-type judgment no (ρ 0.42–0.44); `n_entities` no (ρ ≈ 0). Median ρ 0.68, 13/16 stable. Direction of the class association is preserved for all 16 features. | `v2_stability.csv`, nb 09 Block 7/8 |
| Do V2 features beat V1 / transcript length? | **Yes** — V2 rates vs V1 ΔAUC +0.075 (P = 0.036); vs length +0.074 (P = 0.045). Both were ≈ chance (0.69). | nb 09 Block 22 |
| Do V2 features add information beyond surface statistics? | **Not on held-out data.** CV: +0.063 AUC, ElasticNet-only, P ≈ 0.016; not reproduced by L2 (P 0.08) or SVM (P 0.11). Test: **+0.011, P = 0.42.** SHAP: the top feature is the surface stat `n_type`. | nb 09 Block 22, nb 10 Block 7, `shap_feature_importance.csv` |
| How far below the lexical ceiling? | Frozen primary CV 0.806 / test 0.807 vs TF-IDF CV 0.867 — gap ≈ 0.06 (paired P = 0.001). | nb 09 Block 22 |
| Overfitting? | No — CV→test gap = +0.001. EPV ≈ 3.5 (70 positives / 20 features) but ElasticNet keeps only 11 non-zero. | nb 10 Block 9, nb 09 Block 31 |

**Bottom line for the thesis.** The frozen interpretable model is a solid, honestly-built
artefact: reliable extraction (groundedness 0.99 on test), a leak-free tuned pipeline, a
verified freeze, and a single sealed test giving **AUC 0.807** — above RoBERTa, below TF-IDF.
But its predictive power is essentially the surface statistics: the LLM-extracted features add
no demonstrable information beyond word/type counts on held-out data. The stability check did
its job — it identified and removed the model-dependent features (`specific_verb_ratio`,
`n_generic_verb`, `n_entities`) before they reached the frozen pipeline.

---

## 9. What is covered / what is not

**Covered**
- Second-model stability check with a full re-extraction (qwen3.6-35b, 327/327).
- Two *automatic* gates (degeneracy `zero_share > 0.70`; stability flag `ρ ≥ 0.60`) — no
  eyeballing, reproducible for the methods section.
- Three classifiers (ElasticNet primary, L2 + calibrated Linear-SVM robustness).
- Leak-free pipelines throughout (scaler / vectoriser / OHE fitted inside every CV fold).
- Paired bootstrap for every comparison (never CI-overlap eyeballing).
- Frozen pipeline with a reload-and-assert consistency check and a deprecated-not-deleted
  audit trail for the superseded feature set.
- SHAP explanation of the exact frozen model (LinearExplainer — exact, deterministic).
- One-shot sealed test with an integrity guard (`test_set_used` flag) that blocks re-runs.
- CV→test consistency check (gap +0.001).

**Not covered / caveats an examiner could raise**
- **Only two LLMs, both Qwen-family.** A non-Qwen third model would be a stronger stability
  test; cross-family agreement is unmeasured.
- **Sample size.** 241 train / 61 test → wide CIs. The test paired test (P = 0.42) is
  underpowered; it cannot *prove* the null, only fail to reject it. The CV result (n = 241)
  is the more informative one and it is already marginal / classifier-fragile.
- **No test-set TF-IDF or V1 number in notebook 10** — only surface + frozen primary were run
  on the test set. The 0.862 TF-IDF and 0.742 RoBERTa test figures come from earlier separate
  notebooks, not this pipeline, so the test-set ceiling is not strictly comparable.
- **EPV ≈ 3.5** for the 20-feature frozen model (below the ~10 rule); mitigated by ElasticNet
  sparsity (11 non-zero) and confirmed by the clean CV→test transfer, but worth stating.
- **SHAP background subsampled** to 100/241 documents (a `shap` default warning) — minor,
  affects only the explanation plot precision, not the model or the metrics.
- **`specific_verb_ratio`** was one of the strongest length-independent univariate signals in
  the earlier assessment; the stability check reclassified it as model-dependent and dropped
  it. That is the right call, but it means the "cleanest" interpretable signal did not survive.

---

## 10. Reproducibility notes

- `RUN_LLM=False` re-runs notebooks 09/10 in analysis-only mode from the cached JSONLs.
- `SEED=42` everywhere; CV = 10×5 repeated stratified; bootstrap = 2000 resamples.
- Prompt is single-sourced in `helpers.py` and its SHA (`f52aae5d939d620e`) is asserted in
  09 Block 1, 09 Block 10b, and 10 Block 0 — a drifted prompt stops the run.
- `frozen_spec.json` is the single source of truth for the frozen state; it currently reads
  `test_set_used: true` (sealed 2026-09-04) with `test_results` embedded. Re-running notebook
  09 Block 10 preserves that seal; re-running notebook 10 is blocked at Block 0.
- Shared module: `helpers.py` (config constants, `V2_PROMPT`, `StreamingLocalProvider`,
  `make_provider`, `cv_proba`, `boot_ci`, `evaluate`, `paired_bootstrap`).
