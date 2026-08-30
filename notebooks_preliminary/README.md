# Preliminary experiment — LLM features for cognitive-impairment screening

Six notebooks prepared for the thesis consultation. Run them in order; each writes to
`outputs/` and the next one reads from there.

## Setup

Put these beside the notebooks:

```
fileDataset/        # unzipped fileDataset.zip  (overview/, train/, test/)
OutputsQwen/        # existing Qwen feature CSVs
```

`pip install llm-feature-gen scikit-learn xgboost matplotlib pandas scipy`

## The rule followed throughout

The labelled **test set is never used** — not for feature design, not for model selection,
not for reporting. Notebooks 03–06 load `train/` only. `overview/` (= `train.extra`, 86
unlabelled documents) is used for LLM feature discovery and prompt experiments, so the
schema is designed without ever seeing the class structure.

## The notebooks

| # | Notebook | What it answers |
|---|---|---|
| 01 | `01_dataset_overview.ipynb` | What is in the corpus; duplicates, length, class balance |
| 02 | `02_llm_feature_extraction.ipynb` | Discovery + generation with `llm-feature-gen` |
| 03 | `03_feature_inspection.ipynb` | Are the LLM features degenerate, length proxies, or discriminative? |
| 04 | `04_ml_baselines.ipynb` | How much signal does each representation carry? |
| 05 | `05_final_comparison.ipynb` | The comparison table and figure |
| 06 | `06_v2_features_demo.ipynb` | Proposed v2 features: counts, rates, evidence spans |

Notebooks 02 and 06 have `RUN_LLM = False` at the top so they execute without the endpoint,
using the feature table already in `OutputsQwen/`. Set it to `True` and fill in your API key
to regenerate. Notebook 06 also uses a transparent rule-based stand-in for the v2 constructs
so the direction can be evaluated before spending LLM calls.

Figures were stripped from the Drive copies of notebooks 05 and 06 (an exact copy of a large
base64 image cannot be guaranteed through this upload path). All numeric results and tables
are intact; re-run the plotting cells to render the figures, or use the PNGs in `outputs/`.

## Headline results (10×5-fold CV on 241 training documents)

| Representation | AUC | Balanced acc. | Macro F1 |
|---|---|---|---|
| Majority class | 0.500 | 0.500 | 0.415 |
| Transcript length only | 0.693 | 0.639 | 0.606 |
| LLM features v1 (logreg) | 0.694 | 0.632 | 0.620 |
| LLM features v1 (XGBoost) | 0.657 | 0.574 | 0.576 |
| Surface statistics | 0.742 | 0.672 | 0.644 |
| Surface + LLM features | 0.757 | 0.703 | 0.693 |
| **v2 counts (rule-based stand-in)** | **0.835** | **0.783** | **0.761** |
| TF-IDF word 1–2gram | 0.842 | 0.752 | 0.760 |
| TF-IDF char 3–5gram | 0.868 | 0.753 | 0.758 |

Three things to take from this:

1. **The v1 categorical LLM features do not beat transcript length** (0.694 vs 0.693), even
   though seven of the ten features are individually associated with diagnosis after
   multiple-testing correction. The constructs are sound; the 3–5 level encoding discards them.
2. **Flexible models do worse, not better,** on those features — 241 documents produce 221
   distinct feature vectors, so a tree ensemble memorises rather than generalises.
3. **Count-based features close most of the gap to TF-IDF.** Even a crude keyword
   implementation of the v2 constructs reaches 0.835. That is the argument for spending LLM
   budget on the v2 schema.

## Caveats to state at the consultation

- All confidence intervals are wide; 241 documents supports "indistinguishable", not rankings.
- The v2 numbers come from a keyword stand-in whose lists were written after looking at the
  training data. They are optimistic. The LLM version, which never sees labels, is the next
  experiment.
- The transcripts carry no participant ID, so age, sex and education cannot be controlled for.
  Worth requesting that key.
