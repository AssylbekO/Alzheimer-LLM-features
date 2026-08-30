# Local development setup

This project runs entirely locally — no Google Colab and no Google Drive.

## 1. System prerequisites

* **Python** — the version this environment was set up with is 3.14.
* **libomp** (macOS, for XGBoost):

  ```bash
  brew install libomp
  ```

## 2. Virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

The RoBERTa reference notebook (`Referencni-C5-RoBERTa.ipynb`) needs PyTorch and
Transformers, which are large and kept out of the default install:

```bash
pip install -r requirements-roberta.txt
```

## 3. Configuration / secrets

LLM access is configured through environment variables read from a local
`.env` file (never committed):

```bash
cp .env.example .env
# then edit .env and set LOCAL_OPENAI_API_KEY
```

| Variable | Purpose | Secret? |
|---|---|---|
| `LOCAL_OPENAI_BASE_URL` | LiteLLM / OpenAI-compatible endpoint (`https://litellm.vse.cz/v1`) | no |
| `LOCAL_MODEL_TEXT` | model name (`qwen3.5:122b`) | no |
| `LOCAL_OPENAI_API_KEY` | API key for the endpoint | **yes** |

`llm_feature_gen.providers.local_provider.LocalProvider` calls `load_dotenv()`
on import, so `LocalProvider()` with no arguments picks these up automatically.
The notebooks that build the provider explicitly read
`os.environ["LOCAL_OPENAI_API_KEY"]` — the key is never hard-coded.

## 4. Data

The transcript corpus ships with the repo as `fileDataset.zip` (and
`notebooks_preliminary/fileDataset.zip`). Notebooks unpack it next to
themselves on first run; the extracted `fileDataset/`, `overview/`, `train/`,
`test/` folders are git-ignored. `Data_lindat_zipformer_ft2_lm-extra06_allFeatures.pkl`
is the input for the RoBERTa reference notebook. `OutputsQwen/` holds
pre-computed Qwen feature tables used when a notebook runs with `RUN_LLM = False`.

## 5. Smoke test

```bash
python - <<'PY'
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from llm_feature_gen.providers.local_provider import LocalProvider

p = LocalProvider()
print("provider:", p.base_url, p.text_model)

tr = pd.read_csv("OutputsQwen/train_all_feature_values.csv")
te = pd.read_csv("OutputsQwen/test_all_feature_values.csv")
cols = [c for c in tr.columns if c not in {"File", "Class", "raw_llm_output"}]
prep = lambda d: d[cols].astype("string").fillna("missing")
le = LabelEncoder()
m = Pipeline([("p", ColumnTransformer([("c", OneHotEncoder(handle_unknown="ignore"), cols)])),
             ("clf", XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                                   eval_metric="logloss", random_state=42))])
m.fit(prep(tr), le.fit_transform(tr["Class"]))
pred = le.inverse_transform(m.predict(prep(te)))
print(classification_report(te["Class"].astype(str), pred, digits=3))
PY
```
