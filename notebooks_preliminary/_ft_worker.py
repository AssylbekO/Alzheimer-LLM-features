"""Standalone fine-tuning worker for notebook 12's fine-tuned classifier arm.

Runs as a SEPARATE PROCESS per model (invoked via subprocess from the notebook), not
imported. This is a deliberate isolation measure, not a style choice: sequentially
fine-tuning two different transformer architectures on MPS within one process was
found to reliably hang (backward pass on the second model never completes -- process
sits in an uninterruptible-sleep state; confirmed via an isolated repro before writing
this file). Giving each model its own fresh process/MPS context avoids the hang
entirely, at the cost of a bit of duplicated setup code between this file and Block 2
of the notebook (the doc-loading logic below is intentionally identical to Block 2's).

Usage: python _ft_worker.py <model_key>
Writes: fileDataset/outputs/<model_key>_finetuned_oof.npz  (cv_proba, test_proba, y_train, y_test)
"""
import sys, re, json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          get_linear_schedule_with_warmup, set_seed)
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import AdamW
from sklearn.model_selection import train_test_split, StratifiedKFold

SEED = 42

MODEL_CONFIGS = {
    "fernet_c5_roberta":  "fav-kky/FERNET-C5-RoBERTa",
    "robeczech_base":     "ufal/robeczech-base",
    "czert_b_base_cased": "UWB-AIR/Czert-B-base-cased",
    "mbert_base_cased":   "google-bert/bert-base-multilingual-cased",
    "small_e_czech":      "Seznam/small-e-czech",
}

FT_N_REPEATS, FT_N_SPLITS = 1, 5          # single 5-fold CV -- see notebook Block 7 markdown
FT_MAX_LENGTH, FT_BATCH_SIZE, FT_EPOCHS = 256, 8, 8
FT_LR, FT_WARMUP_RATIO, FT_PATIENCE = 1e-5, 0.1, 2
FT_VAL_FRACTION, FT_WEIGHT_DECAY = 0.20, 0.01


def read_split(folder, label):
    return [{"file": f.name, "label": label,
             "text": f.read_text(encoding="utf-8", errors="replace").strip()}
            for f in sorted(folder.glob("*.txt"))]


def load_docs(data_dir):
    train_docs = pd.DataFrame(read_split(data_dir / "train" / "negative", 0) +
                              read_split(data_dir / "train" / "positive", 1))
    train_docs = train_docs.sort_values("file").reset_index(drop=True)   # == Block 2
    test_docs = pd.DataFrame(read_split(data_dir / "test" / "negative", 0) +
                             read_split(data_dir / "test" / "positive", 1))
    assert len(train_docs) == 241 and len(test_docs) == 61
    return (train_docs.text.tolist(), train_docs.label.values.astype(int),
            test_docs.text.tolist(), test_docs.label.values.astype(int))


def make_loader(tok, texts, labels, shuffle, seed, device):
    enc = tok(list(texts), padding=True, truncation=True, max_length=FT_MAX_LENGTH,
              return_tensors="pt")
    ds = TensorDataset(enc["input_ids"], enc["attention_mask"],
                       torch.tensor(labels, dtype=torch.long))
    gen = torch.Generator(); gen.manual_seed(seed)
    return DataLoader(ds, batch_size=FT_BATCH_SIZE, shuffle=shuffle,
                      generator=gen if shuffle else None)


def run_loss(model, loader, device):
    model.eval(); tot = 0.0
    with torch.no_grad():
        for ids, mask, lab in loader:
            out = model(input_ids=ids.to(device), attention_mask=mask.to(device),
                        labels=lab.to(device))
            tot += out.loss.item()
    return tot / max(len(loader), 1)


def predict_proba(model, loader, device):
    model.eval(); probs = []
    with torch.no_grad():
        for ids, mask, _ in loader:
            out = model(input_ids=ids.to(device), attention_mask=mask.to(device))
            probs.append(torch.softmax(out.logits, dim=1)[:, 1].cpu().numpy())
    return np.concatenate(probs)


def finetune_predict(hf_name, train_texts, train_labels, predict_texts, seed, device, tag=""):
    """Fine-tune fresh on (train_texts, train_labels) with an 80/20 fold-internal val
    split for early stopping, then predict_proba on predict_texts. predict_texts labels
    are never seen -- neither for training nor for model selection."""
    # seeds python/numpy/torch (+CUDA) before the model is built, so the randomly
    # initialised classification head and dropout masks are identical across re-runs.
    set_seed(seed)
    tr_idx, va_idx = train_test_split(np.arange(len(train_texts)), test_size=FT_VAL_FRACTION,
                                      random_state=seed, stratify=train_labels)
    tok = AutoTokenizer.from_pretrained(hf_name)
    model = AutoModelForSequenceClassification.from_pretrained(hf_name, num_labels=2).to(device)

    inner_tr = make_loader(tok, [train_texts[i] for i in tr_idx], train_labels[tr_idx],
                           True, seed, device)
    inner_va = make_loader(tok, [train_texts[i] for i in va_idx], train_labels[va_idx],
                           False, seed, device)
    pred_loader = make_loader(tok, predict_texts, np.zeros(len(predict_texts), dtype=int),
                              False, seed, device)

    opt = AdamW(model.parameters(), lr=FT_LR, weight_decay=FT_WEIGHT_DECAY)
    total_steps = len(inner_tr) * FT_EPOCHS
    sched = get_linear_schedule_with_warmup(
        opt, num_warmup_steps=int(FT_WARMUP_RATIO * total_steps), num_training_steps=total_steps)

    best_val, best_state, bad_epochs, stopped_epoch = float("inf"), None, 0, FT_EPOCHS
    for ep in range(FT_EPOCHS):
        model.train()
        for ids, mask, lab in inner_tr:
            opt.zero_grad(set_to_none=True)
            out = model(input_ids=ids.to(device), attention_mask=mask.to(device),
                        labels=lab.to(device))
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step(); sched.step()
        val_loss = run_loss(model, inner_va, device)
        if val_loss < best_val - 1e-4:
            best_val, bad_epochs = val_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= FT_PATIENCE:
                stopped_epoch = ep + 1
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"    {tag}: stopped at epoch {stopped_epoch}/{FT_EPOCHS}, best val_loss={best_val:.4f}",
          flush=True)
    return predict_proba(model, pred_loader, device)


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in MODEL_CONFIGS:
        print(f"usage: python _ft_worker.py <{'|'.join(MODEL_CONFIGS)}>")
        sys.exit(2)
    key = sys.argv[1]
    hf_name = MODEL_CONFIGS[key]

    data_dir = Path("fileDataset")
    out_dir = data_dir / "outputs"
    texts_train, y_train, texts_test, y_test = load_docs(data_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else
                          ("mps" if torch.backends.mps.is_available() else "cpu"))
    # pin cuDNN to deterministic kernels (no-op on MPS/CPU); costs a little speed.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[{key}] device={device}  hf={hf_name}  protocol={FT_N_REPEATS}x{FT_N_SPLITS}-fold CV",
          flush=True)

    acc = np.zeros(len(y_train))
    for r in range(FT_N_REPEATS):
        skf = StratifiedKFold(n_splits=FT_N_SPLITS, shuffle=True, random_state=SEED + r)
        for fold_i, (tr_i, va_i) in enumerate(skf.split(np.zeros(len(y_train)), y_train)):
            proba_fold = finetune_predict(
                hf_name, [texts_train[i] for i in tr_i], y_train[tr_i],
                [texts_train[i] for i in va_i], seed=SEED + r, device=device,
                tag=f"fold {fold_i + 1}/{FT_N_SPLITS}")
            acc[va_i] += proba_fold
    p_cv = acc / FT_N_REPEATS

    p_test = finetune_predict(hf_name, texts_train, y_train, texts_test, seed=SEED,
                              device=device, tag="test fit")

    out_path = out_dir / f"{key}_finetuned_oof.npz"
    np.savez(out_path, cv_proba=p_cv, test_proba=p_test, y_train=y_train, y_test=y_test)
    print(f"[{key}] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
