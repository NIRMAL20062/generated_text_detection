"""Fine-tuned RoBERTa detector (role B): human (0) vs machine (1).

    python -m src.train_roberta --setting indist [--dev]
    python -m src.train_roberta --setting paraphrase_orig --predict-only --model outputs/roberta/indist/model

Training (default): fine-tune `roberta-base` on the setting's `train` ids, pick the checkpoint with the best val
AUROC, then score the val + test ids and write `results/roberta/<setting>/preds.csv` + `run_info.json`
(contract: docs/TASKS.md §0.3, score = P(machine), higher = more AI).

--predict-only: no training. Score a setting with an already trained model. This is how the `paraphrase*`
settings are scored (they have no train split: the `indist` model is reused, TASKS.md §0.2).

Why these choices (docs/TASKS.md §B):
  - class-weighted cross-entropy: dedup keeps every machine text but each human text once, so machine texts
    outnumber humans; weights stop the model from just answering "machine".
  - best checkpoint on val AUROC: with 1 epoch, val is checked several times during the epoch and the best
    weights are kept (val is never the test set).
  - fp16 only on a GPU (autocast); CPU runs in fp32.
--dev: tiny stratified subsets, max_len 32, finishes in about 2 minutes on CPU. Dev outputs go to results_dev/.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import ROOT, SEED, load_split, load_texts

RESULTS_DIR = ROOT / "results"
DEV_RESULTS_DIR = ROOT / "results_dev"   # src.eval --dev refuses predictions inside results/
MODEL_ROOT = ROOT / "outputs" / "roberta"  # gitignored: weights never go to git
DEV_CAPS = {"train": 32, "val": 24, "test": 24}  # rows per split in --dev
DEV_MAX_LEN = 32
NO_TRAIN_PREFIX = "paraphrase"  # these settings have an empty train split: use --predict-only


# ---------------------------------------------------------------- small pure helpers (unit-tested)

def class_weights(labels) -> list[float]:
    """Weights for the loss, inverse to class frequency: w_c = n / (2 * n_c). Balanced data gives [1, 1]."""
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=2)
    if (counts == 0).any():
        raise ValueError(f"need both classes in the train split, got counts {counts.tolist()}")
    return (counts.sum() / (2.0 * counts)).tolist()


def stratified_cap(df: pd.DataFrame, n: int, seed: int = SEED) -> pd.DataFrame:
    """At most n rows with both labels kept (n // 2 each when available); used only for --dev."""
    parts = []
    for label in (0, 1):
        sub = df[df["label"] == label].sort_values("id")
        parts.append(sub.sample(n=min(len(sub), n // 2), random_state=seed))
    return pd.concat(parts).sort_values("id")


def eval_steps(steps_per_epoch: int, epochs: int, evals_per_epoch: int) -> set[int]:
    """1-based optimiser steps after which val AUROC is computed. The last step is always included."""
    steps = {max(1, round(e * steps_per_epoch + k * steps_per_epoch / evals_per_epoch))
             for e in range(epochs) for k in range(1, evals_per_epoch + 1)}
    steps.add(steps_per_epoch * epochs)
    return {s for s in steps if s <= steps_per_epoch * epochs}


def auroc_or_none(y, s):
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, s)) if len(set(np.asarray(y).tolist())) == 2 else None


# ---------------------------------------------------------------- model helpers

def pick_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def collate(seqs: list[list[int]], pad_id: int, device: str):
    """Pad a list of token-id lists to the longest one. Returns (input_ids, attention_mask) tensors."""
    import torch

    longest = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), longest), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), longest), dtype=torch.long)
    for r, s in enumerate(seqs):
        ids[r, :len(s)] = torch.tensor(s)
        mask[r, :len(s)] = 1
    return ids.to(device), mask.to(device)


def predict_scores(model, seqs, pad_id: int, device: str, batch_size: int, amp: bool) -> np.ndarray:
    """P(machine) for every sequence, in the original order. Batches by length so padding stays small."""
    import torch

    model.eval()
    order = np.argsort([len(s) for s in seqs])
    out = np.zeros(len(seqs), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            ids, mask = collate([seqs[i] for i in idx], pad_id, device)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=amp):
                logits = model(input_ids=ids, attention_mask=mask).logits
            out[idx] = torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()
    return out


def train_model(model, train_seqs, train_y, val_seqs, val_y, pad_id: int, device: str, args, amp: bool) -> dict:
    """Fine-tune in place. Keeps the weights with the best val AUROC and loads them back at the end."""
    import torch
    from transformers import get_linear_schedule_with_warmup

    weights = torch.tensor(class_weights(train_y), dtype=torch.float32, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps_per_epoch = math.ceil(len(train_seqs) / args.batch_size)
    total = steps_per_epoch * args.epochs
    sched = get_linear_schedule_with_warmup(optim, int(0.06 * total), total)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    checks = eval_steps(steps_per_epoch, args.epochs, args.evals_per_epoch)
    rng = np.random.default_rng(SEED)
    y = np.asarray(train_y)
    best, best_state, step, t0 = None, None, 0, time.time()
    for epoch in range(args.epochs):
        model.train()
        perm = rng.permutation(len(train_seqs))
        for start in range(0, len(perm), args.batch_size):
            idx = perm[start:start + args.batch_size]
            ids, mask = collate([train_seqs[i] for i in idx], pad_id, device)
            labels = torch.tensor(y[idx], dtype=torch.long, device=device)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=amp):
                logits = model(input_ids=ids, attention_mask=mask).logits
            loss = loss_fn(logits.float(), labels)
            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()
            sched.step()
            step += 1
            if step % 50 == 0 or step == 1:
                print(f"  epoch {epoch + 1} step {step}/{total} loss {loss.item():.4f} ({time.time() - t0:.0f}s)", flush=True)
            if step in checks:
                auc = auroc_or_none(val_y, predict_scores(model, val_seqs, pad_id, device, args.eval_batch_size, amp))
                print(f"  step {step}: val AUROC {auc if auc is None else round(auc, 4)}", flush=True)
                model.train()
                if auc is not None and (best is None or auc > best):
                    best = auc
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"best_val_auroc": best, "train_seconds": round(time.time() - t0, 1), "steps": step}


# ---------------------------------------------------------------- bookkeeping

def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def hardware(device: str) -> str:
    if device == "cuda":
        import torch

        return torch.cuda.get_device_name(0)
    return "CPU"


def rows_for(split: dict, part: str, by_id: pd.DataFrame, cap: int | None) -> pd.DataFrame:
    """The rows of one split part, in a stable order; fails loudly if an id has no text."""
    ids = split.get(part, [])
    missing = [i for i in ids if i not in by_id.index]
    if missing:
        raise KeyError(f"{len(missing)} {part} ids not in load_texts(), e.g. {missing[:3]}"
                       " (paraphrased ids need data/processed/paraphrased.parquet)")
    df = by_id.loc[ids].reset_index()
    return stratified_cap(df, cap) if cap else df


# ---------------------------------------------------------------- main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--setting", required=True, help="split name, e.g. indist, logo_chatGPT, xdomain_arxiv, paraphrase_orig")
    ap.add_argument("--dev", action="store_true", help="tiny subsets on data/sample_dev.parquet; CPU, about 2 minutes")
    ap.add_argument("--predict-only", action="store_true", help="no training: score the setting with --model")
    ap.add_argument("--model", default=None,
                    help="training: base model name (default roberta-base). --predict-only: path of a saved model dir")
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--eval-batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--evals-per-epoch", type=int, default=2, help="how often val AUROC is checked for the best checkpoint")
    ap.add_argument("--no-save", action="store_true", help="do not write model weights to outputs/")
    args = ap.parse_args(argv)
    if args.predict_only and not args.model:
        ap.error("--predict-only needs --model <path to a saved model directory>")
    if not args.predict_only and args.setting.startswith(NO_TRAIN_PREFIX):
        ap.error(f"{args.setting} has no train split: score it with --predict-only --model outputs/roberta/indist/model")

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = pick_device()
    amp = device == "cuda"  # fp16 autocast only on a GPU
    max_len = min(args.max_len, DEV_MAX_LEN) if args.dev else args.max_len
    start = time.time()

    texts = load_texts(dev=args.dev).drop_duplicates("id").set_index("id")
    split = load_split(args.setting, dev=args.dev)
    caps = DEV_CAPS if args.dev else {}
    val_df = rows_for(split, "val", texts, caps.get("val"))
    test_df = rows_for(split, "test", texts, caps.get("test"))

    source = args.model if args.predict_only else (args.model or "roberta-base")
    tok = AutoTokenizer.from_pretrained(source)
    model = AutoModelForSequenceClassification.from_pretrained(source, num_labels=2).to(device)
    pad_id = tok.pad_token_id

    def encode(df):
        return tok(df["text"].tolist(), truncation=True, max_length=max_len)["input_ids"]

    val_seqs, test_seqs = encode(val_df), encode(test_df)
    info = {"best_val_auroc": None, "train_seconds": 0.0, "steps": 0}
    n_train = 0
    if not args.predict_only:
        train_df = rows_for(split, "train", texts, caps.get("train"))
        n_train = len(train_df)
        print(f"{args.setting}: train {n_train} / val {len(val_df)} / test {len(test_df)}, "
              f"model={source}, device={device}, max_len={max_len}", flush=True)
        info = train_model(model, encode(train_df), train_df["label"].tolist(), val_seqs, val_df["label"].tolist(),
                           pad_id, device, args, amp)
        if not args.no_save and not args.dev:
            out_model = MODEL_ROOT / args.setting / "model"
            model.save_pretrained(out_model)
            tok.save_pretrained(out_model)
            print(f"saved model to {out_model}", flush=True)

    val_scores = predict_scores(model, val_seqs, pad_id, device, args.eval_batch_size, amp)
    t_pred = time.time()
    test_scores = predict_scores(model, test_seqs, pad_id, device, args.eval_batch_size, amp)
    pred_seconds = time.time() - t_pred

    preds = pd.concat([
        pd.DataFrame({"id": val_df["id"], "split": "val", "score": val_scores}),
        pd.DataFrame({"id": test_df["id"], "split": "test", "score": test_scores}),
    ], ignore_index=True)
    out = (DEV_RESULTS_DIR if args.dev else RESULTS_DIR) / "roberta" / args.setting
    out.mkdir(parents=True, exist_ok=True)
    preds.to_csv(out / "preds.csv", index=False)
    run_info = {
        "method": "roberta", "setting": args.setting, "git_commit": git_commit(), "seed": SEED, "dev": args.dev,
        "predict_only": args.predict_only, "model": source, "max_len": max_len, "batch_size": args.batch_size,
        "lr": args.lr, "epochs": 0 if args.predict_only else args.epochs, "fp16": amp,
        "class_weights": None if args.predict_only else class_weights(train_df["label"].tolist()),
        "n_train": n_train, "n_val": len(val_df), "n_test": len(test_df),
        "best_val_auroc": info["best_val_auroc"], "train_seconds": info["train_seconds"], "steps": info["steps"],
        "seconds_per_text_inference": round(pred_seconds / max(len(test_df), 1), 5),
        "hardware": hardware(device), "total_runtime_seconds": round(time.time() - start, 1),
    }
    (out / "run_info.json").write_text(json.dumps(run_info, indent=2))
    print(f"wrote {out}/preds.csv ({len(preds)} rows) and run_info.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
