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
--dev: tiny stratified train subset (val/test whole), max_len 32, finishes in about 2 minutes on CPU. Dev outputs go to results_dev/.

The idea in plain words: RoBERTa is a pretrained Transformer that already "knows" English. We put a small 2-class
head on top (human / machine) and train the whole network for one epoch on labelled texts, so it learns which
patterns separate the two. Unlike the zero-shot scores (src/zeroshot.py), it needs training data, so it is very
strong on generators it has seen and weaker on unseen ones (the LOGO settings test exactly that).

Input:  src.data.load_texts() + splits/<setting>.json (train/val/test ids).
Output: results/roberta/<setting>/preds.csv (id,split,score for every val + test id) and run_info.json
        (git commit, seed, hyperparameters, hardware, runtime); model weights in outputs/roberta/<setting>/model
        (gitignored). Then grade with `python -m src.eval results/roberta/<setting>/preds.csv`.

Pipeline of main(): load texts and split -> tokenise (truncate to max_len tokens) -> train_model() with
class-weighted loss, keeping the best-val-AUROC weights -> predict_scores() on val and test -> write files.
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
DEV_CAPS = {"train": 32}  # rows per split in --dev; val/test stay whole (src.eval needs 100% test coverage)
DEV_MAX_LEN = 32
NO_TRAIN_PREFIX = "paraphrase"  # these settings have an empty train split: use --predict-only


# ---------------------------------------------------------------- small pure helpers (unit-tested)

def class_weights(labels) -> list[float]:
    """Weights for the loss, inverse to class frequency: w_c = n / (2 * n_c). Balanced data gives [1, 1].

    Worked example: 100 humans and 200 machines -> n = 300, w_human = 300 / 200 = 1.5, w_machine = 300 / 400 = 0.75.
    A mistake on a (rarer) human text now costs twice as much as one on a machine text, so the model cannot get a
    low loss by answering "machine" for everything. Both classes must be present, or the weight would be infinite.
    """
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=2)
    if (counts == 0).any():
        raise ValueError(f"need both classes in the train split, got counts {counts.tolist()}")
    return (counts.sum() / (2.0 * counts)).tolist()


def stratified_cap(df: pd.DataFrame, n: int, seed: int = SEED) -> pd.DataFrame:
    """At most n rows with both labels kept (n // 2 each when available); used only for --dev.

    Why stratified: a random 32 rows could, by bad luck, contain only one class, and class_weights() would then
    fail. Taking n // 2 per label guarantees both. A smaller class is not oversampled (no row is repeated).
    """
    parts = []
    for label in (0, 1):
        # sort first so the same seed always picks the same rows, whatever order the parquet file had
        sub = df[df["label"] == label].sort_values("id")
        parts.append(sub.sample(n=min(len(sub), n // 2), random_state=seed))
    return pd.concat(parts).sort_values("id")


def eval_steps(steps_per_epoch: int, epochs: int, evals_per_epoch: int) -> set[int]:
    """1-based optimiser steps after which val AUROC is computed. The last step is always included.

    Why check during the epoch: we train for only 1 epoch, so "pick the best epoch" is impossible; instead val
    AUROC is measured evals_per_epoch times (default 2: halfway and at the end) and the best weights are kept.
    Example: 100 steps, 1 epoch, 2 checks -> {50, 100}.
    """
    # evenly spaced checks inside every epoch; round() because steps are whole numbers, max(1, .) so never step 0
    steps = {max(1, round(e * steps_per_epoch + k * steps_per_epoch / evals_per_epoch))
             for e in range(epochs) for k in range(1, evals_per_epoch + 1)}
    steps.add(steps_per_epoch * epochs)
    return {s for s in steps if s <= steps_per_epoch * epochs}


def auroc_or_none(y, s):
    """Val AUROC of scores s for labels y, or None if y has only one class.

    AUROC = the chance that a random machine text gets a higher score than a random human text (1.0 = perfect,
    0.5 = coin flip). It is undefined with one class (there is no pair to compare), which can happen on a tiny
    --dev val set; returning None lets training continue instead of crashing. Used only to pick the checkpoint;
    the reported metrics come from src.eval.
    """
    from sklearn.metrics import roc_auc_score   # imported here so importing this module stays fast for tests

    return float(roc_auc_score(y, s)) if len(set(np.asarray(y).tolist())) == 2 else None


# ---------------------------------------------------------------- model helpers

def pick_device() -> str:
    """'cuda' when a GPU is visible (Kaggle T4), else 'cpu' (local --dev runs). torch is imported lazily so the
    pure helpers above can be unit-tested without torch installed."""
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def collate(seqs: list[list[int]], pad_id: int, device: str):
    """Pad a list of token-id lists to the longest one. Returns (input_ids, attention_mask) tensors.

    Texts have different lengths but a batch must be a rectangle, so short ones are filled with pad_id. The
    attention mask (1 = real token, 0 = padding) tells the model to ignore the padding, so padding never changes
    a prediction. Example: [[5, 6, 7], [5, 6]] -> ids [[5, 6, 7], [5, 6, pad]], mask [[1, 1, 1], [1, 1, 0]].
    """
    import torch

    longest = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), longest), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), longest), dtype=torch.long)
    for r, s in enumerate(seqs):
        ids[r, :len(s)] = torch.tensor(s)
        mask[r, :len(s)] = 1
    return ids.to(device), mask.to(device)


def predict_scores(model, seqs, pad_id: int, device: str, batch_size: int, amp: bool) -> np.ndarray:
    """P(machine) for every sequence, in the original order. Batches by length so padding stays small.

    This probability is the `score` in preds.csv (higher = more likely AI, as the contract requires).
    """
    import torch

    model.eval()   # turns off dropout, so the same text always gets the same score
    # sort by length: texts of similar length share a batch, so little compute is wasted on padding
    order = np.argsort([len(s) for s in seqs])
    out = np.zeros(len(seqs), dtype=np.float64)
    with torch.no_grad():   # no gradients needed for prediction: faster and uses less GPU memory
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            ids, mask = collate([seqs[i] for i in idx], pad_id, device)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=amp):
                logits = model(input_ids=ids, attention_mask=mask).logits
            # softmax turns the 2 logits into probabilities; column 1 = class "machine". Writing to out[idx]
            # puts each score back at its text's original position (undoing the length sort).
            out[idx] = torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()
    return out


def train_model(model, train_seqs, train_y, val_seqs, val_y, pad_id: int, device: str, args, amp: bool) -> dict:
    """Fine-tune in place. Keeps the weights with the best val AUROC and loads them back at the end.

    Takes the model, tokenised train/val texts with labels, and the CLI args (lr, batch size, epochs, checks per
    epoch). Returns {"best_val_auroc", "train_seconds", "steps"} for run_info.json.
    Val is used only to choose the checkpoint, never to update weights; test is never seen here.
    """
    import torch
    from transformers import get_linear_schedule_with_warmup

    # class-weighted cross-entropy: machine texts outnumber humans ~1.9 : 1, see class_weights()
    weights = torch.tensor(class_weights(train_y), dtype=torch.float32, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    # AdamW = Adam with weight decay applied correctly; the standard optimiser for fine-tuning Transformers
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps_per_epoch = math.ceil(len(train_seqs) / args.batch_size)
    total = steps_per_epoch * args.epochs
    # learning rate rises from 0 over the first 6% of steps (warm-up: big early updates could wreck the pretrained
    # weights), then falls linearly to 0 at the last step
    sched = get_linear_schedule_with_warmup(optim, int(0.06 * total), total)
    # fp16 numbers can underflow to 0 for tiny gradients; the scaler multiplies the loss before backward() and
    # divides again before the update. Disabled (a no-op) on CPU.
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    checks = eval_steps(steps_per_epoch, args.epochs, args.evals_per_epoch)
    rng = np.random.default_rng(SEED)   # seeded shuffle: the same run twice sees batches in the same order
    y = np.asarray(train_y)
    best, best_state, step, t0 = None, None, 0, time.time()
    for epoch in range(args.epochs):
        model.train()   # dropout on during training
        perm = rng.permutation(len(train_seqs))   # new random order of training texts every epoch
        for start in range(0, len(perm), args.batch_size):
            idx = perm[start:start + args.batch_size]
            ids, mask = collate([train_seqs[i] for i in idx], pad_id, device)
            labels = torch.tensor(y[idx], dtype=torch.long, device=device)
            # forward pass in fp16 on the GPU (about 2x faster on a T4); the loss is computed in fp32 for accuracy
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=amp):
                logits = model(input_ids=ids, attention_mask=mask).logits
            loss = loss_fn(logits.float(), labels)
            optim.zero_grad(set_to_none=True)   # clear last step's gradients
            scaler.scale(loss).backward()        # gradients of the (scaled) loss
            scaler.unscale_(optim)               # back to true gradient size before clipping
            # gradient clipping: cap the overall gradient size at 1.0 so one odd batch cannot make a huge jump
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim)                   # update the weights (skipped if fp16 overflowed)
            scaler.update()
            sched.step()                         # move the learning rate one step along the warm-up/decay schedule
            step += 1
            if step % 50 == 0 or step == 1:
                print(f"  epoch {epoch + 1} step {step}/{total} loss {loss.item():.4f} ({time.time() - t0:.0f}s)", flush=True)
            if step in checks:
                # checkpoint selection: score all of val, keep a copy of the weights if this is the best so far
                auc = auroc_or_none(val_y, predict_scores(model, val_seqs, pad_id, device, args.eval_batch_size, amp))
                print(f"  step {step}: val AUROC {auc if auc is None else round(auc, 4)}", flush=True)
                model.train()   # predict_scores() switched to eval mode; switch back
                if auc is not None and (best is None or auc > best):
                    best = auc
                    # copy to CPU memory: a GPU copy of all weights would double the GPU memory used
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)   # end with the best checkpoint, not simply the last one
    return {"best_val_auroc": best, "train_seconds": round(time.time() - t0, 1), "steps": step}


# ---------------------------------------------------------------- bookkeeping

def git_commit() -> str:
    """The current git commit hash, saved in run_info.json so every result can be traced back to the exact code
    that produced it (AGENTS.md rule). 'unknown' if git is not available, e.g. a zip download on Kaggle."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def hardware(device: str) -> str:
    """GPU name (e.g. 'Tesla T4') or 'CPU', for run_info.json: runtimes are only comparable on the same hardware."""
    if device == "cuda":
        import torch

        return torch.cuda.get_device_name(0)
    return "CPU"


def rows_for(split: dict, part: str, by_id: pd.DataFrame, cap: int | None) -> pd.DataFrame:
    """The rows of one split part, in a stable order; fails loudly if an id has no text.

    `split` is the loaded splits/<setting>.json, `part` is 'train', 'val' or 'test', `by_id` is load_texts()
    indexed by id, and `cap` limits the rows (--dev only). A missing id is an error rather than a silent skip:
    a skipped test text would make src.eval reject the predictions anyway, and later is harder to debug.
    """
    ids = split.get(part, [])
    missing = [i for i in ids if i not in by_id.index]
    if missing:
        raise KeyError(f"{len(missing)} {part} ids not in load_texts(), e.g. {missing[:3]}"
                       " (paraphrased ids need data/processed/paraphrased.parquet)")
    df = by_id.loc[ids].reset_index()
    return stratified_cap(df, cap) if cap else df


# ---------------------------------------------------------------- main

def main(argv=None) -> int:
    """Command-line entry point: train (or load) the model for one setting and write preds.csv + run_info.json.

    `argv` lets tests call main([...]) directly instead of running a subprocess. Returns 0 on success.
    """
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
    # catch impossible combinations before the slow model download / GPU setup starts
    if args.predict_only and not args.model:
        ap.error("--predict-only needs --model <path to a saved model directory>")
    if not args.predict_only and args.setting.startswith(NO_TRAIN_PREFIX):
        ap.error(f"{args.setting} has no train split: score it with --predict-only --model outputs/roberta/indist/model")

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    # SEED = 42 everywhere: the classifier head's random start and the shuffling are the same on every run
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = pick_device()
    amp = device == "cuda"  # fp16 autocast only on a GPU
    max_len = min(args.max_len, DEV_MAX_LEN) if args.dev else args.max_len
    start = time.time()

    # data only through the shared loaders (AGENTS.md rule); index by id for fast lookup of each split's rows
    texts = load_texts(dev=args.dev).drop_duplicates("id").set_index("id")
    split = load_split(args.setting, dev=args.dev)
    caps = DEV_CAPS if args.dev else {}
    val_df = rows_for(split, "val", texts, caps.get("val"))
    test_df = rows_for(split, "test", texts, caps.get("test"))

    # training starts from pretrained roberta-base (a fresh 2-class head is added on top);
    # --predict-only loads a model we fine-tuned earlier (e.g. the indist model for the paraphrase settings)
    source = args.model if args.predict_only else (args.model or "roberta-base")
    tok = AutoTokenizer.from_pretrained(source)
    model = AutoModelForSequenceClassification.from_pretrained(source, num_labels=2).to(device)
    pad_id = tok.pad_token_id

    def encode(df):
        """Text -> token ids, cut to max_len tokens (256 by default: long texts keep only their beginning, which
        keeps training affordable on a T4; a limitation noted in the report)."""
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
        # save the weights (outputs/ is gitignored) so the paraphrase settings can reuse the indist model
        if not args.no_save and not args.dev:
            out_model = MODEL_ROOT / args.setting / "model"
            model.save_pretrained(out_model)
            tok.save_pretrained(out_model)
            print(f"saved model to {out_model}", flush=True)

    # score val as well as test: src.eval picks the accuracy/F1 threshold on val, then applies it to test
    val_scores = predict_scores(model, val_seqs, pad_id, device, args.eval_batch_size, amp)
    t_pred = time.time()   # time test scoring alone, for the compute-cost comparison in the report
    test_scores = predict_scores(model, test_seqs, pad_id, device, args.eval_batch_size, amp)
    pred_seconds = time.time() - t_pred

    preds = pd.concat([
        pd.DataFrame({"id": val_df["id"], "split": "val", "score": val_scores}),
        pd.DataFrame({"id": test_df["id"], "split": "test", "score": test_scores}),
    ], ignore_index=True)
    # preds.csv format is the contract (TASKS.md §0.3): id,split,score; dev results never go into results/
    out = (DEV_RESULTS_DIR if args.dev else RESULTS_DIR) / "roberta" / args.setting
    out.mkdir(parents=True, exist_ok=True)
    preds.to_csv(out / "preds.csv", index=False)
    # run_info.json: everything needed to reproduce or explain this run (commit, seed, hyperparameters, runtime)
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
