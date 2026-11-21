"""Zero-shot detectors (role C): perplexity and log-rank under a small causal LM.

    python -m src.zeroshot --method ppl|logrank [--dev]

Zero-shot methods don't train, so every text is scored ONCE, the scores are cached in
``results/zeroshot_cache/<method>.parquet`` (``id, score``), and then one ``preds.csv`` per setting is written by
selecting that setting's val/test ids from the cache (see docs/TASKS.md §0.3 and §C).

Score convention: higher = more likely AI.
  ppl     : mean token log-likelihood (machine text is more probable under an LM)
  logrank : -(mean log-rank of the actual tokens)  (machine text uses higher-ranked tokens)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import DEV_SPLITS_DIR, ROOT, SEED, SPLITS_DIR, load_split, load_texts

METHODS = ("ppl", "logrank")
RESULTS_DIR = ROOT / "results"
CACHE_DIR = RESULTS_DIR / "zeroshot_cache"
NOT_SETTINGS = {"detectgpt_subset"}  # split-dir json files that are not evaluation settings


# ---------------------------------------------------------------- scoring

def pick_device():
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def load_lm(model_name: str, device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float16 if device == "cuda" else torch.float32
    )
    model.to(device).eval()
    return tok, model


def token_stats(model, input_ids, attention_mask, method: str):
    """Per-text score for one padded batch. input_ids[:, 0] is the BOS token, so every real token is scored."""
    import torch

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1].float()
        targets = input_ids[:, 1:]
        mask = attention_mask[:, 1:].float()
        n = mask.sum(1).clamp(min=1)
        if method == "ppl":
            logp = torch.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            return ((logp * mask).sum(1) / n).cpu().numpy()  # mean log-likelihood
        target_logit = logits.gather(-1, targets.unsqueeze(-1))
        rank = (logits > target_logit).sum(-1).float() + 1.0  # 1 = the LM's top choice
        return (-(torch.log(rank) * mask).sum(1) / n).cpu().numpy()  # -mean log-rank


def score_texts(df: pd.DataFrame, method: str, tok, model, device: str, max_tokens: int, batch_size: int,
                cache_path: Path, save_every: int = 20) -> dict:
    """Score the rows of ``df`` that are not in the cache yet; append to the cache. Returns timing info."""
    import torch

    cached = pd.read_parquet(cache_path) if cache_path.exists() else pd.DataFrame(columns=["id", "score"])
    todo = df[~df["id"].isin(set(cached["id"]))]
    info = {"n_scored": int(len(todo)), "n_cached": int(len(df) - len(todo)), "seconds": 0.0}
    if todo.empty:
        return info

    bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ids = tok(todo["text"].tolist(), add_special_tokens=False, truncation=True, max_length=max_tokens)["input_ids"]
    seqs = [[bos] + x for x in ids]
    order = np.argsort([len(s) for s in seqs])  # similar lengths per batch = less padding
    todo_ids = todo["id"].tolist()

    new_rows, t0 = [], time.time()
    for b, start in enumerate(range(0, len(order), batch_size)):
        idx = order[start:start + batch_size]
        longest = max(len(seqs[i]) for i in idx)
        input_ids = torch.full((len(idx), longest), pad, dtype=torch.long)
        attn = torch.zeros((len(idx), longest), dtype=torch.long)
        for r, i in enumerate(idx):
            input_ids[r, :len(seqs[i])] = torch.tensor(seqs[i])
            attn[r, :len(seqs[i])] = 1
        scores = token_stats(model, input_ids.to(device), attn.to(device), method)
        new_rows += [(todo_ids[i], float(s)) for i, s in zip(idx, scores)]
        if (b + 1) % save_every == 0:  # a Kaggle timeout must not lose progress
            _write_cache(cached, new_rows, cache_path)
            print(f"  {len(new_rows)}/{len(todo)} scored ({time.time() - t0:.0f}s)", flush=True)
    _write_cache(cached, new_rows, cache_path)
    info["seconds"] = time.time() - t0
    return info


def _write_cache(cached: pd.DataFrame, new_rows: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame(new_rows, columns=["id", "score"])
    pd.concat([cached, new], ignore_index=True).drop_duplicates("id", keep="last").to_parquet(path, index=False)


# ---------------------------------------------------------------- per-setting outputs

def list_settings(dev: bool) -> list[str]:
    d = DEV_SPLITS_DIR if dev else SPLITS_DIR
    return sorted(p.stem for p in d.glob("*.json") if p.stem not in NOT_SETTINGS)


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


def write_setting(method: str, setting: str, scores: pd.Series, dev: bool, run_info: dict) -> int:
    """preds.csv (val + test rows only; train is never needed) + run_info.json for one setting."""
    split = load_split(setting, dev=dev)
    rows = [(i, name) for name in ("val", "test") for i in split.get(name, [])]
    preds = pd.DataFrame(rows, columns=["id", "split"])
    preds["score"] = preds["id"].map(scores)
    if preds["score"].isna().any():
        missing = preds.loc[preds["score"].isna(), "id"].head(3).tolist()
        raise KeyError(f"{setting}: {int(preds['score'].isna().sum())} split ids have no score, e.g. {missing}")
    out = (ROOT / "results_dev" if dev else RESULTS_DIR) / method / setting  # eval.py rejects --dev preds inside results/
    out.mkdir(parents=True, exist_ok=True)
    preds.to_csv(out / "preds.csv", index=False)
    (out / "run_info.json").write_text(json.dumps({**run_info, "setting": setting, "n_val": int((preds.split == "val").sum()),
                                                   "n_test": int((preds.split == "test").sum())}, indent=2))
    return len(preds)


# ---------------------------------------------------------------- main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", required=True, choices=METHODS)
    ap.add_argument("--dev", action="store_true", help="data/sample_dev.parquet + splits/dev; CPU-friendly (<2 min)")
    ap.add_argument("--model", default=None, help="scorer LM (default: gpt2-medium; gpt2 with --dev)")
    ap.add_argument("--max-tokens", type=int, default=None, help="default 256; 24 with --dev (CPU smoke test)")
    ap.add_argument("--batch-size", type=int, default=None, help="default 16 on GPU, 8 on CPU")
    ap.add_argument("--settings", nargs="*", default=None, help="only these settings (default: every split file)")
    args = ap.parse_args(argv)

    import torch

    torch.manual_seed(SEED)
    device = pick_device()
    model_name = args.model or ("gpt2" if args.dev else "gpt2-medium")
    max_tokens = args.max_tokens or (24 if args.dev else 256)
    batch_size = args.batch_size or (16 if device == "cuda" else 8)
    start = time.time()

    df = load_texts(dev=args.dev)
    settings = args.settings or list_settings(args.dev)
    # score only ids that some setting needs as val/test; the train part is never used by a zero-shot method
    needed = set()
    for s in settings:
        sp = load_split(s, dev=args.dev)
        needed.update(sp.get("val", []))
        needed.update(sp.get("test", []))
    df = df[df["id"].isin(needed)]
    print(f"{args.method}: {len(df)} texts, {len(settings)} settings, model={model_name}, device={device}", flush=True)

    cache_path = CACHE_DIR / f"{args.method}{'_dev' if args.dev else ''}.parquet"
    tok, model = load_lm(model_name, device)
    info = score_texts(df, args.method, tok, model, device, max_tokens, batch_size, cache_path)
    print(f"scored {info['n_scored']} new texts ({info['n_cached']} from cache) in {info['seconds']:.0f}s", flush=True)

    scores = pd.read_parquet(cache_path).set_index("id")["score"]
    run_info = {
        "method": args.method, "git_commit": git_commit(), "seed": SEED, "model": model_name,
        "max_tokens": max_tokens, "batch_size": batch_size, "dev": args.dev,
        "hardware": hardware(device),
        "seconds_per_text": (info["seconds"] / info["n_scored"]) if info["n_scored"] else None,
        "total_runtime_seconds": round(time.time() - start, 1),
    }
    for s in settings:
        n = write_setting(args.method, s, scores, args.dev, run_info)
        print(f"  {s}: {n} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
