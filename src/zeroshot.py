"""Zero-shot detectors (role C): perplexity and log-rank under a small causal LM.

    python -m src.zeroshot --method ppl|logrank|fastdetectgpt [--dev] [--full]

Zero-shot methods don't train, so every text is scored ONCE, the scores are cached in
``results/zeroshot_cache/<method>__<model>__<max_tokens>.parquet`` (``id, score``), and then one ``preds.csv`` per setting is written by
selecting that setting's val/test ids from the cache (see docs/TASKS.md §0.3 and §C).

Score convention: higher = more likely AI.
  ppl     : mean token log-likelihood (machine text is more probable under an LM)
  logrank : -(mean log-rank of the actual tokens)  (machine text uses higher-ranked tokens)
  fastdetectgpt : Fast-DetectGPT (Bao et al., ICLR 2024), the one-forward-pass fallback for DetectGPT. It measures
          how far the text's log-likelihood sits above what the LM expects for a *sampled* text (conditional
          probability curvature), analytically instead of with 20 T5 perturbations:
              score = (sum_t log p(x_t) - sum_t E[log p]) / sqrt(sum_t Var[log p])
          with E and Var taken over the LM's own next-token distribution at each position. One model plays both
          the sampling and scoring role here. Only the DetectGPT subset + paraphrase test ids are scored unless
          --full is given.

Input:  texts via src.data.load_texts() and the split files splits/<setting>.json via load_split()
        (data/sample_dev.parquet + splits/dev/ with --dev).
Output: results/<method>/<setting>/preds.csv   (id, split, score) for every val + test id
        results/<method>/<setting>/run_info.json   (git commit, seed, model, max_tokens, runtime, hardware)
        results/zeroshot_cache/<method>__<model>__<max_tokens>.parquet   (id, score; the cache)
        With --dev everything goes to results_dev/ instead. Score the preds with: python -m src.eval <preds.csv>

Why "zero-shot": there is no training and no labels are used. A small pretrained language model (LM, a model that
predicts the next token of a text) reads each text, and we turn its next-token probabilities into ONE number.
Humans and machines get different numbers on average, so the number alone can separate them (AUROC in eval.py).
A "token" is a word piece (GPT-2 splits "unhappiness" into several tokens).

The three scores, in plain words. Let p(x_t) be the probability the LM gave to the token that really came next.
- ppl (mean log-likelihood): average of log p(x_t) over the tokens. Example: three tokens with probabilities
  0.5, 0.25, 0.125 give logs -0.69, -1.39, -2.08 and a mean of -1.39. Closer to 0 = the LM found the text
  predictable. Machine text is predictable, so a HIGHER (less negative) value means more AI. (The name "ppl"
  is historical: perplexity is exp(-mean log-likelihood), so we just use the same number with the sign kept.)
- logrank: at every position, sort all vocabulary tokens by the LM's probability. The rank is the place of the
  real token in that list (1 = the LM's top choice). We average log(rank) and flip the sign. Machine text
  keeps picking top choices (rank near 1, log-rank near 0), so after the sign flip HIGHER = more AI.
- fastdetectgpt: a text can be probable just because it is plain, not because a machine wrote it. So we compare
  the text with what the LM expects for a typical sampled text. At each position the LM's own distribution
  gives mu = E[log p] and sigma^2 = Var[log p] (the average and spread of log p over all tokens it might pick).
  score = (sum of real log p - sum of mu) / sqrt(sum of sigma^2). Machine text sits ABOVE the LM's own
  expectation (positive score); human text is more surprising, so it sits at or below it. Dividing by sigma
  makes scores of long and short texts comparable. HIGHER = more AI.

Why scores are cached: running the LM is the slow part (GPU minutes for the real data), while writing one
preds.csv per setting is instant. Many settings share the same val/test ids, so each text is scored once and
reused. The cache is also saved every few batches, so a crash or a Kaggle timeout does not lose the work.
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

METHODS = ("ppl", "logrank", "fastdetectgpt")
PARAPHRASE_SETTINGS = ("paraphrase_orig", "paraphrase", "paraphrase_human")
RESULTS_DIR = ROOT / "results"
CACHE_DIR = RESULTS_DIR / "zeroshot_cache"
DEV_CACHE_DIR = ROOT / "results_dev" / "zeroshot_cache"
NOT_SETTINGS = {"detectgpt_subset"}  # split-dir json files that are not evaluation settings


# ---------------------------------------------------------------- scoring

def pick_device():
    """Return "cuda" if a GPU is available, else "cpu".

    Why: the real run (gpt2-medium on all texts) needs a GPU; --dev is small enough for a laptop CPU.
    torch is imported inside the function so that importing this file stays fast (tests need no GPU setup).
    """
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def load_lm(model_name: str, device: str):
    """Download (first time) and load a pretrained causal LM and its tokenizer. Returns (tokenizer, model).

    "Causal" means the model predicts each token from the tokens before it, which is exactly what we need to
    ask "how probable was this text?". The model is only read, never trained: .eval() switches off dropout so
    the same text always gets the same score.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    # float16 (half precision) is faster and uses half the GPU memory; CPUs do not do float16 well, so use float32 there
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float16 if device == "cuda" else torch.float32
    )
    model.to(device).eval()
    return tok, model


def token_stats(model, input_ids, attention_mask, method: str):
    """Per-text score for one padded batch. input_ids[:, 0] is the BOS token, so every real token is scored.

    Takes the LM, a (batch, length) tensor of token ids, a same-shape 0/1 mask (1 = real token, 0 = padding) and
    the method name. Returns a numpy array with one score per text; higher = more AI for all three methods.
    This is the only place where the maths lives; tests/test_zeroshot.py checks it against hand calculations.
    """
    import torch

    # no_grad: we only read the model, so do not keep the memory needed for training
    with torch.no_grad():
        # logits = raw scores for every vocabulary token at every position. Row t predicts token t+1, so drop the
        # last row (nothing follows it). .float() avoids float16 rounding problems in the softmax and sums.
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1].float()
        # targets: the token that really came next at each position
        targets = input_ids[:, 1:]
        # mask: 1 for real tokens, 0 for padding, so padding never counts
        mask = attention_mask[:, 1:].float()
        # n: number of real tokens per text (min 1 avoids dividing by zero)
        n = mask.sum(1).clamp(min=1)
        if method == "ppl":
            # log_softmax turns logits into log-probabilities; gather picks the one of the real next token
            logp = torch.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            return ((logp * mask).sum(1) / n).cpu().numpy()  # mean log-likelihood
        if method == "fastdetectgpt":
            # Fast-DetectGPT: compare the real log p with the mean (mu) and variance (sigma^2) of log p under
            # the LM's OWN next-token distribution, i.e. E[log p] = sum_v p(v) log p(v) over the whole vocabulary.
            # Tiny example: if the LM gives 0.9 / 0.1 to two tokens, mu = 0.9*log0.9 + 0.1*log0.1 = -0.33.
            # logp_all: log p(v) for every vocabulary token v
            logp_all = torch.log_softmax(logits, dim=-1)
            logp = logp_all.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            p = logp_all.exp()
            mean_ref = (p * logp_all).sum(-1)  # E[log p(x~)] at each position
            var_ref = (p * logp_all ** 2).sum(-1) - mean_ref ** 2  # Var[log p(x~)]
            # numerator: how far the real text sits above the LM's expectation, summed over real tokens
            num = ((logp - mean_ref) * mask).sum(1)
            # denominator: sqrt of the summed variances; clamp(1e-8) avoids dividing by zero
            den = (var_ref * mask).sum(1).clamp(min=1e-8).sqrt()
            return (num / den).cpu().numpy()  # higher = more AI
        # method == "logrank" (the only one left). Rank = 1 + number of tokens the LM liked MORE than the real one.
        # Ties count as rank 1 because only strictly larger logits are counted. Logits are compared directly:
        # softmax keeps the order, so ranks from logits equal ranks from probabilities.
        target_logit = logits.gather(-1, targets.unsqueeze(-1))
        rank = (logits > target_logit).sum(-1).float() + 1.0  # 1 = the LM's top choice
        return (-(torch.log(rank) * mask).sum(1) / n).cpu().numpy()  # -mean log-rank


def score_texts(df: pd.DataFrame, method: str, tok, model, device: str, max_tokens: int, batch_size: int,
                cache_path: Path, save_every: int = 20) -> dict:
    """Score the rows of ``df`` that are not in the cache yet; append to the cache. Returns timing info.

    ``df`` needs columns id and text. Returns {"n_scored", "n_cached", "seconds"}. Why a cache: see the file
    docstring; a rerun, or another setting that needs the same ids, costs no LM time.
    """
    import torch

    cached = pd.read_parquet(cache_path) if cache_path.exists() else pd.DataFrame(columns=["id", "score"])
    todo = df[~df["id"].isin(set(cached["id"]))]
    info = {"n_scored": int(len(todo)), "n_cached": int(len(df) - len(todo)), "seconds": 0.0}
    if todo.empty:
        return info

    # BOS = "beginning of sequence" token. We put it in front of every text so the LM also gives a probability to the
    # very first real token. GPT-2 has no pad token, so its end-of-text token is borrowed for padding (the mask hides it).
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    # cut every text to max_tokens: it bounds the cost, and all texts are judged on the same-sized piece
    ids = tok(todo["text"].tolist(), add_special_tokens=False, truncation=True, max_length=max_tokens)["input_ids"]
    seqs = [[bos] + x for x in ids]
    order = np.argsort([len(s) for s in seqs])  # similar lengths per batch = less padding
    todo_ids = todo["id"].tolist()

    new_rows, t0 = [], time.time()
    for b, start in enumerate(range(0, len(order), batch_size)):
        idx = order[start:start + batch_size]
        longest = max(len(seqs[i]) for i in idx)
        # build one rectangular batch: shorter texts are filled with pad ids, and attn marks real tokens with 1
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
    """Save old cached scores + the new (id, score) rows to one parquet file at ``path``.

    Called every few batches and once at the end, so a crash keeps what was already scored. drop_duplicates
    keeps the newest score if an id was somehow scored twice.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame(new_rows, columns=["id", "score"])
    pd.concat([cached, new], ignore_index=True).drop_duplicates("id", keep="last").to_parquet(path, index=False)


# ---------------------------------------------------------------- per-setting outputs

def list_settings(dev: bool) -> list[str]:
    """Names of all evaluation settings (the split file names without .json), sorted.

    Why: --settings is optional; by default we write predictions for every setting that has a split file.
    detectgpt_subset is a list of ids, not a setting, so it is left out (NOT_SETTINGS).
    """
    d = DEV_SPLITS_DIR if dev else SPLITS_DIR
    return sorted(p.stem for p in d.glob("*.json") if p.stem not in NOT_SETTINGS)


def git_commit() -> str:
    """Current git commit hash, or "unknown" if git is missing or this is not a repo.

    Why: it is written to run_info.json so every result file can be traced to the exact code that made it.
    """
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def hardware(device: str) -> str:
    """Name of the GPU (for example "Tesla T4" in results/ppl/indist/run_info.json) or "CPU".

    Why: it goes into run_info.json, so the reported seconds per text can be understood.
    """
    if device == "cuda":
        import torch

        return torch.cuda.get_device_name(0)
    return "CPU"


def write_setting(method: str, setting: str, scores: pd.Series, dev: bool, run_info: dict,
                  partial: bool = False) -> int:
    """preds.csv (val + test rows only; train is never needed) + run_info.json for one setting."""
    # take the ids this setting uses for val and test, and look up their cached score; no LM call happens here
    split = load_split(setting, dev=dev)
    rows = [(i, name) for name in ("val", "test") for i in split.get(name, [])]
    preds = pd.DataFrame(rows, columns=["id", "split"])
    preds["score"] = preds["id"].map(scores)
    if partial:  # subset-only methods: keep the ids that were scored; a setting with no scored test id is skipped
        preds = preds.dropna(subset=["score"])
        if not (preds["split"] == "test").any():
            return 0
    # a missing score would silently become a wrong row in the evaluation, so stop with a clear error instead
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
    """Command-line entry point: score the needed texts (using the cache), then write one preds.csv per setting.

    ``argv`` is a list of arguments (None = read sys.argv; tests can pass their own). Returns 0 on success.
    Steps: read arguments, choose the ids to score, load the LM, score (cached), write the outputs.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", required=True, choices=METHODS)
    ap.add_argument("--dev", action="store_true", help="data/sample_dev.parquet + splits/dev; CPU-friendly (<2 min)")
    ap.add_argument("--model", default=None, help="scorer LM (default: gpt2-medium; gpt2 with --dev)")
    ap.add_argument("--max-tokens", type=int, default=None, help="default 256; 24 with --dev (CPU smoke test)")
    ap.add_argument("--batch-size", type=int, default=None, help="default 16 on GPU, 8 on CPU")
    ap.add_argument("--full", action="store_true",
                    help="fastdetectgpt: score every val/test id (like ppl) instead of the DetectGPT subset only")
    ap.add_argument("--settings", nargs="*", default=None, help="only these settings (default: every split file)")
    args = ap.parse_args(argv)

    import torch

    # scoring is deterministic anyway; the fixed seed is recorded in run_info.json
    torch.manual_seed(SEED)
    device = pick_device()
    model_name = args.model or ("gpt2" if args.dev else "gpt2-medium")
    max_tokens = args.max_tokens or (24 if args.dev else 256)
    batch_size = args.batch_size or (8 if args.method == "fastdetectgpt" or device == "cpu" else 16)
    start = time.time()

    df = load_texts(dev=args.dev)
    settings = args.settings or list_settings(args.dev)
    # score only ids that some setting needs as val/test; the train part is never used by a zero-shot method
    needed = set()
    for s in settings:
        sp = load_split(s, dev=args.dev)
        needed.update(sp.get("val", []))
        needed.update(sp.get("test", []))
    partial = args.method == "fastdetectgpt" and not args.full
    if partial:  # TASKS.md §C: the DetectGPT subset + every test id of the paraphrase settings (incl. D's _para ids)
        needed = set(load_split("detectgpt_subset", dev=args.dev))
        for s in settings:
            if s in PARAPHRASE_SETTINGS:
                needed.update(load_split(s, dev=args.dev).get("test", []))
    df = df[df["id"].isin(needed)]
    print(f"{args.method}: {len(df)} texts, {len(settings)} settings, model={model_name}, device={device}", flush=True)

    # the cache key includes the scorer settings: a rerun with another model/max_tokens must not reuse old scores
    cache_name = f"{args.method}__{model_name.replace('/', '--')}__{max_tokens}.parquet"
    cache_path = (DEV_CACHE_DIR if args.dev else CACHE_DIR) / cache_name
    tok, model = load_lm(model_name, device)
    info = score_texts(df, args.method, tok, model, device, max_tokens, batch_size, cache_path)
    print(f"scored {info['n_scored']} new texts ({info['n_cached']} from cache) in {info['seconds']:.0f}s", flush=True)

    scores = pd.read_parquet(cache_path).set_index("id")["score"]
    if partial:
        scores = scores[scores.index.isin(needed)]
    run_info = {
        "method": args.method, "git_commit": git_commit(), "seed": SEED, "model": model_name,
        "max_tokens": max_tokens, "batch_size": batch_size, "dev": args.dev,
        "hardware": hardware(device),
        "seconds_per_text": (info["seconds"] / info["n_scored"]) if info["n_scored"] else None,
        "total_runtime_seconds": round(time.time() - start, 1),
    }
    for s in settings:
        n = write_setting(args.method, s, scores, args.dev, run_info, partial=partial)
        print(f"  {s}: {n} rows" if n else f"  {s}: skipped (no scored test ids)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
