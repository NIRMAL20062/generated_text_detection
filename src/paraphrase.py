"""Paraphrase attack (role D): rewrite machine texts so detectors have to cope with paraphrased AI text.

    python -m src.paraphrase [--dev] [--human-control 200]

What it does (docs/TASKS.md §D):
  1. Takes the 500 machine texts in the `paraphrase_orig` test set (A picked them, so the "before" numbers of
     `paraphrase_orig` and the "after" numbers of `paraphrase` use exactly the same texts).
  2. Paraphrases each text SENTENCE BY SENTENCE with `humarin/chatgpt_paraphraser_on_T5_base` (a long text in
     one go would be truncated), then collapses whitespace like the dataset does (§0.1).
  3. Writes `paraphrased.parquet` (§0.1 schema, id = <orig id>_para, plus `orig_id`).
  4. Writes `splits/paraphrase.json`: same val and the same humans as `paraphrase_orig`, machines swapped for
     their `_para` versions. This is the "after" number.
  5. Optional control (--human-control N, default 200): also paraphrases N of the human test texts (label stays
     0) and writes `splits/paraphrase_human.json` = the 500 machine ORIGINALS + the humans, N of them replaced by
     paraphrases. If detectors flag these as AI, they detect "paraphraser style", not "LLM-written".
  6. Quality check `paraphrase_quality.json`: BLEU of the paraphrase against the original (lower = rewritten
     more), share of texts left unchanged, and length ratio, so the report can show the meaning was kept
     while the wording changed.

Where files go: the paraphrased parquet is written to --out-dir (default outputs/paraphrase/), not into
data/processed/, because on Kaggle that folder is a read-only input. Copy it to data/processed/ locally, and
hand it to Nirmal for a new version of the Kaggle dataset (issue #6). Split files go to splits/ (commit them).
--dev: tiny subset of the dev sample, max 2 minutes on CPU; everything goes to results_dev/paraphrase/.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import subprocess
import time
from collections import Counter
from pathlib import Path

import pandas as pd

from src.data import COLUMNS, ROOT, SEED, SPLITS_DIR, load_split, load_texts, normalize

MODEL_NAME = "humarin/chatgpt_paraphraser_on_T5_base"
DEFAULT_OUT_DIR = ROOT / "outputs" / "paraphrase"
DEV_OUT_DIR = ROOT / "results_dev" / "paraphrase"
QUALITY_PATH = ROOT / "results" / "paraphrase_quality.json"
DEV_QUALITY_PATH = ROOT / "results_dev" / "paraphrase_quality.json"
DEV_MAX_MACHINE, DEV_MAX_HUMAN = 6, 4  # texts per class in --dev
DEV_MAX_SENTENCES = 40                 # --dev: paraphrase only the first sentences of each text


# ---------------------------------------------------------------- pure helpers (unit-tested)

def split_sentences(text: str, min_chars: int = 20) -> list[str]:
    """Split on . ! ? followed by space. Very short pieces ("e.g.", "Fig.") are glued to the next one so the
    paraphraser never sees a fragment. Whitespace-collapsed text in, list of sentences out."""
    pieces = [p for p in re.split(r"(?<=[.!?])\s+", normalize(text)) if p]
    out: list[str] = []
    carry = ""
    for p in pieces:
        p = f"{carry} {p}".strip() if carry else p
        if len(p) < min_chars:
            carry = p
        else:
            out.append(p)
            carry = ""
    if carry:
        if out:
            out[-1] = f"{out[-1]} {carry}"
        else:
            out.append(carry)
    return out


def bleu(references: list[str], hypotheses: list[str], max_n: int = 4) -> float:
    """Corpus BLEU (0-100) of hypotheses against one reference each, on lower-cased whitespace tokens.

    Add-one smoothing for n > 1 so a short text with no 4-gram overlap does not zero the whole score. 100 means
    identical, near 0 means almost no shared wording. We use it to show the paraphrase really changed the words."""
    matches, totals = [0] * max_n, [0] * max_n
    ref_len = hyp_len = 0
    for ref, hyp in zip(references, hypotheses):
        r, h = ref.lower().split(), hyp.lower().split()
        ref_len, hyp_len = ref_len + len(r), hyp_len + len(h)
        for n in range(1, max_n + 1):
            rc = Counter(tuple(r[i:i + n]) for i in range(len(r) - n + 1))
            hc = Counter(tuple(h[i:i + n]) for i in range(len(h) - n + 1))
            matches[n - 1] += sum(min(c, rc[g]) for g, c in hc.items())
            totals[n - 1] += max(len(h) - n + 1, 0)
    if hyp_len == 0 or matches[0] == 0:
        return 0.0
    log_p = 0.0
    for n in range(max_n):
        m, t = (matches[n], totals[n]) if n == 0 else (matches[n] + 1, totals[n] + 1)
        log_p += math.log(m / t) / max_n
    brevity = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / hyp_len)
    return 100.0 * brevity * math.exp(log_p)


def build_splits(orig: dict, machine_ids: list[str], human_ids: list[str], human_para_ids: list[str],
                 suffix: str = "_para") -> tuple[dict, dict | None]:
    """The two split files from the `paraphrase_orig` split (all ids are for the same 1,000 test texts).

    paraphrase       : val as in paraphrase_orig; test = the same humans + the machines replaced by _para.
    paraphrase_human : val as in paraphrase_orig; test = the 500 machine ORIGINALS + the humans, those in
                       `human_para_ids` swapped for their paraphrase. None when no human was paraphrased."""
    val = list(orig["val"])
    paraphrase = {"train": [], "val": val, "test": sorted(human_ids + [m + suffix for m in machine_ids])}
    if not human_para_ids:
        return paraphrase, None
    swapped = {h + suffix for h in human_para_ids}
    human_test = [h for h in human_ids if h + suffix not in swapped] + sorted(swapped)
    return paraphrase, {"train": [], "val": val, "test": sorted(machine_ids + human_test)}


def make_rows(originals: pd.DataFrame, new_texts: list[str], suffix: str = "_para") -> pd.DataFrame:
    """Rows in the §0.1 schema for the paraphrased texts. label / domain / generator / group are copied from the
    original (the group must stay the same so group-aware split checks still hold), id gets the suffix."""
    df = originals[COLUMNS].copy().reset_index(drop=True)
    df["orig_id"] = df["id"]
    df["id"] = df["id"] + suffix
    df["text"] = [normalize(t) for t in new_texts]
    return df[COLUMNS + ["orig_id"]]


# ---------------------------------------------------------------- model

def paraphrase_sentences(sentences: list[str], device: str, batch_size: int, args) -> list[str]:
    """Paraphrase every sentence. Sorted by length so batches have little padding; order restored at the end."""
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device).eval()
    order = sorted(range(len(sentences)), key=lambda i: len(sentences[i]))
    out = [""] * len(sentences)
    t0 = time.time()
    for b, start in enumerate(range(0, len(order), batch_size)):
        idx = order[start:start + batch_size]
        enc = tok([f"paraphrase: {sentences[i]}" for i in idx], return_tensors="pt", padding=True,
                  truncation=True, max_length=args.max_input_tokens).to(device)
        with torch.no_grad():
            gen = model.generate(**enc, do_sample=True, top_p=args.top_p, temperature=args.temperature,
                                 no_repeat_ngram_size=3, max_new_tokens=args.max_new_tokens)
        for i, text in zip(idx, tok.batch_decode(gen, skip_special_tokens=True)):
            out[i] = text.strip() or sentences[i]   # never lose a sentence: fall back to the original
        if b % 20 == 0:
            print(f"  {min(start + batch_size, len(order))}/{len(order)} sentences ({time.time() - t0:.0f}s)", flush=True)
    return out


def paraphrase_texts(texts: list[str], device: str, batch_size: int, args) -> list[str]:
    """Paraphrase whole texts sentence by sentence and join the pieces back in order."""
    per_text = [split_sentences(t) for t in texts]
    if args.dev:
        per_text = [s[:DEV_MAX_SENTENCES] for s in per_text]
    flat = [s for sents in per_text for s in sents]
    new = paraphrase_sentences(flat, device, batch_size, args)
    out, pos = [], 0
    for sents in per_text:
        out.append(" ".join(new[pos:pos + len(sents)]))
        pos += len(sents)
    return out


# ---------------------------------------------------------------- bookkeeping

def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dev", action="store_true", help="tiny subset of the dev sample; outputs in results_dev/paraphrase/")
    ap.add_argument("--human-control", type=int, default=200, help="how many human texts to paraphrase too (0 = skip)")
    ap.add_argument("--out-dir", type=Path, default=None, help="where paraphrased.parquet goes (default outputs/paraphrase/)")
    ap.add_argument("--batch-size", type=int, default=None, help="sentences per batch (default 32 on GPU, 8 on CPU)")
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-input-tokens", type=int, default=128)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    args = ap.parse_args(argv)

    import torch

    torch.manual_seed(SEED)
    rng = random.Random(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = args.batch_size or (32 if device == "cuda" else 8)
    start = time.time()

    texts = load_texts(dev=args.dev).drop_duplicates("id").set_index("id")
    orig = load_split("paraphrase_orig", dev=args.dev)
    test = texts.loc[orig["test"]]
    machine_df = test[test["label"] == 1].reset_index().sort_values("id")
    human_df = test[test["label"] == 0].reset_index().sort_values("id")
    n_h = 0 if args.human_control <= 0 else args.human_control
    if args.dev:
        machine_df, human_df = machine_df.head(DEV_MAX_MACHINE), human_df.head(DEV_MAX_HUMAN)
        n_h = min(n_h, 2)
    print(f"paraphrasing {len(machine_df)} machine texts" + (f" + {min(n_h, len(human_df))} human texts" if n_h else "")
          + f" on {device}", flush=True)

    machine_ids, human_ids = machine_df["id"].tolist(), human_df["id"].tolist()
    human_pick = sorted(rng.sample(human_ids, min(n_h, len(human_ids)))) if n_h else []
    human_pick_df = human_df[human_df["id"].isin(human_pick)]

    new_machine = paraphrase_texts(machine_df["text"].tolist(), device, batch_size, args)
    new_human = paraphrase_texts(human_pick_df["text"].tolist(), device, batch_size, args) if human_pick else []

    rows = make_rows(machine_df, new_machine)
    if human_pick:
        rows = pd.concat([rows, make_rows(human_pick_df, new_human)], ignore_index=True)
    out_dir = DEV_OUT_DIR if args.dev else (args.out_dir or DEFAULT_OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(out_dir / "paraphrased.parquet", index=False)

    sp_para, sp_human = build_splits(orig, machine_ids, human_ids, human_pick)
    split_dir = out_dir if args.dev else SPLITS_DIR   # dev splits must not enter splits/dev: check_splits --dev would reject _para ids
    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / "paraphrase.json").write_text(json.dumps(sp_para, indent=0))
    if sp_human:
        (split_dir / "paraphrase_human.json").write_text(json.dumps(sp_human, indent=0))

    origs = pd.concat([machine_df, human_pick_df]).set_index("id").loc[rows["orig_id"], "text"].tolist()
    new = rows["text"].tolist()
    quality = {
        "model": MODEL_NAME, "seed": SEED, "git_commit": git_commit(), "dev": args.dev, "device": device,
        "sampling": {"top_p": args.top_p, "temperature": args.temperature, "no_repeat_ngram_size": 3},
        "n_machine": len(machine_df), "n_human_paraphrased": len(human_pick_df),
        "bleu_machine": round(bleu(machine_df["text"].tolist(), new_machine), 2),
        "bleu_human": round(bleu(human_pick_df["text"].tolist(), new_human), 2) if new_human else None,
        "unchanged_share": round(sum(a == b for a, b in zip(origs, new)) / max(len(new), 1), 4),
        "mean_length_ratio": round(sum(len(b) / max(len(a), 1) for a, b in zip(origs, new)) / max(len(new), 1), 3),
        "runtime_seconds": round(time.time() - start, 1),
    }
    qpath = DEV_QUALITY_PATH if args.dev else QUALITY_PATH
    qpath.parent.mkdir(parents=True, exist_ok=True)
    qpath.write_text(json.dumps(quality, indent=2))
    print(json.dumps(quality, indent=2))
    print(f"wrote {out_dir / 'paraphrased.parquet'} ({len(rows)} rows), {split_dir / 'paraphrase.json'}"
          + (f", {split_dir / 'paraphrase_human.json'}" if sp_human else "") + f", {qpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
