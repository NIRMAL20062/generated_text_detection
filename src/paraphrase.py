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

Input files : splits/paraphrase_orig.json (which 1,000 test texts to use) and the text table read by
              src.data.load_texts (data/processed/).
Output files: paraphrased.parquet (in --out-dir), splits/paraphrase.json, splits/paraphrase_human.json,
              results/paraphrase_quality.json.

Words used below, in plain English:
  - paraphrase: say the same thing with different words.
  - T5: a text-to-text neural network. We give it "paraphrase: <sentence>" and it writes a new sentence.
  - split file: a small JSON file {"train": [...], "val": [...], "test": [...]} that only lists text ids.
    The detectors read it to know which texts to score. The texts themselves live in the parquet table.
  - BLEU: a score from 0 to 100 for how many words (and word pairs) two texts share. 100 = identical.

How the split files are built (build_splits does the work; all ids come from `paraphrase_orig`):
  paraphrase_orig  : val + test = 500 machine texts (m1, m2, ...) + 500 human texts (h1, h2, ...).
  paraphrase       : same val; test = the same humans + the machines renamed m1_para, m2_para, ...
  paraphrase_human : same val; test = the machine ORIGINALS + the humans, where N humans are renamed h7_para ...
Comparing a detector on paraphrase_orig vs paraphrase shows how much paraphrasing hurts it.
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

# Paraphrasing model from the Hugging Face hub (a T5-base fine-tuned to rewrite ChatGPT-style text).
MODEL_NAME = "humarin/chatgpt_paraphraser_on_T5_base"
# Where files go. Real run: outputs/paraphrase/ for the parquet, results/ for the quality file.
# --dev run: everything goes to results_dev/ so the real files are never overwritten.
DEFAULT_OUT_DIR = ROOT / "outputs" / "paraphrase"
DEV_OUT_DIR = ROOT / "results_dev" / "paraphrase"
QUALITY_PATH = ROOT / "results" / "paraphrase_quality.json"
DEV_QUALITY_PATH = ROOT / "results_dev" / "paraphrase_quality.json"
# Size limits that only apply to --dev, so the smoke test finishes in about 1.5 minutes on CPU.
DEV_MAX_MACHINE, DEV_MAX_HUMAN = 6, 4  # texts per class in --dev
DEV_MAX_SENTENCES = 40                 # --dev: paraphrase only the first sentences of each text
# Why 60: the model reads at most 128 tokens (word pieces). One English word is roughly 1.5 tokens, so 60 words
# is about 90 tokens. A chunk this size is read in full; a longer one would be cut off and its end lost.
MAX_CHUNK_WORDS = 60                   # ~90 T5 tokens: well under --max-input-tokens 128, so nothing is truncated


# ---------------------------------------------------------------- pure helpers (unit-tested)

def split_sentences(text: str, min_chars: int = 20) -> list[str]:
    """Split on . ! ? followed by space. Very short pieces ("e.g.", "Fig.") are glued to the next one so the
    paraphraser never sees a fragment. Whitespace-collapsed text in, list of sentences out.

    Why: the model rewrites one sentence at a time, because a whole long text would be truncated.
    Example: "It rained. We stayed in and read books all day." -> ["It rained. We stayed in and read books all day."]
    ("It rained." has only 10 characters, so it is glued to the next sentence.)
    min_chars: a piece shorter than this many characters counts as a fragment."""
    # Cut after every . ! or ? that is followed by whitespace. The lookbehind (?<=...) keeps the punctuation
    # attached to its sentence. normalize() first turns newlines and double spaces into single spaces.
    pieces = [p for p in re.split(r"(?<=[.!?])\s+", normalize(text)) if p]
    out: list[str] = []
    carry = ""
    # carry = a too-short piece waiting to be glued onto the next piece.
    for p in pieces:
        p = f"{carry} {p}".strip() if carry else p
        if len(p) < min_chars:
            carry = p
        else:
            out.append(p)
            carry = ""
    # The text ended on a short piece: glue it to the last sentence (or keep it alone if it is the only piece).
    if carry:
        if out:
            out[-1] = f"{out[-1]} {carry}"
        else:
            out.append(carry)
    return out


def bleu(references: list[str], hypotheses: list[str], max_n: int = 4) -> float:
    """Corpus BLEU (0-100) of hypotheses against one reference each, on lower-cased whitespace tokens.

    Add-one smoothing for n > 1 so a short text with no 4-gram overlap does not zero the whole score. 100 means
    identical, near 0 means almost no shared wording. We use it to show the paraphrase really changed the words.

    How to read it: BLEU counts how many 1-, 2-, 3- and 4-word runs of the hypothesis also appear in the
    reference, averages them (geometric mean) and punishes a hypothesis that is shorter than the reference.
    Takes: references (originals) and hypotheses (paraphrases), same length, pairwise. Returns one number.
    The real run gave 26.74 for machine texts (see results/paraphrase_quality.json): the wording changed a lot."""
    # matches[n-1] = n-grams of the hypothesis found in its reference; totals[n-1] = n-grams in the hypothesis.
    # Counts are summed over all texts first, then divided once: that is what "corpus" BLEU means.
    matches, totals = [0] * max_n, [0] * max_n
    ref_len = hyp_len = 0
    for ref, hyp in zip(references, hypotheses):
        r, h = ref.lower().split(), hyp.lower().split()
        ref_len, hyp_len = ref_len + len(r), hyp_len + len(h)
        for n in range(1, max_n + 1):
            # Counter of n-grams; min(c, rc[g]) "clips" the credit so repeating one word cannot inflate the score.
            rc = Counter(tuple(r[i:i + n]) for i in range(len(r) - n + 1))
            hc = Counter(tuple(h[i:i + n]) for i in range(len(h) - n + 1))
            matches[n - 1] += sum(min(c, rc[g]) for g, c in hc.items())
            totals[n - 1] += max(len(h) - n + 1, 0)
    # No words, or not even one shared word: the score is 0 (and log(0) below would fail).
    if hyp_len == 0 or matches[0] == 0:
        return 0.0
    # log_p = average of log(precision) over the n-gram sizes.
    log_p = 0.0
    for n in range(max_n):
        m, t = (matches[n], totals[n]) if n == 0 else (matches[n] + 1, totals[n] + 1)
        log_p += math.log(m / t) / max_n
    # Brevity penalty: a very short output would otherwise score high by copying a few words.
    brevity = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / hyp_len)
    return 100.0 * brevity * math.exp(log_p)


def build_splits(orig: dict, machine_ids: list[str], human_ids: list[str], human_para_ids: list[str],
                 suffix: str = "_para") -> tuple[dict, dict | None]:
    """The two split files from the `paraphrase_orig` split (all ids are for the same 1,000 test texts).

    paraphrase       : val as in paraphrase_orig; test = the same humans + the machines replaced by _para.
    paraphrase_human : val as in paraphrase_orig; test = the 500 machine ORIGINALS + the humans, those in
                       `human_para_ids` swapped for their paraphrase. None when no human was paraphrased.

    Takes: orig = the loaded paraphrase_orig split; machine_ids / human_ids = test ids of each class;
    human_para_ids = the humans that were paraphrased; suffix = what is added to a paraphrased id.
    Returns: (paraphrase split, paraphrase_human split or None). Each is a dict train / val / test of id lists.
    Example: machines [m1, m2], humans [h1, h2], human_para_ids [h2] ->
      paraphrase.test       = [h1, h2, m1_para, m2_para]
      paraphrase_human.test = [h1, h2_para, m1, m2]
    Why train is empty: these splits are only for testing detectors that were already trained elsewhere."""
    # val is copied unchanged, so thresholds chosen on val stay comparable with paraphrase_orig.
    val = list(orig["val"])
    # Paraphrased test set = all original humans + every machine id with the suffix. Sorted for a stable file.
    paraphrase = {"train": [], "val": val, "test": sorted(human_ids + [m + suffix for m in machine_ids])}
    # --human-control 0: no human was paraphrased, so there is no control split to write.
    if not human_para_ids:
        return paraphrase, None
    # Control set: keep the machines as they were, replace only the chosen humans by their paraphrase.
    swapped = {h + suffix for h in human_para_ids}
    human_test = [h for h in human_ids if h + suffix not in swapped] + sorted(swapped)
    return paraphrase, {"train": [], "val": val, "test": sorted(machine_ids + human_test)}


def make_rows(originals: pd.DataFrame, new_texts: list[str], suffix: str = "_para") -> pd.DataFrame:
    """Rows in the §0.1 schema for the paraphrased texts. label / domain / generator / group are copied from the
    original (the group must stay the same so group-aware split checks still hold), id gets the suffix.

    Takes: originals = the rows that were paraphrased, new_texts = their paraphrases in the same order.
    Returns: a table with the normal columns plus `orig_id` (the id of the text it was made from).
    Why: the label must stay the same (a paraphrased machine text is still machine, label 1), so the detector
    is scored on whether it still catches it."""
    # copy() so the caller's table is not changed.
    df = originals[COLUMNS].copy().reset_index(drop=True)
    # Keep the link to the original text before the id is renamed.
    df["orig_id"] = df["id"]
    df["id"] = df["id"] + suffix
    df["text"] = [normalize(t) for t in new_texts]
    return df[COLUMNS + ["orig_id"]]


def chunk_long(sentences: list[str], max_words: int = MAX_CHUNK_WORDS) -> list[str]:
    """Cut any sentence longer than max_words into word chunks (reddit/wikihow often have no full stops), so the
    paraphraser's input truncation never silently drops the end of a text. Joining the output gives the input.

    Takes a list of sentences, returns a list of pieces of at most max_words words each.
    Example (max_words=3): ["a b c d e", "short one"] -> ["a b c", "d e", "short one"].
    Why 60 words: see MAX_CHUNK_WORDS above (about 90 tokens, below the 128-token input limit)."""
    out: list[str] = []
    for s in sentences:
        w = s.split()
        # Step through the words 60 at a time. A sentence with no words gives an empty list, so `or [s]`
        # keeps it as it is.
        out += [" ".join(w[i:i + max_words]) for i in range(0, len(w), max_words)] or [s]
    return out


# ---------------------------------------------------------------- model

def paraphrase_sentences(sentences: list[str], device: str, batch_size: int, args) -> list[str]:
    """Paraphrase every sentence. Sorted by length so batches have little padding; order restored at the end.

    Takes: the sentences, the device ("cuda" or "cpu"), the batch size and the command-line args (sampling
    settings). Returns: one paraphrase per input sentence, in the same order.
    Why it exists: this is the only function that runs the neural network."""
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    # The tokenizer turns text into numbers (token ids); eval() switches off training-only behaviour.
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device).eval()
    # Short sentences go together, long ones together: less padding, so faster. `out` is filled by position.
    order = sorted(range(len(sentences)), key=lambda i: len(sentences[i]))
    out = [""] * len(sentences)
    t0 = time.time()
    for b, start in enumerate(range(0, len(order), batch_size)):
        idx = order[start:start + batch_size]
        # The model was trained with the prefix "paraphrase: ". Texts longer than max_length tokens are cut.
        enc = tok([f"paraphrase: {sentences[i]}" for i in idx], return_tensors="pt", padding=True,
                  truncation=True, max_length=args.max_input_tokens).to(device)
        # Sampling settings, in plain words:
        #   do_sample=True        : pick each next word at random (by probability) instead of always the best one,
        #                           so the output is a real rewrite and not a copy.
        #   top_p=0.9 (nucleus)   : only choose among the most likely words that together make up 90% of the
        #                           probability. Rare, odd words are never picked.
        #   temperature=0.8       : below 1 makes likely words even more likely: safer, less wild wording.
        #   no_repeat_ngram_size=3: the same 3 words in a row may not appear twice in one output (stops loops).
        #   max_new_tokens        : upper limit on the length of each paraphrase.
        # No gradients are needed because we are not training.
        with torch.no_grad():
            gen = model.generate(**enc, do_sample=True, top_p=args.top_p, temperature=args.temperature,
                                 no_repeat_ngram_size=3, max_new_tokens=args.max_new_tokens)
        for i, text in zip(idx, tok.batch_decode(gen, skip_special_tokens=True)):
            out[i] = text.strip() or sentences[i]   # never lose a sentence: fall back to the original
        # Progress line every 20 batches, so a long run on Kaggle is visibly alive.
        if b % 20 == 0:
            print(f"  {min(start + batch_size, len(order))}/{len(order)} sentences ({time.time() - t0:.0f}s)", flush=True)
    return out


def paraphrase_texts(texts: list[str], device: str, batch_size: int, args) -> list[str]:
    """Paraphrase whole texts sentence by sentence and join the pieces back in order.

    Takes a list of texts, returns a list of paraphrased texts of the same length and order.
    Why: sentences of all texts are put in one flat list so batches are full, then cut back per text.
    Example: texts ["A. B.", "C."] -> flat sentences [A, B, C] -> paraphrased [A', B', C'] -> ["A' B'", "C'"]."""
    per_text = [chunk_long(split_sentences(t)) for t in texts]
    # --dev only: keep the first 40 sentences of each text so the smoke test is quick.
    if args.dev:
        per_text = [s[:DEV_MAX_SENTENCES] for s in per_text]
    flat = [s for sents in per_text for s in sents]
    new = paraphrase_sentences(flat, device, batch_size, args)
    # Walk through the flat list and hand each text back exactly as many sentences as it gave in.
    out, pos = [], 0
    for sents in per_text:
        out.append(" ".join(new[pos:pos + len(sents)]))
        pos += len(sents)
    return out


# ---------------------------------------------------------------- bookkeeping

def git_commit() -> str:
    """Return the current git commit hash, or "unknown" if git is not available.

    Why: it is written into paraphrase_quality.json so a result can be traced to the exact code that made it.
    On Kaggle there may be no git repository, so any error gives "unknown" instead of crashing the run."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def main(argv=None) -> int:
    """Run the whole paraphrase attack from the command line. Returns 0 on success.

    Takes argv (a list of options; None means use the real command line, a list is handy for tests).
    Steps: pick the texts, paraphrase them, write the parquet, write the two split files, write the quality file.
    Reads splits/paraphrase_orig.json and the text table; writes the files listed at the top of this module."""
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

    # Imported here, not at the top, so the pure helpers (and their tests) work without loading torch.
    import torch

    # Fixed seeds: the same texts are picked and the sampling is repeatable on the same hardware.
    torch.manual_seed(SEED)
    rng = random.Random(SEED)
    # Use the GPU when there is one. A bigger batch fits on a GPU.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = args.batch_size or (32 if device == "cuda" else 8)
    start = time.time()

    # Get the 1,000 test texts of paraphrase_orig (the "before" set). Machine = label 1, human = label 0.
    # Sorting by id makes the order the same on every run.
    texts = load_texts(dev=args.dev).drop_duplicates("id").set_index("id")
    orig = load_split("paraphrase_orig", dev=args.dev)
    test = texts.loc[orig["test"]]
    machine_df = test[test["label"] == 1].reset_index().sort_values("id")
    human_df = test[test["label"] == 0].reset_index().sort_values("id")
    # n_h = how many human texts to paraphrase as the control. --dev shrinks everything to a few texts.
    n_h = 0 if args.human_control <= 0 else args.human_control
    if args.dev:
        machine_df, human_df = machine_df.head(DEV_MAX_MACHINE), human_df.head(DEV_MAX_HUMAN)
        n_h = min(n_h, 2)
    print(f"paraphrasing {len(machine_df)} machine texts" + (f" + {min(n_h, len(human_df))} human texts" if n_h else "")
          + f" on {device}", flush=True)

    # Randomly choose which humans get paraphrased (seeded, so the same ones every run).
    machine_ids, human_ids = machine_df["id"].tolist(), human_df["id"].tolist()
    human_pick = sorted(rng.sample(human_ids, min(n_h, len(human_ids)))) if n_h else []
    human_pick_df = human_df[human_df["id"].isin(human_pick)]

    # The slow part: run the paraphraser. All 500 machine texts, then the chosen humans.
    new_machine = paraphrase_texts(machine_df["text"].tolist(), device, batch_size, args)
    new_human = paraphrase_texts(human_pick_df["text"].tolist(), device, batch_size, args) if human_pick else []

    # Build the table of paraphrased texts (machines first, then the chosen humans) and save it.
    rows = make_rows(machine_df, new_machine)
    if human_pick:
        rows = pd.concat([rows, make_rows(human_pick_df, new_human)], ignore_index=True)
    out_dir = DEV_OUT_DIR if args.dev else (args.out_dir or DEFAULT_OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(out_dir / "paraphrased.parquet", index=False)

    # Build and save the split files that point at the new ids: paraphrase.json is the "after" test set,
    # paraphrase_human.json is the control (see the top of this file for what is inside each).
    sp_para, sp_human = build_splits(orig, machine_ids, human_ids, human_pick)
    split_dir = out_dir if args.dev else SPLITS_DIR   # dev splits must not enter splits/dev: check_splits --dev would reject _para ids
    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / "paraphrase.json").write_text(json.dumps(sp_para, indent=0))
    if sp_human:
        (split_dir / "paraphrase_human.json").write_text(json.dumps(sp_human, indent=0))

    # Quality check: put each original next to its paraphrase, then measure how different they are.
    #   bleu_machine / bleu_human : lower = more rewritten (real run: 26.74 and 23.95).
    #   unchanged_share           : share of texts the model returned word for word (real run: 0.0).
    #   mean_length_ratio         : paraphrase length / original length, near 1 = meaning not cut or padded
    #                               (real run: 0.972).
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
