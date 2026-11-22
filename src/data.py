"""M4 English: download, audit, build the dataset, make group-aware splits, shared loaders.

Contract: docs/TASKS.md §0.1 (dataset columns) and §0.2 (split files).

    python -m src.data --all            # download + build + splits + audit + dev sample
    python -m src.data --check-splits   # assert every split id exists in the data (Kaggle first cell)
    python -m src.data --check-splits --dev

Everyone else should only need `load_texts()` and `load_split()`.

Input:  the 25 English M4 files (5 domains x 5 generators) from github.com/mbzuai-nlp/M4, saved to data/raw/.
Output: data/processed/m4_en.parquet  one row per text: id, text, label (0 human, 1 machine), domain, generator, group
        splits/<setting>.json          {"train": [ids], "val": [ids], "test": [ids]} for every setting
        splits/detectgpt_subset.json   the ids the slow zero-shot method scores
        results/data_audit.md          counts behind every dataset number in the report
        data/sample_dev.parquet + splits/dev/   a ~500-row sample so every script can be tested in minutes

Why `group` matters: M4 has one human text and up to five machine answers written for the same prompt, and the
same human text often appears in several generator files. If those near-copies landed on both sides of a split,
a detector could recognise the topic instead of learning who wrote the text, and test scores would be too high.
So every split puts whole groups, never single rows, into train, val or test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

# Seed for every random choice in the project (sampling, shuffling, model training), so runs repeat exactly.
SEED = 42
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
DATASET_PATH = PROC_DIR / "m4_en.parquet"
PARAPHRASED_PATH = PROC_DIR / "paraphrased.parquet"
PARAPHRASED_FALLBACK = ROOT / "outputs" / "paraphrase" / "paraphrased.parquet"  # src.paraphrase default output
DEV_PATH = ROOT / "data" / "sample_dev.parquet"
SPLITS_DIR = ROOT / "splits"
DEV_SPLITS_DIR = SPLITS_DIR / "dev"
AUDIT_PATH = ROOT / "results" / "data_audit.md"
M4_URL = "https://raw.githubusercontent.com/mbzuai-nlp/M4/main/data/{}"

DOMAINS = ["wikipedia", "wikihow", "reddit", "arxiv", "peerread"]
GENERATORS = ["chatGPT", "davinci", "cohere", "dolly", "bloomz"]
# M4 file names are not consistent (chatGPT/chatgpt, dolly/dolly2).
FILE_GEN = {"chatGPT": ["chatGPT", "chatgpt"], "davinci": ["davinci"], "cohere": ["cohere"],
            "dolly": ["dolly", "dolly2"], "bloomz": ["bloomz"]}
# These generators exist in only one or two domains. Keeping them would make the 5x5 grid uneven, and a
# "leave one generator out" test would be meaningless for a generator that only appears in one domain.
EXCLUDED_FILES = ["arxiv_flant5.jsonl", "reddit_flant5.jsonl", "peerread_llama.jsonl"]
KNOWN_FILES = {
    "wikipedia": ["chatgpt", "davinci", "cohere", "dolly", "bloomz"],
    "wikihow": ["chatGPT", "davinci", "cohere", "dolly2", "bloomz"],
    "reddit": ["chatGPT", "davinci", "cohere", "dolly", "bloomz"],
    "arxiv": ["chatGPT", "davinci", "cohere", "dolly", "bloomz"],
    "peerread": ["chatgpt", "davinci", "cohere", "dolly", "bloomz"],
}
# Field names differ per file. BLOOMZ files store prompt+output in `machine_text`,
# so the dedicated output fields must win.
# peerread_bloomz uses human_reviews / bloom_reviews.
HUMAN_KEYS = ["human_text", "human_reviews", "abstract", "text"]
MACHINE_KEYS = ["machine_answer", "machine_abstract", "bloom_reviews", "machine_text"]
# ORDER MATTERS in both lists: _first() returns the first key found. If "machine_text" came first, BLOOMZ rows
# would use prompt+output as the "machine" text; the copied prompt (human-like words) would blur the label, and
# its fixed prompt template would also be an easy shortcut a detector could learn instead of writing style.
# Texts shorter than this (empty answers, one-word replies) carry no style to detect, so they are dropped.
MIN_CHARS = 50
# Tradeoff: a higher cutoff gives cleaner texts but throws away real short Reddit/WikiHow answers; 50 chars
# (about one sentence) only removes texts with nearly nothing in them.
# The settings written by make_splits() and listed in the audit table. paraphrase/paraphrase_human are not here
# because role D's paraphrase.py writes those.
SETTINGS_TOP = ["indist"] + [f"logo_{g}" for g in GENERATORS] + [f"xdomain_{d}" for d in DOMAINS] + ["paraphrase_orig"]
COLUMNS = ["id", "text", "label", "domain", "generator", "group"]


# ---------------------------------------------------------------- shared loaders (everyone)

def load_texts(dev: bool = False) -> pd.DataFrame:
    """All texts with labels: m4_en.parquet (+ paraphrased.parquet if it exists). Index is not set.

    Why one loader for everybody: if each script read the raw JSONL itself, small differences (other cleaning,
    other field order) would give different texts for the same id, and the four methods would no longer be
    compared on the same data. Scripts then pick rows with load_split() ids.
    """
    if dev:
        # --dev: tiny sample committed to git, so every script can be tested in minutes on CPU, without the
        # full dataset (which is not in git).
        return pd.read_parquet(DEV_PATH)
    parts = [pd.read_parquet(DATASET_PATH)]
    # paraphrase.py writes outputs/paraphrase/ (data/processed is read-only on Kaggle) but its split files go to
    # splits/ at once, so read it from there too; otherwise every paraphrase* setting fails on missing ids.
    para = next((p for p in (PARAPHRASED_PATH, PARAPHRASED_FALLBACK) if p.exists()), None)
    if para is not None:
        parts.append(pd.read_parquet(para))
    # ignore_index=True: both files start their row index at 0; without it the index would repeat (0, 1, ...
    # twice), and .loc lookups by index could return two rows. Rows are found by `id`, never by index.
    return pd.concat(parts, ignore_index=True)


def load_split(setting: str, dev: bool = False) -> dict[str, list[str]]:
    """{'train': [...], 'val': [...], 'test': [...]} for a setting name from TASKS.md §0.2."""
    d = DEV_SPLITS_DIR if dev else SPLITS_DIR
    return json.loads((d / f"{setting}.json").read_text())


# ---------------------------------------------------------------- download

def file_name(domain: str, file_gen: str) -> str:
    """M4's file name for one (domain, generator-as-spelled-in-the-file) cell, e.g. ('reddit', 'dolly') -> 'reddit_dolly.jsonl'."""
    return f"{domain}_{file_gen}.jsonl"


def cell_file(domain: str, gen: str, raw_dir: Path = RAW_DIR) -> Path | None:
    """Path of the raw file for (domain, generator), trying every spelling M4 uses; None if it was not downloaded.

    Why: M4 is inconsistent (wikihow_dolly2.jsonl, wikipedia_chatgpt.jsonl), so callers use our canonical
    generator names and this function finds the actual file.
    """
    for fg in FILE_GEN[gen]:
        p = raw_dir / file_name(domain, fg)
        if p.exists():
            return p
    return None


def download_m4(out_dir: Path = RAW_DIR, retries: int = 3) -> None:
    """Fetch only the 25 English (domain, generator) files. Skips files already complete."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for dom, fgens in KNOWN_FILES.items():
        for fg in fgens:
            name = file_name(dom, fg)
            dest = out_dir / name
            url = M4_URL.format(name)
            for attempt in range(1, retries + 1):
                try:
                    # HEAD asks only for the headers (no body), so we learn the file size cheaply. If the local
                    # file already has that size it is complete and we skip it: rerunning --all does not
                    # download ~1 GB again.
                    req = urllib.request.Request(url, method="HEAD")
                    size = int(urllib.request.urlopen(req, timeout=60).headers.get("Content-Length", -1))
                    if dest.exists() and dest.stat().st_size == size:
                        break
                    print(f"downloading {name} ({size / 1e6:.0f} MB)", flush=True)
                    # Download to a ".part" file first and rename only when complete. If the download dies
                    # halfway, dest never holds a cut-off file that a later run (or build_dataset) would trust.
                    tmp = dest.with_suffix(".part")
                    urllib.request.urlretrieve(url, tmp)
                    # size > 0: when the server sent no Content-Length we got -1 above and cannot check.
                    if size > 0 and tmp.stat().st_size != size:
                        raise OSError(f"size mismatch for {name}")
                    tmp.replace(dest)   # rename is atomic: dest is either the old state or the full new file
                    break
                except OSError as e:    # URLError/HTTPError/timeouts are all subclasses of OSError
                    print(f"  {name}: attempt {attempt} failed: {e}", flush=True)
                    if attempt == retries:
                        raise           # give up loudly: building on a missing file would silently lose a cell
                    # Back off 5 s, 10 s, ...: a short network glitch or GitHub rate limit has time to clear.
                    time.sleep(5 * attempt)


# ---------------------------------------------------------------- parsing

def normalize(text) -> str:
    """Collapse all whitespace. Removes collection artefacts (arXiv hard line breaks, leading spaces,
    Cohere's leading blank lines) that would otherwise be an easy non-linguistic shortcut.

    Example of the shortcut: human arXiv abstracts contain "\\n" every ~80 chars, machine ones don't. A model
    could reach ~100% accuracy just by counting newlines, which says nothing about AI text in general.
    str.split() with no argument splits on ANY run of spaces/tabs/newlines and drops empty pieces, so joining
    with one space gives a single canonical form. Paraphrased texts must get the same normalisation (AGENTS.md),
    or the paraphrase attack would also change formatting, not just wording.
    `str(text)` guards against numbers in the raw JSON; None becomes "" (later dropped by MIN_CHARS)."""
    return " ".join(str(text).split()) if text is not None else ""


def _hash(*parts: str) -> str:
    """Short, stable id (first 16 hex chars of SHA-1) for the joined parts.

    Used to build `group` ids and to detect duplicate texts. Unlike Python's hash(), SHA-1 gives the same value on
    every machine and every run, so the dataset and splits are reproducible.
    (Python's hash() of a str is salted per process by PYTHONHASHSEED, so ids would change every run.)
    The "|" separator keeps ("ab", "c") and ("a", "bc") from hashing the same. 16 hex chars = 64 bits, so a
    collision among ~60k texts is astronomically unlikely; SHA-1's known weaknesses only matter against attackers.
    """
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:16]


def _first(rec: dict, keys: list[str]):
    """Value of the first key in `keys` that the record has and that is not empty; None if there is none.

    Why: M4 files name the same field differently (human_text / abstract / text ...); the key lists at the top
    of this file give the order in which to try them.
    """
    for k in keys:
        # Skip empty values too, not only missing keys: if a record had e.g. "abstract": "" next to a filled
        # "text", taking the empty one would drop a perfectly good record.
        if k in rec and rec[k] not in (None, ""):
            return rec[k]
    return None


def _paper_id(rec: dict) -> str:
    """PeerRead paper id; files disagree on the key and type ('104' vs 104.0)."""
    raw = next((rec[k] for k in ("source_ID", "source_id", "id") if rec.get(k) not in (None, "")), "")
    # Without this, "104" and 104.0 would be two different papers -> two groups -> reviews of the SAME paper
    # could land in both train and test (the leak that groups exist to prevent).
    try:
        f = float(raw)
        return str(int(f)) if f.is_integer() else str(raw)
    except (TypeError, ValueError):
        return str(raw).strip()


def iter_units(path: Path, domain: str):
    """Yield (index, group, human_texts, machine_texts) per raw record, texts normalised.

    One unit = one pair (or, for PeerRead, one paper with lists of reviews)."""
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if not line.strip():
                continue
            rec = json.loads(line)
            h, m = _first(rec, HUMAN_KEYS), _first(rec, MACHINE_KEYS)
            if domain == "peerread":
                hs = [normalize(t) for t in (h or [])]
                ms = [normalize(t) for t in (m or [])]
                # PeerRead: one record = one paper with several human and several machine reviews. All reviews of a
                # paper discuss the same content, so they share one group (venue + paper id). If they were split
                # by row, a test review could be almost a paraphrase of a training review of the same paper.
                paper = f"{rec.get('source', '')}|{_paper_id(rec)}"
                group = _hash("peerread", paper)
            else:
                hs, ms = [normalize(h)], [normalize(m)]
                # Other domains: group = hash of the human text. Each generator wrote its text for the same human
                # prompt/article, so the human text and all machine answers to it end up in one group. Splits are
                # made by group, which keeps a text and its near-copies on the same side of train/test.
                # .lower(): copies of an article that differ only in capitalisation still get ONE group, so they
                # cannot be split across train/test. `domain` is in the hash
                # so that identical short texts in two domains do not merge into one group.
                group = _hash(domain, hs[0].lower())
            yield i, group, hs, ms


# ---------------------------------------------------------------- build

def _cell_rng(seed: int, domain: str, gen: str) -> random.Random:
    # Independent per-cell sampling (not coordinated across generators): if every cell picked the same
    # human texts, almost every group would contain every generator and LOGO train sets would be empty.
    """A random generator seeded from (seed, domain, generator): reproducible, and different for every cell."""
    # (The comment above is written before the docstring, so Python does not see this string as the docstring.)
    # Why a separate Random per cell and not one shared one: with one shared stream, the sample in cell 5
    # would depend on how many draws cells 1-4 made, so fixing one file would reshuffle every later cell.
    # int(hex, 16) turns the 16-hex-char hash into an integer seed.
    return random.Random(int(_hash(str(seed), domain, gen), 16))


def build_dataset(max_per_cell: int = 1000, raw_dir: Path = RAW_DIR, seed: int = SEED):
    """Flatten M4 into the §0.1 schema. Returns (DataFrame, stats dict for the audit)."""
    # stats only feeds write_audit(): every number the report gives about the data comes from here.
    stats = {"raw": {}, "dropped": {}, "kept_units": {}, "missing": [], "dup_humans": {}}
    # rows: plain tuples, turned into ONE DataFrame at the end (appending to a DataFrame row by row is O(n^2)).
    # seen_h / seen_m: hashes of texts already added, for de-duplication (see below).
    rows, seen_h, seen_m = [], set(), set()
    for dom in DOMAINS:
        human_files = defaultdict(set)  # human text hash -> generators whose file contains it
        for gen in GENERATORS:
            path = cell_file(dom, gen, raw_dir)
            if path is None:
                # Missing file: record it (the audit shows it) and carry on, instead of crashing the whole build.
                stats["missing"].append(f"{dom}/{gen}")
                continue
            units, n_raw, dropped = [], 0, Counter()
            for i, group, hs, ms in iter_units(path, dom):
                n_raw += 1
                for t in hs:
                    if len(t) >= MIN_CHARS:
                        human_files[_hash(dom, t.lower())].add(gen)
                hs_ok = [t for t in hs if len(t) >= MIN_CHARS]
                # A machine text identical to its human text is a collection error (the model copied the input): drop it.
                ms_ok = [t for t in ms if len(t) >= MIN_CHARS and t not in hs]
                dropped["short_or_empty_human"] += len(hs) - len(hs_ok)
                dropped["short_empty_or_copied_machine"] += len(ms) - len(ms_ok)
                if not hs_ok or not ms_ok:
                    # Keep only units with BOTH a human and a machine text, so each kept unit adds one of each
                    # and the human:machine balance stays even per cell.
                    dropped["units_without_both_sides"] += 1
                    continue
                units.append((i, group, hs_ok, ms_ok))
            stats["raw"][f"{dom}/{gen}"] = n_raw
            stats["dropped"][f"{dom}/{gen}"] = dict(dropped)
            rng = _cell_rng(seed, dom, gen)
            # PeerRead units are papers with ~4 machine reviews each: cap papers so a cell holds
            # ~max_per_cell machine texts like every other cell, and generators only partly overlap.
            cap = max_per_cell // 4 if dom == "peerread" else max_per_cell
            # Cap each cell at max_per_cell pairs so every generator and domain has a similar weight in the data.
            # Without the cap, a cell with 3,000 rows would weigh 3x a cell with 1,000, and the detector (and the
            # metrics) would mostly reflect the big cells. Tradeoff: less data, but a fair comparison per cell.
            # It also keeps the dataset small enough for 6.6 GB RAM and Kaggle's GPU time.
            if len(units) > cap:
                # sorted(): put the sample back in file order, so row ids come out in a stable order.
                units = sorted(rng.sample(units, cap))
            stats["kept_units"][f"{dom}/{gen}"] = len(units)
            for i, group, hs, ms in units:
                # Only PeerRead has several texts per side, so only its ids get a k suffix (_h0, _h1, ...).
                # Id format: <domain>_<generator>_<record index>_<h|m>[k], e.g. reddit_cohere_0042_m.
                multi = dom == "peerread"
                for k, t in enumerate(hs):
                    key = _hash(dom, t.lower())
                    if key in seen_h:  # same human text already added from another file: keep once
                        # Every generator file repeats the same human article. Keeping all copies would put the
                        # same human text in the data up to 5 times: humans would be over-weighted and the
                        # metrics would count one text several times.
                        continue
                    seen_h.add(key)
                    rows.append((f"{dom}_{gen}_{i:04d}_h{k if multi else ''}", t, 0, dom, gen, group))
                for k, t in enumerate(ms):
                    # gen IS part of the machine key: two generators writing the same sentence are two real
                    # samples; only an exact repeat from the same generator is a duplicate.
                    key = _hash(dom, gen, t)
                    if key in seen_m:
                        continue
                    seen_m.add(key)
                    rows.append((f"{dom}_{gen}_{i:04d}_m{k if multi else ''}", t, 1, dom, gen, group))
        stats["dup_humans"][dom] = {
            "unique_human_texts": len(human_files),
            "in_more_than_one_file": sum(len(g) > 1 for g in human_files.values()),
        }
    df = pd.DataFrame(rows, columns=COLUMNS)
    # Fail now rather than later: every other file (splits, preds.csv) refers to texts by id, so a duplicate id
    # would silently attach one prediction to two texts.
    assert df["id"].is_unique, "duplicate ids"
    return df, stats


# ---------------------------------------------------------------- splits

def _pattern(fracs: tuple[float, ...]) -> list[int]:
    """(0.8, 0.1, 0.1) -> [0]*8 + [1] + [2]: which part each position of a 10-slot cycle goes to."""
    # round(): 0.8 * 10 is 8.000000000000002 in floating point; int() would be fragile, round() gives 8.
    counts = [round(f * 10) for f in fracs]
    assert sum(counts) == 10, f"fractions must be multiples of 0.1: {fracs}"
    return [j for j, c in enumerate(counts) for _ in range(c)]


def _split_groups(strata: pd.Series, fracs, seed: int) -> list[set[str]]:
    """Split groups into len(fracs) parts, stratified.

    Groups are shuffled inside each stratum, strata are visited in a fixed order, and ONE running counter
    walks a repeating pattern across all of them. Remainders carry over between strata, so the totals match
    `fracs` even when every stratum is tiny (rounding each stratum separately gave small strata no val/test)."""
    # `strata` is a Series: index = group id, value = stratum label (e.g. "reddit:chatGPT+dolly").
    # Stratified = every stratum is split ~80/10/10 on its own, so val and test have the same mix of domains and
    # generators as train. A plain random split could, by chance, put few wikihow groups in test.
    # Why not sklearn's GroupShuffleSplit: it is not stratified; StratifiedGroupKFold only gives k equal folds.
    parts = [set() for _ in fracs]
    pattern = _pattern(fracs)
    rng = random.Random(seed)   # own Random object: does not touch (or depend on) the global random state
    pos = 0                     # ONE counter for all strata (see docstring): the remainder carries over
    # sorted(...): visit strata in a fixed order. Dict/groupby order is fine today, but sorting makes the
    # result not depend on pandas internals, so the split files are identical on every machine.
    for _, gs in sorted(strata.groupby(strata).groups.items(), key=lambda kv: kv[0]):
        groups = sorted(gs)     # sort first, THEN shuffle: the shuffle result depends only on the seed
        rng.shuffle(groups)
        for g in groups:
            # pattern = [0]*8+[1]+[2]: positions 0-7 -> train, 8 -> val, 9 -> test, then repeat.
            # Whole group ids go into the part, so all rows of a group end up together (no leak).
            parts[pattern[pos % len(pattern)]].add(g)
            pos += 1
    return parts


def _ids(df: pd.DataFrame, mask) -> list[str]:
    """Sorted list of the ids of the rows selected by `mask`. Sorted so split files are identical on every run."""
    return sorted(df.loc[mask, "id"].tolist())


def _group_strata(df: pd.DataFrame) -> pd.Series:
    """Stratum per group = domain + the set of generators whose machine text is in the group."""
    # Why stratify on the generator SET and not just the domain: a group with machine texts from 3 generators
    # adds 3 machine rows, one with 1 generator adds 1. Without this, test could by chance get mostly the
    # 1-generator groups and have a very different machine:human ratio and generator mix from train.
    m = df[df.label == 1]
    gens = m.groupby("group")["generator"].agg(lambda s: "+".join(sorted(set(s))))
    # .first(): every row of a group has the same domain (domain is in the group hash), so any row will do.
    dom = df.groupby("group")["domain"].first()
    # reindex + fillna("none"): a group with only human texts left has no entry in `gens`; give it its own
    # stratum instead of NaN (NaN would make the whole label NaN and groupby would silently drop the group).
    return (dom + ":" + gens.reindex(dom.index).fillna("none")).rename("stratum")


def make_splits(df: pd.DataFrame, out_dir: Path = SPLITS_DIR, seed: int = SEED,
                n_para: int = 500, n_dgpt_val: int = 500) -> dict[str, dict]:
    """Write every A-owned file of §0.2. Returns {setting: split dict}."""
    out_dir.mkdir(parents=True, exist_ok=True)
    splits: dict[str, dict] = {}
    strata = _group_strata(df)

    # indist: 80/10/10 by group, stratified by domain x generator-set
    # Whole GROUPS go to train/val/test, never single rows (see the group comments in iter_units).
    tr, va, te = _split_groups(strata, (0.8, 0.1, 0.1), seed)
    splits["indist"] = {"train": _ids(df, df.group.isin(tr)), "val": _ids(df, df.group.isin(va)),
                        "test": _ids(df, df.group.isin(te))}

    dom_of_group = df.groupby("group")["domain"].first()
    for gen in GENERATORS:
        # LOGO (leave one generator out): test = every group that has a text by `gen`, with its human texts and
        # only `gen`'s machine texts. Train/val come from the remaining groups, so the detector never sees `gen`.
        test_groups = set(df.loc[(df.label == 1) & (df.generator == gen), "group"])
        # Machine texts of OTHER generators in those test groups are used nowhere: they can't go to test (this
        # setting tests `gen` only) and can't go to train (same group as a test text = leak). That is the price
        # of leak-free splits; the audit shows the resulting sizes.
        test_mask = df.group.isin(test_groups) & ((df.label == 0) | (df.generator == gen))
        # Stratified by domain only: the generator set no longer makes sense once `gen` is removed.
        # This is also why _cell_rng samples each cell independently: if all cells used the same human texts,
        # nearly every group would contain `gen`, all groups would be test groups and train would be empty.
        rest = dom_of_group[~dom_of_group.index.isin(test_groups)]
        # 90/10: no test part needed here (test is fixed above), so all the remaining data goes to train/val.
        tr, va = _split_groups(rest, (0.9, 0.1), seed)
        splits[f"logo_{gen}"] = {"train": _ids(df, df.group.isin(tr)), "val": _ids(df, df.group.isin(va)),
                                 "test": _ids(df, test_mask)}

    for dom in DOMAINS:
        # Cross-domain: test = every text of `dom`; train/val come from the other four domains only.
        # No group crosses domains (domain is part of the group hash), so "all of dom" as test cannot leak.
        # Val is also from the other domains: the threshold must be chosen without seeing the test domain,
        # otherwise the cross-domain result would be optimistic.
        rest = strata[dom_of_group.reindex(strata.index) != dom]
        tr, va = _split_groups(rest, (0.9, 0.1), seed)
        splits[f"xdomain_{dom}"] = {"train": _ids(df, df.group.isin(tr)), "val": _ids(df, df.group.isin(va)),
                                    "test": _ids(df, df.domain == dom)}

    # paraphrase_orig: balanced subset of indist test (machines per domain x generator, humans per domain)
    rng = random.Random(seed)
    test = df[df.id.isin(splits["indist"]["test"])]
    val = df[df.id.isin(splits["indist"]["val"])]
    para_test = _balanced(test, n_para, n_para, rng)
    # Why a subset: paraphrasing each text with an LLM is slow on Kaggle GPUs, so ~500+500 texts are attacked.
    # paraphrase_orig = the SAME texts before paraphrasing, so "orig vs paraphrase" compares like with like and
    # any drop in score is due to the paraphrase, not to a different sample.
    # train is empty: models are trained in `indist` and only evaluated here; val is indist val so the threshold
    # is the same one the indist model uses.
    splits["paraphrase_orig"] = {"train": [], "val": splits["indist"]["val"], "test": sorted(para_test)}

    for name, sp in splits.items():
        (out_dir / f"{name}.json").write_text(json.dumps(sp, indent=0))

    # DetectGPT subset: the paraphrase_orig test texts + a balanced slice of indist val (for thresholds)
    # DetectGPT needs ~100 perturbations x a model pass per text, far too slow for the full test sets.
    # It needs val rows too, because the evaluator picks the threshold on val (TASKS.md §0).
    # The same `rng` is reused after paraphrase_orig, so this subset is also fixed by the seed.
    dgpt = sorted(set(para_test) | set(_balanced(val, n_dgpt_val // 2, n_dgpt_val // 2, rng)))
    (out_dir / "detectgpt_subset.json").write_text(json.dumps(dgpt, indent=0))
    splits["_detectgpt_subset"] = {"ids": dgpt}
    return splits


def _balanced(part: pd.DataFrame, n_machine: int, n_human: int, rng: random.Random) -> list[str]:
    """n_machine spread over domain x generator, n_human spread over domain."""
    out = []
    cells = sorted(part[part.label == 1].groupby(["domain", "generator"]).groups.items())
    doms = sorted(part[part.label == 0].groupby("domain").groups.items())
    for items, n in ((cells, n_machine), (doms, n_human)):
        if not items:
            continue
        # divmod(500, 25) = (20, 0); divmod(500, 3) = (166, 2): the first `extra` cells get one more,
        # so the total is exactly n (when cells are big enough).
        per, extra = divmod(n, len(items))
        for j, (_, idx) in enumerate(items):
            ids = sorted(part.loc[idx, "id"])   # sort before sampling so the sample depends only on the seed
            # (j < extra) is True/False = 1/0. min(): a small cell gives all it has instead of crashing
            # (random.sample raises ValueError if asked for more items than exist).
            out += rng.sample(ids, min(len(ids), per + (j < extra)))
    return out


# ---------------------------------------------------------------- checks

def check_splits(dev: bool = False) -> None:
    """Every id in splits/*.json exists in the data; no id or group spans two parts.

    Why: the split files are committed to git, the data is rebuilt separately (locally, on Kaggle). If the two
    ever drift (other seed, other cleaning), scripts would silently train on fewer texts or crash halfway through
    a GPU run. This check is the first cell of every Kaggle notebook, so a mismatch fails in seconds.
    It also re-proves the no-leak rule on the actual files, not just in the code that wrote them.
    """
    df = load_texts(dev=dev)
    known = set(df.id)
    group = dict(zip(df.id, df.group))
    d = DEV_SPLITS_DIR if dev else SPLITS_DIR
    files = sorted(p for p in d.glob("*.json"))
    assert files, f"no split files in {d}"
    for p in files:
        sp = json.loads(p.read_text())
        if isinstance(sp, list):  # detectgpt_subset.json
            missing = set(sp) - known
            assert not missing, f"{p.name}: {len(missing)} unknown ids, e.g. {sorted(missing)[:3]}"
            continue
        seen_ids, seen_groups = {}, {}
        for part in ("train", "val", "test"):
            ids = sp.get(part, [])
            missing = set(ids) - known
            assert not missing, f"{p.name}/{part}: {len(missing)} unknown ids, e.g. {sorted(missing)[:3]}"
            for i in ids:
                # setdefault(i, part) stores `part` the first time i is seen and returns the stored value after
                # that; if it returns a DIFFERENT part, the id is in two parts (e.g. train and test).
                assert seen_ids.setdefault(i, part) == part, f"{p.name}: id {i} in {seen_ids[i]} and {part}"
                g = group[i]
                if not p.name.startswith("paraphrase"):  # paraphrase*: val=indist val, test from indist test
                    assert seen_groups.setdefault(g, part) == part, f"{p.name}: group {g} in two parts"
        if p.name.startswith("paraphrase"):
            vg = {group[i] for i in sp["val"]}
            # paraphrase* files can't use the per-part group check above (train is empty and test holds
            # paraphrases that share a group with their originals), so check only what matters: val (where
            # the threshold is picked) shares no group with test.
            tg = {group[i] for i in sp["test"] if i in group}
            assert not vg & tg, f"{p.name}: val and test share groups"
    print(f"OK: {len(files)} split files in {d.relative_to(ROOT)} match {len(known)} texts")


# ---------------------------------------------------------------- dev sample

def make_dev_sample(df: pd.DataFrame, per_cell: int = 5, seed: int = SEED) -> pd.DataFrame:
    """~500 rows: `per_cell` groups per (domain, generator) with all their rows."""
    rng = random.Random(seed)
    picked = set()
    m = df[df.label == 1]
    for (_, _), idx in sorted(m.groupby(["domain", "generator"]).groups.items()):
        groups = sorted(set(m.loc[idx, "group"]) - picked)
        picked.update(rng.sample(groups, min(per_cell, len(groups))))
    # Pick whole GROUPS (not rows), so the dev sample can still be split by group like the real data, and
    # every (domain, generator) cell is present: a --dev run exercises every code path, LOGO and xdomain too.
    # `- picked`: don't pick a group twice (it may hold several generators).
    dev = df[df.group.isin(picked)]
    # keep PeerRead from dominating (many reviews per paper): at most 2 rows per group/label/generator
    dev = dev.groupby(["group", "label", "generator"]).head(2)
    return dev.sort_values("id").reset_index(drop=True)[COLUMNS]


# ---------------------------------------------------------------- audit

def write_audit(stats: dict, df: pd.DataFrame, splits: dict, path: Path = AUDIT_PATH) -> None:
    """Write results/data_audit.md: raw counts per file, what cleaning dropped, duplicate human texts, final
    dataset size and every split's sizes and machine:human ratio.

    Why: the report's dataset numbers come from this file, and it lets anyone check the data without rerunning.
    """
    L = ["# M4 English data audit", "", f"Generated by `python -m src.data --all` (seed {SEED}, "
         f"max {stats['max_per_cell']} pairs per cell (PeerRead: {stats['max_per_cell'] // 4} papers), texts < {MIN_CHARS} chars dropped).", ""]
    L += ["## Grid", "", "Raw records per file (a record = one human/machine pair; PeerRead = one paper):", "",
          "| domain | " + " | ".join(GENERATORS) + " |", "|---" * (len(GENERATORS) + 1) + "|"]
    for d in DOMAINS:
        L.append(f"| {d} | " + " | ".join(str(stats["raw"].get(f"{d}/{g}", "MISSING")) for g in GENERATORS) + " |")
    L += ["", f"Missing cells: {', '.join(stats['missing']) or 'none'}. "
          f"Excluded files (other generators): {', '.join(EXCLUDED_FILES)}.", ""]
    L += ["## Cleaning", "",
          "- Whitespace is collapsed in every text (arXiv/PeerRead human texts keep hard line breaks and "
          "leading spaces; machine texts don't, which would be a formatting shortcut).",
          "- BLOOMZ files: human text from `text`/`abstract`, machine text from `machine_answer`/"
          "`machine_abstract` (their `machine_text` field is the prompt + output).",
          "- PeerRead: one row per review (`_h<k>`/`_m<k>`), group = venue + paper id (ids normalised: '104' = 104.0); "
          "duplicate reviews are kept once; at most max_per_cell // 4 papers per cell (~4 machine reviews each).", "",
          "| cell | dropped short/empty human | dropped short/empty/copied machine | units dropped | units kept |",
          "|---|---|---|---|---|"]
    for k, v in stats["dropped"].items():
        L.append(f"| {k} | {v.get('short_or_empty_human', 0)} | {v.get('short_empty_or_copied_machine', 0)} | "
                 f"{v.get('units_without_both_sides', 0)} | {stats['kept_units'][k]} |")
    L += ["", "## Duplicate human texts across generator files (raw, before subsampling)", "",
          "| domain | unique human texts | in more than one generator file |", "|---|---|---|"]
    for d, v in stats["dup_humans"].items():
        L.append(f"| {d} | {v['unique_human_texts']} | {v['in_more_than_one_file']} |")
    L += ["", "So splits are made by `group` (hash of the human text; PeerRead: the paper).", ""]
    L += ["## Final dataset", "", f"{len(df)} texts: {int((df.label == 0).sum())} human, "
          f"{int((df.label == 1).sum())} machine, {df.group.nunique()} groups.", "",
          "| domain | human | " + " | ".join(GENERATORS) + " |", "|---" * (len(GENERATORS) + 2) + "|"]
    for d in DOMAINS:
        sub = df[df.domain == d]
        L.append(f"| {d} | {int((sub.label == 0).sum())} | " + " | ".join(
            str(int(((sub.label == 1) & (sub.generator == g)).sum())) for g in GENERATORS) + " |")
    L += ["", "## Splits (rows, and machine:human ratio)", "",
          "| setting | train | val | test | test humans | ratio train | ratio test |", "|---|---|---|---|---|---|---|"]
    lab = dict(zip(df.id, df.label))
    for name in SETTINGS_TOP:
        sp = splits[name]

        def ratio(ids):
            """Machine:human ratio of a list of ids as text, e.g. '1.87'; '-' for an empty list."""
            m = sum(lab[i] for i in ids)
            return f"{m / max(len(ids) - m, 1):.2f}" if ids else "-"
        th = sum(1 for i in sp["test"] if lab[i] == 0)
        L.append(f"| `{name}` | {len(sp['train'])} | {len(sp['val'])} | {len(sp['test'])} | {th} | "
                 f"{ratio(sp['train'])} | {ratio(sp['test'])} |")
    L += ["", f"`detectgpt_subset.json`: {len(splits['_detectgpt_subset']['ids'])} ids "
          "(the `paraphrase_orig` test set + a balanced slice of `indist` val).",
          "", "TPR@1%FPR is only reported where test humans >= 1,000 (TASKS.md §0.3).", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L))
    print(f"wrote {path.relative_to(ROOT)}")


# ---------------------------------------------------------------- CLI

def main(argv=None) -> int:
    """Command-line entry point (see the module docstring). Returns 0, used as the process exit code.

    --all runs the whole data pipeline in order: download, build m4_en.parquet, write splits, write the audit,
    make the dev sample and its splits, then check every split file against the data.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="download, build, splits, audit, dev sample")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--check-splits", action="store_true", help="validate split files against the data")
    ap.add_argument("--dev", action="store_true", help="with --check-splits: check the dev sample + splits/dev")
    ap.add_argument("--max-per-cell", type=int, default=1000)
    a = ap.parse_args(argv)

    if a.check_splits:
        # Return right away: --check-splits only reads, it must never rebuild data or rewrite split files.
        check_splits(dev=a.dev)
        return 0
    if a.download or a.all:
        download_m4()
    if a.all:
        df, stats = build_dataset(a.max_per_cell)
        stats["max_per_cell"] = a.max_per_cell
        PROC_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(DATASET_PATH, index=False)
        print(f"wrote {DATASET_PATH.relative_to(ROOT)}: {len(df)} rows")
        splits = make_splits(df)
        write_audit(stats, df, splits)
        dev = make_dev_sample(df)
        dev.to_parquet(DEV_PATH, index=False)
        # Same split code on the sample (smaller subsets): dev runs test the real logic, just on less data.
        make_splits(dev, DEV_SPLITS_DIR, n_para=50, n_dgpt_val=50)
        print(f"wrote {DEV_PATH.relative_to(ROOT)}: {len(dev)} rows, splits in {DEV_SPLITS_DIR.relative_to(ROOT)}")
        check_splits()
        check_splits(dev=True)
    if not (a.all or a.download):
        ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
