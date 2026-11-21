"""M4 English: download, audit, build the dataset, make group-aware splits, shared loaders.

Contract: docs/TASKS.md §0.1 (dataset columns) and §0.2 (split files).

    python -m src.data --all            # download + build + splits + audit + dev sample
    python -m src.data --check-splits   # assert every split id exists in the data (Kaggle first cell)
    python -m src.data --check-splits --dev

Everyone else should only need `load_texts()` and `load_split()`.
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

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
DATASET_PATH = PROC_DIR / "m4_en.parquet"
PARAPHRASED_PATH = PROC_DIR / "paraphrased.parquet"
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
MIN_CHARS = 50
SETTINGS_TOP = ["indist"] + [f"logo_{g}" for g in GENERATORS] + [f"xdomain_{d}" for d in DOMAINS] + ["paraphrase_orig"]
COLUMNS = ["id", "text", "label", "domain", "generator", "group"]


# ---------------------------------------------------------------- shared loaders (everyone)

def load_texts(dev: bool = False) -> pd.DataFrame:
    """All texts with labels: m4_en.parquet (+ paraphrased.parquet if it exists). Index is not set."""
    if dev:
        return pd.read_parquet(DEV_PATH)
    parts = [pd.read_parquet(DATASET_PATH)]
    if PARAPHRASED_PATH.exists():
        parts.append(pd.read_parquet(PARAPHRASED_PATH))
    return pd.concat(parts, ignore_index=True)


def load_split(setting: str, dev: bool = False) -> dict[str, list[str]]:
    """{'train': [...], 'val': [...], 'test': [...]} for a setting name from TASKS.md §0.2."""
    d = DEV_SPLITS_DIR if dev else SPLITS_DIR
    return json.loads((d / f"{setting}.json").read_text())


# ---------------------------------------------------------------- download

def file_name(domain: str, file_gen: str) -> str:
    return f"{domain}_{file_gen}.jsonl"


def cell_file(domain: str, gen: str, raw_dir: Path = RAW_DIR) -> Path | None:
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
                    req = urllib.request.Request(url, method="HEAD")
                    size = int(urllib.request.urlopen(req, timeout=60).headers.get("Content-Length", -1))
                    if dest.exists() and dest.stat().st_size == size:
                        break
                    print(f"downloading {name} ({size / 1e6:.0f} MB)", flush=True)
                    tmp = dest.with_suffix(".part")
                    urllib.request.urlretrieve(url, tmp)
                    if size > 0 and tmp.stat().st_size != size:
                        raise OSError(f"size mismatch for {name}")
                    tmp.replace(dest)
                    break
                except OSError as e:
                    print(f"  {name}: attempt {attempt} failed: {e}", flush=True)
                    if attempt == retries:
                        raise
                    time.sleep(5 * attempt)


# ---------------------------------------------------------------- parsing

def normalize(text) -> str:
    """Collapse all whitespace. Removes collection artefacts (arXiv hard line breaks, leading spaces,
    Cohere's leading blank lines) that would otherwise be an easy non-linguistic shortcut."""
    return " ".join(str(text).split()) if text is not None else ""


def _hash(*parts: str) -> str:
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:16]


def _first(rec: dict, keys: list[str]):
    for k in keys:
        if k in rec and rec[k] not in (None, ""):
            return rec[k]
    return None


def _paper_id(rec: dict) -> str:
    """PeerRead paper id; files disagree on the key and type ('104' vs 104.0)."""
    raw = next((rec[k] for k in ("source_ID", "source_id", "id") if rec.get(k) not in (None, "")), "")
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
                paper = f"{rec.get('source', '')}|{_paper_id(rec)}"
                group = _hash("peerread", paper)
            else:
                hs, ms = [normalize(h)], [normalize(m)]
                group = _hash(domain, hs[0].lower())
            yield i, group, hs, ms


# ---------------------------------------------------------------- build

def _cell_rng(seed: int, domain: str, gen: str) -> random.Random:
    # Independent per-cell sampling (not coordinated across generators): if every cell picked the same
    # human texts, almost every group would contain every generator and LOGO train sets would be empty.
    return random.Random(int(_hash(str(seed), domain, gen), 16))


def build_dataset(max_per_cell: int = 1000, raw_dir: Path = RAW_DIR, seed: int = SEED):
    """Flatten M4 into the §0.1 schema. Returns (DataFrame, stats dict for the audit)."""
    stats = {"raw": {}, "dropped": {}, "kept_units": {}, "missing": [], "dup_humans": {}}
    rows, seen_h, seen_m = [], set(), set()
    for dom in DOMAINS:
        human_files = defaultdict(set)  # human text hash -> generators whose file contains it
        for gen in GENERATORS:
            path = cell_file(dom, gen, raw_dir)
            if path is None:
                stats["missing"].append(f"{dom}/{gen}")
                continue
            units, n_raw, dropped = [], 0, Counter()
            for i, group, hs, ms in iter_units(path, dom):
                n_raw += 1
                for t in hs:
                    if len(t) >= MIN_CHARS:
                        human_files[_hash(dom, t.lower())].add(gen)
                hs_ok = [t for t in hs if len(t) >= MIN_CHARS]
                ms_ok = [t for t in ms if len(t) >= MIN_CHARS and t not in hs]
                dropped["short_or_empty_human"] += len(hs) - len(hs_ok)
                dropped["short_empty_or_copied_machine"] += len(ms) - len(ms_ok)
                if not hs_ok or not ms_ok:
                    dropped["units_without_both_sides"] += 1
                    continue
                units.append((i, group, hs_ok, ms_ok))
            stats["raw"][f"{dom}/{gen}"] = n_raw
            stats["dropped"][f"{dom}/{gen}"] = dict(dropped)
            rng = _cell_rng(seed, dom, gen)
            # PeerRead units are papers with ~4 machine reviews each: cap papers so a cell holds
            # ~max_per_cell machine texts like every other cell, and generators only partly overlap.
            cap = max_per_cell // 4 if dom == "peerread" else max_per_cell
            if len(units) > cap:
                units = sorted(rng.sample(units, cap))
            stats["kept_units"][f"{dom}/{gen}"] = len(units)
            for i, group, hs, ms in units:
                multi = dom == "peerread"
                for k, t in enumerate(hs):
                    key = _hash(dom, t.lower())
                    if key in seen_h:  # same human text already added from another file: keep once
                        continue
                    seen_h.add(key)
                    rows.append((f"{dom}_{gen}_{i:04d}_h{k if multi else ''}", t, 0, dom, gen, group))
                for k, t in enumerate(ms):
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
    assert df["id"].is_unique, "duplicate ids"
    return df, stats


# ---------------------------------------------------------------- splits

def _pattern(fracs: tuple[float, ...]) -> list[int]:
    """(0.8, 0.1, 0.1) -> [0]*8 + [1] + [2]: which part each position of a 10-slot cycle goes to."""
    counts = [round(f * 10) for f in fracs]
    assert sum(counts) == 10, f"fractions must be multiples of 0.1: {fracs}"
    return [j for j, c in enumerate(counts) for _ in range(c)]


def _split_groups(strata: pd.Series, fracs, seed: int) -> list[set[str]]:
    """Split groups into len(fracs) parts, stratified.

    Groups are shuffled inside each stratum, strata are visited in a fixed order, and ONE running counter
    walks a repeating pattern across all of them. Remainders carry over between strata, so the totals match
    `fracs` even when every stratum is tiny (rounding each stratum separately gave small strata no val/test)."""
    parts = [set() for _ in fracs]
    pattern = _pattern(fracs)
    rng = random.Random(seed)
    pos = 0
    for _, gs in sorted(strata.groupby(strata).groups.items(), key=lambda kv: kv[0]):
        groups = sorted(gs)
        rng.shuffle(groups)
        for g in groups:
            parts[pattern[pos % len(pattern)]].add(g)
            pos += 1
    return parts


def _ids(df: pd.DataFrame, mask) -> list[str]:
    return sorted(df.loc[mask, "id"].tolist())


def _group_strata(df: pd.DataFrame) -> pd.Series:
    """Stratum per group = domain + the set of generators whose machine text is in the group."""
    m = df[df.label == 1]
    gens = m.groupby("group")["generator"].agg(lambda s: "+".join(sorted(set(s))))
    dom = df.groupby("group")["domain"].first()
    return (dom + ":" + gens.reindex(dom.index).fillna("none")).rename("stratum")


def make_splits(df: pd.DataFrame, out_dir: Path = SPLITS_DIR, seed: int = SEED,
                n_para: int = 500, n_dgpt_val: int = 500) -> dict[str, dict]:
    """Write every A-owned file of §0.2. Returns {setting: split dict}."""
    out_dir.mkdir(parents=True, exist_ok=True)
    splits: dict[str, dict] = {}
    strata = _group_strata(df)

    # indist: 80/10/10 by group, stratified by domain x generator-set
    tr, va, te = _split_groups(strata, (0.8, 0.1, 0.1), seed)
    splits["indist"] = {"train": _ids(df, df.group.isin(tr)), "val": _ids(df, df.group.isin(va)),
                        "test": _ids(df, df.group.isin(te))}

    dom_of_group = df.groupby("group")["domain"].first()
    for gen in GENERATORS:
        test_groups = set(df.loc[(df.label == 1) & (df.generator == gen), "group"])
        test_mask = df.group.isin(test_groups) & ((df.label == 0) | (df.generator == gen))
        rest = dom_of_group[~dom_of_group.index.isin(test_groups)]
        tr, va = _split_groups(rest, (0.9, 0.1), seed)
        splits[f"logo_{gen}"] = {"train": _ids(df, df.group.isin(tr)), "val": _ids(df, df.group.isin(va)),
                                 "test": _ids(df, test_mask)}

    for dom in DOMAINS:
        rest = strata[dom_of_group.reindex(strata.index) != dom]
        tr, va = _split_groups(rest, (0.9, 0.1), seed)
        splits[f"xdomain_{dom}"] = {"train": _ids(df, df.group.isin(tr)), "val": _ids(df, df.group.isin(va)),
                                    "test": _ids(df, df.domain == dom)}

    # paraphrase_orig: balanced subset of indist test (machines per domain x generator, humans per domain)
    rng = random.Random(seed)
    test = df[df.id.isin(splits["indist"]["test"])]
    val = df[df.id.isin(splits["indist"]["val"])]
    para_test = _balanced(test, n_para, n_para, rng)
    splits["paraphrase_orig"] = {"train": [], "val": splits["indist"]["val"], "test": sorted(para_test)}

    for name, sp in splits.items():
        (out_dir / f"{name}.json").write_text(json.dumps(sp, indent=0))

    # DetectGPT subset: the paraphrase_orig test texts + a balanced slice of indist val (for thresholds)
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
        per, extra = divmod(n, len(items))
        for j, (_, idx) in enumerate(items):
            ids = sorted(part.loc[idx, "id"])
            out += rng.sample(ids, min(len(ids), per + (j < extra)))
    return out


# ---------------------------------------------------------------- checks

def check_splits(dev: bool = False) -> None:
    """Every id in splits/*.json exists in the data; no id or group spans two parts."""
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
                assert seen_ids.setdefault(i, part) == part, f"{p.name}: id {i} in {seen_ids[i]} and {part}"
                g = group[i]
                if not p.name.startswith("paraphrase"):  # paraphrase*: val=indist val, test from indist test
                    assert seen_groups.setdefault(g, part) == part, f"{p.name}: group {g} in two parts"
        if p.name.startswith("paraphrase"):
            vg = {group[i] for i in sp["val"]}
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
    dev = df[df.group.isin(picked)]
    # keep PeerRead from dominating (many reviews per paper): at most 2 rows per group/label/generator
    dev = dev.groupby(["group", "label", "generator"]).head(2)
    return dev.sort_values("id").reset_index(drop=True)[COLUMNS]


# ---------------------------------------------------------------- audit

def write_audit(stats: dict, df: pd.DataFrame, splits: dict, path: Path = AUDIT_PATH) -> None:
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
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="download, build, splits, audit, dev sample")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--check-splits", action="store_true", help="validate split files against the data")
    ap.add_argument("--dev", action="store_true", help="with --check-splits: check the dev sample + splits/dev")
    ap.add_argument("--max-per-cell", type=int, default=1000)
    a = ap.parse_args(argv)

    if a.check_splits:
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
        make_splits(dev, DEV_SPLITS_DIR, n_para=50, n_dgpt_val=50)
        print(f"wrote {DEV_PATH.relative_to(ROOT)}: {len(dev)} rows, splits in {DEV_SPLITS_DIR.relative_to(ROOT)}")
        check_splits()
        check_splits(dev=True)
    if not (a.all or a.download):
        ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
