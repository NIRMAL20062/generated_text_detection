"""Shared evaluator: turns one preds.csv into one metrics.json (contract: docs/TASKS.md §0.3).

Usage:
    python -m src.eval results/<method>/<setting>/preds.csv                 # full data
    python -m src.eval results/<method>/<setting>/preds.csv --by generator  # + AUROC per generator
    python -m src.eval results/<method>/<setting>/preds.csv --dev           # dev sample (fast check)

Why one shared file: if every detector computed its own metrics, small differences (another threshold,
another CI method) would make the tables unfair. Here every method is scored by exactly the same code.

What it reports (all on the TEST rows):
- AUROC, the main metric, plus a bootstrap 95% confidence interval.
- Accuracy and macro-F1, using a threshold chosen on the VAL rows (never on test). If val has fewer
  than 50 texts of either class, these are null and only AUROC is reported.
- TPR at 1% FPR, only if test has at least 1,000 human texts (otherwise too noisy, so null).
- AUROC per domain and per generator, saved in metrics.json; --by also prints one of them.

Input format (preds.csv), one row per text:
    id,split,score            split is "val" or "test"; score: HIGHER = MORE LIKELY AI
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve

from src.data import DEV_SPLITS_DIR, ROOT, SEED, SPLITS_DIR, load_texts

SUBSET_METHODS = {"detectgpt", "fastdetectgpt"}   # too expensive to score every text: test may be partial
N_BOOT = 1000            # how many bootstrap resamples for the AUROC confidence interval
MIN_VAL_PER_CLASS = 50   # fewer val texts than this per class -> threshold is unreliable -> null
MIN_TEST_HUMANS = 1000   # 1% of 1,000 humans = 10 texts; with fewer, TPR@1%FPR is noise -> null
TARGET_FPR = 0.01        # "wrongly flag at most 1% of human texts"


# =============================================================================================
# 1. Loading and checking the predictions
#    A wrong preds.csv should crash loudly here, not quietly produce a wrong number in the report.
# =============================================================================================

def load_preds(path: Path, texts: pd.DataFrame) -> pd.DataFrame:
    """Read preds.csv, check it is valid, and attach the true label (and domain/generator/group).

    The detector only gives us (id, split, score). The true answer (label 0 = human, 1 = machine)
    comes from the dataset, so a detector can never "grade itself".
    """
    preds = pd.read_csv(path, dtype={"id": str, "split": str})

    # Check 1: the three required columns exist.
    missing_cols = {"id", "split", "score"} - set(preds.columns)
    if missing_cols:
        raise ValueError(f"{path}: missing columns {sorted(missing_cols)}")
    # Check 2: each text appears once (a duplicate would count twice in the metrics).
    if preds["id"].duplicated().any():
        raise ValueError(f"{path}: duplicate ids, e.g. {preds.loc[preds['id'].duplicated(), 'id'].iloc[0]}")
    # Check 3: only val and test rows. Train rows must never be scored as if they were unseen.
    bad_split = set(preds["split"]) - {"val", "test"}
    if bad_split:
        raise ValueError(f"{path}: split must be 'val' or 'test', got {sorted(bad_split)}")
    # Check 4: every score is a real number (a NaN would silently break AUROC).
    if not np.isfinite(preds["score"].to_numpy(dtype=float)).all():
        raise ValueError(f"{path}: score has NaN or inf values")

    # Attach the true labels. how="left" keeps every prediction; ids missing from the dataset get
    # NaN labels, which Check 5 catches. validate="one_to_one" guards against duplicate dataset ids.
    meta = texts[["id", "label", "domain", "generator", "group"]]
    merged = preds.merge(meta, on="id", how="left", validate="one_to_one")
    # Check 5: every id exists in the dataset (catches typos and ids from an old data version).
    unknown = merged["label"].isna()
    if unknown.any():
        raise ValueError(f"{path}: {unknown.sum()} ids not in load_texts(), e.g. {merged.loc[unknown, 'id'].iloc[0]}")
    merged["label"] = merged["label"].astype(int)
    return merged


def check_against_split(preds: pd.DataFrame, method: str, setting: str, dev: bool) -> dict:
    """Check every predicted id sits in the same part (val/test) of the setting's split file.

    This catches the worst mistake: scoring a text the model trained on as if it were test data.
    The setting name comes from the folder: results/<method>/<setting>/preds.csv, so a misspelled
    folder has no split file and is an error (otherwise the leakage check would be silently skipped).

    Coverage = the share of the split's val/test that was scored. Test must be complete, because a
    detector that skipped some test texts would be graded on an easier, different test set. The one
    exception is DetectGPT (SUBSET_METHODS), which scores only a fixed subsample by design (TASKS.md C).
    Incomplete val only gets a warning: it affects the threshold, not which texts are graded.
    """
    split_file = (DEV_SPLITS_DIR if dev else SPLITS_DIR) / f"{setting}.json"
    if not split_file.exists():
        raise ValueError(f"no split file {split_file}: is the folder name '{setting}' a real setting (TASKS.md §0.2)?")
    split = json.loads(split_file.read_text())
    coverage = {}
    for part in ("val", "test"):
        ids = set(preds.loc[preds["split"] == part, "id"])
        allowed = set(split.get(part, []))
        stray = ids - allowed          # ids the detector called "val"/"test" that are not in that part
        if stray:
            raise ValueError(f"{len(stray)} '{part}' ids are not in {split_file.name}[{part}], e.g. {next(iter(stray))}")
        coverage[part] = round(len(ids) / len(allowed), 4) if allowed else None
    if coverage["test"] is not None and coverage["test"] < 1 and method not in SUBSET_METHODS:
        raise ValueError(f"only {coverage['test']:.1%} of {split_file.name}[test] was scored; "
                         f"every method except {sorted(SUBSET_METHODS)} must score the whole test set")
    if coverage["val"] is not None and coverage["val"] < 1:
        print(f"warning: only {coverage['val']:.1%} of {split_file.name}[val] was scored", file=sys.stderr)
    return coverage


# =============================================================================================
# 2. The metrics
#    Convention everywhere: y = true labels (0 human, 1 machine), s = scores (higher = more AI).
# =============================================================================================

def auroc(y: np.ndarray, s: np.ndarray) -> float | None:
    """Area under the ROC curve = the chance that a random machine text scores higher than a random
    human text.

    1.0 = perfect ranking, 0.5 = coin flip, 0.0 = perfectly reversed (still a signal, just flipped).
    It needs no threshold, which is why it is our main metric. Returns None if only one class is
    present, because then there is nothing to rank against.
    """
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, s))


def bootstrap_auroc_ci(y: np.ndarray, s: np.ndarray, groups: np.ndarray | None = None,
                       n_boot: int = N_BOOT, seed: int = SEED):
    """95% confidence interval for AUROC by bootstrapping.

    Idea: our test set is one sample of the world. Draw n_boot new test sets by picking items
    *with replacement* (some twice, some not at all), compute AUROC on each, and keep the middle 95%
    of those values. A wide interval means the test set is too small to trust.
    A fixed seed makes the interval identical on every run.

    We resample whole GROUPS, not single rows: texts in one group are related (a human text and its
    machine twin on the same topic, or all reviews of one PeerRead paper). Resampling rows would treat
    them as independent and make the interval look narrower (more certain) than it really is.
    Without `groups`, each row is its own group.
    """
    rng = np.random.default_rng(seed)
    if groups is None:
        groups = np.arange(len(y))
    # row numbers belonging to each group, computed once
    _, group_idx = np.unique(groups, return_inverse=True)
    members = np.split(np.argsort(group_idx, kind="stable"), np.cumsum(np.bincount(group_idx))[:-1])
    n_groups = len(members)
    stats = []
    for _ in range(n_boot):
        picked = rng.integers(0, n_groups, n_groups)              # n_groups random groups, repeats allowed
        idx = np.concatenate([members[g] for g in picked])        # all rows of the picked groups
        if y[idx].min() == y[idx].max():       # resample drew only one class: AUROC undefined, skip it
            continue
        stats.append(roc_auc_score(y[idx], s[idx]))
    if not stats:
        return None, None
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def best_threshold(y: np.ndarray, s: np.ndarray) -> float:
    """The threshold t (predict "machine" if score >= t) with the highest macro-F1. We call it on VAL.

    Why val: choosing t on test would tune the answer to the exam it is graded on.
    Why macro-F1: it averages the F1 of both classes, so a detector can't score well by calling
    everything "machine" (the majority class here).

    How: try every distinct score as t, all at once. Sort the scores high -> low. Calling the top k
    texts "machine" gives:
        TP = machine texts in the top k        FP = k - TP     (humans wrongly flagged)
        FN = all machine texts - TP            TN = all humans - FP
    np.cumsum computes TP for every k in one go, so no slow Python loop is needed.

    Worked example: scores [0.9, 0.8, 0.6, 0.4, 0.1], labels [1, 1, 0, 1, 0]
        t = 0.8 -> top 2 = two machines: TP=2 FP=0 FN=1 TN=2 -> macro-F1 0.80 (the best, so t = 0.8)
    (tests/test_eval.py checks this example.)
    """
    order = np.argsort(-s, kind="stable")        # row positions sorted by score, highest first
    s_sorted, y_sorted = s[order], y[order]
    P, N = y.sum(), len(y) - y.sum()             # total machine texts, total human texts
    tp = np.cumsum(y_sorted)                     # TP if we cut after position k (for every k)
    fp = np.arange(1, len(y) + 1) - tp
    fn, tn = P - tp, N - fp
    # F1 = 2TP / (2TP + FP + FN); for the human class the roles of TP and TN swap.
    # np.maximum(..., 1) avoids dividing by zero.
    f1_machine = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    f1_human = 2 * tn / np.maximum(2 * tn + fn + fp, 1)
    macro = (f1_machine + f1_human) / 2
    # Two texts with the same score must get the same prediction, so only cut where the score changes.
    valid = np.append(s_sorted[:-1] != s_sorted[1:], True)
    k = np.flatnonzero(valid)[np.argmax(macro[valid])]
    return float(s_sorted[k])


def tpr_at_fpr(y: np.ndarray, s: np.ndarray, target: float = TARGET_FPR) -> float:
    """True positive rate at a fixed false positive rate: the share of machine texts we still catch
    when we may wrongly flag at most `target` (1%) of human texts.

    Why it matters: in real use, falsely accusing a student of using AI is costly. A detector with a
    high AUROC can still be useless at 1% FPR.
    roc_curve gives every (FPR, TPR) point; we take the best TPR among points with FPR <= target.
    """
    fpr, tpr, _ = roc_curve(y, s)
    return float(tpr[fpr <= target].max())


def breakdown(test: pd.DataFrame, by: str, texts: pd.DataFrame) -> dict:
    """AUROC separately for each domain, or for each generator, on the test rows.

    Shows *where* a detector fails, e.g. fine on Wikipedia but weak on Reddit.

    Domain: easy, because human and machine texts both carry their real domain.
    Generator: a human text has no generator of its own. After dedup, the human row's `generator`
    field is just whichever file we kept it from, so we must not use it. Instead a human text counts
    for generator g if its group (= its pair) contains a g machine text anywhere in the dataset
    (TASKS.md §0.1).
    """
    out = {}
    machines = test[test["label"] == 1]
    humans = test[test["label"] == 0]
    for value in sorted(machines[by].unique()):
        m = machines[machines[by] == value]
        if by == "domain":
            h = humans[humans["domain"] == value]
        else:
            # every group in the whole dataset that contains a machine text from this generator
            gen_groups = set(texts.loc[(texts["label"] == 1) & (texts["generator"] == value), "group"])
            h = humans[humans["group"].isin(gen_groups)]
        part = pd.concat([m, h])
        out[value] = {
            "auroc": auroc(part["label"].to_numpy(), part["score"].to_numpy()),
            "n_machine": len(m),
            "n_human": len(h),
        }
    return out


# =============================================================================================
# 3. Putting it together
# =============================================================================================

def evaluate(preds: pd.DataFrame, texts: pd.DataFrame) -> dict:
    """All metrics for one preds file (already loaded and checked). Returns the metrics.json dict."""
    val, test = preds[preds["split"] == "val"], preds[preds["split"] == "test"]
    if test.empty:
        raise ValueError("no test rows in preds.csv")
    yv, sv = val["label"].to_numpy(), val["score"].to_numpy(dtype=float)
    yt, st = test["label"].to_numpy(), test["score"].to_numpy(dtype=float)
    n_test_h, n_test_m = int((yt == 0).sum()), int((yt == 1).sum())
    n_val_h, n_val_m = int((yv == 0).sum()), int((yv == 1).sum())

    # Threshold-free metrics: always reported.
    lo, hi = bootstrap_auroc_ci(yt, st, groups=test["group"].to_numpy())
    m = {
        "auroc": auroc(yt, st),
        "auroc_ci95": [lo, hi],
        "threshold": None,
        "accuracy": None,
        "macro_f1": None,
        "tpr_at_1pct_fpr": None,
        "n_test": len(test), "n_test_human": n_test_h, "n_test_machine": n_test_m,
        "n_val": len(val), "n_val_human": n_val_h, "n_val_machine": n_val_m,
    }
    # Threshold metrics: only if val is big enough to pick a trustworthy threshold.
    # The threshold is chosen on val, then applied unchanged to test.
    if min(n_val_h, n_val_m) >= MIN_VAL_PER_CLASS:
        t = best_threshold(yv, sv)
        pred = (st >= t).astype(int)
        m.update(threshold=t, accuracy=float(accuracy_score(yt, pred)),
                 macro_f1=float(f1_score(yt, pred, average="macro")))
    # Low-FPR metric: only if there are enough humans for "1% of humans" to mean something.
    if n_test_h >= MIN_TEST_HUMANS and n_test_m > 0:
        m["tpr_at_1pct_fpr"] = tpr_at_fpr(yt, st)
    m["by_domain"] = breakdown(test, "domain", texts)
    m["by_generator"] = breakdown(test, "generator", texts)
    return m


# =============================================================================================
# 4. Command line
# =============================================================================================

def _fmt(x) -> str:
    """Print a number with 4 decimals, or 'null' if it was not computed."""
    return "null" if x is None else f"{x:.4f}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("preds", type=Path, help="results/<method>/<setting>/preds.csv")
    ap.add_argument("--by", choices=["domain", "generator"], help="also print AUROC per domain or generator")
    ap.add_argument("--dev", action="store_true", help="labels from data/sample_dev.parquet, splits from splits/dev/")
    a = ap.parse_args(argv)

    # The folder names tell us what this run is: results/<method>/<setting>/preds.csv
    setting, method = a.preds.parent.name, a.preds.parent.parent.name
    # Dev runs must never land in results/, where their metrics.json would overwrite real results
    # and could end up in the report tables.
    if a.dev and a.preds.resolve().is_relative_to(ROOT / "results"):
        raise ValueError(f"--dev results must not be inside results/ (got {a.preds}); write them to results_dev/")
    texts = load_texts(dev=a.dev)
    preds = load_preds(a.preds, texts)
    coverage = check_against_split(preds, method, setting, a.dev)
    metrics = {"method": method, "setting": setting, "dev": a.dev, "coverage": coverage, **evaluate(preds, texts)}

    # metrics.json goes next to preds.csv; make_tables.py collects all of them later.
    out = a.preds.parent / "metrics.json"
    out.write_text(json.dumps(metrics, indent=2) + "\n")
    lo, hi = metrics["auroc_ci95"]
    print(f"{method}/{setting}: AUROC {_fmt(metrics['auroc'])} [{_fmt(lo)}, {_fmt(hi)}]  "
          f"acc {_fmt(metrics['accuracy'])}  macro-F1 {_fmt(metrics['macro_f1'])}  "
          f"TPR@1%FPR {_fmt(metrics['tpr_at_1pct_fpr'])}  "
          f"(test {metrics['n_test_human']} human / {metrics['n_test_machine']} machine)")
    if a.by:
        for value, r in metrics[f"by_{a.by}"].items():
            print(f"  {value:<10} AUROC {_fmt(r['auroc'])}  ({r['n_human']} human / {r['n_machine']} machine)")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
