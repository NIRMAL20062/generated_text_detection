"""Baselines: TF-IDF + logistic regression, and a length-only sanity check (TASKS.md, role A).

    python -m src.baseline_tfidf                          # tfidf_lr on every setting -> results/tfidf_lr/<setting>/
    python -m src.baseline_tfidf --method length          # length-only LR            -> results/length/<setting>/
    python -m src.baseline_tfidf --setting indist logo_dolly
    python -m src.baseline_tfidf --dev                    # dev sample, < 2 min on CPU -> results_dev/...

Then score each run with the shared evaluator:
    python -m src.eval results/tfidf_lr/indist/preds.csv

Input:  texts via src.data.load_texts() (data/processed/m4_en.parquet, + paraphrased.parquet if present) and the
        split files splits/<setting>.json via load_split() (data/sample_dev.parquet + splits/dev/ with --dev).
Output: results/<method>/<setting>/preds.csv   (id, split, score) for every val + test id; score = P(machine)
        results/<method>/<setting>/run_info.json   (git commit, seed, hyperparameters, sizes, runtime, hardware)
        results/tfidf_lr/<setting>/top_features.csv   (tfidf_lr only)

Why these baselines: a cheap, transparent model sets the bar that RoBERTa and the zero-shot methods must beat,
and its weights can be read to see what it relies on. The length check tells us whether a trivial shortcut
exists in the data.

Methods
- tfidf_lr: word 1-2-grams + char 3-5-grams (TF-IDF, sublinear tf), LogisticRegression(class_weight='balanced').
  Also writes top_features.csv: the 30 features pushing most towards "AI" and the 30 most towards "human".
- length:   LogisticRegression(class_weight='balanced') on log(characters) and log(words) only. If this
  scores high, the data has a length shortcut and every other detector's number must be read with that in mind.

Settings: every split file A makes (indist, logo_*, xdomain_*, paraphrase_orig) plus D's paraphrase /
paraphrase_human once their split files exist. paraphrase* have no train part: they are scored with the
model trained on indist train (TASKS.md §0.2), so that model is trained once and reused.

Memory: char n-grams over whole texts need several GB of RAM, so run the full data on Kaggle/Colab. On a small
laptop, --max-chars 2000 (longer than the median text, ~1,500 chars) keeps only the start of each text for the
vectorisers. The default 0 uses whole texts. The value is recorded in run_info.json.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data import DEV_SPLITS_DIR, ROOT, SEED, SETTINGS_TOP, SPLITS_DIR, load_split, load_texts

# --method choices. 'length' is not a real detector: it measures how far text length alone gets you.
METHODS = ["tfidf_lr", "length"]
PARAPHRASE_SETTINGS = ["paraphrase_orig", "paraphrase", "paraphrase_human"]  # no train part: use the indist model
# setting -> setting whose train part trains its model. Settings not listed train on their own train part.
TRAIN_SOURCE = {s: "indist" for s in PARAPHRASE_SETTINGS}
N_TOP = 30

# TF-IDF hyperparameters (recorded in run_info.json)
# TF-IDF in plain words: each text becomes a long vector with one number per word (or character n-gram).
# The number is high when the n-gram is frequent in THIS text (tf) but rare across all texts (idf), so common
# words like "the" get little weight. sublinear_tf uses 1 + log(count) instead of the raw count, so a word
# repeated 20 times does not count 20x. min_df drops n-grams seen in too few texts (mostly typos and noise).
# Word 1-2-grams catch vocabulary ("additionally", "in conclusion"); char 3-5-grams catch spelling,
# punctuation and word pieces, which also work on words never seen in training.
WORD = dict(analyzer="word", ngram_range=(1, 2), min_df=2, max_features=200_000, sublinear_tf=True,
            lowercase=True, dtype=np.float32)
CHAR = dict(analyzer="char", ngram_range=(3, 5), min_df=3, max_features=300_000, sublinear_tf=True,
            lowercase=True, dtype=np.float32)


# ---------------------------------------------------------------- models

class TfidfLR:
    """word + char TF-IDF, concatenated, then balanced logistic regression."""

    def __init__(self, C: float, max_chars: int):
        """Store C and max_chars and create the two (still empty) vectorisers.

        C is the inverse regularisation strength of the logistic regression: small C = strong penalty on large
        weights (simpler model), large C = weaker penalty. fit() may replace it with the best C found on val.
        """
        self.C, self.max_chars = C, max_chars
        self.word, self.char = TfidfVectorizer(**WORD), TfidfVectorizer(**CHAR)

    def _cut(self, texts: pd.Series) -> pd.Series:
        """Keep only the first max_chars characters of each text (all of it when max_chars is 0).

        Why: char n-grams over very long texts need a lot of RAM; cutting lets the baseline run on a small laptop.
        The same cut is applied at fit and at score time, so train and test see texts the same way.
        """
        return texts.str.slice(0, self.max_chars) if self.max_chars else texts

    def _lr(self, C: float) -> LogisticRegression:
        """A fresh logistic regression with the given C.

        class_weight='balanced' weights each class by 1 / its frequency. There are about twice as many machine as
        human texts, so without it the model could lean towards always saying "machine". liblinear is a solver
        that works well on large sparse matrices like ours, and random_state=SEED keeps runs reproducible.
        """
        return LogisticRegression(C=C, class_weight="balanced", solver="liblinear", max_iter=1000, random_state=SEED)

    def fit(self, texts: pd.Series, y: np.ndarray, val_texts: pd.Series | None = None,
            val_y: np.ndarray | None = None, Cs: list[float] | None = None) -> "TfidfLR":
        """Vectorise once. With val data and several Cs, keep the C with the best val AUROC (never test)."""
        t = self._cut(texts)
        # Vectorisers are fitted on TRAIN texts only; the word and char columns are placed side by side.
        X = sp.hstack([self.word.fit_transform(t), self.char.fit_transform(t)], format="csr")
        self.val_auroc = {}
        # Pick C on VAL by AUROC (needs both classes in val). Test is never looked at, so the test score stays honest.
        # The vectorising above is done once and reused for every C, which is the expensive part.
        if Cs and len(Cs) > 1 and val_texts is not None and len(set(val_y)) == 2:
            Xv = self._X(val_texts)
            best = None
            for C in Cs:
                lr = self._lr(C).fit(X, y)
                auc = float(roc_auc_score(val_y, lr.predict_proba(Xv)[:, 1]))
                self.val_auroc[str(C)] = round(auc, 5)
                print(f"    C={C}: val AUROC {auc:.4f}", flush=True)
                if best is None or auc > best[0]:
                    best = (auc, C, lr)
            _, self.C, self.lr = best
        else:
            self.lr = self._lr(self.C).fit(X, y)
        return self

    def _X(self, texts: pd.Series):
        """Turn texts into the feature matrix with the vectorisers already fitted on train.

        transform() (not fit_transform) is used on purpose: the vocabulary and IDF weights come from train only,
        so nothing about val or test texts leaks into the features.
        """
        t = self._cut(texts)
        return sp.hstack([self.word.transform(t), self.char.transform(t)], format="csr")

    def score(self, texts: pd.Series) -> np.ndarray:
        """P(machine) for each text, between 0 and 1. Higher = more likely AI, as preds.csv requires."""
        return self.lr.predict_proba(self._X(texts))[:, 1]          # P(machine): higher = more AI

    def params(self) -> dict:
        """Hyperparameters to record in run_info.json, so every number in the report can be traced to its settings."""
        return {"C": self.C, "val_auroc_by_C": self.val_auroc, "max_chars": self.max_chars, "solver": "liblinear", "class_weight": "balanced",
                "word": _jsonable(WORD), "char": _jsonable(CHAR),
                "n_features": len(self.word.vocabulary_) + len(self.char.vocabulary_)}

    def top_features(self, n: int = N_TOP) -> pd.DataFrame:
        """The n features with the largest positive weights (push towards AI) and the n most negative (towards human).

        Why: a linear model can be read directly, which lets us see WHAT it relies on. This is how we found that
        the strongest "human" features are page leftovers such as "references" and "_url_0_" (see the report's
        Limitations). Example row: direction=ai, rank=1, feature="word:additionally", coef > 0.
        """
        names = np.concatenate([["word:" + w for w in self.word.get_feature_names_out()],
                                ["char:" + c for c in self.char.get_feature_names_out()]])
        coef = self.lr.coef_[0]
        # In logistic regression, P(machine) grows with the weighted sum of features: a positive weight pushes a
        # text towards "machine", a negative one towards "human". Sorting the weights gives both ends.
        order = np.argsort(coef)
        ai, human = order[::-1][:n], order[:n]
        return pd.DataFrame({
            "direction": ["ai"] * n + ["human"] * n,
            "rank": list(range(1, n + 1)) * 2,
            "feature": np.concatenate([names[ai], names[human]]),
            "coef": np.concatenate([coef[ai], coef[human]]),
        })


class LengthLR:
    """Length-only shortcut check: log(1 + chars), log(1 + words) -> balanced logistic regression."""

    def __init__(self):
        """Scale the two features to mean 0 / std 1, then a balanced logistic regression.

        Scaling makes the two features comparable, so neither dominates just because its numbers are bigger.
        """
        self.model = make_pipeline(StandardScaler(),
                                   LogisticRegression(class_weight="balanced", random_state=SEED))

    @staticmethod
    def _feats(texts: pd.Series) -> np.ndarray:
        """Two features per text: log(1 + number of characters) and log(1 + number of words).

        The log stops a few very long texts from dominating: 1,000 vs 2,000 chars matters as much as 10,000 vs 20,000.
        """
        return np.log1p(np.column_stack([texts.str.len(), texts.str.split().str.len()]).astype(float))

    def fit(self, texts: pd.Series, y: np.ndarray, *_, **__) -> "LengthLR":
        """Train on (texts, labels). Extra arguments (val data, Cs) are accepted and ignored so that run_setting()
        can call both models the same way; there is no hyperparameter to tune here.
        """
        self.model.fit(self._feats(texts), y)
        return self

    def score(self, texts: pd.Series) -> np.ndarray:
        """P(machine) from length alone. Higher = more likely AI."""
        return self.model.predict_proba(self._feats(texts))[:, 1]

    def params(self) -> dict:
        """Features and learned weights, for run_info.json. A positive coef means longer texts look more machine-like."""
        lr = self.model[-1]
        return {"features": ["log1p(chars)", "log1p(words)"], "class_weight": "balanced",
                "coef": lr.coef_[0].round(4).tolist(), "intercept": float(lr.intercept_[0])}

    def top_features(self, n: int = N_TOP):
        """None: with only two features there is nothing to rank, so no top_features.csv is written."""
        return None


def make_model(method: str, C: float, max_chars: int):
    """Return an untrained model for --method ('tfidf_lr' or 'length'); both have fit/score/params/top_features."""
    return TfidfLR(C, max_chars) if method == "tfidf_lr" else LengthLR()


# ---------------------------------------------------------------- helpers

def _jsonable(d: dict) -> dict:
    """Copy of a vectoriser settings dict that json.dumps can write: types become their names, tuples become lists.

    Example: {'dtype': np.float32, 'ngram_range': (1, 2)} -> {'dtype': 'float32', 'ngram_range': [1, 2]}.
    """
    return {k: (v.__name__ if isinstance(v, type) else list(v) if isinstance(v, tuple) else v) for k, v in d.items()}


def _git_commit() -> str | None:
    """The current git commit hash, or None outside a git checkout.

    Why: run_info.json must record which version of the code produced a result (AGENTS.md hard rules).
    """
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _hardware() -> str:
    """A short description of the CPU and OS, recorded in run_info.json next to the runtime."""
    return f"CPU: {platform.processor() or platform.machine()} x{os.cpu_count()} ({platform.platform()})"


def available_settings(dev: bool) -> list[str]:
    """A's settings, plus D's paraphrase splits if they have been delivered."""
    d = DEV_SPLITS_DIR if dev else SPLITS_DIR
    return SETTINGS_TOP + [s for s in PARAPHRASE_SETTINGS if s not in SETTINGS_TOP and (d / f"{s}.json").exists()]


# ---------------------------------------------------------------- run

def run_setting(setting: str, method: str, texts: pd.DataFrame, out_root: Path, dev: bool,
                C: list[float], max_chars: int, cache: dict) -> Path:
    """Train on the setting's train part (or reuse the indist model), score val + test, write the files."""
    t0 = time.time()
    split = load_split(setting, dev=dev)
    # paraphrase* settings have no train part: they reuse the indist model, so the same detector is compared
    # before and after paraphrasing. The cache keeps that model so it is trained only once.
    train_setting = TRAIN_SOURCE.get(setting, setting)
    if train_setting in cache:
        model, n_train, train_secs = cache[train_setting]
        print(f"[{method}/{setting}] reusing the {train_setting} model", flush=True)
    else:
        train_ids = load_split(train_setting, dev=dev)["train"]
        if not train_ids:
            raise ValueError(f"{train_setting}: empty train split")
        tr = texts.loc[train_ids]
        print(f"[{method}/{setting}] training on {train_setting} train: {len(tr)} texts", flush=True)
        t_fit = time.time()
        val_ids = load_split(train_setting, dev=dev)["val"]
        va = texts.loc[val_ids]
        model = make_model(method, C[0], max_chars).fit(tr["text"], tr["label"].to_numpy(),
                                                        va["text"], va["label"].to_numpy(), Cs=C)
        n_train, train_secs = len(tr), round(time.time() - t_fit, 1)
        cache[train_setting] = (model, n_train, train_secs)

    # Score val AND test: src.eval picks the decision threshold on val, then reports metrics on test.
    ids = split["val"] + split["test"]
    missing = [i for i in ids if i not in texts.index]
    if missing:
        raise KeyError(f"{setting}: {len(missing)} ids have no text, e.g. {missing[:3]} "
                       "(paraphrased ids need data/processed/paraphrased.parquet)")
    preds = pd.DataFrame({
        "id": ids,
        "split": ["val"] * len(split["val"]) + ["test"] * len(split["test"]),
        "score": model.score(texts.loc[ids, "text"]),
    })

    out = out_root / method / setting
    out.mkdir(parents=True, exist_ok=True)
    preds.to_csv(out / "preds.csv", index=False, float_format="%.6f")
    top = model.top_features()
    if top is not None:
        top.to_csv(out / "top_features.csv", index=False, float_format="%.4f")
    run_info = {
        "method": method, "setting": setting, "trained_on": f"{train_setting}[train]", "dev": dev,
        "git_commit": _git_commit(), "seed": SEED, "hyperparameters": model.params(),
        "n_train": n_train, "n_val": len(split["val"]), "n_test": len(split["test"]),
        "train_seconds": train_secs, "runtime_seconds": round(time.time() - t0, 1),
        "hardware": _hardware(),
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__, "pandas": pd.__version__},
    }
    (out / "run_info.json").write_text(json.dumps(run_info, indent=2) + "\n")
    print(f"[{method}/{setting}] wrote {out}/preds.csv ({len(preds)} rows)", flush=True)
    return out / "preds.csv"


def main(argv=None) -> int:
    """Command-line entry point: parse arguments, load texts, train and score every requested setting.

    Returns 0 on success (used as the process exit code). See the module docstring for example commands.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", choices=METHODS, default="tfidf_lr")
    ap.add_argument("--setting", nargs="+", help="setting names (default: every available setting)")
    ap.add_argument("--dev", action="store_true", help="data/sample_dev.parquet + splits/dev, output to results_dev/")
    ap.add_argument("--C", type=float, nargs="+", default=[0.25, 1.0, 4.0, 16.0],
                    help="tfidf_lr: inverse regularisation strength(s); with several, the best on val AUROC is kept")
    ap.add_argument("--max-chars", type=int, default=0, help="tfidf_lr: use only the first N chars (0 = all)")
    ap.add_argument("--out-dir", type=Path, help="default: results/ (results_dev/ with --dev)")
    a = ap.parse_args(argv)

    out_root = a.out_dir or (ROOT / ("results_dev" if a.dev else "results"))
    # Guard: dev-sample numbers are meaningless and must never overwrite or mix with real results.
    if a.dev and out_root.resolve().is_relative_to((ROOT / "results").resolve()):
        raise ValueError("--dev output must not go inside results/ (it would mix with real results)")
    settings = a.setting or available_settings(a.dev)
    texts = load_texts(dev=a.dev).set_index("id")

    # Train the indist model first when paraphrase* settings will reuse it.
    settings = sorted(settings, key=lambda s: (s != "indist", s in PARAPHRASE_SETTINGS))
    cache: dict = {}
    for s in settings:
        run_setting(s, a.method, texts, out_root, a.dev, a.C, a.max_chars, cache)
        if s not in TRAIN_SOURCE.values():
            cache.pop(s, None)                 # only the indist model is reused; free the rest
    print(f"done. Next: python -m src.eval {out_root.relative_to(ROOT) if out_root.is_relative_to(ROOT) else out_root}"
          f"/{a.method}/<setting>/preds.csv{' --dev' if a.dev else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
