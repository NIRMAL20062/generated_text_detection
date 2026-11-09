"""Tests for src/eval.py. Run from the repo root:

    python -m pytest tests/ -q

How these tests are built (use the same recipe for your own files):
  1. Known answers: inputs where we already know the right output (a perfect detector must get
     AUROC 1.0, a reversed one 0.0, a hand-worked example must give the hand-worked threshold).
  2. Bad inputs: broken files must raise an error, never produce a quiet wrong number.
  3. End to end: run the real command on the committed dev sample, like a teammate would.
None of these tests need data/processed/, so they run on a fresh clone in a few seconds.
"""
import json

import numpy as np
import pandas as pd
import pytest

from src import eval as ev
from src.data import load_split, load_texts


# ---------------------------------------------------------------- 1. known answers

def test_auroc_known_values():
    y = np.array([0, 0, 1, 1])
    assert ev.auroc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0      # machines always score higher
    assert ev.auroc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0      # perfectly reversed
    assert ev.auroc(y, np.array([0.5, 0.5, 0.5, 0.5])) == 0.5      # all tied = no information
    assert ev.auroc(np.array([1, 1]), np.array([0.3, 0.7])) is None  # one class only: undefined


def test_random_scores_give_auroc_near_half():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 5000)
    assert abs(ev.auroc(y, rng.random(5000)) - 0.5) < 0.03


def test_bootstrap_ci_contains_estimate_and_is_reproducible():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 500)
    s = y + rng.normal(0, 1, 500)                 # a noisy but useful detector
    lo, hi = ev.bootstrap_auroc_ci(y, s, n_boot=200)
    assert lo < ev.auroc(y, s) < hi
    assert (lo, hi) == ev.bootstrap_auroc_ci(y, s, n_boot=200)   # fixed seed -> same interval


def test_best_threshold_hand_worked_example():
    # Worked out by hand in the best_threshold docstring: the best cut is t = 0.8 (macro-F1 0.80).
    y = np.array([1, 1, 0, 1, 0])
    s = np.array([0.9, 0.8, 0.6, 0.4, 0.1])
    assert ev.best_threshold(y, s) == 0.8


def test_best_threshold_never_splits_tied_scores():
    y = np.array([1, 0, 1, 0])
    s = np.array([0.7, 0.7, 0.7, 0.2])
    t = ev.best_threshold(y, s)
    pred = s >= t
    assert pred[0] == pred[1] == pred[2]          # same score -> same prediction


def test_tpr_at_1pct_fpr():
    # 100 humans scored 0.00 .. 0.99; 5 machines far above them, 5 far below.
    # At <= 1% FPR we may flag at most 1 human, so we catch exactly the 5 high machines: TPR = 0.5.
    y = np.array([0] * 100 + [1] * 10)
    s = np.concatenate([np.arange(100) / 100, [2.0] * 5, [-1.0] * 5])
    assert ev.tpr_at_fpr(y, s) == 0.5


def _toy_texts():
    """A tiny fake dataset. Group g1 = one human text paired with a chatGPT and a dolly machine text."""
    return pd.DataFrame({
        "id":        ["h1", "m1", "m2", "h2", "m3"],
        "label":     [0, 1, 1, 0, 1],
        "domain":    ["reddit"] * 5,
        "generator": ["dolly", "chatGPT", "dolly", "dolly", "dolly"],   # h1 kept from the dolly file
        "group":     ["g1", "g1", "g1", "g2", "g2"],
    })


def test_breakdown_by_generator_uses_groups_not_the_human_generator_field():
    texts = _toy_texts()
    test = texts.assign(score=[0.1, 0.9, 0.8, 0.2, 0.7], split="test")
    out = ev.breakdown(test, "generator", texts)
    # h1's generator field says "dolly", but it still counts for chatGPT because group g1 holds m1.
    assert out["chatGPT"] == {"auroc": 1.0, "n_machine": 1, "n_human": 1}
    assert out["dolly"]["n_human"] == 2 and out["dolly"]["n_machine"] == 2


# ---------------------------------------------------------------- 2. bad inputs must fail loudly

@pytest.mark.parametrize("rows, message", [
    ("id,split,score\nh1,test,0.1\nh1,test,0.2\n", "duplicate ids"),
    ("id,split,score\nh1,train,0.1\n", "split must be"),
    ("id,split,score\nh1,test,nan\n", "NaN"),
    ("id,split,score\nnot_a_real_id,test,0.1\n", "not in load_texts"),
    ("id,score\nh1,0.1\n", "missing columns"),
])
def test_load_preds_rejects_bad_files(tmp_path, rows, message):
    path = tmp_path / "preds.csv"
    path.write_text(rows)
    with pytest.raises(ValueError, match=message):
        ev.load_preds(path, _toy_texts())


# ---------------------------------------------------------------- 3. end to end on the dev sample

def _write_dev_preds(tmp_path, setting, score_fn):
    """Write results/<method>/<setting>/preds.csv for the dev split, like a real detector would."""
    texts = load_texts(dev=True).set_index("id")
    split = load_split(setting, dev=True)
    ids = split["val"] + split["test"]
    folder = tmp_path / "fake_method" / setting
    folder.mkdir(parents=True)
    pd.DataFrame({
        "id": ids,
        "split": ["val"] * len(split["val"]) + ["test"] * len(split["test"]),
        "score": score_fn(texts.loc[ids]),
    }).to_csv(folder / "preds.csv", index=False)
    return folder / "preds.csv"


def test_cli_on_dev_sample_perfect_detector(tmp_path):
    path = _write_dev_preds(tmp_path, "indist", lambda df: df["label"].to_numpy())
    assert ev.main([str(path), "--dev", "--by", "domain"]) == 0
    m = json.loads((path.parent / "metrics.json").read_text())
    assert m["method"] == "fake_method" and m["setting"] == "indist"
    assert m["auroc"] == 1.0
    assert m["tpr_at_1pct_fpr"] is None           # dev test set has far fewer than 1,000 humans
    assert set(m["by_domain"]) <= {"wikipedia", "wikihow", "reddit", "arxiv", "peerread"}


def test_cli_rejects_ids_from_the_wrong_split(tmp_path):
    # Predictions made for indist, but saved under logo_dolly: must be caught.
    path = _write_dev_preds(tmp_path, "indist", lambda df: df["label"].to_numpy())
    wrong = tmp_path / "fake_method" / "logo_dolly"
    wrong.mkdir()
    (wrong / "preds.csv").write_text(path.read_text())
    with pytest.raises(ValueError, match="not in logo_dolly.json"):
        ev.main([str(wrong / "preds.csv"), "--dev"])


# ---------------------------------------------------------------- 4. review fixes

def test_group_bootstrap_is_wider_when_rows_in_a_group_are_related():
    # 200 groups of 5 identical rows: really only 200 independent samples, not 1,000.
    rng = np.random.default_rng(2)
    y = np.repeat(rng.integers(0, 2, 200), 5)
    s = y + np.repeat(rng.normal(0, 1, 200), 5)
    groups = np.repeat(np.arange(200), 5)
    lo_row, hi_row = ev.bootstrap_auroc_ci(y, s, n_boot=300)
    lo_grp, hi_grp = ev.bootstrap_auroc_ci(y, s, groups=groups, n_boot=300)
    assert (hi_grp - lo_grp) > 1.5 * (hi_row - lo_row)


def test_misspelled_setting_folder_is_an_error(tmp_path):
    path = _write_dev_preds(tmp_path, "indist", lambda df: df["label"].to_numpy())
    typo = tmp_path / "fake_method" / "indsit"
    typo.mkdir()
    (typo / "preds.csv").write_text(path.read_text())
    with pytest.raises(ValueError, match="no split file"):
        ev.main([str(typo / "preds.csv"), "--dev"])


@pytest.mark.parametrize("method, should_fail", [("roberta", True), ("detectgpt", False)])
def test_partial_test_set_only_allowed_for_detectgpt(tmp_path, method, should_fail):
    path = _write_dev_preds(tmp_path, "logo_dolly", lambda df: df["label"].to_numpy())
    preds = pd.read_csv(path)
    test_rows = preds.index[preds["split"] == "test"]
    partial = tmp_path / method / "logo_dolly"
    partial.mkdir(parents=True)
    preds.drop(test_rows[:3]).to_csv(partial / "preds.csv", index=False)   # skip 3 test texts
    if should_fail:
        with pytest.raises(ValueError, match="must score the whole test set"):
            ev.main([str(partial / "preds.csv"), "--dev"])
    else:
        assert ev.main([str(partial / "preds.csv"), "--dev"]) == 0


def test_dev_run_refuses_to_write_into_results(tmp_path):
    path = ev.ROOT / "results" / "fake_method" / "indist" / "preds.csv"   # never created: check fires first
    with pytest.raises(ValueError, match="must not be inside results/"):
        ev.main([str(path), "--dev"])


def test_evaluate_threshold_metrics_when_val_is_big_enough():
    # The dev sample has < 50 val texts per class, so build a synthetic set that reaches the
    # accuracy / macro-F1 branch. Machine scores ~ N(1, 1), human scores ~ N(0, 1).
    rng = np.random.default_rng(3)
    n = 400
    label = rng.integers(0, 2, n)
    df = pd.DataFrame({
        "id": [f"t{i}" for i in range(n)], "label": label, "score": label + rng.normal(0, 1, n),
        "split": np.where(np.arange(n) < 200, "val", "test"),
        "domain": "reddit", "generator": "chatGPT", "group": [f"g{i}" for i in range(n)],
    })
    m = ev.evaluate(df, df)
    assert m["threshold"] is not None and 0.6 < m["accuracy"] < 0.85 and 0.6 < m["macro_f1"] < 0.85
    assert m["tpr_at_1pct_fpr"] is None           # far fewer than 1,000 test humans
