"""Tests for src/baseline_tfidf.py on the committed dev sample (no data/processed needed).

    python -m pytest tests/ -q
"""
import json

import pandas as pd
import pytest

from src import baseline_tfidf as bl
from src import eval as ev
from src.data import load_split


@pytest.mark.parametrize("method", bl.METHODS)
def test_dev_run_writes_files_the_evaluator_accepts(tmp_path, method):
    assert bl.main(["--dev", "--method", method, "--setting", "indist", "logo_dolly", "paraphrase_orig",
                    "--out-dir", str(tmp_path)]) == 0
    for setting in ("indist", "logo_dolly", "paraphrase_orig"):
        out = tmp_path / method / setting
        preds = pd.read_csv(out / "preds.csv")
        split = load_split(setting, dev=True)
        # exactly the split's val + test ids, nothing from train
        assert sorted(preds["id"]) == sorted(split["val"] + split["test"])
        assert preds["score"].between(0, 1).all()
        info = json.loads((out / "run_info.json").read_text())
        assert info["seed"] == 42 and info["method"] == method
        # the shared evaluator must accept the file (checks leakage, coverage, labels)
        assert ev.main([str(out / "preds.csv"), "--dev"]) == 0
        assert json.loads((out / "metrics.json").read_text())["auroc"] is not None


def test_paraphrase_settings_reuse_the_indist_model(tmp_path):
    bl.main(["--dev", "--setting", "paraphrase_orig", "indist", "--out-dir", str(tmp_path)])
    info = json.loads((tmp_path / "tfidf_lr" / "paraphrase_orig" / "run_info.json").read_text())
    assert info["trained_on"] == "indist[train]"
    # same model -> identical scores for the shared val ids
    a = pd.read_csv(tmp_path / "tfidf_lr" / "indist" / "preds.csv").set_index("id")
    b = pd.read_csv(tmp_path / "tfidf_lr" / "paraphrase_orig" / "preds.csv").set_index("id")
    shared = a.index.intersection(b.index)
    assert len(shared) > 0 and (a.loc[shared, "score"] == b.loc[shared, "score"]).all()


def test_top_features_has_30_each_way(tmp_path):
    bl.main(["--dev", "--setting", "indist", "--out-dir", str(tmp_path)])
    top = pd.read_csv(tmp_path / "tfidf_lr" / "indist" / "top_features.csv")
    assert (top["direction"].value_counts() == 30).all()
    assert (top.loc[top.direction == "ai", "coef"] > 0).all()


def test_dev_output_refuses_results_dir():
    with pytest.raises(ValueError, match="must not go inside results/"):
        bl.main(["--dev", "--out-dir", str(bl.ROOT / "results")])


def test_C_is_picked_on_val_and_recorded(tmp_path):
    bl.main(["--dev", "--setting", "indist", "--C", "0.1", "10", "--out-dir", str(tmp_path)])
    hp = json.loads((tmp_path / "tfidf_lr" / "indist" / "run_info.json").read_text())["hyperparameters"]
    assert set(hp["val_auroc_by_C"]) == {"0.1", "10.0"}
    assert hp["C"] == max((v, float(k)) for k, v in hp["val_auroc_by_C"].items())[1]
