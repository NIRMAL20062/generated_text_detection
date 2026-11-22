"""error_analysis picks the right rows (issue #31)."""
import json

import pandas as pd
import pytest

from src import error_analysis as ea
from src.data import load_split, load_texts


def _preds():
    # threshold 0.5. Humans h1..h4, machines m1..m4; two val rows that must be ignored.
    return pd.DataFrame({
        "id":    ["h1", "h2", "h3", "h4", "m1", "m2", "m3", "m4", "hv", "mv"],
        "split": ["test"] * 8 + ["val", "val"],
        "score": [0.95, 0.60, 0.50, 0.10, 0.05, 0.40, 0.49, 0.90, 0.99, 0.01],
        "label": [0, 0, 0, 0, 1, 1, 1, 1, 0, 1],
    })


def test_picks_highest_scored_humans_and_lowest_scored_machines():
    fp, fn = ea.find_errors(_preds(), threshold=0.5, n=2)
    assert fp["id"].tolist() == ["h1", "h2"]         # 0.95, 0.60; h3 (=0.5, also wrong) is third; val 'hv' ignored
    assert fn["id"].tolist() == ["m1", "m2"]         # 0.05, 0.40; 'mv' is val, m4 is correct
    fp, fn = ea.find_errors(_preds(), threshold=0.5, n=10)
    assert fp["id"].tolist() == ["h1", "h2", "h3"]   # score == threshold counts as "machine" (src.eval rule)
    assert fn["id"].tolist() == ["m1", "m2", "m3"]


def test_report_counts_and_markdown_escaping():
    p = _preds().assign(domain="reddit", generator="bloomz", text="a | b " * 100)
    md = ea.report("roberta", "indist", p, 0.5, 1)
    assert "3 of 4 human texts called machine" in md and "3 of 4 machine texts called human" in md
    assert "| `h1` | reddit | – | 0.95 | 600 |" in md          # humans show no generator
    assert "| `m1` | reddit | bloomz | 0.05 | 600 |" in md
    assert "a \\| b" in md                                    # a '|' in the text must not break the table


def test_missing_threshold_fails_loudly(tmp_path):
    d = tmp_path / "fastdetectgpt" / "logo_bloomz"
    d.mkdir(parents=True)
    (d / "preds.csv").write_text("id,split,score\n")
    (d / "metrics.json").write_text(json.dumps({"threshold": None}))
    with pytest.raises(SystemExit, match="no threshold"):
        ea.main(["--method", "fastdetectgpt", "--setting", "logo_bloomz", "--dev", "--results", str(tmp_path)])


def test_end_to_end_on_dev_sample(tmp_path):
    texts = load_texts(dev=True).set_index("id")
    split = load_split("indist", dev=True)
    rows = [(i, part, float(texts.at[i, "label"]) * 0.5 + 0.25) for part in ("val", "test") for i in split[part]]
    d = tmp_path / "tfidf_lr" / "indist"
    d.mkdir(parents=True)
    pd.DataFrame(rows, columns=["id", "split", "score"]).to_csv(d / "preds.csv", index=False)
    (d / "metrics.json").write_text(json.dumps({"threshold": 0.5}))   # humans 0.25, machines 0.75 -> no errors
    assert ea.main(["--method", "tfidf_lr", "--setting", "indist", "--dev", "--results", str(tmp_path)]) == 0
    md = (d / "errors.md").read_text(encoding="utf-8")
    assert "Test errors: 0 of" in md and md.count("_none_") == 2
