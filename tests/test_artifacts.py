"""artifacts.py counts patterns correctly (issue #32)."""
import pandas as pd

from src import artifacts as ar


def _fake():
    return pd.DataFrame({
        "id": ["h1", "h2", "m1", "m2", "h3_para"],
        "text": ["Text. , more. References", "See _url_0_ here", "Clean text.", "Uses references too",
                 "References"],
        "label": [0, 0, 1, 1, 0],
        "domain": ["wikipedia", "reddit", "wikipedia", "reddit", "wikipedia"],
    })


def test_counts_on_four_rows():
    s = ar.pattern_shares(_fake().iloc[:4]).set_index(["pattern", "domain", "label"])
    assert s.loc[("references", "all", 0), "n_with"] == 1 and s.loc[("references", "all", 0), "share"] == 0.5
    assert s.loc[("references", "all", 1), "n_with"] == 1                    # case-insensitive: "references"
    assert s.loc[("references", "wikipedia", 0), "share"] == 1.0
    assert s.loc[("references", "reddit", 1), "share"] == 1.0
    assert s.loc[("_URL_N_", "reddit", 0), "share"] == 1.0
    assert s.loc[("_URL_N_", "all", 1), "n_with"] == 0
    assert s.loc[("'. ,' (stray comma)", "all", 0), "n_with"] == 1
    assert s.loc[("external links", "all", 0), "n"] == 2


def test_markdown_tables():
    md = ar.to_markdown(ar.pattern_shares(_fake().iloc[:4]), "fake")
    assert "| _URL_N_ | 50.0% | 0.0% | ∞ |" in md
    assert "| references | 50.0% | 50.0% | 1.0× |" in md
    assert "| references | 100.0% / 0.0% | 0.0% / 100.0% |" in md           # wikipedia, reddit


def test_end_to_end_dev(tmp_path):
    out = tmp_path / "artifacts.md"
    assert ar.main(["--dev", "--out", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert "## All domains" in text and all(p in text for p in ar.PATTERNS)
