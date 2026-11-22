"""make_tables on fake metrics.json files (TASKS.md §D: build it against dummy metrics)."""
import json

from src import make_tables as mt


def _write(root, method, setting, auroc):
    """Create a fake results/<method>/<setting>/metrics.json, so the tables can be tested without running any detector."""
    d = root / method / setting
    d.mkdir(parents=True)
    (d / "metrics.json").write_text(json.dumps({"auroc": auroc, "auroc_ci95": [auroc - .02, auroc + .02],
                                                "accuracy": .8, "macro_f1": .79, "tpr_at_1pct_fpr": None, "n_test": 10}))


def test_tables_fill_present_cells_and_dash_missing(tmp_path):
    """If this fails, the Markdown tables would show wrong numbers, a wrong paraphrase drop, a LOGO mean over an
    incomplete row, or methods in the wrong order."""
    _write(tmp_path, "tfidf_lr", "indist", .95)
    _write(tmp_path, "tfidf_lr", "logo_chatGPT", .80)
    _write(tmp_path, "tfidf_lr", "paraphrase_orig", .90)
    _write(tmp_path, "tfidf_lr", "paraphrase", .60)
    _write(tmp_path, "ppl", "indist", .70)
    text = mt.build(tmp_path, tmp_path / "tables.md", None)
    assert "| tfidf_lr | 0.950 [0.930, 0.970] | 0.800 | 0.790 | – |" in text
    assert "| tfidf_lr | 0.900 | 0.600 | 0.300 | – |" in text      # paraphrase drop
    logo_row = [l for l in text.split("## Table 2")[1].splitlines() if l.startswith("| tfidf_lr")][0]
    assert "0.800" in logo_row and logo_row.rstrip().endswith("– |")  # mean is '–' until every generator is in
    assert text.index("| tfidf_lr") < text.index("| ppl")           # METHOD_ORDER


def test_latex_tables_match_report_layout(tmp_path):
    """results/tables.tex rows use the report's names, 3 decimals and '--' for missing (issue #33)."""
    _write(tmp_path, "tfidf_lr", "indist", .95)
    _write(tmp_path, "tfidf_lr", "paraphrase_orig", .90)
    _write(tmp_path, "tfidf_lr", "paraphrase", .60)
    _write(tmp_path, "fastdetectgpt", "logo_bloomz", .406)
    mt.build(tmp_path, tmp_path / "tables.md", None)
    tex = (tmp_path / "tables.tex").read_text(encoding="utf-8")
    assert r"  TF-IDF + LR                & 0.950 [0.930, 0.970] & 0.800 & 0.790 & -- \\" in tex
    assert r"  TF-IDF + LR                & 0.900 & 0.600 & 0.300 & -- \\" in tex          # Table 4 drop
    assert r"  Fast-DetectGPT$^\dagger$   & -- & -- & -- & -- & 0.406 & -- \\" in tex   # Table 2, BLOOMZ column
    assert tex.count(r"\begin{table}") == 4 and tex.count(r"\hline\hline") == 8
    for label in ("tab:indist", "tab:logo", "tab:xdomain", "tab:para"):
        assert rf"\label{{{label}}}" in tex


def test_fig3_paraphrase_bars(tmp_path):
    """Fig 3 is written and its data has one row per method, one column per paraphrase setting (issue #30)."""
    _write(tmp_path, "logrank", "paraphrase_orig", .86)
    _write(tmp_path, "logrank", "paraphrase", .70)
    _write(tmp_path, "roberta", "paraphrase_human", .97)
    df = mt.load_metrics(tmp_path)
    bars = mt.paraphrase_bars(df)
    assert list(bars.index) == ["logrank", "roberta"]
    assert bars.loc["logrank"].tolist()[:2] == [.86, .70]
    mt.build(tmp_path, tmp_path / "tables.md", tmp_path / "figures")
    assert (tmp_path / "figures" / "fig3_paraphrase.png").stat().st_size > 0
