"""make_tables on fake metrics.json files (TASKS.md §D: build it against dummy metrics)."""
import json

from src import make_tables as mt


def _write(root, method, setting, auroc):
    d = root / method / setting
    d.mkdir(parents=True)
    (d / "metrics.json").write_text(json.dumps({"auroc": auroc, "auroc_ci95": [auroc - .02, auroc + .02],
                                                "accuracy": .8, "macro_f1": .79, "tpr_at_1pct_fpr": None, "n_test": 10}))


def test_tables_fill_present_cells_and_dash_missing(tmp_path):
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
