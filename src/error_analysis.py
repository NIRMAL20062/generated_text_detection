"""Error analysis: the most confident mistakes of one detector in one setting (issues #27, #31).

    python -m src.error_analysis --method roberta --setting logo_bloomz          # -> results/roberta/logo_bloomz/errors.md
    python -m src.error_analysis --method tfidf_lr --setting indist --n 10
    python -m src.error_analysis --method tfidf_lr --setting indist --dev        # reads results_dev/

Input:  results/<method>/<setting>/preds.csv (id,split,score) and metrics.json (its `threshold`), texts and labels
        from src.data.load_texts().
Output: a Markdown section printed to stdout and written to results/<method>/<setting>/errors.md with
        - the n most confident FALSE POSITIVES: human texts with the highest scores among those at or above the
          threshold (the detector was surest they were AI, and they were not), and
        - the n most confident FALSE NEGATIVES: machine texts with the lowest scores among those below the
          threshold (the detector was surest they were human).
        For each: id, domain, generator, score, length in characters and the first 200 characters.

Why a script instead of hand-picked examples: anyone can regenerate exactly the same examples, and "most confident"
is a fixed rule, so we cannot cherry-pick the mistakes that tell a nice story.

Only TEST rows are used, with the threshold the shared evaluator chose on VAL (predict "machine" if score >= t, the
same rule as src.eval.best_threshold), so the errors are exactly the ones behind the reported accuracy and F1.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from src.data import ROOT, load_texts
from src.eval import load_preds

SNIPPET_CHARS = 200


def find_errors(preds: pd.DataFrame, threshold: float, n: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(false positives, false negatives), the n most confident of each, from labelled test predictions.

    `preds` needs columns id, split, score, label (0 human, 1 machine). Most confident = furthest past the
    threshold on the wrong side: highest-scored humans and lowest-scored machines. Ties are broken by id so the
    output is the same on every run.
    """
    test = preds[preds["split"] == "test"]
    fp = test[(test["label"] == 0) & (test["score"] >= threshold)]
    fn = test[(test["label"] == 1) & (test["score"] < threshold)]
    fp = fp.sort_values(["score", "id"], ascending=[False, True]).head(n)
    fn = fn.sort_values(["score", "id"], ascending=[True, True]).head(n)
    return fp, fn


def _snippet(text: str) -> str:
    """First 200 characters, made safe for a Markdown table cell."""
    s = " ".join(str(text).split())[:SNIPPET_CHARS]
    return s.replace("\\", "\\\\").replace("|", "\\|").replace("`", "'")


def _table(rows: pd.DataFrame) -> str:
    """Markdown table of error rows. A human text has no generator of its own (after dedup its `generator` is just
    the file it was kept from, see src.eval.breakdown), so for humans the column shows '–'."""
    if rows.empty:
        return "_none_"
    lines = ["| id | domain | generator | score | chars | first 200 characters |", "|---|---|---|---|---|---|"]
    for r in rows.itertuples(index=False):
        gen = r.generator if r.label == 1 else "–"
        lines.append(f"| `{r.id}` | {r.domain} | {gen} | {r.score:.4g} | {len(str(r.text))} | {_snippet(r.text)} |")
    return "\n".join(lines)


def report(method: str, setting: str, preds: pd.DataFrame, threshold: float, n: int) -> str:
    """The Markdown section for one method/setting: counts, then the two error tables."""
    fp, fn = find_errors(preds, threshold, n)
    test = preds[preds["split"] == "test"]
    n_fp = int(((test["label"] == 0) & (test["score"] >= threshold)).sum())
    n_fn = int(((test["label"] == 1) & (test["score"] < threshold)).sum())
    n_h, n_m = int((test["label"] == 0).sum()), int((test["label"] == 1).sum())
    return "\n\n".join([
        f"## {method} / {setting}",
        f"Threshold {threshold:.4g} (chosen on val by `src.eval`; predict machine if score >= threshold). "
        f"Test errors: {n_fp} of {n_h} human texts called machine (false positives), "
        f"{n_fn} of {n_m} machine texts called human (false negatives).",
        f"### {len(fp)} most confident false positives (human text, highest scores)\n\n" + _table(fp),
        f"### {len(fn)} most confident false negatives (machine text, lowest scores)\n\n" + _table(fn),
    ]) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", required=True, help="results/<method>/, e.g. roberta, tfidf_lr, logrank")
    ap.add_argument("--setting", required=True, help="results/<method>/<setting>/, e.g. indist, logo_bloomz")
    ap.add_argument("--n", type=int, default=5, help="examples per error type (default 5)")
    ap.add_argument("--dev", action="store_true", help="dev sample texts, results_dev/ by default")
    ap.add_argument("--results", type=Path, help="default: results/ (results_dev/ with --dev)")
    ap.add_argument("--out", type=Path, help="default: <results>/<method>/<setting>/errors.md")
    a = ap.parse_args(argv)

    run_dir = (a.results or ROOT / ("results_dev" if a.dev else "results")) / a.method / a.setting
    preds_path, metrics_path = run_dir / "preds.csv", run_dir / "metrics.json"
    for p in (preds_path, metrics_path):
        if not p.exists():
            raise SystemExit(f"missing {p}: run the detector and `python -m src.eval {preds_path}` first")
    threshold = json.loads(metrics_path.read_text())["threshold"]
    if threshold is None:   # src.eval leaves it null when val has < 50 texts per class (e.g. Fast-DetectGPT LOGO)
        raise SystemExit(f"{metrics_path} has no threshold (val too small), so there are no errors to analyse")

    texts = load_texts(dev=a.dev)
    preds = load_preds(preds_path, texts)       # validates preds.csv and attaches label/domain/generator
    preds = preds.merge(texts[["id", "text"]], on="id", how="left", validate="one_to_one")
    text = report(a.method, a.setting, preds, threshold, a.n)

    out = a.out or run_dir / "errors.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print(text)
    print(f"wrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
