"""Report tables and figures from every results/<method>/<setting>/metrics.json (TASKS.md §D, issue #14).

    python -m src.make_tables                 # results/ -> results/tables.md, results/tables.tex, results/figures/*.png
    python -m src.make_tables --results results_dev

Table 1: in-distribution AUROC [95% CI], accuracy, macro-F1, TPR@1%FPR per method
Table 2: LOGO AUROC, method x held-out generator, + mean
Table 3: cross-domain AUROC, method x held-out domain, + mean
Table 4: paraphrase attack: paraphrase_orig -> paraphrase AUROC and the drop (+ paraphrase_human control)
Fig 1: grouped bars (method x setting group); Fig 2: LOGO heatmap;
Fig 3: paraphrase attack, AUROC on paraphrase_orig / paraphrase / paraphrase_human per method (issue #30).
results/tables.tex holds Tables 1-4 as LaTeX in the layout of report/report.tex (issue #33), so the report's numbers
can be pasted (or \\input) straight from the evaluator's output and never disagree with metrics.json.
A missing method/setting shows as "–" ("--" in LaTeX), so the script runs before every result is in.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from src.data import DOMAINS, GENERATORS, ROOT

# Row order in every table and figure: the shortcut check first, then simple -> strong
# (TF-IDF baseline, the three zero-shot scores, fine-tuned RoBERTa last).
METHOD_ORDER = ["length", "tfidf_lr", "ppl", "logrank", "fastdetectgpt", "roberta"]


def load_metrics(results: Path) -> pd.DataFrame:
    """One row per results/<method>/<setting>/metrics.json, in METHOD_ORDER.

    Columns: method, setting, auroc, lo/hi (95% CI), accuracy, macro_f1, tpr1 (TPR at 1% FPR), n_test.
    Why read metrics.json and never preds.csv: every number in the report must come from the one shared evaluator
    (src/eval.py). The method and setting names come from the folder names, so no table is typed in by hand.
    A metric the evaluator left null (e.g. Fast-DetectGPT TPR, too few humans) stays None and prints as '–'.
    """
    rows = []
    for f in sorted(results.glob("*/*/metrics.json")):
        m = json.loads(f.read_text())
        lo, hi = m.get("auroc_ci95") or [None, None]   # CI can be null (AUROC undefined): keep both ends missing
        rows.append({"method": f.parent.parent.name, "setting": f.parent.name, "auroc": m.get("auroc"),
                     "lo": lo, "hi": hi, "accuracy": m.get("accuracy"), "macro_f1": m.get("macro_f1"),
                     "tpr1": m.get("tpr_at_1pct_fpr"), "n_test": m.get("n_test")})
    df = pd.DataFrame(rows, columns=["method", "setting", "auroc", "lo", "hi", "accuracy", "macro_f1", "tpr1", "n_test"])
    # sort by position in METHOD_ORDER; an unknown method (a new experiment) goes last instead of crashing
    order = {m: i for i, m in enumerate(METHOD_ORDER)}
    return df.assign(_o=df["method"].map(lambda m: order.get(m, len(order)))).sort_values(["_o", "method"]).drop(columns="_o")


def f(x, d=3) -> str:
    """Format a number with d decimals (0.95123 -> '0.951'); a missing result becomes '–'.

    Why '–' instead of an error: the script must run before every method/setting is finished, so the team could
    see partial tables early. A dash also makes "not run / not defined" visibly different from a real 0.
    """
    return "–" if x is None or pd.isna(x) else f"{x:.{d}f}"


def md(df: pd.DataFrame) -> str:
    """A DataFrame as a GitHub Markdown table (header, |---| separator, one line per row) for results/tables.md.
    Written by hand instead of df.to_markdown() so we need no extra package (tabulate)."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


def methods(df: pd.DataFrame) -> list[str]:
    """The methods present in df, once each, keeping METHOD_ORDER (dict.fromkeys drops duplicates but keeps order,
    which set() would not)."""
    return list(dict.fromkeys(df["method"]))


def grid(df: pd.DataFrame, settings: list[str], labels: list[str]) -> pd.DataFrame:
    """method x setting AUROC matrix (NaN where missing), plus a 'mean' column. Used for Tables 2 (LOGO) and 3
    (cross-domain): `settings` are the folder names (logo_chatGPT, ...), `labels` the column titles (chatGPT, ...).

    Why the mean is only shown when every setting is present: a mean over 4 of 5 held-out generators would quietly
    leave out a hard one (e.g. BLOOMZ) and look better than it is, and it could not be compared across methods.
    """
    piv = df.pivot_table(index="method", columns="setting", values="auroc", aggfunc="first")
    # reindex fixes the order and adds empty (NaN) cells for settings that have not been run yet
    piv = piv.reindex(index=methods(df), columns=settings)
    piv.columns = labels
    piv["mean"] = piv.mean(axis=1, skipna=False)   # mean only when every setting is present
    return piv


def table1(df):
    """Table 1, in distribution: AUROC with its 95% CI, accuracy, macro-F1 and TPR at 1% FPR, one row per method.

    AUROC (threshold-free ranking quality) is the main metric; the CI comes from the evaluator's group bootstrap.
    Accuracy and F1 use the threshold src.eval chose on val. TPR@1%FPR = share of machine texts caught when at
    most 1% of humans may be wrongly flagged, the number that matters if a detector is used to accuse people.
    """
    d = df[df.setting == "indist"]
    return pd.DataFrame({"method": d.method, "AUROC [95% CI]": [f"{f(a)} [{f(l)}, {f(h)}]" for a, l, h in zip(d.auroc, d.lo, d.hi)],
                         "accuracy": d.accuracy.map(f), "macro-F1": d.macro_f1.map(f), "TPR@1%FPR": d.tpr1.map(f)})


def table4(df):
    """Table 4, paraphrase attack: AUROC before (paraphrase_orig) and after (paraphrase) paraphrasing the same 500
    machine texts, the drop, and the human-paraphrase control (paraphrase_human).

    drop = orig - paraphrased, so a positive drop means the attack worked (e.g. log-rank 0.860 -> 0.699 = 0.160)
    and a negative one means the detector got slightly better. Because the humans and the machine texts' topics
    are identical before and after, the change is caused by the paraphrasing alone. Methods without either
    paraphrase result are left out of this table.
    """
    rows = []
    for m in methods(df):
        a = df[(df.method == m)].set_index("setting")["auroc"]
        o, p, h = (a.get(s) for s in ("paraphrase_orig", "paraphrase", "paraphrase_human"))
        if o is None and p is None:
            continue
        drop = o - p if o is not None and p is not None else None   # a drop needs both numbers
        rows.append({"method": m, "orig AUROC": f(o), "paraphrased AUROC": f(p), "drop": f(drop),
                     "human-paraphrase control": f(h)})
    return pd.DataFrame(rows, columns=["method", "orig AUROC", "paraphrased AUROC", "drop", "human-paraphrase control"])


def figures(df, logo: pd.DataFrame, out: Path) -> list[Path]:
    """Write the report figures into `out` and return their paths. A figure with no data at all is skipped.

    Fig 1: per method, grouped bars of AUROC in distribution, mean over the 5 LOGO settings, mean over the 5
           cross-domain settings, and after paraphrasing; shows at a glance where each detector loses accuracy.
    Fig 2: heatmap of the LOGO grid (Table 2): rows = detectors, columns = held-out generator, colour = AUROC.
    Fig 3: paraphrase attack (Table 4) as bars, see paraphrase_bars().
    The dashed line at AUROC 0.5 is chance level (a coin flip).
    """
    import matplotlib
    matplotlib.use("Agg")   # draw to files only, no window: works on Kaggle and servers without a screen
    import matplotlib.pyplot as plt
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    # Fig 1: each group is a filter on the setting name; groupby().mean() averages e.g. the 5 logo_* settings.
    # Unlike Table 2's mean, this one averages whatever is present, so the figure can be drawn early.
    groups = {"in-dist": df.setting.eq("indist"), "LOGO (mean)": df.setting.str.startswith("logo_"),
              "cross-domain (mean)": df.setting.str.startswith("xdomain_"), "paraphrased": df.setting.eq("paraphrase")}
    bars = pd.DataFrame({g: df[mask].groupby("method")["auroc"].mean() for g, mask in groups.items()}).reindex(methods(df))
    if not bars.dropna(how="all").empty:
        ax = bars.plot.bar(figsize=(9, 4.5), rot=0, width=0.8)
        ax.set_ylabel("AUROC"); ax.set_ylim(0, 1); ax.axhline(0.5, color="grey", lw=0.8, ls="--")
        ax.set_title("Detector AUROC by setting", pad=28)
        ax.legend(fontsize=8, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
        plt.tight_layout(); p = out / "fig1_auroc_by_setting.png"; plt.savefig(p, dpi=150); plt.close(); paths.append(p)
    # Fig 2: the heatmap shows only the 5 generators (the mean is in Table 2); height grows with the number of rows
    h = logo.drop(columns="mean")
    if not h.dropna(how="all").empty:
        fig, ax = plt.subplots(figsize=(7, 0.5 * len(h) + 1.5))
        # colour scale from 0.5 (chance) to 1 (perfect); anything below chance gets the darkest colour
        im = ax.imshow(h.to_numpy(dtype=float), vmin=0.5, vmax=1, cmap="viridis", aspect="auto")
        ax.set_xticks(range(h.shape[1]), h.columns); ax.set_yticks(range(len(h)), h.index)
        for i in range(h.shape[0]):
            for j in range(h.shape[1]):
                v = h.iat[i, j]
                # write the value in each cell: white text on dark cells (< 0.8), black on light ones
                ax.text(j, i, f(v, 2), ha="center", va="center", color="w" if pd.notna(v) and v < 0.8 else "k", fontsize=8)
        fig.colorbar(im, label="AUROC"); ax.set_title("LOGO: AUROC on the held-out generator")
        plt.tight_layout(); p = out / "fig2_logo_heatmap.png"; plt.savefig(p, dpi=150); plt.close(); paths.append(p)
    para = paraphrase_bars(df)
    if not para.dropna(how="all").empty:
        ax = para.plot.bar(figsize=(9, 4.5), rot=0, width=0.8, color=["#4c72b0", "#dd8452", "#8c8c8c"])
        ax.set_ylabel("AUROC"); ax.set_ylim(0, 1); ax.axhline(0.5, color="grey", lw=0.8, ls="--")
        ax.set_title("Paraphrase attack: AUROC before and after paraphrasing", pad=28)
        ax.legend(fontsize=8, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
        plt.tight_layout(); p = out / "fig3_paraphrase.png"; plt.savefig(p, dpi=150); plt.close(); paths.append(p)
    return paths


PARA_BARS = {"paraphrase_orig": "original texts", "paraphrase": "machine texts paraphrased",
             "paraphrase_human": "human texts paraphrased (control)"}


def paraphrase_bars(df: pd.DataFrame) -> pd.DataFrame:
    """method x {original, paraphrased machine, paraphrased human} AUROC for Fig 3 (NaN where missing).

    Why three bars: 'original -> paraphrased machine' is the attack (likelihood scores fall by ~0.16, supervised
    detectors barely move); the human control shows *why* the supervised ones don't move: paraphrasing human text
    also makes it look machine-written to them, so they react to the paraphraser's style, not to authorship.
    """
    piv = df.pivot_table(index="method", columns="setting", values="auroc", aggfunc="first")
    piv = piv.reindex(index=methods(df), columns=list(PARA_BARS))
    piv.columns = list(PARA_BARS.values())
    return piv


# ---------------------------------------------------------------- LaTeX (report/report.tex layout)

TEX_METHOD = {"length": "Length only", "tfidf_lr": "TF-IDF + LR", "ppl": "Log-likelihood", "logrank": "Log-rank",
              "fastdetectgpt": r"Fast-DetectGPT$^\dagger$", "roberta": "RoBERTa"}
TEX_GEN = {"chatGPT": "ChatGPT", "davinci": "davinci", "cohere": "Cohere", "dolly": "Dolly", "bloomz": "BLOOMZ"}
TEX_DOM = {"wikipedia": "Wikipedia", "wikihow": "wikiHow", "reddit": "Reddit", "arxiv": "arXiv", "peerread": "PeerRead"}


def tex_num(x) -> str:
    """3 decimals, '--' for a missing value (the LaTeX twin of f())."""
    return "--" if x is None or pd.isna(x) else f"{x:.3f}"


def tex_table(caption: str, label: str, header: list[str], rows: list[list[str]]) -> str:
    """One table in the report's style: double rules top and bottom, single rule under the header.

    Method names are padded to 26 characters exactly as in report/report.tex, so a plain `diff` of the two files
    shows only real number changes.
    """
    spec = "l" + "c" * (len(header) - 1)
    lines = [r"\begin{table}", rf"  \caption{{{caption}}}", rf"  \begin{{tabular}}{{{spec}}}", r"  \hline\hline",
             "  " + " & ".join(header) + r" \\", r"  \hline"]
    lines += [f"  {r[0]:<26} & " + " & ".join(r[1:]) + r" \\" for r in rows]
    lines += [r"  \hline\hline", r"  \end{tabular}", rf"  \label{{{label}}}", r"\end{table}"]
    return "\n".join(lines)


def latex(df: pd.DataFrame, logo: pd.DataFrame, xdom: pd.DataFrame) -> str:
    """Tables 1-4 as LaTeX, same captions, labels and column order as report/report.tex."""
    name = lambda m: TEX_METHOD.get(m, m.replace("_", r"\_"))
    d = df[df.setting == "indist"]
    t1 = [[name(m), f"{tex_num(a)} [{tex_num(lo)}, {tex_num(hi)}]", tex_num(acc), tex_num(f1), tex_num(t)]
          for m, a, lo, hi, acc, f1, t in zip(d.method, d.auroc, d.lo, d.hi, d.accuracy, d.macro_f1, d.tpr1)]
    grid_rows = lambda g: [[name(m)] + [tex_num(v) for v in r] for m, r in zip(g.index, g.to_numpy(dtype=float))]
    t4 = []
    for m in methods(df):
        a = df[df.method == m].set_index("setting")["auroc"]
        o, p, h = (a.get(s) for s in ("paraphrase_orig", "paraphrase", "paraphrase_human"))
        if o is None and p is None:
            continue
        t4.append([name(m), tex_num(o), tex_num(p), tex_num(o - p if o is not None and p is not None else None),
                   tex_num(h)])
    tables = [
        tex_table(r"In-distribution test results (\texttt{indist}).", "tab:indist",
                  ["Method", r"AUROC [95\% CI]", "Accuracy", "Macro-F1", r"TPR@1\%FPR"], t1),
        tex_table("Leave-one-generator-out AUROC (column = held-out generator).", "tab:logo",
                  ["Method"] + [TEX_GEN[g] for g in GENERATORS] + ["Mean"], grid_rows(logo)),
        tex_table("Cross-domain AUROC (column = held-out domain).", "tab:xdomain",
                  ["Method"] + [TEX_DOM[x] for x in DOMAINS] + ["Mean"], grid_rows(xdom)),
        tex_table("Paraphrase attack: AUROC on the same 500 machine and 500 human texts before and after "
                  "paraphrasing.", "tab:para",
                  ["Method", "Original", "Paraphrased", "Drop", "Human-paraphrase control"], t4),
    ]
    head = "% Generated by `python -m src.make_tables` from results/*/*/metrics.json. Do not edit by hand.\n"
    return head + "\n\n".join(tables) + "\n"


def build(results: Path, out_md: Path, fig_dir: Path | None) -> str:
    """Read every metrics.json under `results`, write tables.md and tables.tex next to out_md, and the figures into
    fig_dir (skipped when None, e.g. in tests). Returns the Markdown text so tests can check rows directly.
    Fails with a clear message if there are no results at all, instead of writing empty tables.
    """
    df = load_metrics(results)
    if df.empty:
        raise SystemExit(f"no metrics.json under {results}/<method>/<setting>/; run src.eval first")
    logo = grid(df, [f"logo_{g}" for g in GENERATORS], GENERATORS)
    xdom = grid(df, [f"xdomain_{d}" for d in DOMAINS], DOMAINS)
    # turn a numeric grid into formatted strings ('0.951' / '–') with the method name as a normal first column
    fmt = lambda g: g.map(f).reset_index().rename(columns={"index": "method"})
    text = "\n\n".join([
        f"# Results tables (generated by `python -m src.make_tables` from {results.name}/)",
        "## Table 1: in-distribution (test)\n" + md(table1(df)),
        "## Table 2: leave-one-generator-out AUROC (column = held-out generator)\n" + md(fmt(logo)),
        "## Table 3: cross-domain AUROC (column = held-out domain)\n" + md(fmt(xdom)),
        "## Table 4: paraphrase attack (same 500 machine texts before/after)\n" + md(table4(df)),
    ]) + "\n"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(text, encoding="utf-8")
    out_tex = out_md.with_suffix(".tex")
    out_tex.write_text(latex(df, logo, xdom), encoding="utf-8")
    print(f"wrote {out_tex}")
    if fig_dir is not None:
        for p in figures(df, logo, fig_dir):
            print(f"wrote {p}")
    print(f"wrote {out_md}")
    return text


def main(argv=None) -> int:
    """Command-line entry point. --results picks the folder to read (default results/; results_dev/ for dev
    runs) and the outputs are written inside that same folder. `argv` lets tests call main([...]) directly."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=ROOT / "results")
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args(argv)
    build(a.results, a.results / "tables.md", None if a.no_figures else a.results / "figures")
    return 0


if __name__ == "__main__":
    sys.exit(main())
