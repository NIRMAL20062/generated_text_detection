"""Report tables and figures from every results/<method>/<setting>/metrics.json (TASKS.md §D, issue #14).

    python -m src.make_tables                 # results/ -> results/tables.md + results/figures/*.png
    python -m src.make_tables --results results_dev

Table 1: in-distribution AUROC [95% CI], accuracy, macro-F1, TPR@1%FPR per method
Table 2: LOGO AUROC, method x held-out generator, + mean
Table 3: cross-domain AUROC, method x held-out domain, + mean
Table 4: paraphrase attack: paraphrase_orig -> paraphrase AUROC and the drop (+ paraphrase_human control)
Fig 1: grouped bars (method x setting group); Fig 2: LOGO heatmap.
A missing method/setting shows as "–", so the script runs before every result is in.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from src.data import DOMAINS, GENERATORS, ROOT

METHOD_ORDER = ["length", "tfidf_lr", "ppl", "logrank", "fastdetectgpt", "roberta"]


def load_metrics(results: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(results.glob("*/*/metrics.json")):
        m = json.loads(f.read_text())
        lo, hi = m.get("auroc_ci95") or [None, None]
        rows.append({"method": f.parent.parent.name, "setting": f.parent.name, "auroc": m.get("auroc"),
                     "lo": lo, "hi": hi, "accuracy": m.get("accuracy"), "macro_f1": m.get("macro_f1"),
                     "tpr1": m.get("tpr_at_1pct_fpr"), "n_test": m.get("n_test")})
    df = pd.DataFrame(rows, columns=["method", "setting", "auroc", "lo", "hi", "accuracy", "macro_f1", "tpr1", "n_test"])
    order = {m: i for i, m in enumerate(METHOD_ORDER)}
    return df.assign(_o=df["method"].map(lambda m: order.get(m, len(order)))).sort_values(["_o", "method"]).drop(columns="_o")


def f(x, d=3) -> str:
    return "–" if x is None or pd.isna(x) else f"{x:.{d}f}"


def md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


def methods(df: pd.DataFrame) -> list[str]:
    return list(dict.fromkeys(df["method"]))


def grid(df: pd.DataFrame, settings: list[str], labels: list[str]) -> pd.DataFrame:
    """method x setting AUROC matrix (NaN where missing) with a mean over the settings that exist."""
    piv = df.pivot_table(index="method", columns="setting", values="auroc", aggfunc="first")
    piv = piv.reindex(index=methods(df), columns=settings)
    piv.columns = labels
    piv["mean"] = piv.mean(axis=1, skipna=False)   # mean only when every setting is present
    return piv


def table1(df):
    d = df[df.setting == "indist"]
    return pd.DataFrame({"method": d.method, "AUROC [95% CI]": [f"{f(a)} [{f(l)}, {f(h)}]" for a, l, h in zip(d.auroc, d.lo, d.hi)],
                         "accuracy": d.accuracy.map(f), "macro-F1": d.macro_f1.map(f), "TPR@1%FPR": d.tpr1.map(f)})


def table4(df):
    rows = []
    for m in methods(df):
        a = df[(df.method == m)].set_index("setting")["auroc"]
        o, p, h = (a.get(s) for s in ("paraphrase_orig", "paraphrase", "paraphrase_human"))
        if o is None and p is None:
            continue
        drop = o - p if o is not None and p is not None else None
        rows.append({"method": m, "orig AUROC": f(o), "paraphrased AUROC": f(p), "drop": f(drop),
                     "human-paraphrase control": f(h)})
    return pd.DataFrame(rows, columns=["method", "orig AUROC", "paraphrased AUROC", "drop", "human-paraphrase control"])


def figures(df, logo: pd.DataFrame, out: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    groups = {"in-dist": df.setting.eq("indist"), "LOGO (mean)": df.setting.str.startswith("logo_"),
              "cross-domain (mean)": df.setting.str.startswith("xdomain_"), "paraphrased": df.setting.eq("paraphrase")}
    bars = pd.DataFrame({g: df[mask].groupby("method")["auroc"].mean() for g, mask in groups.items()}).reindex(methods(df))
    if not bars.dropna(how="all").empty:
        ax = bars.plot.bar(figsize=(9, 4.5), rot=0, width=0.8)
        ax.set_ylabel("AUROC"); ax.set_ylim(0, 1); ax.axhline(0.5, color="grey", lw=0.8, ls="--")
        ax.set_title("Detector AUROC by setting", pad=28)
        ax.legend(fontsize=8, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
        plt.tight_layout(); p = out / "fig1_auroc_by_setting.png"; plt.savefig(p, dpi=150); plt.close(); paths.append(p)
    h = logo.drop(columns="mean")
    if not h.dropna(how="all").empty:
        fig, ax = plt.subplots(figsize=(7, 0.5 * len(h) + 1.5))
        im = ax.imshow(h.to_numpy(dtype=float), vmin=0.5, vmax=1, cmap="viridis", aspect="auto")
        ax.set_xticks(range(h.shape[1]), h.columns); ax.set_yticks(range(len(h)), h.index)
        for i in range(h.shape[0]):
            for j in range(h.shape[1]):
                v = h.iat[i, j]
                ax.text(j, i, f(v, 2), ha="center", va="center", color="w" if pd.notna(v) and v < 0.8 else "k", fontsize=8)
        fig.colorbar(im, label="AUROC"); ax.set_title("LOGO: AUROC on the held-out generator")
        plt.tight_layout(); p = out / "fig2_logo_heatmap.png"; plt.savefig(p, dpi=150); plt.close(); paths.append(p)
    return paths


def build(results: Path, out_md: Path, fig_dir: Path | None) -> str:
    df = load_metrics(results)
    if df.empty:
        raise SystemExit(f"no metrics.json under {results}/<method>/<setting>/; run src.eval first")
    logo = grid(df, [f"logo_{g}" for g in GENERATORS], GENERATORS)
    xdom = grid(df, [f"xdomain_{d}" for d in DOMAINS], DOMAINS)
    fmt = lambda g: g.map(f).reset_index().rename(columns={"index": "method"})
    text = "\n\n".join([
        f"# Results tables (generated by `python -m src.make_tables` from {results.name}/)",
        "## Table 1: in-distribution (test)\n" + md(table1(df)),
        "## Table 2: leave-one-generator-out AUROC (column = held-out generator)\n" + md(fmt(logo)),
        "## Table 3: cross-domain AUROC (column = held-out domain)\n" + md(fmt(xdom)),
        "## Table 4: paraphrase attack (same 500 machine texts before/after)\n" + md(table4(df)),
    ]) + "\n"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(text)
    if fig_dir is not None:
        for p in figures(df, logo, fig_dir):
            print(f"wrote {p}")
    print(f"wrote {out_md}")
    return text


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=ROOT / "results")
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args(argv)
    build(a.results, a.results / "tables.md", None if a.no_figures else a.results / "figures")
    return 0


if __name__ == "__main__":
    sys.exit(main())
