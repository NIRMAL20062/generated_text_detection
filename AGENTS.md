# Guide for AI coding agents (and new humans)

Read this before changing anything. It tells you what the project is, which docs are the source of truth,
and the rules every change must follow.

## 1. What this project is

Course project **P20** (Deep Learning, Team 20): detect whether a text is **human-written or LLM-generated**,
and test whether detectors still work on **unseen generators**, **unseen domains** and **paraphrased** AI text,
using the English part of the **M4** benchmark. Three detectors: TF-IDF + logistic regression →
fine-tuned RoBERTa → zero-shot perplexity / log-rank / DetectGPT. The main output is comparison tables and
figures for a report, not an app.

**Deadline: Sat 22 Nov 2026, 11:59 pm IST** (report + this repo). Scope cuts beat missing the deadline.

## 2. Read in this order

1. [docs/PROJECT_BRIEF.md](docs/PROJECT_BRIEF.md): the instructor's original assignment, word for word, and how
   each line maps to the repo. **If anything conflicts, the brief wins.**
2. [docs/TASKS.md](docs/TASKS.md) **§0 "The contract"**: data schema, split files, `preds.csv` format, setting
   names. This is binding, and code that breaks it breaks everyone else's code.
3. [docs/ROADMAP.md](docs/ROADMAP.md): why each method and setting exists, the timeline, and the risks.
4. [CONTRIBUTING.md](CONTRIBUTING.md): branches, PRs, Kaggle runs, bringing results back.
5. `results/data_audit.md`: actual dataset counts. It's on the `a-data` branch until that branch is merged.

## 3. The pipeline

```
src/data.py ──► data/processed/m4_en.parquet  +  splits/<setting>.json
                     │
   ┌─────────────────┼──────────────────┬───────────────────┐
   ▼                 ▼                  ▼                   ▼
baseline_tfidf.py  train_roberta.py   zeroshot.py         paraphrase.py
(A, CPU)           (B, Kaggle GPU)    (C, Kaggle GPU)     (D → paraphrased.parquet, splits/paraphrase*.json)
   │                 │                  │
   └──► results/<method>/<setting>/preds.csv + run_info.json
                     │
              src/eval.py ──► metrics.json        (the ONLY metric code)
                     │
          src/make_tables.py ──► results/figures/ + report tables
```

Settings (exact strings): `indist`, `logo_<gen>` ×5, `xdomain_<dom>` ×5, `paraphrase_orig`, `paraphrase`,
`paraphrase_human` (optional). Generators: `chatGPT davinci cohere dolly bloomz`. Domains:
`wikipedia wikihow reddit arxiv peerread`. Methods: `length tfidf_lr roberta ppl logrank fastdetectgpt`
(`fastdetectgpt` as fallback).

## 4. Hard rules

- **Follow the contract** in TASKS.md §0. Don't invent new column names, file paths or setting names.
- **Load data only through** `src.data.load_texts()` / `load_split()`. Never re-read raw JSONL in other scripts
  or hand-filter split files.
- **Splits are by `group`**, never by row. Never move a text between train/val/test by hand.
- **`score` in preds.csv: higher = more likely AI.** Always include `val` rows (the evaluator picks the threshold
  on val).
- **Never write your own metric code.** Use `python -m src.eval <preds.csv>`.
- **Every script needs `--dev`**, which runs on `data/sample_dev.parquet` in under 2 minutes on CPU. Run it
  before any GPU job and before opening a PR.
- **Any new text** (paraphrases) gets the same whitespace normalisation: `" ".join(text.split())`.
- `SEED = 42` everywhere. Record `git_commit`, seed, hyperparameters, runtime and hardware in `run_info.json`.
- **Notebooks hold no logic.** They `git clone`, install, and call `python -m src.<script>`.
- **Never commit** `data/` (except `data/sample_dev.parquet`), model weights or `results/zeroshot_cache/`.
  **Do commit** `splits/` and `results/**/*.csv|json`.

## 5. Ownership and git

| Role | Person (GitHub) | Branch | Owns |
|---|---|---|---|
| A: Data & Baseline (lead) | Nirmal Kumar (NIRMAL20062) | `a-data` | `src/data.py`, `src/eval.py`, `src/baseline_tfidf.py`, `splits/` (except D's) |
| B: RoBERTa | Vivek Kumar (VIVEK342004) | `b-roberta` | `src/train_roberta.py`, `notebooks/kaggle_roberta.ipynb` |
| C: Zero-shot | Ritesh Jha (RITESHJHA2244) | `c-zeroshot` | `src/zeroshot.py`, `notebooks/kaggle_zeroshot.ipynb` |
| D: Attack & Results | Vishal Kumar (kumar-vishal01) | `d-attack` | `src/paraphrase.py`, `src/make_tables.py`, `splits/paraphrase*.json` |

- Ask which role you are working for, and only edit files that role owns. Changing a shared file or the
  contract needs the team's agreement first.
- Work on the role's branch. **Nobody pushes straight to `main`**: open a PR, and another member reviews it.
- Commits come from the member's own GitHub account (the repo is graded on contributions). Commit messages
  start with the area: `data:`, `baseline:`, `eval:`, `roberta:`, `zeroshot:`, `attack:`, `tables:`, `docs:`.

## 6. Environment and compute

- Python 3.11, `uv venv --python 3.11 && uv pip install -r requirements.txt`. `torch` is not pinned, because
  Kaggle provides it.
- The local machine has **no GPU, about 6.6 GB of RAM and little disk space**. Run data prep, TF-IDF, eval and
  tables locally. Run RoBERTa, zero-shot scoring and paraphrasing on Kaggle (private dataset
  `team20-m4-processed`).

## 7. Finding the current status

Status changes daily, so check it rather than trusting a snapshot:
`git log --oneline --all | head`, `gh pr list`, the checkboxes in [docs/TASKS.md](docs/TASKS.md), and which
`results/<method>/<setting>/metrics.json` files exist.
