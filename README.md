# Detecting AI-Generated Text Across Domains and Generators (P20)

Deep Learning course project, Team 20. We detect whether a text is human-written or machine-generated, and test
whether detectors still work on **unseen generators**, **unseen domains**, and **paraphrased** AI text, using the
[M4](https://github.com/mbzuai-nlp/M4) benchmark (English).

**Detectors:** TF-IDF + logistic regression (baseline) → fine-tuned RoBERTa → zero-shot perplexity / log-rank / DetectGPT.
**Settings:** in-distribution, leave-one-generator-out, leave-one-domain-out, paraphrase attack.

## Team

| Role | Member | GitHub |
|---|---|---|
| A: Data & Baseline (lead) | Nirmal Kumar | [@NIRMAL20062](https://github.com/NIRMAL20062) |
| B: RoBERTa | Vivek Kumar | [@VIVEK342004](https://github.com/VIVEK342004) |
| C: Zero-shot | Ritesh Jha | [@RITESHJHA2244](https://github.com/RITESHJHA2244) |
| D: Attack & Results | Vishal Kumar | [@kumar-vishal01](https://github.com/kumar-vishal01) |

## Docs

- [docs/PROJECT_BRIEF.md](docs/PROJECT_BRIEF.md): the instructor's original assignment, word for word, mapped to the repo
- [AGENTS.md](AGENTS.md): start here if you are an AI coding agent (or new to the repo)
- [docs/ROADMAP.md](docs/ROADMAP.md): what we're building, the evaluation settings, timeline, risks
- [docs/TASKS.md](docs/TASKS.md): who codes what, and the shared data/split/prediction formats (read §0 first)
- [CONTRIBUTING.md](CONTRIBUTING.md): how we work day to day (branches, PRs, Kaggle runs, getting results back)

## Setup

```bash
git clone https://github.com/NIRMAL20062/generated_text_detection.git
cd generated_text_detection
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -r requirements.txt   # plus torch if you run models locally
```

## How to reproduce

Every script has `--dev`, which runs on the committed `data/sample_dev.parquet` in under 2 minutes on CPU and
writes to `results_dev/`. Full runs need the data (step 1) and, for the model steps, a GPU (we used Kaggle T4).
On Kaggle, attach the `team20-m4-processed` dataset and run `notebooks/kaggle_all.ipynb`, which does steps 3–6.

```bash
# 1. Data: download M4, build data/processed/m4_en.parquet, splits/*.json and results/data_audit.md
python -m src.data --all
python -m src.data --check-splits          # every split id exists, no group crosses train/val/test

# 2. Baselines (CPU, ~1-2 h for all settings)
python -m src.baseline_tfidf --method length
python -m src.baseline_tfidf --method tfidf_lr

# 3. Paraphrase attack (GPU): outputs/paraphrase/paraphrased.parquet + splits/paraphrase*.json
python -m src.paraphrase

# 4. Zero-shot detectors (GPU)
python -m src.zeroshot --method ppl
python -m src.zeroshot --method logrank
python -m src.zeroshot --method fastdetectgpt

# 5. RoBERTa (GPU): indist first, its model is reused for the paraphrase settings
python -m src.train_roberta --setting indist
python -m src.train_roberta --setting paraphrase --predict-only --model outputs/roberta/indist/model
python -m src.train_roberta --setting logo_chatGPT --no-save     # ... every logo_* and xdomain_*

# 6. Metrics for every run, then the report tables and figures
for f in results/*/*/preds.csv; do python -m src.eval "$f"; done
python -m src.make_tables                  # results/tables.md + results/figures/

# tests
python -m pytest tests/ -q
```

## References

1. E. Mitchell et al., *DetectGPT: Zero-Shot Machine-Generated Text Detection using Probability Curvature*, ICML 2023. https://arxiv.org/abs/2301.11305
2. Y. Wang et al., *M4: Multi-generator, Multi-domain, and Multi-lingual Black-Box Machine-Generated Text Detection*, EACL 2024. https://arxiv.org/abs/2305.14902
