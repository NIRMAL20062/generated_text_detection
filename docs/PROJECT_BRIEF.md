# Project Brief (from the instructor)

This is the original project assignment, copied **word for word** from the course project list. It is the
source of truth for scope: if any other doc disagrees with this page, this page wins. The column names on the
left are ours; the course sheet listed the fields in this order without labels.

| Field | As given |
|---|---|
| **Project ID** | P20 |
| **Title** | Detecting AI-Generated Text Across Domains and Generators |
| **Modality** | Text |
| **Task** | Distinguish human-written from LLM-generated text and test generalisation to unseen generators. |
| **Dataset** | M4 dataset (github.com/mbzuai-nlp/M4) |
| **Methods** | TF-IDF + LR baseline → fine-tuned RoBERTa detector → zero-shot perplexity / DetectGPT-style scoring; leave-one-generator-out. |
| **Evaluation** | Cross-generator and cross-domain robustness; effect of paraphrasing attacks. |
| **References** | [1] Mitchell et al., DetectGPT: Zero-Shot Machine-Generated Text Detection using Probability Curvature, ICML 2023 — https://arxiv.org/abs/2301.11305 <br> [2] Wang et al., M4: Multi-generator, Multi-domain, and Multi-lingual Black-Box Machine-Generated Text Detection, EACL 2024 — https://arxiv.org/abs/2305.14902 |

**Course logistics:** Deep Learning course (Sitare), Team 20, instructor Chhavi Sharma.
Deliverables are a **project report** and **this GitHub repo** (one per group). The report format and upload
link will be shared by the instructor. Deadline: **Sat 22 Nov 2026, 11:59 pm IST**. Presentations and viva
start on 23 Nov 2026.

## How each requirement maps to the repo

Every line of the brief must end up as code, results and a report section. This table is the checklist.

| Brief requirement | Where it is implemented | Result it produces | Owner |
|---|---|---|---|
| Distinguish human vs LLM text | binary label `0` = human, `1` = machine in `data/processed/m4_en.parquet` ([TASKS.md §0.1](TASKS.md)) | n/a | A |
| M4 dataset | `src/data.py`: English M4, 5 domains × 5 generators, 1,000 machine texts per cell | `results/data_audit.md` | A |
| TF-IDF + LR baseline | `src/baseline_tfidf.py` (+ `--method length` sanity check) | `results/tfidf_lr/`, `results/length/` | A |
| Fine-tuned RoBERTa detector | `src/train_roberta.py` (Kaggle GPU) | `results/roberta/` | B |
| Zero-shot perplexity / DetectGPT-style scoring | `src/zeroshot.py --method ppl\|logrank\|detectgpt` | `results/ppl/`, `results/logrank/`, `results/detectgpt/` | C |
| Leave-one-generator-out | `splits/logo_<gen>.json` ×5 | Table 2 (method × held-out generator) | A builds splits; A, B, C run |
| Cross-generator robustness | same as LOGO | Table 2 + LOGO heatmap | all |
| Cross-domain robustness | `splits/xdomain_<dom>.json` ×5 | Table 3 (method × held-out domain) | A builds splits; A, B, C run |
| Effect of paraphrasing attacks | `src/paraphrase.py` → `splits/paraphrase.json`, compared to `splits/paraphrase_orig.json` | Table 4 (AUROC before → after) | D makes data; A, B, C score |
| Shared evaluation | `src/eval.py` (AUROC + 95% CI, acc, macro-F1, TPR@1%FPR) | `metrics.json` per run | A |
| Final tables and figures | `src/make_tables.py` | `results/figures/`, report tables | D |
| References [1], [2] | cited in README, ROADMAP and the report | n/a | all |

## What we added beyond the brief (and why)

These are our choices, not instructor requirements. Cut them first if time runs out.

- **Log-rank** zero-shot score: almost free once perplexity is computed.
- **Length-only baseline**: detects whether the data has a length shortcut.
- **Group-aware splits**: the same human text appears in several M4 files. Without these splits it would leak
  from train into test and inflate the scores.
- **Fast-DetectGPT**: a fallback, used only if DetectGPT is too slow.
- **Human-paraphrase control** (optional): checks whether the detectors flag "paraphraser style" rather than
  "LLM-written" text.
- **English only, 1,000 machine texts per cell**: keeps the work within our compute budget (no local GPU,
  Kaggle quota).
