# P20 — Detecting AI-Generated Text Across Domains and Generators

**Team 20:** NIRMAL20062 · VIVEK342004 · RITESHJHA2244 · kumar-vishal01
**Deadline:** Sat 22 Nov 2026, 11:59 pm IST (report + this GitHub repo) · **Viva:** from 23 Nov

---

## 1. What we are building (in one paragraph)

A system that reads a piece of text and says **"human-written" or "machine-generated"**. Building a detector
that works on the data it was trained on is easy. **The real question is whether it still works on text from
an LLM it has never seen, on a topic or domain it has never seen, or after the AI text has been paraphrased
to dodge detection.** We build three detectors of increasing sophistication and stress-test all three in
these settings. The main result of the project is a set of comparison tables and plots showing where each
detector breaks.

## 2. The data: M4

M4 (Wang et al., EACL 2024) is a benchmark of paired human and machine texts. We use the **English** part:

| | |
|---|---|
| **Domains (5)** | Wikipedia, WikiHow, Reddit (ELI5), arXiv abstracts, PeerRead (paper reviews) |
| **Generators (5)** | ChatGPT, text-davinci-003, Cohere, Dolly-v2, BLOOMZ |
| **Format** | one JSONL file per (domain, generator), each row holding a human text and a machine text on the same prompt or topic |
| **Source** | https://github.com/mbzuai-nlp/M4 |

This grid (domain × generator) is what makes cross-domain and cross-generator experiments possible.
To fit our compute budget, we **subsample to at most 1,000 pairs per cell** and use **fixed random seeds** so
everyone gets the same splits.

**Two day-1 checks before building splits:**
- **Grid audit:** count rows per (domain, generator). Some cells may be missing or small, and some domains
  include extra generators (e.g. GPT-4). We use only the cells we actually have and document the rest.
- **Leak check:** the *same human text* often appears in several generator files of a domain. If we split
  naively, a human text can land in train *and* test, which inflates the scores, especially for LOGO. So we
  **give every text a group key (a hash of the normalised human text) and split by group**, and we report how
  many human texts are duplicated across files.

## 3. The three detectors

### 3.1 TF-IDF + Logistic Regression (baseline)
- Turn each text into a vector of word/character n-gram frequencies (TF-IDF), then fit a linear classifier.
- **Why:** fast (minutes on a laptop), interpretable (we can show which n-grams signal "AI"), and the
  standard baseline that every deep model must beat.
- **Expectation:** strong in-distribution, weak on unseen generators and domains, because it memorises
  surface vocabulary.
- **Sanity baseline alongside it:** a classifier that uses *only text length* (characters/tokens). If length
  alone gets a high AUROC in some setting, the data has a shortcut and we must say so in the report.

### 3.2 Fine-tuned RoBERTa (supervised deep model)
- Take pretrained `roberta-base`, add a classification head, and fine-tune on labelled human/AI texts.
- **Why:** this is the deep-learning core of the course project, and the strongest detector in the M4 paper.
- **Expectation:** best in-distribution, but it can still overfit to the generators it saw during training.
  Leave-one-generator-out (LOGO) tests whether it does.
- **Compute:** needs a GPU, so we run it on Kaggle or Colab (about 20–40 min per run on a T4 with max_len 256).

### 3.3 Zero-shot scoring: perplexity, then DetectGPT (no training)
- Run the text through an open LLM (e.g. GPT-2) and measure how *likely* the LLM finds it.
  - **Log-likelihood / perplexity:** AI text tends to have **lower perplexity**, because LLMs choose
    high-probability words.
  - **Log-rank:** the average rank of each actual token in the model's prediction list.
  - **DetectGPT (Mitchell et al., ICML 2023):** make about 20–50 small rewrites of the text with T5
    (mask spans and refill them). For **machine text**, almost every rewrite *drops* the log-probability a
    lot, because the original sits at a local peak ("probability curvature"). For **human text**, the change
    is small and random. Score = log p(original) − mean log p(rewrites).
- **Why:** needs **no labelled data**, so in principle it is not tied to any training generator.
  This makes it an interesting contrast with RoBERTa under LOGO.
- **Compute:** DetectGPT is expensive (N rewrites × forward passes per text). So we score **one fixed,
  balanced subsample once** (about 1,500 texts spread evenly over domain × generator × label, plus the
  paraphrased set), then evaluate each setting on the part of that subsample inside its val/test split.
  Perplexity and log-rank are cheap, so they score everything.
- **Fallback:** if DetectGPT is too slow, use **Fast-DetectGPT** (Bao et al., ICLR 2024). It gives a similar
  curvature score from a single forward pass.

## 4. Evaluation settings (the core of the report)

| Setting | Train on | Test on | Question it answers |
|---|---|---|---|
| **In-distribution** | all domains & generators (train split) | same mix (test split) | Upper bound: how good is it in easy conditions? |
| **Leave-one-generator-out (LOGO)** | 4 generators | the held-out 5th (repeat ×5) | Does it work on an **unseen LLM**? |
| **Cross-domain** | 4 domains | the held-out 5th domain (repeat ×5) | Does it work on an **unseen kind of writing**? |
| **Paraphrase attack** | in-distribution model | AI test texts rewritten by a paraphraser | Can a simple rewrite **fool** the detector? |

Zero-shot methods have no training step, but we still report them on the *same test sets*
(choosing the threshold on held-out data only) so the tables line up.

**Metrics:** AUROC (main metric, threshold-free), plus accuracy and macro-F1, with **bootstrap 95% confidence
intervals** for AUROC. TPR at 1% FPR matters in practice (false accusations against humans are costly), but it
is only meaningful with enough human test texts. We report it only where the test set has **at least 1,000
human texts**, so not for the small DetectGPT subsamples.

**Paraphrase attack:** take a fixed set of 500 AI + 500 human test texts, feed the AI ones through an
off-the-shelf paraphrase model (e.g. a T5/Pegasus paraphraser, or DIPPER if it fits on the GPU), and score the
set with all detectors **before and after** paraphrasing (`paraphrase_orig` vs `paraphrase`). Using the same
texts both times makes the AUROC drop a fair comparison.
*Optional control:* also paraphrase about 200 human texts. If the detectors then call those "AI", they are
detecting "paraphraser style", not "LLM-written".

## 5. Compute plan

The local machine has no GPU and little RAM or disk, so we split the work:

| Runs locally (CPU) | Runs on Kaggle / Colab (GPU) |
|---|---|
| data download, subsampling, split files | RoBERTa fine-tuning (in-dist + 5 LOGO + cross-domain) |
| TF-IDF + LR (all settings) | perplexity / DetectGPT scoring |
| results aggregation, plots | paraphrase generation |

Each of the 4 team members has their own Kaggle GPU quota (about 30 h/week), so we split GPU jobs across
accounts. All code lives in this repo; notebooks only `git clone` the repo and call the scripts.

- **Data on Kaggle:** the processed data is gitignored, so A uploads it once as a private Kaggle Dataset shared
  with the team, and notebooks attach it.
- **Results back:** download `results/…` from the notebook output, then commit and open a PR from your own
  machine and account.
- **Test cheaply first:** every script has a `--dev` flag that runs on a ~500-row sample on CPU in under
  2 minutes, so bugs are caught before spending GPU quota.
- Every run records the git commit, seed and hyperparameters in `run_info.json`.
  Full details: [TASKS.md §0.4](TASKS.md).

## 6. Planned repo structure

```
generated_text_detection/
├── README.md              # how to reproduce everything
├── requirements.txt       # Python 3.11; torch not pinned (Kaggle provides it)
├── data/                  # (gitignored) raw M4 + processed parquet; shared via a private Kaggle Dataset
├── splits/                # small committed files of IDs per split → everyone uses identical splits
├── src/
│   ├── data.py            # load M4, audit grid, dedupe, subsample, build group-aware splits
│   ├── eval.py            # shared evaluator: AUROC (+bootstrap CI), acc, F1, TPR@1%FPR
│   ├── baseline_tfidf.py  # TF-IDF + LR, plus the length-only sanity baseline
│   ├── train_roberta.py
│   ├── zeroshot.py        # perplexity, log-rank, DetectGPT (Fast-DetectGPT fallback)
│   ├── paraphrase.py
│   └── make_tables.py     # every metrics.json → final tables + figures
├── notebooks/             # thin Kaggle/Colab runners + analysis
├── results/<method>/<setting>/{preds.csv, metrics.json}   # committed
├── results/figures/
└── docs/                  # ROADMAP.md (this file), TASKS.md (who codes what)
```

**Rule:** every experiment writes `results/<method>/<setting>/preds.csv`, `src/eval.py` turns that into
`metrics.json` next to it, and `src/make_tables.py` builds every table from those files. Exact formats are in
[TASKS.md §0](TASKS.md).

## 7. Timeline & owners

Roles A–D and exactly what each person codes are in [TASKS.md](TASKS.md).

| Dates | Milestone | Owner |
|---|---|---|
| **Fri 7 Nov** | Day-0 bootstrap commit on `main` (docs, `.gitignore`, `requirements.txt`, `src/__init__.py`, README stub) → everyone branches | A |
| **Sat 8 – Sun 9 Nov** | M4 download, grid audit + leak check, dev sample, group-aware stratified splits, shared `eval.py`, Kaggle Dataset shared | A (lead) + all review |
| **Sun 9 – Wed 12 Nov** | TF-IDF + LR and the length baseline on in-dist, LOGO and cross-domain, plus top-feature analysis | A |
| **Sun 9 – Sat 15 Nov** | RoBERTa: in-dist by 12 Nov, 5× LOGO by 14 Nov, 5× cross-domain by 15 Nov | B |
| **Sun 9 – Fri 14 Nov** | Zero-shot: perplexity + log-rank on everything by 12 Nov, then DetectGPT on the fixed subsample | C |
| **Sun 9 – Thu 13 Nov** | Build the paraphrased test set (+ optional human control) → on `main` by 13 Nov | D |
| **Thu 13 – Sun 16 Nov** | Every detector re-scored on `paraphrase` | A, B, C |
| **Wed 12 – Sun 16 Nov** | **Draft the report sections as results come in** (method + setup sections can be written right away) | each owner |
| **Sun 16 – Wed 19 Nov** | Final tables and figures, error analysis, Discussion section | D + all |
| **Wed 19 – Sat 22 Nov** | Report polish, README, final push **before 22 Nov 11:59 pm** | all |
| **by Fri 21 Nov** | 10-min walkthrough by each owner, so everyone can explain all methods at the viva | all |
| **Sun 23 Nov →** | Presentation / viva | all |

**Checkpoint rule:** if DetectGPT or paraphrasing is running late by 16 Nov, ship the cheaper version
(log-likelihood/log-rank only; a smaller paraphrase subset) rather than miss the deadline.

## 8. Report outline (draft)

1. Introduction: why detecting AI text matters; why generalisation is the hard part
2. Related work: M4, DetectGPT, (optionally Fast-DetectGPT, GLTR)
3. Dataset: M4 English subset, subsampling, split construction
4. Methods: TF-IDF+LR, RoBERTa, zero-shot (perplexity, log-rank, DetectGPT), paraphrase attack
5. Experimental setup: settings table, metrics, hyperparameters, compute
6. Results: in-dist, LOGO, cross-domain, paraphrase (one table + one figure each)
7. Discussion: which detector generalises best and why; failure cases; limitations
8. Conclusion
9. References · team contributions

## 9. Risks

| Risk | Mitigation |
|---|---|
| GPU quota runs out | spread jobs over 4 Kaggle accounts; cap max_len at 256; use 1 epoch where enough |
| DetectGPT too slow | fewer perturbations (e.g. 20), smaller fixed subsample, smaller scorer model, or switch to Fast-DetectGPT |
| Train/test leakage (duplicate human texts) | group-aware splits by human-text hash; report the duplicate count |
| Length or format shortcuts | length-only baseline; mention it in Discussion if it scores high |
| Low local disk (≈9 GB free) | download only English M4 files; keep raw data out of git |
| Results don't match the paper | that's fine, as long as we report it honestly and explain why (subsampling, smaller models) |
| Someone can't explain others' parts at the viva | each owner gives a 10-min walkthrough to the team by 21 Nov |

## References

1. E. Mitchell et al., *DetectGPT: Zero-Shot Machine-Generated Text Detection using Probability Curvature*, ICML 2023. https://arxiv.org/abs/2301.11305
2. Y. Wang et al., *M4: Multi-generator, Multi-domain, and Multi-lingual Black-Box Machine-Generated Text Detection*, EACL 2024. https://arxiv.org/abs/2305.14902
3. G. Bao et al., *Fast-DetectGPT: Efficient Zero-Shot Detection of Machine-Generated Text via Conditional Probability Curvature*, ICLR 2024. https://arxiv.org/abs/2310.05130 (fallback only)
