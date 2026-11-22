# Viva notes: role A (data, splits, evaluator, baselines)

Plain-English notes so you can explain every part you own. Each answer points to the file that proves it.

## The project in 4 sentences
We train and test detectors that say whether a text was written by a human or an LLM. The data is M4 English: 5
domains (Wikipedia, wikiHow, Reddit, arXiv, PeerRead) × 5 generators (ChatGPT, davinci, Cohere, Dolly, BLOOMZ).
The question is not "can we detect AI text?" (in-distribution is easy) but "**does detection still work on a
generator, a domain or a paraphrase the detector has never seen?**". Every detector writes the same `preds.csv`
format, and one shared script (`src/eval.py`) grades them all the same way.

## Pipeline (who makes what)
```
M4 raw files ─► src/data.py ─► m4_en.parquet (38,241 texts) + splits/*.json + data_audit.md     (A)
                                   │
       ┌───────────────┬───────────┼──────────────┬──────────────────┐
  baseline_tfidf (A)  train_roberta (B)  zeroshot (C)   paraphrase (D) ─► paraphrased texts + paraphrase splits
       └───────────────┴───────────┬──────────────┘
                           preds.csv per method/setting
                                   ▼
                     src/eval.py (A) ─► metrics.json ─► make_tables.py ─► report tables + figures
```

## Likely questions and answers

**Q: Why split by "group" instead of randomly?**
The same human text appears in up to 5 M4 files, once per generator it was paired with. With a random split, a human
text (or a machine text written for the same prompt) could be in training *and* test, so the detector would be graded
on text it has effectively seen. We give each human text and all its machine twins one `group` id and keep each group
entirely in train, val or test. Evidence: `results/data_audit.md` (3,000 of 3,000 Reddit human texts appear in more
than one file), and `python -m src.data --check-splits` checks that no group crosses splits.

**Q: Why collapse whitespace?**
arXiv and PeerRead human texts had hard line breaks; machine texts didn't. A detector could reach a high score just by
spotting newlines, which is a formatting shortcut and not a real signal. We collapse all whitespace in every text.

**Q: What do LOGO and cross-domain mean?**
LOGO (leave-one-generator-out): train on 4 generators, test only on the 5th. The test groups keep only their human
texts and the held-out generator's texts, so no test human text was seen in training. Cross-domain is the same idea
with domains. This answers the brief's "generalisation to unseen generators" and "cross-domain robustness".

**Q: Why is AUROC the main metric?**
It measures how well the scores rank machine texts above human texts, over all thresholds. It doesn't depend on the
class ratio (we have about 1.9 machine texts per human text) or on picking a threshold. 0.5 = chance, 1.0 = perfect.

**Q: Where does the threshold for accuracy / F1 come from?**
From the **validation** set: we pick the threshold with the best val macro-F1, then apply it unchanged to test.
Picking it on test would tune the answer to the exam. If val has fewer than 50 texts of a class, we report no
accuracy or F1 at all rather than an unreliable one (`src/eval.py`, `MIN_VAL_PER_CLASS`).

**Q: What is the 95% confidence interval?**
Bootstrap: draw 1,000 new test sets by resampling, compute AUROC on each, take the 2.5th and 97.5th percentiles. We
resample whole **groups**, because texts in a group are related, so resampling rows would make the interval too narrow.

**Q: What is TPR at 1% FPR and why report it?**
The share of machine texts caught when at most 1% of humans are falsely flagged. In real use, wrongly accusing a
student is costly, and a detector with a good AUROC can still be weak at 1% FPR. Only reported with at least 1,000
test humans; otherwise 1% would be under 10 texts.

**Q: How does the evaluator stop cheating or mistakes?**
Labels come from the dataset, never from the detector. It crashes if `preds.csv` has a train id, misses any test id
(except Fast-DetectGPT, which scores a fixed subset by design), has duplicates or NaN scores, or if the folder name
isn't a real setting.

**Q: Explain the TF-IDF baseline.**
TF-IDF turns each text into a sparse vector of word 1–2-gram and character 3–5-gram weights; logistic regression
learns which n-grams point to "AI". `class_weight='balanced'` makes up for machine texts outnumbering humans. C
(regularisation strength) is chosen from {0.25, 1, 4, 16} by **val** AUROC. `top_features.csv` shows the 30 most
"AI" and 30 most "human" n-grams.

**Q: Why a length-only baseline?**
It's a shortcut check. If a model that only sees text length scores well, part of every detector's score in that
setting may come from length rather than style. On the dev sample it was close to chance in-distribution (0.55) but
around 0.79 for held-out BLOOMZ and Reddit, so we flag those settings. (Confirm with the full-data numbers.)

**Q: Why Fast-DetectGPT instead of DetectGPT?**
DetectGPT needs about 20 T5 rewrites and 21 scoring passes per text, which didn't fit the Kaggle GPU quota.
Fast-DetectGPT measures the same "probability curvature" analytically in one pass. The brief asks for
"DetectGPT-style scoring", so it qualifies, and the report says so as a limitation.

**Q: How do the paraphrase settings work?**
`paraphrase_orig` = 500 machine + 500 human test texts. `paraphrase` = the same texts, with the 500 machine texts
replaced by T5 paraphrases. Same humans, same model (the `indist` one, no retraining), so the AUROC drop is caused
by the paraphrase alone.

**Q: Is it reproducible?**
Seed 42 everywhere. Every run writes `run_info.json` (git commit, hyperparameters, hardware, runtime, versions). The
README lists every command, and every script has `--dev` for a 2-minute check. 50+ tests run with `pytest`.

**Q: What are the limitations?**
English only; at most 1,000 machine texts per cell; one seed; one paraphraser; GPT-2 medium as the zero-shot scorer;
256-token truncation for RoBERTa and zero-shot; Fast-DetectGPT on a 1,500-text subset only (wider CIs).

## Numbers to know (from `results/data_audit.md`)
- 38,241 texts = 13,243 human + 24,998 machine, 11,278 groups.
- `indist`: 30,602 train / 3,801 val / 3,838 test. Each LOGO test set ≈ 10,100 texts.
- Your results: fill these in from `results/tables.md` once the full run is committed.
