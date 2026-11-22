# Report draft: Dataset, Setup and Baselines (role A)

> Draft for issues #8 and #15. Numbers marked `__` are blanks: fill them from `results/<method>/<setting>/metrics.json`
> (or `results/tables.md`, made by `python -m src.make_tables`) after the full Kaggle run. Never report a number
> that is not in those files. All dataset numbers below come from `results/data_audit.md`.

## 1. Dataset

We use the English part of **M4** (Wang et al., EACL 2024): human texts paired with texts generated for the same
prompt by five LLMs (**ChatGPT, davinci-003, Cohere, Dolly-v2, BLOOMZ**) in five domains (**Wikipedia, wikiHow,
Reddit ELI5, arXiv abstracts, PeerRead reviews**). The full 5 × 5 grid exists. We leave out the files that
only exist for some domains (`arxiv_flant5`, `reddit_flant5`, `peerread_llama`), so every setting covers the same
generators.

**Sampling.** At most 1,000 pairs per (domain, generator) cell (PeerRead: 250 papers per cell, about 4 machine
reviews each), with a fixed seed (42). Texts under 50 characters, and machine texts that copy the prompt, are dropped.

**Final size:** 38,241 texts: 13,243 human and 24,998 machine (1,000 machine texts per cell), in 11,278 groups.

**Cleaning choices that matter for a fair result:**

- **Whitespace is collapsed in every text.** arXiv and PeerRead human texts contain hard line breaks and leading
  spaces, but the machine texts don't. Without this step, a detector could score highly just by spotting newlines.
  The paraphrased texts get the same normalisation.
- **BLOOMZ files** store the prompt plus the output in `machine_text`. We take the output-only field instead.
- **Duplicate human texts.** The same human text appears in up to five generator files (one per generator it was
  paired with). We keep it once. Every text gets a `group` id: a hash of the human text, or for PeerRead, of
  the paper. A human text and all of its machine twins share a group.

## 2. Evaluation settings and splits

**Splits are made by group, never by row**, so no human text, and no machine text written for the same prompt,
appears in both training and test. A random row split would leak near-duplicates and inflate every score.

| Setting | Train / val | Test | Question it answers |
|---|---|---|---|
| `indist` | 80 / 10 % of groups, stratified by domain × generator | other 10 % | How well does it work on data like the training data? |
| `logo_<gen>` (×5) | the other 4 generators | only `<gen>` (plus the humans of its groups) | Does it generalise to an **unseen generator**? |
| `xdomain_<dom>` (×5) | the other 4 domains | only `<dom>` | Does it generalise to an **unseen domain**? |
| `paraphrase_orig` → `paraphrase` | trained on `indist` | the same 500 machine + 500 human test texts, machine texts before / after paraphrasing | Does it survive a **paraphrase attack**? |

In LOGO, the test groups keep only their human texts and the held-out generator's texts. Other generators' texts
from those groups are dropped rather than moved to training, so no test human text is seen in training. Split
sizes and class ratios per setting are in `results/data_audit.md`. For example, `indist` has 30,602 / 3,801 / 3,838
texts and each LOGO test set has about 10,100 texts.

**Class balance.** Deduplication keeps each human text once, so machine texts outnumber human texts (about 1.9 : 1
in `indist`). Trained models compensate with class weights. AUROC, our main metric, does not depend on the ratio.

## 3. Evaluation protocol (`src/eval.py`, shared by every method)

Every detector writes one `preds.csv` (`id, split, score`, higher = more likely AI) per setting. One shared script
computes every metric, so all methods are graded identically.

- **Labels come from the dataset, not from the detector.** The evaluator also refuses a file that scores a train id,
  skips any test id, or contains NaN scores.
- **AUROC** on test, with a **95 % bootstrap CI** that resamples whole groups, because texts in a group are not
  independent.
- **Accuracy and macro-F1** at a threshold chosen **on val** (best val macro-F1), then applied unchanged to test.
  Choosing it on test would tune the answer to the exam.
- **TPR at 1 % FPR**: the share of machine texts caught when at most 1 % of humans are falsely flagged. It is only
  reported when the test set has at least 1,000 humans.

## 4. Baselines (`src/baseline_tfidf.py`)

**TF-IDF + logistic regression (`tfidf_lr`).** Word 1–2-grams (min_df 2, up to 200k features) and character
3–5-grams (min_df 3, up to 300k features), sublinear TF, concatenated, then logistic regression with
`class_weight='balanced'`. The regularisation strength C is chosen from {0.25, 1, 4, 16} by **val** AUROC, separately
for each setting. The chosen C and the val AUROC for every C are in each `run_info.json`. The `paraphrase*`
settings have no training data, so they are scored with the `indist` model. `top_features.csv` lists the 30
n-grams that push most towards "AI" and the 30 that push most towards "human".

**Length-only (`length`).** Logistic regression on just log(characters) and log(words). This is a **shortcut
check**, not a real detector. If it scores high in a setting, part of any detector's score there may come from
text length rather than style.

### Results (fill from `results/tables.md`)

| Setting | `tfidf_lr` AUROC [95 % CI] | `length` AUROC |
|---|---|---|
| in-distribution | __ [__, __] | __ |
| LOGO mean | __ | __ |
| cross-domain mean | __ | __ |
| paraphrase_orig → paraphrase | __ → __ | __ → __ |

**What to write once the numbers are in (check each claim against the tables first):**

- How far `tfidf_lr` drops from in-distribution to LOGO and cross-domain, and which held-out generator or domain is
  hardest.
- Where `length` is clearly above 0.5. On the dev sample it was about 0.55 in-distribution but about 0.79 for
  `logo_bloomz` and `xdomain_reddit`, a sign of a length shortcut there. Confirm this on the full data before
  claiming it.
- The top features: whether they look like style (e.g. "additionally", "overall") or topic or formatting artifacts.

## 5. Reproducibility

Every run writes `run_info.json` with the git commit, seed (42), hyperparameters, hardware, runtime and library
versions. The exact commands are in the README ("How to reproduce").
