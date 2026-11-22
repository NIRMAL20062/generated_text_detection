# Report draft: Dataset, Setup and Baselines (role A)

> Draft for issues #8 and #15. Result numbers are read from `results/<method>/<setting>/metrics.json`
> (`results/tables.md`). All dataset numbers come from `results/data_audit.md`.

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
| in-distribution | 0.990 [0.987, 0.992] | 0.591 |
| LOGO mean | 0.928 | 0.582 |
| cross-domain mean | 0.877 | 0.605 |
| paraphrase_orig → paraphrase | 0.991 → 0.994 | 0.589 → 0.613 |

**Findings** (the full text is in `report/report.tex`):

- TF-IDF + LR is near-perfect in distribution (0.990) and drops on unseen generators (mean 0.928; held-out BLOOMZ
  0.814) and unseen domains (mean 0.877; held-out arXiv 0.825).
- The length-only model is near chance in distribution (0.591) but reaches 0.764 with BLOOMZ held out and 0.768 and
  0.744 on held-out Reddit and wikiHow: a length shortcut in those settings.
- The paraphrase attack does not lower TF-IDF (0.991 → 0.994), but paraphrased *human* texts are scored as more
  machine-like (`paraphrase_human` 0.964): the model reacts to the paraphraser's style.
- The strongest "human" features are formatting remnants of the M4 sources (`references`, `external links`,
  `_url_0_`, `. ,`), so in-distribution scores are likely optimistic. The selected C was 16, the top of the grid, in
  every setting.

## 5. Limitations (state these plainly in the report)

- **DetectGPT.** We ran Fast-DetectGPT (Bao et al., ICLR 2024), the one-forward-pass analytic form of DetectGPT's
  probability curvature, instead of perturbation DetectGPT (reference [1]), which needs about 20 T5 rewrites and
  21 scorer passes per text. The brief asks for "DetectGPT-style scoring", which this satisfies, but the numbers
  are not those of the original method. Fast-DetectGPT scores only the 1,500-id subset plus the paraphrase test
  sets, so its LOGO and cross-domain numbers rest on fewer texts (wider CIs).
- **Scorer model.** The zero-shot scorer is GPT-2 medium, truncated to 256 tokens. None of the M4 generators is
  GPT-2, so the zero-shot methods are black-box here.
- **Scope.** English only, at most 1,000 machine texts per cell, one paraphraser (T5), one seed. Every setting
  that could not be run in the compute budget must appear in the tables as "not run (compute)", never as a blank.
- **RoBERTa** is trained for 1 epoch with max length 256, so long texts are truncated.

## 6. Reproducibility

Every run writes `run_info.json` with the git commit, seed (42), hyperparameters, hardware, runtime and library
versions. The exact commands are in the README ("How to reproduce").
