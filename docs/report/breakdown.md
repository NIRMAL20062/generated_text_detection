# Per-domain and per-generator breakdown (`indist`, issue #28)

Test AUROC of three detectors on the in-distribution test set, split by domain and by generator. Every number is
copied from `by_domain` / `by_generator` in `results/<method>/indist/metrics.json` (written by `src.eval`), rounded
to 3 decimals. The "all" column is the overall `indist` AUROC from Table 1.

How the evaluator splits (see `breakdown()` in `src/eval.py`):
- **by domain:** machine and human test texts of that domain.
- **by generator:** that generator's machine texts against the human texts of the *same groups* (the humans its
  texts were written for). A human text has no generator of its own, so the humans differ per column and the
  columns are not a partition of the test set.

## Method × domain

| method | wikipedia | wikihow | reddit | arxiv | peerread | all |
|---|---|---|---|---|---|---|
| tfidf_lr | 0.996 | 0.999 | 0.994 | 0.946 | 0.993 | 0.990 |
| logrank | 0.873 | 0.817 | 0.922 | 0.809 | 0.958 | 0.857 |
| roberta | 0.988 | 0.998 | 0.998 | 0.969 | 0.997 | 0.991 |
| *n test (machine / human)* | 500 / 293 | 501 / 260 | 497 / 259 | 501 / 260 | 500 / 267 | 2499 / 1339 |

## Method × generator

| method | chatGPT | davinci | cohere | dolly | bloomz | all |
|---|---|---|---|---|---|---|
| tfidf_lr | 0.998 | 0.990 | 0.993 | 0.977 | 0.992 | 0.990 |
| logrank | 0.963 | 0.820 | 0.945 | 0.822 | 0.781 | 0.857 |
| roberta | 0.999 | 0.985 | 0.996 | 0.982 | 0.993 | 0.991 |
| *n test (machine / human)* | 499 / 508 | 505 / 521 | 498 / 523 | 497 / 519 | 500 / 527 | 2499 / 1339 |

## What stands out

**arXiv is the hardest domain for every detector** (TF-IDF 0.946, RoBERTa 0.969, log-rank 0.809), the only domain
where the supervised detectors fall below 0.98; PeerRead is the easiest for log-rank (0.958) and wikiHow for the
supervised ones (0.999, 0.998). **Log-rank's in-distribution weakness is generator-specific:** it is strong on ChatGPT (0.963)
and Cohere (0.945) but near 0.8 on davinci, Dolly and BLOOMZ, whereas the supervised detectors stay above 0.977 on
every generator once they have seen it in training. BLOOMZ is easy in distribution for TF-IDF and RoBERTa (0.992,
0.993) but is the hardest *unseen* generator for both (0.814, 0.881 in Table 2), so their in-distribution success on
BLOOMZ comes from having seen its style, not from a property that transfers.
