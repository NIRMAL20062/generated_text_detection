# Report draft: Zero-shot methods (role C)

> Draft for issue #15. All numbers below are read from `results/<method>/<setting>/metrics.json`
> and `run_info.json` after the Kaggle run. Do not report a number that is not in those files.

## 1. Zero-shot detection: the idea

A zero-shot detector needs no training data and no labels. It asks a language model (the *scorer*) how it
"feels" about a text, and turns that into one number. We use GPT-2 medium as the scorer, truncate every text to
256 tokens, and define the score so that **higher means more likely machine-written** for every method.

Why this can work: a language model generates text by picking likely next tokens. Text written by a model
therefore tends to look likely *to a model*, while human text contains more surprising word choices.

| Method | Score | Intuition |
|---|---|---|
| **Perplexity (`ppl`)** | mean log-likelihood of the text's tokens under the scorer | Machine text is more probable, so its log-likelihood is higher (less negative). |
| **Log-rank (`logrank`)** | minus the mean log of each token's rank among the scorer's predictions (rank 1 = its top choice) | Machine text uses top-ranked tokens more often, so its log-rank is smaller and the score is higher. Ranks are less sensitive to a few extreme probabilities than raw likelihood. |
| **DetectGPT (full)** | log p(x) compared with the log p of many slightly rewritten copies of x | See below. Not run: too slow for our budget. |
| **Fast-DetectGPT (`fastdetectgpt`)** | analytic version of the same idea, one forward pass per text | See below. This is the fallback we ran. |

## 2. Probability curvature (DetectGPT and Fast-DetectGPT)

DetectGPT (Mitchell et al., ICML 2023) is built on one observation: text sampled from a language model sits near
a *local peak* of that model's log-probability. If you make small rewrites of a machine text, almost every
rewrite becomes **less** likely, because the original was already near the top. For a human text, the rewrites
are about as likely as the original, or more, because the text was not sitting on a peak.

So the detector measures the *curvature* of the log-probability around the text:

    score = log p(x) - average of log p(x') over rewrites x' of x   (normalised by their spread)

A large positive score means "x is a sharp peak", which suggests machine text. The cost is that every text needs
N rewrites (the paper uses T5 mask-filling, we planned N = 20) and N+1 forward passes through the scorer.

Fast-DetectGPT (Bao et al., ICLR 2024) removes the rewrites. Instead of sampling rewrites of the whole text, it
looks at each token position and asks what the model *expected* to see there. At position t the model gives a
distribution over the next token. From it we can compute, in closed form:

- the expected log-probability E_t of a token drawn from that distribution, and
- its variance V_t.

Then:

    score = ( sum_t log p(x_t)  -  sum_t E_t ) / sqrt( sum_t V_t )

Read it as: "how much more likely are the actual tokens than the tokens the model itself would have picked,
measured in standard deviations?" This needs a single forward pass, which is why it is far cheaper. We use the
same model (GPT-2 medium) as both sampler and scorer, which is also the main (white-box) setting of the Fast-DetectGPT paper; the paper's black-box variant uses a separate sampling model.

Our implementation is checked against a plain-numpy reference in `tests/test_zeroshot.py`.

## 3. What was scored

- `ppl`, `logrank`: every val and test text of every setting.
- `fastdetectgpt`: only the fixed DetectGPT subset (`splits/detectgpt_subset.json`, 1,500 texts) plus the
  test ids of the paraphrase settings. LOGO and cross-domain test sets therefore contain only the part of the
  subset that falls inside them, so their confidence intervals are wider. AUROC with its bootstrap CI is the
  headline metric for this method; threshold metrics are reported only where val is large enough
  (see `src/eval.py`).
- Zero-shot methods have no training step. For `paraphrase` and `paraphrase_orig` nothing is retrained.

## 4. Results (fill from metrics.json)

| Setting | ppl AUROC | logrank AUROC | Fast-DetectGPT AUROC |
|---|---|---|---|
| `indist` | 0.845 [0.833, 0.856] | 0.857 [0.846, 0.868] | 0.789 [0.761, 0.819] |
| `paraphrase_orig` | 0.846 [0.822, 0.870] | 0.860 [0.836, 0.882] | 0.789 [0.761, 0.819] |
| `paraphrase` | 0.683 [0.648, 0.717] | 0.699 [0.665, 0.732] | 0.743 [0.714, 0.773] |
| LOGO (mean of 5) | 0.860 | 0.870 | 0.797 |
| cross-domain (mean of 5) | 0.874 | 0.885 | 0.798 |

Observations to write once the numbers exist (delete any that the numbers do not support):
- Which method generalises best to an unseen generator, and which generator is hardest?
- Does the paraphrase attack lower the AUROC of each method, and by how much (`paraphrase_orig` -> `paraphrase`)?
- Do zero-shot methods drop less than trained detectors when the generator or domain changes, since they never
  saw any generator?

## 5. Compute cost

Take these from `run_info.json` (`seconds_per_text`, `hardware`, `total_runtime_seconds`):

| Method | Forward passes per text | Seconds per text | Hardware | Texts scored |
|---|---|---|---|---|
| `ppl` | 1 | 0.0154 | Tesla T4 | 38,941 |
| `logrank` | 1 | 0.0158 | Tesla T4 | 38,941 |
| `fastdetectgpt` | 1 (plus the extra softmax work) | 0.0188 | Tesla T4 | 2,200 |
| full DetectGPT (not run) | 21 for N = 20 rewrites, plus a T5 pass per rewrite | not measured | n/a | n/a |

State plainly that full DetectGPT was not run, and why: it needs about N times the compute of `ppl` plus T5
rewrites, which did not fit the GPU quota and deadline, so we ran the Fast-DetectGPT fallback that the project
plan allows. Do not quote a measured cost for full DetectGPT.

## 6. Caveats to state

- One scorer model (GPT-2 medium) for everything. A stronger scorer could change the ranking of methods.
- Texts are truncated to 256 tokens, so long arXiv and PeerRead texts are judged on their opening part only.
- `fastdetectgpt` uses one model as sampler and scorer (the paper's white-box setting); a stronger or different sampling model could change its numbers.
- Subset evaluation gives wider confidence intervals than the full-set `ppl` and `logrank` numbers; do not compare
  the AUROC values across methods without looking at the CIs.

## References
1. E. Mitchell et al., DetectGPT, ICML 2023. https://arxiv.org/abs/2301.11305
2. G. Bao et al., Fast-DetectGPT, ICLR 2024. https://arxiv.org/abs/2310.05130
