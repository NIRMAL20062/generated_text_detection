# Report draft: RoBERTa detector (role B)

> Draft for issue #15. The method text below describes what `src/train_roberta.py` actually does. Numbers marked
> `__` are blanks: fill them from `results/roberta/<setting>/metrics.json` and `run_info.json` after the Kaggle
> run. Do not report a number that is not in those files.

## 1. Method

The supervised detector fine-tunes a pretrained language model to classify a text as human (0) or machine (1).
We use `roberta-base` with a two-class classification head on top of its first-token representation. Unlike the
zero-shot methods, it learns from labelled examples, so it can pick up generator-specific habits. That is exactly
what the leave-one-generator-out (LOGO) and cross-domain settings are designed to test.

| Choice | Value | Why |
|---|---|---|
| Model | `roberta-base` | Standard strong encoder for text classification, fits one T4 GPU |
| Input length | 256 tokens (truncated) | Same limit as the zero-shot scorer; keeps training time within the GPU quota |
| Batch size / learning rate | 16 / 2e-5 | Usual fine-tuning values for RoBERTa |
| Optimiser and schedule | AdamW (weight decay 0.01), linear decay with 6% warm-up, gradient clipping at 1.0 | Standard recipe, stable for a short run |
| Epochs | 1 | Time is limited; one pass over the training texts (30,602 for `indist`, about 19,000 for each LOGO run, `results/data_audit.md`) |
| Precision | fp16 autocast on the GPU only | Faster and lighter on memory |
| Loss | cross-entropy with class weights `n / (2 * n_c)` from the train split | Deduplication keeps every machine text but each human text once, so machine texts outnumber humans; the weights stop the model from just answering "machine" |
| Checkpoint | the weights with the best validation AUROC | Val AUROC is computed twice per epoch and the best weights are restored at the end. The test set is never used to choose anything |
| Seed | 42 everywhere | Reproducibility |

The score written to `preds.csv` is the softmax probability of the machine class, so higher means more likely AI,
as in every other method. The shared evaluator then picks a decision threshold on the validation split and
reports AUROC with a bootstrap 95% confidence interval on the test split.

## 2. Settings

- `indist`: train on 80% of the groups, test on 10% held out by group, so a human text and its machine twin never
  sit on both sides of the split.
- `logo_<generator>` (five runs): the model is trained on four generators and tested on the fifth, which it has
  never seen. The held-out generator's texts, and the human texts they were paired with, are kept out of training.
- `xdomain_<domain>` (five runs): trained on four domains, tested on the fifth.
- `paraphrase_orig`, `paraphrase`, `paraphrase_human`: no retraining. The `indist` model is reused with
  `--predict-only`, so the only thing that changes between `paraphrase_orig` and `paraphrase` is that the 500
  machine texts are replaced by their paraphrases. That makes the drop in AUROC attributable to the paraphrase.

## 3. Results (fill from metrics.json)

| Setting | AUROC (95% CI) |
|---|---|
| `indist` | __ |
| `paraphrase_orig` | __ |
| `paraphrase` | __ |
| `paraphrase_human` | __ |

| Held-out generator (LOGO) | AUROC | Held-out domain | AUROC |
|---|---|---|---|
| chatGPT | __ | wikipedia | __ |
| davinci | __ | wikihow | __ |
| cohere | __ | reddit | __ |
| dolly | __ | arxiv | __ |
| bloomz | __ | peerread | __ |

Points to discuss once the numbers exist (delete any that the numbers do not support):
- How much does AUROC fall from `indist` to the held-out generators? Which generator is hardest, and is the drop
  larger than for the zero-shot methods, which never trained on any generator?
- Does cross-domain transfer fail more than cross-generator transfer, or the other way round?
- How much does the paraphrase attack reduce AUROC (`paraphrase_orig` -> `paraphrase`)? Does `paraphrase_human`
  show that the model flags paraphrased human text as machine, which would mean it partly detects "paraphraser
  style" and not "LLM-written"?

## 4. Compute cost

Take these from `run_info.json` (`train_seconds`, `seconds_per_text_inference`, `hardware`):

| Run | Training time | Inference per text | Hardware |
|---|---|---|---|
| `indist` | __ | __ | __ |
| `logo_*` (mean of 5) | __ | __ | __ |
| `xdomain_*` (mean of 5) | __ | __ | __ |

For the cost comparison with the zero-shot methods: RoBERTa pays a training cost once per setting (about __ minutes
on a T4), after which scoring is as cheap as one forward pass per text, similar to `ppl` and `logrank`.

## 5. Caveats to state

- One epoch and one seed per setting: the confidence intervals come from bootstrapping the test set, not from
  repeating training, so they do not capture training variance.
- Texts are truncated to 256 tokens, so long arXiv and PeerRead texts are judged on their opening part only.
- Hyperparameters were not tuned; the values above are common defaults.
- LOGO and cross-domain test sets contain fewer texts than `indist`, so their intervals are wider.
- The `paraphrase*` settings reuse the `indist` model and score only the 1,000-text test sets of those settings.
