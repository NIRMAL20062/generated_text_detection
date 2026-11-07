# Team 20: Who Codes What

> Proposed assignment. Swap roles freely, but **edit this file when you do** so it stays the source of truth.
> Big picture: [ROADMAP.md](ROADMAP.md). Step-by-step git + Kaggle commands: [CONTRIBUTING.md](../CONTRIBUTING.md).
> Final deadline: **Sat 22 Nov, 11:59 pm IST**.

| Role | Person | Owns | One-line job |
|---|---|---|---|
| **A: Data & Baseline (lead)** | NIRMAL20062 | `src/data.py`, `src/eval.py`, `src/baseline_tfidf.py`, `splits/` (except D's two files) | Build the dataset and splits everyone uses, the shared evaluator, and TF-IDF+LR |
| **B: RoBERTa** | VIVEK342004 | `src/train_roberta.py`, `notebooks/kaggle_roberta.ipynb` | Fine-tune RoBERTa on every setting (GPU) |
| **C: Zero-shot** | RITESHJHA2244 | `src/zeroshot.py`, `notebooks/kaggle_zeroshot.ipynb` | Perplexity, log-rank, and DetectGPT scores (GPU) |
| **D: Attack & Results** | kumar-vishal01 | `src/paraphrase.py`, `src/make_tables.py`, `notebooks/kaggle_paraphrase.ipynb`, `splits/paraphrase.json`, `splits/paraphrase_human.json` | Paraphrase attack, final tables and plots, report assembly |

---

## Day 0 (A, today): bootstrap commit

`main` has no commits yet, so nobody can branch. A first pushes one commit with:
`docs/`, `.gitignore`, `requirements.txt`, an empty `src/__init__.py` (needed for `python -m src.x`),
`splits/.gitkeep`, `results/.gitkeep`, `notebooks/.gitkeep`, and a `README.md` stub. **Then everyone branches.**

**Environment:** Python **3.11** (matches Kaggle). Locally: `uv venv --python 3.11 && uv pip install -r requirements.txt`.
`requirements.txt` pins `transformers`, `datasets`, `scikit-learn`, `pandas`, `pyarrow`, but **not `torch`**
(Kaggle already has it, and reinstalling it wastes quota; install it locally yourself if you need it).

---

## 0. The contract (read this first, all of you)

If everyone follows these formats, nobody waits on anybody else's code.

### 0.1 Data files (produced by A, plus D's paraphrases)
`data/processed/m4_en.parquet`, one row per text:

| column | type | example |
|---|---|---|
| `id` | str | `arxiv_chatGPT_0042_m` (`_h` = human, `_m` = machine) |
| `text` | str | … |
| `label` | int | `0` = human, `1` = machine |
| `domain` | str | `wikipedia` · `wikihow` · `reddit` · `arxiv` · `peerread` |
| `generator` | str | `chatGPT` · `davinci` · `cohere` · `dolly` · `bloomz` (human rows: the generator of their paired file) |
| `group` | str | hash of the normalised **human** text of the pair. Both texts of a pair share it, and so does every copy of the same human text across generator files |

**Grid:** the full 5 domains × 5 generators exists in M4 English. The extra files `arxiv_flant5`,
`reddit_flant5` and `peerread_llama` are **excluded**.

**PeerRead exception:** a PeerRead row is one paper with a *list* of reviews (e.g. 3 human, 4 machine), not a
1:1 pair. Each review becomes its own row (id suffix `_h<k>` / `_m<k>`, e.g. `peerread_chatGPT_0042_m3`), and
`group` = hash of the **paper** (title / source id), so every review of one paper stays in the same split.
`max_per_cell` counts papers for PeerRead. Exact handling and counts are in `results/data_audit.md`.

Duplicate human texts across generator files are **kept once** in the dataset (the dropped copies' machine
twins keep the same `group`). After dedup, a human row's `generator` value is arbitrary. So **LOGO splits and any
`--by generator` breakdown pick human rows by `group` membership** (humans whose group contains that generator's
machine text), never by the human row's `generator` field.

`data/processed/paraphrased.parquet` (made by D): same columns, ids end in `_para`, plus an `orig_id` column.

**Everyone loads text through one function:** `src.data.load_texts()` returns both files concatenated.
`eval.py`, the detectors and `make_tables.py` all use it, so the `_para` rows have labels everywhere.

**Class balance:** dedup keeps each human text once but every machine text, so there can be up to about
5 machine texts per human text. A reports the class ratio per split in `data_audit.md`. Trained models
compensate: LR uses `class_weight='balanced'`, RoBERTa uses class-weighted cross-entropy. AUROC (our main
metric) isn't affected by the ratio.

### 0.2 Split files (committed to git)
`splits/<setting>.json` → `{"train": [ids], "val": [ids], "test": [ids]}`

**Splits are made by `group`, never by row.** A group never spans two of train/val/test.
For `logo_<gen>`: groups that contain a `<gen>` machine text go to test, keeping **only** their human row(s) and
`<gen>` machine row(s). The other generators' machine rows in those groups are **dropped** (never moved to train),
so test holds only the unseen generator and no human text seen in training reappears.
`xdomain_<dom>` works the same way, but by domain (groups never cross domains, so nothing is dropped).

Setting names (use these exact strings everywhere):

| setting | made by | meaning |
|---|---|---|
| `indist` | A | 80/10/10 by group, **stratified by domain × generator** so small cells (e.g. dolly, bloomz) appear in val and test |
| `logo_<gen>` ×5 | A | test = only `<gen>` (+ humans of its groups); train/val = the other 4 |
| `xdomain_<dom>` ×5 | A | test = only `<dom>`; train/val = the other 4 domains |
| `paraphrase_orig` | A | val = `indist` val; test = 500 machine + 500 human texts from the `indist` test (balanced over domain × generator), **original** texts. This is the fair "before" number |
| `paraphrase` | D | same val and same 500 humans; the 500 machine texts replaced by their `_para` versions. This is the "after" number |
| `paraphrase_human` | D | *(optional control)* the same 500 machine originals + ~200 of the humans replaced by paraphrased versions (label stays 0) |
| `detectgpt_subset.json` | A | not a setting: a list of ~1,500 ids drawn from **`indist` val + test**, balanced over domain × generator × label (seed 42) |

Train-free methods (zero-shot) and already-trained models are evaluated on `paraphrase*` using the `indist` model.

### 0.3 Prediction files (produced by A, B, C, read by D)
`results/<method>/<setting>/preds.csv`:

```
id,split,score
arxiv_chatGPT_0042_m,test,0.93
```
- `split` ∈ `val`, `test`. **Always include val rows**: the evaluator picks the threshold on val.
- `score`: **higher = more likely AI**. (For zero-shot, flip the sign if needed so this holds.)
- `<method>` names: `length`, `tfidf_lr`, `roberta`, `ppl`, `logrank`, `detectgpt` (or `fastdetectgpt` if we fall back).

Next to it, every method writes `results/<method>/<setting>/run_info.json`:
`git_commit` (`git rev-parse HEAD`), `seed`, key hyperparameters, runtime, and hardware (CPU or GPU type).

Then anyone runs `python -m src.eval results/<method>/<setting>/preds.csv` → writes `metrics.json`
(AUROC with a bootstrap 95% CI, accuracy, macro-F1, and TPR@1%FPR **only if the test set has ≥1,000 human texts**,
otherwise `null`). **Nobody writes their own metric code.**

### 0.4 Running things: `--dev` flag, Kaggle data, getting results back
- **`--dev` on every script:** runs on `data/sample_dev.parquet` and finishes in **under 2 minutes on CPU**.
  Test your code with `--dev` before spending GPU quota. PR reviewers run it too.
- **Data on Kaggle:** `data/` is gitignored, so A uploads `data/processed/*.parquet` as a **private Kaggle
  Dataset `team20-m4-processed`**, shared with the other three. Notebooks attach that dataset, link it to
  `data/processed/`, and `git clone` the repo for the code and `splits/`. The first cell asserts that every id in
  `splits/*.json` exists in the data. D re-uploads a new dataset version after adding `paraphrased.parquet`.
- **The repo is public**, so `git clone` on Kaggle needs no token. If we ever make it private, store a GitHub
  token in Kaggle Secrets.
- **Getting results back:** download `results/<method>/<setting>/` from the notebook's Output tab, copy it into
  your local clone, commit it **from your own machine**, and open a PR. Model weights stay in Kaggle output or
  Drive.

### 0.5 Git rules
- Your own branch: `a-data`, `b-roberta`, `c-zeroshot`, `d-attack`. Open a PR into `main`, and someone else approves it.
- **Commit and open PRs from your own GitHub account.** The repo is graded and the report has a contributions
  section, so don't send code to someone else to push.
- Only edit files you own. To change the contract or a shared file, ask in the group first.
- **Never commit** `data/` or model weights (they're in `.gitignore`). The **one exception** is the small
  `data/sample_dev.parquet` (~500 rows), which is committed so everyone can run `--dev` right after `git pull`.
  **Do commit** `splits/` and `results/**/*.csv|json` (but not `results/zeroshot_cache/`, which is gitignored).
- Fixed seed everywhere: `SEED = 42`. No `configs/` folder: CLI arguments + `run_info.json` record everything.

---

## A: Data & Baseline · NIRMAL20062

Everyone depends on you first, so **do the bootstrap commit today, ship a dev sample by 8 Nov**, and the full data by 9 Nov.

**Code**
- [ ] `src/data.py`
  - `download_m4(out_dir)`: fetch only the English JSONL files (disk is tight)
  - `audit(raw_dir)`: **do this first.** Print and save `results/data_audit.md`: rows per (domain, generator), any missing cells, extra generators (e.g. GPT-4, which we exclude), **how many human texts appear in more than one generator file**, and (after splitting) the **class ratio per split**. Share it in the group chat; it decides the final grid.
  - `build_dataset(max_per_cell=1000) -> DataFrame`: flatten human/machine pairs into the §0.1 schema, add the `group` hash, dedupe human texts, drop empty or very short texts (<50 chars), and subsample per (domain, generator)
  - `make_splits(df)`: write every A-owned file in §0.2 (`indist`, `logo_*`, `xdomain_*`, `paraphrase_orig`, `detectgpt_subset.json`), **splitting by `group`**. Add an assert that no group appears in two parts of any split.
  - `load_texts()`: the shared loader from §0.1
  - `make_dev_sample()`: write `data/sample_dev.parquet` (~500 rows, all domains and generators)
  - CLI: `python -m src.data --all`
- [ ] Upload `data/processed/` as the Kaggle Dataset `team20-m4-processed` and share it with B, C and D (§0.4)
- [ ] `src/eval.py`: reads preds.csv + `load_texts()`, picks the threshold on val (max F1), reports test metrics → `metrics.json`: AUROC + bootstrap 95% CI (1,000 resamples), accuracy, macro-F1, TPR@1%FPR (only if ≥1,000 test humans). Include a `--by domain|generator` breakdown.
- [ ] `src/baseline_tfidf.py`: word 1–2-grams + char 3–5-grams, `LogisticRegression(class_weight='balanced')`, run on **every** setting, write preds.csv + run_info.json. Also save the top 30 "AI" and top 30 "human" features (good report material).
  - Same script, `--method length`: logistic regression on text length only (characters + words). This is the shortcut sanity check.
- [ ] `requirements.txt`, `.gitignore`, README "how to reproduce" section

**Deadlines**
| By | Deliverable |
|---|---|
| **Fri 7 Nov** | Day-0 bootstrap commit on `main` |
| Sat 8 Nov | `results/data_audit.md` + `data/sample_dev.parquet` shared → B, C, D start coding against it |
| Sun 9 Nov | full dataset + all A-owned split files + `load_texts()` + `eval.py` on `main`; Kaggle Dataset shared |
| Wed 12 Nov | TF-IDF + length results for `indist`, `logo_*`, `xdomain_*`, `paraphrase_orig` |
| Fri 14 Nov | TF-IDF + length on `paraphrase` (and `paraphrase_human`) once D delivers them on 13 Nov |

**Report sections:** Dataset, Experimental setup, Baseline results

---

## B: RoBERTa · VIVEK342004

**Code**
- [ ] `src/train_roberta.py --setting <name> [--dev]`
  - `roberta-base`, max_len 256, batch 16, lr 2e-5, 2 epochs (1 if time is short), fp16
  - **class-weighted cross-entropy** (weights from the train split's class ratio)
  - early stopping or best checkpoint chosen on val AUROC
  - write `results/roberta/<setting>/preds.csv` (val + test, score = softmax P(machine)) + `run_info.json`
  - `--predict-only --model <path>`: score a setting with an already-trained model (for the `paraphrase*` settings)
- [ ] `notebooks/kaggle_roberta.ipynb`: attach the Kaggle Dataset, `git clone`, `pip install -r requirements.txt`, then a loop over settings calling the script. **No logic inside the notebook.**
- [ ] Save model weights to Kaggle output / Google Drive only (not git). **Keep the `indist` model for your own `paraphrase*` runs.**

**Run order (11 training runs + 3 prediction-only runs; stop early if GPU quota runs out)**
1. `indist` ← do this first, by **Wed 12 Nov**. Right after it, score `paraphrase_orig` with the same model.
2. `logo_*` ×5 ← most important for our research question, by **Fri 14 Nov**
3. `xdomain_*` ×5 ← by **Sat 15 Nov**
4. `paraphrase` (+ `paraphrase_human`): `--predict-only` with the `indist` model, by **Sun 16 Nov**

**Report sections:** RoBERTa method, cross-generator results discussion

---

## C: Zero-shot · RITESHJHA2244

Zero-shot methods don't train, so **score each text once**, cache the scores, then write one preds.csv per
setting by selecting the ids from that setting's split file.

**Code**
- [ ] `src/zeroshot.py --method ppl|logrank|detectgpt [--dev]`
  - scorer LM: `gpt2-medium` (fall back to `gpt2` if it's slow), truncate texts to 256 tokens
  - `ppl`: mean token log-likelihood (score = log-lik, because higher log-lik means more AI-like)
  - `logrank`: mean log-rank of the actual tokens (score = −logrank)
  - `detectgpt`: perturb with `t5-large` mask-filling (span length 2, ~15% of words masked),
    n=20 perturbations; score = (logp(x) − mean logp(x̃)) / std (the normalised version)
  - cache scores to `results/zeroshot_cache/<method>.parquet` (`id, score`) so a Kaggle timeout doesn't lose progress (skip ids that are already cached)
  - then write `results/<method>/<setting>/preds.csv` + `run_info.json` for every setting
- [ ] `notebooks/kaggle_zeroshot.ipynb`: same rule as B, so the notebook only calls the scripts
- [ ] **What to score:**
  - `ppl`, `logrank`: every text in `load_texts()` (cheap)
  - `detectgpt`: only the ids in `splits/detectgpt_subset.json` (made by A from `indist` val + test, so `indist`
    gets about 750 test texts and each LOGO/xdomain test gets roughly a fifth of the subsample), **plus** all ids in
    the `paraphrase_orig`, `paraphrase` and `paraphrase_human` test sets. Record the time per text (reporting the
    compute cost is itself a result).
- [ ] Fallback: if DetectGPT can't finish by 14 Nov, implement `--method fastdetectgpt` (Fast-DetectGPT, one forward pass) instead.

**Deadlines**
| By | Deliverable |
|---|---|
| Wed 12 Nov | `ppl` + `logrank` for every setting except the D-made `paraphrase*` ones |
| Fri 14 Nov | `detectgpt` on the subset + `paraphrase_orig` → preds for those settings |
| Sun 16 Nov | all three methods on `paraphrase` (+ `paraphrase_human`) |

**Report sections:** Zero-shot methods (explain probability curvature), compute-cost comparison

---

## D: Attack & Results · kumar-vishal01

**Code**
- [ ] `src/paraphrase.py [--dev]`
  - paraphrase the **500 machine texts in the `paraphrase_orig` test set** (A picks them, so the "before" and
    "after" use exactly the same texts)
  - model: start with `humarin/chatgpt_paraphraser_on_T5_base`; try DIPPER only if GPU memory allows.
    Paraphrase sentence by sentence so long texts don't get truncated
  - write `data/processed/paraphrased.parquet` (§0.1 schema, id suffix `_para`, plus `orig_id`), then upload a new version of the Kaggle Dataset
  - write `splits/paraphrase.json` (§0.2: same val and humans as `paraphrase_orig`, machines swapped for `_para`)
  - sanity check: report BLEU or BERTScore vs the original, so we can show the meaning was kept
  - *optional control:* paraphrase ~200 of those **human** texts too (label stays 0) → `splits/paraphrase_human.json`.
    If detectors flag these as AI, they're detecting "paraphraser style" rather than "LLM-written".
- [ ] `src/make_tables.py`: reads every `results/*/*/metrics.json` →
  - Table 1: in-dist AUROC per method (± CI)
  - Table 2: LOGO AUROC (method × held-out generator) + average
  - Table 3: cross-domain AUROC (method × held-out domain)
  - Table 4: paraphrase AUROC **`paraphrase_orig` → `paraphrase`**, with the drop (+ the human control if done)
  - Fig 1: grouped bar chart (method × setting); Fig 2: LOGO heatmap → saved to `results/figures/`
  - write `make_tables.py` against **fake metrics.json files** first, so it's ready before the real results arrive
- [ ] Own the report document: set up the template once the instructor shares the format, and merge everyone's sections

**Deadlines**
| By | Deliverable |
|---|---|
| Wed 12 Nov | `make_tables.py` working on dummy metrics |
| Thu 13 Nov | `paraphrased.parquet` (Kaggle Dataset updated) + `splits/paraphrase.json` (+ `paraphrase_human.json`) on `main` → A, B, C score them |
| Wed 19 Nov | final tables + figures from the real results |

**Report sections:** Paraphrase attack, Results (all tables and figures), Discussion, Conclusion (written with everyone)

---

## Dependency map (who waits on whom)

```
A: bootstrap commit (7 Nov) ──► everyone can branch
A: data audit + dev sample (8 Nov) ──► B, C, D start coding, testing with --dev
A: full data + splits + eval.py + Kaggle Dataset (9 Nov) ──► B, C run real experiments; D starts paraphrasing
D: paraphrased set + split files (13 Nov) ──► A, B, C score `paraphrase` / `paraphrase_human`
A, B, C: preds + metrics (≤16 Nov) ──► D builds final tables (19 Nov)
everyone: report sections (≤16 Nov drafts, 20 Nov final) ──► D merges → final submission (22 Nov)
```

## Report: write as you go
Don't leave writing to 19–22 Nov. Each owner drafts their **Method** section by **Wed 12 Nov** (it doesn't
need results) and adds their **Results** paragraph as soon as their numbers land (by **Sun 16 Nov**).
D merges everything into the final document from 16 Nov.

## Daily sync (5 min, group chat is fine)
1. What I pushed yesterday
2. What I'm doing today
3. Am I blocked? On whom?

## Before the viva (by 21 Nov)
Each person gives a **10-minute walkthrough** of their part to the other three.
The examiners can ask anyone about any method.
