# How we work (Team 20)

This page covers the **steps**: git, Kaggle, PRs. The **rules about formats** (dataset columns, split files,
`preds.csv`) are in [docs/TASKS.md §0](docs/TASKS.md). Read that first.

## 1. One-time setup (each member)

1. Accept the GitHub collaborator invite (check your email or github.com/notifications).
2. Clone and install:
   ```bash
   git clone https://github.com/NIRMAL20062/generated_text_detection.git
   cd generated_text_detection
   uv venv --python 3.11 && source .venv/bin/activate
   uv pip install -r requirements.txt        # add torch yourself if you run models locally
   ```
3. Make sure your commits show your own name. The repo is graded on who did what:
   ```bash
   git config user.name  "Your Name"
   git config user.email "the-email-on-your-github-account"
   ```
4. Create your branch **once**:

   | Role | Branch |
   |---|---|
   | A: NIRMAL20062 | `a-data` |
   | B: VIVEK342004 | `b-roberta` |
   | C: RITESHJHA2244 | `c-zeroshot` |
   | D: kumar-vishal01 | `d-attack` |

   ```bash
   git switch -c b-roberta      # use your own branch name
   git push -u origin b-roberta
   ```

## 2. Every working day

```bash
git switch b-roberta
git pull origin main             # get everyone else's merged work first
# ... code, only in files you own (TASKS.md role table) ...
python -m src.train_roberta --dev --setting indist   # your script with --dev: must finish in < 2 min on CPU
git add <your files>
git commit -m "roberta: add class-weighted loss"
git push
```

- **Small commits, clear messages:** start with your area (`data:`, `roberta:`, `zeroshot:`, `attack:`,
  `tables:`, `docs:`).
- **Never `git add -A` blindly.** Check `git status` first. `data/` and model weights must never be committed
  (`.gitignore` blocks most of it; `data/sample_dev.parquet` is the only exception).
- Post the 3-line daily sync in the group chat: pushed yesterday / doing today / blocked on whom.

## 3. Getting your work into `main` (pull request)

1. Push your branch, then open a PR into `main`:
   ```bash
   gh pr create --base main --fill      # or use the "Compare & pull request" button on GitHub
   ```
2. Fill in the checklist the PR template gives you.
3. **Someone else** reviews it. The reviewer pulls the branch, runs the script with `--dev`, and approves.
   ```bash
   gh pr checkout <number> && python -m src.<script> --dev
   ```
4. The author merges after approval (use **Squash and merge**). Nobody pushes straight to `main`.
5. Keep PRs small, one feature each, so they get reviewed the same day.

**Need to change a shared file or the contract** (`src/data.py` loader, `src/eval.py`, `docs/TASKS.md §0`)?
Ask in the group chat first. The owner makes the change.

## 4. Running on Kaggle (GPU jobs: B, C, D)

1. **New notebook** → Settings: **GPU T4**, **Internet on**.
2. **Add data** → our private dataset **`team20-m4-processed`** (A shares it).
3. First cells (keep all logic in `src/`, never in the notebook):
   ```python
   !git clone -b main https://github.com/NIRMAL20062/generated_text_detection.git
   %cd generated_text_detection
   !pip install -q -r requirements.txt
   !mkdir -p data && ln -s /kaggle/input/team20-m4-processed data/processed
   !python -m src.data --check-splits      # asserts every split id exists in the data
   !python -m src.train_roberta --setting indist
   ```
   To test code that isn't merged yet, clone your branch instead of `main` (`-b b-roberta`).
4. **Bring results back:** download `results/<method>/<setting>/` from the notebook's **Output** tab, copy it
   into your local clone, then commit and open a PR from your own machine.
   Model weights stay in Kaggle Output or Google Drive, never in git.
5. Check your GPU quota (about 30 h/week per account). Tell the group if you're running low, so someone else can
   take a run.

## 5. Evaluating

Nobody writes metric code. After your `preds.csv` exists:
```bash
python -m src.eval results/<method>/<setting>/preds.csv          # writes metrics.json next to it
python -m src.eval results/<method>/<setting>/preds.csv --by generator
```
Commit `preds.csv`, `metrics.json` and `run_info.json` together.

## 6. `.py` or notebook? Both, with different jobs

| | `src/*.py` | `notebooks/*.ipynb` |
|---|---|---|
| **Job** | **does** the work: all logic, training, scoring | **runs and shows** the work: GPU runs, plots, demo |
| **Why** | importable, testable, clean PR diffs, what examiners read | Kaggle/Colab need them; good for figures and explaining |
| **Rule** | every function lives here | no functions or logic: only `!python -m src.<script>` calls and plotting |

Notebooks hide state (cells run out of order) and make unreadable git diffs, so logic inside one can't be
reviewed or tested. A notebook should still work if you "Restart & Run All".

**Where to run what** (local laptop has no GPU and ~6 GB RAM):

| Runs fine locally on CPU | Needs a GPU (Kaggle, or Colab as backup) |
|---|---|
| `data.py`, `eval.py`, TF-IDF + length baselines, `make_tables.py`, every `--dev` run | RoBERTa training, perplexity / DetectGPT scoring on full data, paraphrasing |

Kaggle and Colab run the same notebook code, so if one is down or out of quota, switch to the other.

## 7. Testing your code

Run all tests from the repo root before every PR (takes seconds, no GPU or full data needed):
```bash
python -m pytest tests/ -q
```
Write `tests/test_<your_script>.py` for your own file with the same three kinds of test as
[tests/test_eval.py](tests/test_eval.py):

1. **Known answers:** inputs where you already know the right output.
   e.g. a perfect detector must get AUROC 1.0; a hand-worked example must give the hand-worked result.
2. **Bad inputs fail loudly:** a broken file or wrong setting must raise an error, never give a quiet wrong number.
3. **End to end on the dev sample:** call your script's `main([... "--dev"])` and check the output files exist
   and follow the contract (`id,split,score`, val + test rows, ids from the right split).

Ideas per role:
- **Baseline (A):** length-only scores the same as `text.str.len()` ranking; TF-IDF beats AUROC 0.5 on dev.
- **RoBERTa (B):** train on ~20 dev texts for a few steps: training loss must go down (proves the loop learns);
  `preds.csv` scores are probabilities in [0, 1].
- **Zero-shot (C):** a repeated, very predictable text gets a higher log-likelihood than shuffled words;
  the cache skips ids already scored.
- **Attack (D):** paraphrased ids end in `_para`, keep `orig_id`, whitespace is collapsed;
  `make_tables.py` builds every table from fake `metrics.json` files.

Before spending GPU time, always run your script with `--dev` locally first.

## 8. Comments: write for a teammate who has never seen your file

The examiners can ask any of us about any file, so the code has to teach itself.
- Every file starts with a docstring: what it does, how to run it, its input and output.
- Every function has a docstring saying what it takes, what it returns and **why** it exists.
- Explain the idea behind a method in plain words where it's used (what AUROC means, why the threshold is
  picked on val, what probability curvature is). A small worked example is even better.
- Comment the **why**, not the what: `# skip: AUROC is undefined with one class` beats `# if statement`.
- See [src/eval.py](src/eval.py) for the style.

## 9. Key dates

| Date | What |
|---|---|
| Sat 8 Nov | A: `data/sample_dev.parquet` + `results/data_audit.md`. Everyone starts coding against the sample |
| Sun 9 Nov | A: full data, splits, `eval.py` on `main`; Kaggle dataset shared |
| Wed 12 Nov | Method sections of the report drafted (each owner) |
| Sun 16 Nov | All results in `results/` |
| Fri 21 Nov | 10-min walkthroughs |
| **Sat 22 Nov, 11:59 pm IST** | Report + repo final |

Full timeline: [docs/ROADMAP.md §7](docs/ROADMAP.md).
