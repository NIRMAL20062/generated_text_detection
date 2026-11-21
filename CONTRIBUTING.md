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

0. **One-time: verify your phone on Kaggle** (kaggle.com → Settings → Phone verification). Without it, Kaggle
   silently turns **Internet off** even when you switch it on, and `git clone` fails with
   `Could not resolve host: github.com`.
1. **One-time: get access to the data.** Comment your Kaggle username on issue #6. A shares the private dataset
   **`nirmal222/team20-m4-processed`** (one file, `m4_en.parquet`, 44 MB) with you.
2. **New notebook** → Settings: **GPU T4**, **Internet on** → **Add Input** → search `team20-m4-processed`.
   It mounts at `/kaggle/input/datasets/nirmal222/team20-m4-processed/`. The cell below finds it either way.
3. First cells (keep all logic in `src/`, never in the notebook):
   ```python
   !git clone -b main https://github.com/NIRMAL20062/generated_text_detection.git
   %cd generated_text_detection
   !pip install -q -r requirements.txt
   !mkdir -p data && ln -sfn "$(dirname "$(find /kaggle/input -name m4_en.parquet | head -1)")" data/processed
   !ls -l data/processed/                   # must show m4_en.parquet
   !python -m src.data --check-splits      # must print: OK: 13 split files in splits match 38241 texts
   !python -m src.train_roberta --setting indist
   ```
   To test code that isn't merged yet, clone your branch instead of `main` (`-b b-roberta`).

   **If `pip install` fails or breaks torch:** skip that line. `src/data.py`, `src/eval.py` and the baselines
   were tested on Kaggle's preinstalled packages (Python 3.13, pandas 2.3.3, scikit-learn 1.6.1, 21 Nov).
   `run_info.json` records the versions you actually used.
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

## 6. Key dates

| Date | What |
|---|---|
| Sat 8 Nov | A: `data/sample_dev.parquet` + `results/data_audit.md`. Everyone starts coding against the sample |
| Sun 9 Nov | A: full data, splits, `eval.py` on `main`; Kaggle dataset shared |
| Wed 12 Nov | Method sections of the report drafted (each owner) |
| Sun 16 Nov | All results in `results/` |
| Fri 21 Nov | 10-min walkthroughs |
| **Sat 22 Nov, 11:59 pm IST** | Report + repo final |

Full timeline: [docs/ROADMAP.md §7](docs/ROADMAP.md).
