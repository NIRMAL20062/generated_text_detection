## What this PR does
<!-- One or two sentences. Link the TASKS.md item, e.g. "A: make_splits()" -->

## How I tested it
<!-- The exact command(s), e.g. `python -m src.baseline_tfidf --dev --setting indist` -->

## Checklist
- [ ] I only changed files my role owns (see the role table in docs/TASKS.md), or the group agreed to the change
- [ ] The script runs with `--dev` in under 2 minutes on CPU
- [ ] No `data/` files (except `data/sample_dev.parquet`) or model weights are committed
- [ ] Output follows the contract in docs/TASKS.md §0 (`preds.csv` columns, setting names, score = higher means AI)
- [ ] For results: `preds.csv` + `metrics.json` (from `src.eval`) + `run_info.json` are committed together
- [ ] Commits are from my own GitHub account

## Reviewer
<!-- Reviewer: run `gh pr checkout <n>` and the --dev command above, then approve -->
