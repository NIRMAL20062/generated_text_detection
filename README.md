# Detecting AI-Generated Text Across Domains and Generators (P20)

Deep Learning course project, Team 20. We detect whether a text is human-written or machine-generated, and test
whether detectors still work on **unseen generators**, **unseen domains**, and **paraphrased** AI text, using the
[M4](https://github.com/mbzuai-nlp/M4) benchmark (English).

**Detectors:** TF-IDF + logistic regression (baseline) → fine-tuned RoBERTa → zero-shot perplexity / log-rank / DetectGPT.
**Settings:** in-distribution, leave-one-generator-out, leave-one-domain-out, paraphrase attack.

## Team

| Role | Member | GitHub |
|---|---|---|
| A: Data & Baseline (lead) | Nirmal Kumar | [@NIRMAL20062](https://github.com/NIRMAL20062) |
| B: RoBERTa | Vivek Kumar | [@VIVEK342004](https://github.com/VIVEK342004) |
| C: Zero-shot | Ritesh Jha | [@RITESHJHA2244](https://github.com/RITESHJHA2244) |
| D: Attack & Results | Vishal Kumar | [@kumar-vishal01](https://github.com/kumar-vishal01) |

## Docs

- [docs/ROADMAP.md](docs/ROADMAP.md): what we're building, the evaluation settings, timeline, risks
- [docs/TASKS.md](docs/TASKS.md): who codes what, and the shared data/split/prediction formats (read §0 first)

## Setup

```bash
git clone https://github.com/NIRMAL20062/generated_text_detection.git
cd generated_text_detection
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -r requirements.txt   # plus torch if you run models locally
```

## How to reproduce

_To be filled in as the scripts land (data → baselines → RoBERTa → zero-shot → paraphrase → tables)._

## References

1. E. Mitchell et al., *DetectGPT: Zero-Shot Machine-Generated Text Detection using Probability Curvature*, ICML 2023. https://arxiv.org/abs/2301.11305
2. Y. Wang et al., *M4: Multi-generator, Multi-domain, and Multi-lingual Black-Box Machine-Generated Text Detection*, EACL 2024. https://arxiv.org/abs/2305.14902
