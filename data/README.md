# Data

This directory does **not** contain raw dataset files. All datasets are loaded directly from HuggingFace at runtime using the configurations in `configs/datasets.yaml`.

## Dataset sources

| Dataset       | HuggingFace ID                       | License        |
|---------------|--------------------------------------|----------------|
| GSM8K         | `openai/gsm8k`                       | MIT            |
| MATH-500      | `HuggingFaceH4/MATH-500`             | MIT            |
| MMLU-Pro      | `TIGER-Lab/MMLU-Pro`                 | MIT            |
| MBPP          | `google-research-datasets/mbpp`      | CC-BY-4.0      |
| HumanEval     | `openai/openai_humaneval`            | MIT            |
| TriviaQA      | `mandarjoshi/trivia_qa`              | Apache-2.0     |

## Reproducibility

Dataset sampling uses `random_state=42` for the subsets that are sampled (GSM8K, MATH-500, MMLU-Pro, TriviaQA). The MBPP splits and HumanEval are used in full.

To verify your local dataset versions match those used in the paper, check the dataset accuracy and class balance numbers in `results/dataset_accuracy.csv` after running `scripts/02_grade_responses.py`. These should match Appendix D, Table D2 in the paper.

## Why no data files

Including raw dataset files in the repo would (a) bloat the repo size, (b) duplicate data already available canonically on HuggingFace, and (c) risk license issues. Pulling fresh from HuggingFace at runtime guarantees you're using the canonical version.
