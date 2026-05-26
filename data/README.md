# Data

This repository nor project contains any raw dataset files. All datasets are loaded directly from HuggingFace at runtime using the configurations in `configs/datasets.yaml`.

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

To verify your local dataset versions match those used in the paper, check the dataset accuracy and class balance numbers in `results/dataset_accuracy.csv` after running `scripts/02_grade_responses.py`. 
