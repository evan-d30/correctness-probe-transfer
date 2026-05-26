"""Prompt templates for each dataset.

Matches Appendix F (Table F1) of the paper. Each dataset uses a tailored
prompt format with an explicit output anchor to support deterministic
answer extraction.
"""

PROMPT_TEMPLATES = {
    "gsm8k": (
        "Solve the following problem step by step. "
        "End with: \"Final Answer: \" followed by your numeric answer.\n\n"
        "Problem: {question}\n\n"
        "Solution:"
    ),
    "math500": (
        "Solve the following problem step by step. "
        "Put your final answer in \\boxed{{...}}.\n\n"
        "Problem: {problem}\n\n"
        "Solution:"
    ),
    "mmlu_pro": (
        "Reason step by step about the following multiple-choice question, "
        "then end with \"Answer: \" followed by the letter of the correct option.\n\n"
        "Question: {question}\n"
        "Options:\n{options}\n\n"
        "Reasoning:"
    ),
    "mbpp_test": (
        "Write a Python solution for the following problem. "
        "Return only the code in a python block.\n\n"
        "Problem: {text}\n\n"
        "```python\n"
    ),
    "mbpp_train": (
        "Write a Python solution for the following problem. "
        "Return only the code in a python block.\n\n"
        "Problem: {text}\n\n"
        "```python\n"
    ),
    "humaneval": (
        "Write a Python solution for the following problem. "
        "Return only the code in a python block. "
        "Preserve the required function signature.\n\n"
        "{prompt}\n\n"
        "```python\n"
    ),
    "triviaqa": (
        "Answer the following question. "
        "End with: \"Final Answer: \" followed by your answer.\n\n"
        "Question: {question}\n\n"
        "Answer:"
    ),
}


# Answer markers used for the pre-answer-marker hidden state extraction
# and for output parsing during grading.
ANSWER_MARKERS = {
    "gsm8k": "Final Answer:",
    "math500": "\\boxed",
    "mmlu_pro": "Answer:",
    "mbpp_test": None,            # code datasets have no textual marker
    "mbpp_train": None,
    "humaneval": None,
    "triviaqa": "Final Answer:",
}


def build_prompt(dataset_name: str, example: dict) -> str:
    """Build the prompt string for a single example.

    Parameters
    ----------
    dataset_name : str
        One of the seven dataset keys.
    example : dict
        Raw example from the dataset (with fields matching the template).

    Returns
    -------
    str
        The formatted prompt, ready to pass through the model's chat template.
    """
    if dataset_name not in PROMPT_TEMPLATES:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    template = PROMPT_TEMPLATES[dataset_name]

    if dataset_name == "mmlu_pro":
        # MMLU-Pro has an options list to format
        options_text = "\n".join(
            f"  {chr(65 + i)}. {opt}" for i, opt in enumerate(example["options"])
        )
        return template.format(question=example["question"], options=options_text)

    return template.format(**example)
