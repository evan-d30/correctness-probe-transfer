"""Multiple-choice correctness grading for MMLU-Pro.

Implements Appendix G.2 of the paper. Parses the model's final answer as a
multiple-choice letter, prioritizing the "Answer:" output anchor and
constraining valid predictions to the available option letters.
"""

import re
from typing import Optional


def extract_mc_answer(
    generated_text: str,
    valid_letters: list[str],
) -> Optional[str]:
    """Extract a multiple-choice letter answer from the response.

    Strategy:
      1. Look for "Answer:" anchor followed by a valid letter.
      2. Look for a standalone valid letter near the end of the text.

    Parameters
    ----------
    generated_text : str
        The model's generation.
    valid_letters : list[str]
        Available option letters (e.g., ["A", "B", "C", "D"] or ["A".."J"]).

    Returns
    -------
    str or None
        The extracted letter, or None if no valid letter was found.
    """
    letter_alt = "|".join(re.escape(l) for l in valid_letters)

    # Primary: "Answer: X" anchor
    match = re.search(rf"Answer:\s*\(?({letter_alt})\)?", generated_text)
    if match:
        return match.group(1)

    # Secondary: any standalone valid letter near the end of the text
    tail = generated_text[-200:]
    tail_matches = re.findall(rf"\b({letter_alt})\b", tail)
    if tail_matches:
        return tail_matches[-1]

    return None


def grade_mmlu_pro(
    generated_text: str,
    gold_answer: str,
    valid_letters: list[str],
) -> bool:
    """Grade an MMLU-Pro response against the gold letter answer."""
    predicted = extract_mc_answer(generated_text, valid_letters)
    if predicted is None:
        return False
    return predicted.strip().upper() == gold_answer.strip().upper()
