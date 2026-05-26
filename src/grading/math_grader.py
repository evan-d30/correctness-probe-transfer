"""Math correctness grading for GSM8K and MATH-500.

Implements Appendix G.1 of the paper. Math tasks use multi-pattern answer
extraction with symbolic equivalence checking when possible.
"""

import re
from typing import Optional

import sympy
from sympy.parsing.sympy_parser import parse_expr


def extract_gsm8k_answer(generated_text: str) -> Optional[str]:
    """Extract the final numeric answer from a GSM8K response.

    Looks for "Final Answer:" anchor and extracts the trailing number.
    """
    match = re.search(r"Final Answer:\s*([+-]?\d[\d,]*(?:\.\d+)?)", generated_text)
    if match:
        return match.group(1).replace(",", "")
    # Fallback: last number in the text
    numbers = re.findall(r"[+-]?\d+(?:\.\d+)?", generated_text)
    return numbers[-1] if numbers else None


def grade_gsm8k(generated_text: str, gold_answer: str) -> bool:
    """Grade a GSM8K response against the gold numeric answer."""
    predicted = extract_gsm8k_answer(generated_text)
    if predicted is None:
        return False
    try:
        return abs(float(predicted) - float(gold_answer)) < 1e-6
    except ValueError:
        return predicted.strip() == gold_answer.strip()


def extract_boxed_answer(generated_text: str) -> Optional[str]:
    """Extract the contents of \\boxed{...} from a MATH-500 response.

    Handles nested braces by counting depth.
    """
    idx = generated_text.find("\\boxed{")
    if idx == -1:
        return None
    start = idx + len("\\boxed{")
    depth = 1
    for i in range(start, len(generated_text)):
        c = generated_text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return generated_text[start:i].strip()
    return None


def normalize_math_expression(expr: str) -> str:
    """Apply standard normalizations before symbolic comparison."""
    expr = expr.strip()
    expr = expr.replace("\\!", "").replace("\\,", "").replace("\\;", "")
    expr = expr.replace("\\left", "").replace("\\right", "")
    expr = expr.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    expr = expr.replace("^{\\circ}", "")
    return expr


def grade_math500(generated_text: str, gold_answer: str) -> bool:
    """Grade a MATH-500 response using SymPy-based symbolic equivalence.

    Falls back to normalized string comparison if symbolic parsing fails.
    """
    predicted = extract_boxed_answer(generated_text)
    if predicted is None:
        return False

    predicted_norm = normalize_math_expression(predicted)
    gold_norm = normalize_math_expression(gold_answer)

    # Try symbolic equivalence
    try:
        pred_expr = parse_expr(predicted_norm)
        gold_expr = parse_expr(gold_norm)
        diff = sympy.simplify(pred_expr - gold_expr)
        return diff == 0
    except (sympy.SympifyError, SyntaxError, TypeError, ValueError):
        pass

    # Fallback: normalized string comparison
    return predicted_norm == gold_norm
