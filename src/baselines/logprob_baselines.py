"""Logprob-based confidence baselines.
"""

import numpy as np
from sklearn.metrics import roc_auc_score


def mean_logp(token_logps: list[float]) -> float:
    """Mean log probability across generated tokens (Equation E10)."""
    if not token_logps:
        return 0.0
    return float(np.mean(token_logps))


def final_logp(token_logps: list[float]) -> float:
    """Log probability of the final generated token (Equation E11)."""
    if not token_logps:
        return 0.0
    return float(token_logps[-1])


def mean_entropy_negated(token_entropies: list[float]) -> float:
    """Negated mean entropy across generated tokens (Equation E12).

    Negated so that higher score corresponds to higher predicted correctness.
    """
    if not token_entropies:
        return 0.0
    return float(-np.mean(token_entropies))


def max_entropy_negated(token_entropies: list[float]) -> float:
    """Negated max entropy across generated tokens (Equation E13)."""
    if not token_entropies:
        return 0.0
    return float(-np.max(token_entropies))


BASELINE_FUNCTIONS = {
    "mean_logp": mean_logp,
    "final_logp": final_logp,
    "mean_entropy_negated": mean_entropy_negated,
    "max_entropy_negated": max_entropy_negated,
}


def compute_baseline_aucs(
    token_logps_per_example: list[list[float]],
    token_entropies_per_example: list[list[float]],
    labels: np.ndarray,
) -> dict[str, float]:
    """Compute AUC for each of the four baselines against correctness labels.

    Parameters
    ----------
    token_logps_per_example : list[list[float]]
        For each example, the per-token log probabilities of the generated text.
    token_entropies_per_example : list[list[float]]
        For each example, the per-token entropies of the next-token distribution.
    labels : np.ndarray
        Binary correctness labels.

    Returns
    -------
    dict[str, float]
        Baseline name -> AUC.
    """
    results = {}
    for name, fn in BASELINE_FUNCTIONS.items():
        if name in ("mean_logp", "final_logp"):
            scores = [fn(logps) for logps in token_logps_per_example]
        else:
            scores = [fn(ents) for ents in token_entropies_per_example]
        results[name] = float(roc_auc_score(labels, scores))
    return results


def best_baseline_auc(baseline_aucs: dict[str, float]) -> tuple[str, float]:
    """Return the (name, AUC) of the best-performing baseline."""
    best_name = max(baseline_aucs, key=baseline_aucs.get)
    return best_name, baseline_aucs[best_name]


def probe_vs_baseline_gap(probe_auc: float, baseline_aucs: dict[str, float]) -> float:
    """Compute the probe-vs-baseline gap (Equation 5).

    Positive values mean the probe outperforms the best baseline.
    """
    _, best_auc = best_baseline_auc(baseline_aucs)
    return probe_auc - best_auc
