"""Probe training and evaluation.

Trains a binary logistic regression probe on hidden states from a single
layer to predict response-level correctness. Matches Section 4.1 of the paper.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


@dataclass
class ProbeResult:
    """Result of training a probe at one layer on one dataset."""
    weights: np.ndarray            # shape (hidden_dim,)
    bias: float
    cv_auc_mean: float             # mean AUC across CV folds
    cv_auc_std: float              # std AUC across CV folds
    layer: int
    n_examples: int
    n_positive: int


def train_probe(
    hidden_states: np.ndarray,
    labels: np.ndarray,
    n_folds: int = 5,
    C: float = 1.0,
    max_iter: int = 2000,
    random_state: int = 42,
) -> ProbeResult:
    """Train a logistic regression probe with stratified cross-validation.

    Parameters
    ----------
    hidden_states : np.ndarray
        Shape (n_examples, hidden_dim). Hidden states at a single layer.
    labels : np.ndarray
        Shape (n_examples,). Binary correctness labels in {0, 1}.
    n_folds : int
        Number of stratified CV folds. Default 5.
    C : float
        Inverse L2 regularization strength.
    max_iter : int
        Max L-BFGS iterations.
    random_state : int
        Seed for fold splits and probe initialization.

    Returns
    -------
    ProbeResult
        Probe weights (fit on full dataset), bias, and CV AUC statistics.
    """
    fold_aucs = []
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    for train_idx, val_idx in skf.split(hidden_states, labels):
        model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            class_weight="balanced",
            random_state=random_state,
        )
        model.fit(hidden_states[train_idx], labels[train_idx])
        val_scores = model.predict_proba(hidden_states[val_idx])[:, 1]
        fold_aucs.append(roc_auc_score(labels[val_idx], val_scores))

    # Refit on full dataset for transfer evaluation
    final_model = LogisticRegression(
        C=C,
        max_iter=max_iter,
        class_weight="balanced",
        random_state=random_state,
    )
    final_model.fit(hidden_states, labels)

    return ProbeResult(
        weights=final_model.coef_[0],
        bias=float(final_model.intercept_[0]),
        cv_auc_mean=float(np.mean(fold_aucs)),
        cv_auc_std=float(np.std(fold_aucs)),
        layer=-1,  # set by caller
        n_examples=len(labels),
        n_positive=int(labels.sum()),
    )


def select_best_layer_constrained(
    per_layer_aucs: dict[int, float],
    n_layers: int,
) -> int:
    """Select the best layer constrained to the upper half of the network.

    Matches Equation 2 in the paper:
        l*(D) = argmax_{l in {floor(L/2), ..., L}} AUC_CV(D, l)

    Parameters
    ----------
    per_layer_aucs : dict[int, float]
        Mapping from layer index to in-domain CV AUC.
    n_layers : int
        Total number of layers L.

    Returns
    -------
    int
        Selected layer index.
    """
    lower_bound = n_layers // 2
    constrained = {l: auc for l, auc in per_layer_aucs.items() if l >= lower_bound}
    if not constrained:
        raise ValueError(
            f"No layers found in upper half (>= {lower_bound}) of {n_layers} layers"
        )
    return max(constrained, key=constrained.get)


def evaluate_probe_on_target(
    probe_weights: np.ndarray,
    probe_bias: float,
    target_hidden_states: np.ndarray,
    target_labels: np.ndarray,
) -> float:
    """Evaluate a trained probe on target-dataset hidden states.

    Used to fill in off-diagonal cells of the transfer matrix (Equation 3).

    Parameters
    ----------
    probe_weights : np.ndarray
        Probe weights from source dataset, shape (hidden_dim,).
    probe_bias : float
        Probe bias from source dataset.
    target_hidden_states : np.ndarray
        Target-dataset hidden states at the source's best layer, shape (n_target, hidden_dim).
    target_labels : np.ndarray
        Target-dataset correctness labels.

    Returns
    -------
    float
        AUC on the target dataset.
    """
    scores = target_hidden_states @ probe_weights + probe_bias
    return float(roc_auc_score(target_labels, scores))
