"""Cosine similarity between probe weight vectors.

Implements Section 4.5 of the paper: compute pairwise cosine similarity
between probes trained at a model-specific shared layer.
"""

import numpy as np


def cosine_similarity(w_a: np.ndarray, w_b: np.ndarray) -> float:
    """Compute cosine similarity between two probe weight vectors.

    Matches Equation 4 in the paper:
        cos(w_a, w_b) = (w_a^T w_b) / (||w_a|| ||w_b||)

    Parameters
    ----------
    w_a, w_b : np.ndarray
        Probe weight vectors, shape (hidden_dim,).

    Returns
    -------
    float
        Cosine similarity in [-1, 1].
    """
    norm_a = np.linalg.norm(w_a)
    norm_b = np.linalg.norm(w_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(w_a, w_b) / (norm_a * norm_b))


def pairwise_cosine_matrix(probe_weights: dict[str, np.ndarray]) -> np.ndarray:
    """Compute the full pairwise cosine matrix for a set of probes.

    Parameters
    ----------
    probe_weights : dict[str, np.ndarray]
        Mapping from dataset name to probe weight vector. All vectors must
        come from probes trained at the same shared layer (so they live in
        the same representation space).

    Returns
    -------
    np.ndarray
        Symmetric matrix of shape (n_datasets, n_datasets) with pairwise
        cosine similarities. Order matches insertion order of probe_weights.
    """
    names = list(probe_weights.keys())
    n = len(names)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            matrix[i, j] = cosine_similarity(
                probe_weights[names[i]],
                probe_weights[names[j]],
            )
    return matrix
