"""I/O utilities for the correctness-probe-transfer pipeline.

This module is intentionally a stub. Concrete I/O routines will be filled in
as the pipeline is built out. The conventions below define the expected
directory layout and file formats so that downstream scripts can rely on
consistent paths.

Recommended directory layout
----------------------------
results/
    generations/
        {model}/
            {dataset}/
                generations.jsonl       # one JSON per example
    hidden_states/
        {model}/
            {dataset}/
                example_{idx:05d}.npz   # four positions x n_layers x hidden_dim
    probes/
        {model}/
            {dataset}/
                layer_{ell}_seed_{s}.npz   # weights, bias, cv_auc_mean, cv_auc_std
    transfer_matrices/
        {model}_transfer.csv
        averaged_transfer.csv
    cosines/
        {model}_cosines.csv
    baselines/
        {model}_baselines.csv
"""

from pathlib import Path
from typing import Any


def generations_path(results_root: Path, model: str, dataset: str) -> Path:
    """Path to the generations JSONL for a (model, dataset) pair."""
    return results_root / "generations" / model / dataset / "generations.jsonl"


def hidden_state_path(
    results_root: Path,
    model: str,
    dataset: str,
    example_idx: int,
) -> Path:
    """Path to the hidden-state NPZ for a single example."""
    return (
        results_root
        / "hidden_states"
        / model
        / dataset
        / f"example_{example_idx:05d}.npz"
    )


def probe_path(
    results_root: Path,
    model: str,
    dataset: str,
    layer: int,
    seed: int,
) -> Path:
    """Path to a stored probe (weights + bias + CV stats)."""
    return (
        results_root
        / "probes"
        / model
        / dataset
        / f"layer_{layer}_seed_{seed}.npz"
    )


def transfer_matrix_path(results_root: Path, model: str) -> Path:
    """Path to a per-model transfer matrix CSV."""
    return results_root / "transfer_matrices" / f"{model}_transfer.csv"


def averaged_transfer_path(results_root: Path) -> Path:
    """Path to the cross-model averaged transfer matrix CSV."""
    return results_root / "transfer_matrices" / "averaged_transfer.csv"


def cosine_matrix_path(results_root: Path, model: str) -> Path:
    """Path to a per-model cosine similarity matrix CSV."""
    return results_root / "cosines" / f"{model}_cosines.csv"


def baseline_results_path(results_root: Path, model: str) -> Path:
    """Path to per-model probe-vs-baseline results CSV."""
    return results_root / "baselines" / f"{model}_baselines.csv"


def ensure_parent_dir(path: Path) -> Path:
    """Ensure the parent directory of `path` exists, then return `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
