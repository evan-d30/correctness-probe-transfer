"""Build 7x7 transfer matrices for each model.

This is stage 5 of the pipeline. Assumes stages 1-4 have already produced:
  - Generated responses (stage 1)
  - Correctness labels (stage 2)
  - Hidden state activations (stage 3)
  - Per-(model, dataset) probes at all layers (stage 4)

Output:
  results/transfer_matrices/{model}_transfer.csv  - 7x7 transfer AUC per model
  results/transfer_matrices/averaged_transfer.csv  - element-wise average

Usage:
    python scripts/05_build_transfer_matrices.py
    python scripts/05_build_transfer_matrices.py --models qwen3-8b
    python scripts/05_build_transfer_matrices.py --seeds 42 7 13 100 0
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Imports from src/ would go here, e.g.:
# from src.probes.train import evaluate_probe_on_target, select_best_layer_constrained
# from src.utils.io import load_probe, load_hidden_states, load_labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["qwen3-8b", "llama-3.1-8b", "gemma-2-9b"],
        help="Models to process",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[
            "gsm8k",
            "math500",
            "mmlu_pro",
            "mbpp_test",
            "mbpp_train",
            "humaneval",
            "triviaqa",
        ],
        help="Datasets to include in the transfer matrix",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42],
        help="Random seeds to evaluate over (multi-seed averaging)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/transfer_matrices"),
        help="Output directory for transfer matrices",
    )
    parser.add_argument(
        "--probe-config",
        type=Path,
        default=Path("configs/probe_config.yaml"),
        help="Path to probe configuration YAML",
    )
    return parser.parse_args()


def build_transfer_matrix(
    model: str,
    datasets: list[str],
    seed: int,
) -> pd.DataFrame:
    """Build a single (n_datasets x n_datasets) transfer matrix for one model.

    For each (source, target) pair:
      1. Train probe on source at source's constrained-best layer
      2. Evaluate probe on target at the SAME layer
      3. For diagonal cells, report 5-fold CV AUC instead

    Returns a DataFrame with rows = sources, columns = targets, values = AUC.
    """
    matrix = pd.DataFrame(
        index=datasets,
        columns=datasets,
        dtype=float,
    )

    for source in datasets:
        # TODO: Load source probe at constrained best layer
        # probe = load_probe(model, source, layer=source_best_layer, seed=seed)
        # source_best_layer = ...

        for target in datasets:
            if source == target:
                # Diagonal: use 5-fold CV AUC
                # matrix.loc[source, target] = load_cv_auc(model, source, seed=seed)
                pass
            else:
                # Off-diagonal: evaluate source probe on target activations
                # at the same source layer (Equation 3)
                # target_hidden = load_hidden_states(model, target, layer=source_best_layer)
                # target_labels = load_labels(model, target)
                # matrix.loc[source, target] = evaluate_probe_on_target(
                #     probe.weights, probe.bias, target_hidden, target_labels
                # )
                pass

    return matrix


def average_matrices(matrices: list[pd.DataFrame]) -> pd.DataFrame:
    """Element-wise average across per-model matrices.

    Matches Equation E8 in the paper.
    """
    stacked = np.stack([m.values for m in matrices], axis=0)
    avg = stacked.mean(axis=0)
    return pd.DataFrame(avg, index=matrices[0].index, columns=matrices[0].columns)


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    with open(args.probe_config) as f:
        config = yaml.safe_load(f)

    per_model_matrices = []
    for model in args.models:
        print(f"Building transfer matrix for {model}...")

        # Build per-seed matrices then average
        seed_matrices = [
            build_transfer_matrix(model, args.datasets, seed)
            for seed in args.seeds
        ]

        if len(args.seeds) > 1:
            mean_matrix = average_matrices(seed_matrices)
            std_matrix = pd.DataFrame(
                np.stack([m.values for m in seed_matrices]).std(axis=0),
                index=args.datasets,
                columns=args.datasets,
            )
            std_matrix.to_csv(args.results_dir / f"{model}_transfer_std.csv")
        else:
            mean_matrix = seed_matrices[0]

        mean_matrix.to_csv(args.results_dir / f"{model}_transfer.csv")
        per_model_matrices.append(mean_matrix)

    # Cross-model average
    print("Averaging across models...")
    avg = average_matrices(per_model_matrices)
    avg.to_csv(args.results_dir / "averaged_transfer.csv")
    print(f"Wrote averaged transfer matrix to {args.results_dir / 'averaged_transfer.csv'}")


if __name__ == "__main__":
    main()
