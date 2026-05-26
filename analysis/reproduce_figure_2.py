"""Reproduce Figure 2: the averaged 7x7 transfer matrix.

Reads results/transfer_matrices/averaged_transfer.csv and renders the heatmap
shown in Figure 2 of the paper.

Usage:
    python analysis/reproduce_figure_2.py
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Display names matching the paper
DATASET_DISPLAY = {
    "gsm8k": "gsm8k",
    "math500": "math500",
    "mmlu_pro": "mmlu_pro",
    "mbpp_test": "mbpp_test",
    "mbpp_train": "mbpp_train",
    "humaneval": "humaneval",
    "triviaqa": "triviaqa",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/transfer_matrices/averaged_transfer.csv"),
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=Path("results/figures/figure_2_averaged_transfer.png"),
    )
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=Path("results/figures/figure_2_averaged_transfer.pdf"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_png.parent.mkdir(parents=True, exist_ok=True)

    matrix = pd.read_csv(args.input, index_col=0)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        vmin=0.50,
        vmax=0.95,
        cbar_kws={"label": "AUC"},
        ax=ax,
    )
    ax.set_xlabel("Target dataset")
    ax.set_ylabel("Source dataset")
    ax.set_title("Average 7-dataset transfer matrix (3 models)")
    plt.tight_layout()

    fig.savefig(args.output_png, dpi=300)
    fig.savefig(args.output_pdf)
    print(f"Wrote {args.output_png} and {args.output_pdf}")


if __name__ == "__main__":
    main()
