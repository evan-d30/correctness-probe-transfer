# Probe direction cosine similarities

This directory contains the output cosine similarity matrices from `scripts/06_compute_cosines.py`.

## Files (generated, not committed)

- `qwen3-8b_cosines.csv` — 7×7 matrix of pairwise cosines at the shared layer (26)
- `llama-3.1-8b_cosines.csv` — 7×7 matrix at the shared layer (16)
- `gemma-2-9b_cosines.csv` — 7×7 matrix at the shared layer (41)

Rows and columns are dataset names; cell values are cosine similarities between
the probe weight vectors trained on each dataset at the model's shared layer.
