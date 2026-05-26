# Transfer matrices

This directory contains the output 7×7 transfer matrices from `scripts/05_build_transfer_matrices.py`.

## Files (generated, not committed)

- `qwen3-8b_transfer.csv` — 7×7 matrix for Qwen3-8B
- `llama-3.1-8b_transfer.csv` — 7×7 matrix for Llama-3.1-8B
- `gemma-2-9b_transfer.csv` — 7×7 matrix for Gemma-2-9B
- `averaged_transfer.csv` — element-wise average across the three models
- `transfer_multiseed_mean.csv` — mean across 5 seeds
- `transfer_multiseed_std.csv` — std deviation across 5 seeds

Rows = source datasets, columns = target datasets, cell values = transfer AUC.
