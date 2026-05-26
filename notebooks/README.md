# Notebooks

This directory contains interactive notebooks for exploring and verifying the pipeline.

## `smoke_test.ipynb`

A minimal end-to-end pipeline verification that runs in 3–5 minutes on a free Colab T4 GPU.

**What it does:**

1. Loads Qwen2.5-1.5B-Instruct (small enough to fit on a free-tier GPU)
2. Generates greedy responses to 50 GSM8K problems while recording hidden states
3. Trains a logistic regression probe at every layer
4. Plots per-layer AUC for predicting response correctness

**Why it exists:**

- **Reproducibility check.** If the notebook runs and shows a clear AUC peak above 0.5 in the middle-to-late layers, your environment is set up correctly and the central per-layer pattern from the paper's Section 5.5 reproduces on a small model.
- **Tutorial.** Each cell is annotated with what's happening and why. A reader can understand the pipeline mechanics without running the full multi-model experiments.
- **Sanity check.** If the AUC is flat near 0.5, something is wrong — either the environment, the answer extractor, or class imbalance. The notebook includes troubleshooting notes at the bottom.

**Important notes:**

- The smoke test uses **Qwen2.5-1.5B-Instruct**, not the three 7-9B models used in the paper. The qualitative per-layer pattern should be similar, but the absolute AUC values will differ.
- Only **50 GSM8K problems** are used. Full reproduction requires the full datasets and models described in the main README.
- Results from a successful smoke-test run are committed in the notebook so you can see the expected output without running it.

**How to run:**

On Google Colab:
1. Open `smoke_test.ipynb` in Colab
2. Set `Runtime → Change runtime type → T4 GPU`
3. Run all cells

Locally (requires a CUDA GPU with at least 6 GB VRAM):
```bash
pip install -r ../requirements.txt
jupyter notebook smoke_test.ipynb
```

**Expected runtime:** 3–5 minutes on a T4 GPU. Most of that is the initial model download (~3 GB).
