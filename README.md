# Do Correctness Probes Transfer? A Cross-Family Study of LLM Hidden States

This repository contains code and analysis for the paper:

> **Do Correctness Probes Transfer? A Cross-Family Study of LLM Hidden States**
> *Anonymous submission to EMNLP*

We study out-of-distribution transfer of response-level correctness probes across three open-weight LLM families (Qwen3-8B, Llama-3.1-8B, Gemma-2-9B), seven datasets, and four task families.

## Smoke test

We recommend running the full pipeline, verify your environment with the smoke test as we did with our own project as well.

\`\`\`bash
jupyter notebook notebooks/smoke_test.ipynb
\`\`\`

This runs a minimal end-to-end pipeline on Qwen2.5-1.5B with 50 GSM8K problems in about 3–5 minutes on a free Colab T4 GPU. If the per-layer AUC plot shows a clear peak above 0.5 in the middle-to-late layers, your environment is set up correctly and you can proceed to the full pipeline.

See `notebooks/README.md` for details.

## Key findings

1. **Within-family transfer is stronger than cross-family transfer** (0.075 AUC gap in the averaged matrix), with the clearest pattern in code generation.
2. **Probe weight directions are nearly orthogonal across most dataset pairs**, even when predictive transfer is strong — suggesting correctness is not encoded along a single canonical direction.
3. **Probes outperform logprob baselines mainly in-domain and within-family**; on cross-family transfer, target-side confidence measures are typically stronger.
4. **A layer-selection confound** can make early-layer signals appear transferable in-domain but fail cross-source.

## Repository structure

```
.
├── analysis/        # Scripts to reproduce paper tables and figures
├── configs/         # YAML configs for models, datasets, and probe hyperparameters
├── data/            # Dataset loading (no raw data committed)
└── results/         # Output directory for computed results
├── scripts/         # Pipeline scripts (run in order: 01 → 07)
├── src/             # Core library code
│   ├── generation/  # Response generation
│   ├── grading/     # Dataset-specific correctness graders
│   ├── activations/ # Hidden state extraction
│   ├── probes/      # Probe training, transfer, cosine analysis
│   └── baselines/   # Logprob baseline computation
```

## Installation

```bash
git clone https://github.com/<USER>/correctness-probe-transfer.git
cd correctness-probe-transfer
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
```

You will also need:
- A HuggingFace account with access to Llama-3.1-8B-Instruct (gated)
- An NVIDIA GPU with at least 24GB VRAM (we used a single H100 80GB)
- Approximately 200GB of disk space for hidden-state activations across all models and datasets

## Reproducing the paper

The pipeline runs in seven stages. Each stage writes intermediate artifacts to `results/`.

```bash
# 1. Generate responses for all (model, dataset) pairs
python scripts/01_generate_responses.py --config configs/models.yaml

# 2. Grade responses using dataset-specific evaluators
python scripts/02_grade_responses.py

# 3. Extract hidden states via generate-then-replay
python scripts/03_extract_activations.py

# 4. Train per-layer probes on each (model, dataset) pair
python scripts/04_train_probes.py

# 5. Build 7x7 transfer matrices for each model
python scripts/05_build_transfer_matrices.py

# 6. Compute pairwise probe-weight cosine similarities
python scripts/06_compute_cosines.py

# 7. Compute logprob baselines and probe-vs-baseline gaps
python scripts/07_compute_baselines.py
```

After the pipeline completes, reproduce paper figures and tables:

```bash
python analysis/reproduce_table_2.py    # In-domain AUCs
python analysis/reproduce_table_3.py    # Within vs. cross-family summary
python analysis/reproduce_figure_2.py   # Averaged 7x7 transfer matrix
python analysis/reproduce_figure_3.py   # Cosine similarity matrices
python analysis/reproduce_figure_4.py   # Per-layer AUC trajectories
```

## Models and datasets

| Model           | HuggingFace ID                 | Layers | Hidden dim |
|-----------------|--------------------------------|--------|------------|
| Qwen3-8B        | Qwen/Qwen3-8B                  | 36     | 4096       |
| Llama-3.1-8B    | meta-llama/Llama-3.1-8B-Instruct | 32   | 4096       |
| Gemma-2-9B      | google/gemma-2-9b-it           | 42     | 3584       |

| Dataset       | Family          | N    |
|---------------|-----------------|------|
| GSM8K         | Math            | 1000 |
| MATH-500      | Math            | 500  |
| MMLU-Pro      | Multiple-choice | 1000 |
| MBPP-test     | Code            | 257  |
| MBPP-train    | Code            | 120  |
| HumanEval     | Code            | 164  |
| TriviaQA      | Factual         | 1000 |

## Compute requirements

Approximate runtime on a single NVIDIA H100 80GB:

- Generation (all model/dataset pairs): ~24 hours
- Activation extraction: ~12 hours
- Probe training (full transfer matrices, 5 seeds): ~6 hours
- Analysis and figures: <1 hour

Total disk usage: ~200GB for hidden states (float16), plus ~5GB for generated responses and probe weights.

## Citation

To release soon.

## License

This code is released under the MIT License. See `LICENSE` for details.

Note that the underlying models (Qwen3, Llama-3.1, Gemma-2) and datasets (GSM8K, MATH-500, MMLU-Pro, MBPP, HumanEval, TriviaQA) have their own licenses; please consult their respective sources.

## Contact

To release soon.
