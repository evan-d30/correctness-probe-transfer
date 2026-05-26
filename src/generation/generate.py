"""Response generation orchestration.

Coordinates the generation pass across (model, dataset) pairs. The actual
per-example generation logic lives in src/activations/extract.py since
generation and hidden-state extraction are coupled by the generate-then-replay
procedure (Section 4.2 of the paper).
"""

from pathlib import Path
from typing import Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.activations.extract import extract_for_example
from src.generation.prompts import ANSWER_MARKERS, build_prompt


def apply_chat_template(tokenizer: AutoTokenizer, prompt: str) -> str:
    """Wrap a raw prompt in the model's chat template.

    All three models in the paper use their default chat template without
    system-prompt modification (Section 3.1).
    """
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def load_model_and_tokenizer(
    huggingface_id: str,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load a model and its tokenizer from HuggingFace.

    Parameters
    ----------
    huggingface_id : str
        HuggingFace model identifier (e.g., "Qwen/Qwen3-8B").
    dtype : torch.dtype
        Model weights dtype. Default float16.
    device : str
        Device to place the model on.
    """
    tokenizer = AutoTokenizer.from_pretrained(huggingface_id)
    model = AutoModelForCausalLM.from_pretrained(
        huggingface_id,
        torch_dtype=dtype,
        device_map=device,
    )
    model.eval()
    return model, tokenizer


def generate_dataset(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset_name: str,
    examples: Iterable[dict],
    output_dir: Path,
    max_new_tokens: int = 2048,
    device: str = "cuda",
) -> None:
    """Generate responses and extract hidden states for one (model, dataset) pair.

    For each example, runs the generate-then-replay procedure and writes:
      - The generated text and token statistics
      - The hidden states at four positions, all layers

    Parameters
    ----------
    model, tokenizer
        Loaded HuggingFace model and tokenizer.
    dataset_name : str
        Key into PROMPT_TEMPLATES and ANSWER_MARKERS.
    examples : Iterable[dict]
        Raw dataset examples. Order will be preserved in output files.
    output_dir : Path
        Where to write per-example outputs.
    max_new_tokens : int
        Generation cap (2048 in the paper).
    device : str
        Device.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    answer_marker = ANSWER_MARKERS[dataset_name]

    for example_idx, example in enumerate(examples):
        raw_prompt = build_prompt(dataset_name, example)
        prompt = apply_chat_template(tokenizer, raw_prompt)

        generation, hidden = extract_for_example(
            model,
            tokenizer,
            prompt,
            answer_marker=answer_marker,
            max_new_tokens=max_new_tokens,
            device=device,
        )

        # TODO: serialize generation + hidden states to output_dir.
        # Suggested layout:
        #   output_dir / "generations.jsonl"  (one JSON per example with text + token stats)
        #   output_dir / "hidden" / f"example_{example_idx:05d}.npz"  (the four positions)
        pass
