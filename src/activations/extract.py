"""Hidden state extraction via generate-then-replay.

Implements Section 4.2 of the paper. The pipeline is two-pass:

  1. Generate up to 2048 tokens with greedy decoding, recording token-level
     log probabilities and entropies for the baselines (Section 4.6).
     Hidden states are NOT stored during generation.

  2. Replay the concatenated prompt + generation in a single forward pass
     with output_hidden_states=True, then extract hidden states at four
     positions from every layer.

The four extraction positions (Equations E4-E7) are:
  - end of input question
  - middle of generation
  - pre-answer marker (or penultimate generated token if no marker)
  - final generated token

All reported probe results in the paper use the pre-answer-marker position.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch.nn.functional import log_softmax
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class GenerationOutput:
    """Output of the first pass: greedy generation with token-level statistics."""
    prompt_ids: torch.Tensor          # shape (prompt_len,)
    generated_ids: torch.Tensor       # shape (gen_len,)
    generated_text: str
    token_logps: list[float]          # per-token log prob of the actually generated token
    token_entropies: list[float]      # per-token entropy of the full next-token distribution


@dataclass
class HiddenStateOutput:
    """Output of the second pass: hidden states at four positions, all layers.

    Each tensor has shape (n_layers, hidden_dim) -- one row per layer.
    """
    end_of_question: np.ndarray
    middle_of_generation: np.ndarray
    pre_answer_marker: np.ndarray
    final_generated_token: np.ndarray
    marker_found: bool                 # False if we fell back to penultimate token


def generate_with_token_stats(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 2048,
    device: str = "cuda",
) -> GenerationOutput:
    """First pass: greedy generation while recording token-level statistics.

    Implements Equations E2-E3 from the paper. Records the log probability of
    the actually generated token (l_k) and the Shannon entropy of the full
    next-token distribution (H_k) at each position.

    Parameters
    ----------
    model : AutoModelForCausalLM
        Causal LM with output_scores=True supported.
    tokenizer : AutoTokenizer
        Matching tokenizer.
    prompt : str
        The (already chat-template-formatted) prompt string.
    max_new_tokens : int
        Maximum number of new tokens to generate.
    device : str
        Device to run on.

    Returns
    -------
    GenerationOutput
        Tokens, text, and per-token logp / entropy lists.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_ids = inputs["input_ids"][0]
    prompt_len = prompt_ids.shape[0]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # The "scores" tuple has one entry per generated token: each is the raw
    # logits over the vocabulary BEFORE that token was sampled.
    scores = outputs.scores  # tuple of length gen_len, each shape (1, vocab_size)
    generated_ids = outputs.sequences[0, prompt_len:]  # shape (gen_len,)

    token_logps: list[float] = []
    token_entropies: list[float] = []
    for k, score in enumerate(scores):
        # log p(v | context) for all v in vocab
        log_probs = log_softmax(score[0], dim=-1)  # shape (vocab_size,)
        # log p of the actually generated token
        gen_token_id = generated_ids[k].item()
        token_logps.append(float(log_probs[gen_token_id].item()))
        # Shannon entropy: H = -sum p(v) log p(v)
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum().item()
        token_entropies.append(float(entropy))

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return GenerationOutput(
        prompt_ids=prompt_ids.cpu(),
        generated_ids=generated_ids.cpu(),
        generated_text=generated_text,
        token_logps=token_logps,
        token_entropies=token_entropies,
    )


def find_answer_marker_position(
    generated_text: str,
    generated_ids: torch.Tensor,
    tokenizer: AutoTokenizer,
    marker: Optional[str],
) -> Optional[int]:
    """Find the token-index in the generation where the answer marker first appears.

    Returns the position m such that the pre-answer-marker hidden state should
    be taken at token index (prompt_len + m - 1).

    Parameters
    ----------
    generated_text : str
        The decoded generation string.
    generated_ids : torch.Tensor
        Token IDs of the generation.
    tokenizer : AutoTokenizer
        Matching tokenizer.
    marker : str or None
        The dataset-specific answer anchor (e.g. "Final Answer:", "Answer:",
        "\\boxed"). If None, no marker is sought.

    Returns
    -------
    int or None
        The 1-indexed token position of the first character of the marker
        within the generation, or None if marker is not present.
    """
    if marker is None:
        return None

    char_idx = generated_text.find(marker)
    if char_idx == -1:
        return None

    # Walk the tokens until we cover char_idx characters of decoded text
    cumulative_text = ""
    for token_pos, token_id in enumerate(generated_ids):
        cumulative_text += tokenizer.decode([token_id.item()], skip_special_tokens=True)
        if len(cumulative_text) > char_idx:
            return token_pos + 1  # 1-indexed for the m definition in Eq E6

    return None


def extract_hidden_states(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    generation: GenerationOutput,
    answer_marker: Optional[str],
    device: str = "cuda",
) -> HiddenStateOutput:
    """Second pass: replay prompt + generation, extract hidden states at four positions.

    Implements Equations E4-E7 from the paper. Returns hidden states from
    every transformer layer (excluding the embedding layer) at four positions.

    Parameters
    ----------
    model : AutoModelForCausalLM
        Same model used for generation. Must support output_hidden_states=True.
    tokenizer : AutoTokenizer
        Matching tokenizer.
    generation : GenerationOutput
        Result of the first generation pass.
    answer_marker : str or None
        Dataset-specific answer anchor for the pre-answer-marker position.
    device : str
        Device to run on.

    Returns
    -------
    HiddenStateOutput
        Hidden states at four positions, each of shape (n_layers, hidden_dim),
        stored as float16 numpy arrays.
    """
    prompt_len = generation.prompt_ids.shape[0]
    gen_len = generation.generated_ids.shape[0]
    full_ids = torch.cat([generation.prompt_ids, generation.generated_ids]).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(full_ids, output_hidden_states=True)

    # hidden_states is a tuple of length (n_layers + 1): the first entry is the
    # embedding layer output; we EXCLUDE it per Section 4.2 of the paper.
    hidden_states = outputs.hidden_states[1:]  # each of shape (1, seq_len, hidden_dim)

    # Stack into a single tensor of shape (n_layers, seq_len, hidden_dim)
    all_layers = torch.stack([h[0] for h in hidden_states], dim=0)

    # Compute the four position indices into the seq_len axis
    T = prompt_len
    K = gen_len

    # Position 1: end of input question (Eq E4)
    pos_end_q = T - 1  # last token of prompt; 0-indexed

    # Position 2: middle of generation (Eq E5)
    pos_middle = T + (K // 2)
    pos_middle = min(pos_middle, T + K - 1)

    # Position 3: pre-answer marker (Eq E6)
    marker_pos = find_answer_marker_position(
        generation.generated_text,
        generation.generated_ids,
        tokenizer,
        answer_marker,
    )
    marker_found = marker_pos is not None
    if marker_found:
        # T + m - 1, 0-indexed
        pos_marker = T + marker_pos - 1
    else:
        # Fallback: penultimate generated token, T + K - 1 (0-indexed)
        pos_marker = T + K - 2 if K >= 2 else T + K - 1
    pos_marker = min(max(pos_marker, T), T + K - 1)

    # Position 4: final generated token (Eq E7)
    pos_final = T + K - 1

    def gather(pos: int) -> np.ndarray:
        return all_layers[:, pos, :].to(torch.float16).cpu().numpy()

    return HiddenStateOutput(
        end_of_question=gather(pos_end_q),
        middle_of_generation=gather(pos_middle),
        pre_answer_marker=gather(pos_marker),
        final_generated_token=gather(pos_final),
        marker_found=marker_found,
    )


def extract_for_example(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    answer_marker: Optional[str],
    max_new_tokens: int = 2048,
    device: str = "cuda",
) -> tuple[GenerationOutput, HiddenStateOutput]:
    """Convenience wrapper: run both passes for a single example.

    Returns
    -------
    (GenerationOutput, HiddenStateOutput)
        The generation result (text, ids, token statistics) and the hidden
        states at the four extraction positions.
    """
    generation = generate_with_token_stats(
        model, tokenizer, prompt, max_new_tokens=max_new_tokens, device=device
    )
    hidden = extract_hidden_states(
        model, tokenizer, generation, answer_marker, device=device
    )
    return generation, hidden
