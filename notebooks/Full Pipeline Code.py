# -*- coding: utf-8 -*-
"""phase4_v2_updated_qwen3_hf.ipynb

## Cell 1 — Volume mount verification + folder structure
"""

import os, sys, shutil
from pathlib import Path
from datetime import datetime

WORKSPACE = Path('/workspace')
PHASE4_DIR = WORKSPACE / 'phase4_outputs'
RAW_DIR = PHASE4_DIR / 'raw'
FINAL_DIR = PHASE4_DIR / 'final'

stat = shutil.disk_usage(WORKSPACE)
total_gb, free_gb = stat.total/1024**3, stat.free/1024**3
print(f'Total at {WORKSPACE}: {total_gb:.1f} GB  |  Free: {free_gb:.1f} GB')
if total_gb < 30:
    sys.exit('STOP: /workspace appears to be on container disk, not volume.')

test = WORKSPACE / '_mount_test.txt'
test.write_text('ok'); assert test.exists(); test.unlink()

for sub in ['raw/activations','raw/generations','raw/logprobs','raw/labels','raw/diagnostics','raw/errors',
            'final/results/transfer_matrices','final/results/probes',
            'final/results/baselines','final/results/direction_similarity',
            'final/figures','final/audits','final/logs','final/configs']:
    (PHASE4_DIR / sub).mkdir(parents=True, exist_ok=True)

(PHASE4_DIR / 'DELETE_AFTER_VERIFY.txt').write_text(
    f'Created: {datetime.now().isoformat()}\n\n'
    'DO NOT delete /raw/ until ALL checked:\n'
    '[ ] 1. Labels audited for all 6 datasets\n'
    '[ ] 2. Transfer matrices sanity-checked\n'
    '[ ] 3. Figures saved correctly\n'
    '[ ] 4. Audit files complete\n'
    '[ ] 5. Probe weights pickled\n'
    '[ ] 6. Final results downloaded AND verified on laptop\n'
)
print(f'\nFolder structure ready at {PHASE4_DIR}')

"""## Cell 2 — Install dependencies"""

!pip install -q -U transformers datasets accelerate scikit-learn sympy 'antlr4-python3-runtime==4.11' matplotlib seaborn requests reportlab huggingface_hub

"""## Cell 3 — Imports and configuration"""

import re, json, time, pickle, traceback, subprocess, tempfile
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import requests
from getpass import getpass
from huggingface_hub import login
from sympy import sympify, simplify
from sympy.parsing.latex import parse_latex

MODELS = [
    {'name': 'Qwen/Qwen3-8B',                     'short': 'qwen3'},
    {'name': 'meta-llama/Llama-3.1-8B-Instruct', 'short': 'llama'},
    {'name': 'google/gemma-2-9b-it',             'short': 'gemma'},
]

# Honest dataset names. mbpp_train is NOT LiveCodeBench — it's a separate MBPP split
# used as the 'harder code' partner of mbpp_test for the within-family pair.
DATASETS = ['gsm8k', 'math500', 'mmlu_pro', 'mbpp_test', 'mbpp_train', 'triviaqa']
PER_DATASET_N = {
    'gsm8k': 1000, 'math500': 500, 'mmlu_pro': 1000,
    'mbpp_test': 500, 'mbpp_train': 500, 'triviaqa': 1000,
}

MAX_NEW_TOKENS    = 2048
CHECKPOINT_EVERY  = 50
SEED              = 42
DEVICE            = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE             = torch.bfloat16 if DEVICE == 'cuda' else torch.float32  # bf16 for H100

# Hugging Face token
# Paste your token when prompted. This is needed for gated models like Llama/Gemma.
# It is stored only in this runtime environment variable, not hard-coded into the notebook file.
HF_TOKEN = os.environ.get('HF_TOKEN', '').strip()
if not HF_TOKEN:
    HF_TOKEN = getpass('Paste your Hugging Face token (input hidden): ').strip()
    os.environ['HF_TOKEN'] = HF_TOKEN
if HF_TOKEN:
    login(token=HF_TOKEN)
    print('Hugging Face login complete.')
else:
    print('No Hugging Face token provided. Gated models may fail to load.')

NTFY_CHANNEL = "phase4"

print(f'Device: {DEVICE}  Dtype: {DTYPE}')
print(f'Datasets: {DATASETS}')
print(f'Notifications: {"enabled" if NTFY_CHANNEL else "disabled"}')

# Save the configuration to disk for reproducibility
with open(FINAL_DIR / 'configs' / 'phase4_config.json', 'w') as f:
    json.dump({
        'models': [m['name'] for m in MODELS],
        'datasets': DATASETS,
        'per_dataset_n': PER_DATASET_N,
        'max_new_tokens': MAX_NEW_TOKENS,
        'dtype': str(DTYPE),
        'seed': SEED,
        'created': datetime.now().isoformat(),
    }, f, indent=2)

"""## Cell 4 — Notification helper (now includes error notifications)"""

def notify(message, title='Phase 4', priority='default'):
    print(f'[NOTIFY {priority}] {message}')
    if not NTFY_CHANNEL:
        return
    try:
        requests.post(
            f'https://ntfy.sh/{NTFY_CHANNEL}',
            data=message.encode('utf-8'),
            headers={'Title': title, 'Priority': priority},
            timeout=5,
        )
    except Exception as e:
        print(f'  (notification failed: {e})')

notify('Phase 4 v2 setup complete')

"""## Cell 5 — Dataset registry

Six datasets, honestly named. Each provides a loader, prompt formatter, predicted-answer extractor, and correctness checker.
"""

# ===== Chat template helper =====
def apply_chat_template_safely(tok, messages):
    """Apply chat template; disables Qwen3 thinking mode when supported."""
    model_name = getattr(tok, 'name_or_path', '') or ''
    is_qwen3 = 'qwen3' in model_name.lower()
    if is_qwen3:
        # Qwen3 supports thinking/non-thinking modes. We want comparable standard instruct output.
        if not messages or messages[0].get('role') != 'system':
            messages = [{'role': 'system', 'content': 'You are a helpful assistant. /no_think'}] + messages
        try:
            return tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            # Older tokenizer versions may not expose enable_thinking.
            return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# ===== Math answer utilities (shared by GSM8K, MATH-500) =====
def extract_boxed(text):
    if text is None: return None
    results, i = [], 0
    while True:
        idx = text.find(r'\boxed{', i)
        if idx < 0: break
        start = idx + len(r'\boxed{')
        depth, j = 1, start
        while j < len(text) and depth > 0:
            if text[j] == '{': depth += 1
            elif text[j] == '}': depth -= 1
            j += 1
        if depth == 0: results.append(text[start:j-1])
        i = j
    return results[-1].strip() if results else None

def extract_math_answer(text):
    if text is None: return None
    b = extract_boxed(text)
    if b is not None: return b.strip()
    m = re.findall(r'Final Answer\s*:?\s*(.+)', text, flags=re.IGNORECASE)
    if m: return m[-1].strip().split('\n')[0].strip().rstrip('.,')
    m = re.findall(r'(?:the answer is|answer is|answer:)\s*(.+)', text, flags=re.IGNORECASE)
    if m: return m[-1].strip().split('\n')[0].strip().rstrip('.,')
    return None

def normalize_math(s):
    if s is None: return None
    s = str(s).strip()
    b = extract_boxed(s)
    if b is not None: s = b.strip()
    s = s.replace('$','').replace('\\$','').replace(',','')
    s = re.sub(r'\\text\{[^}]*\}','', s)
    s = re.sub(r'\\,','', s)
    s = s.replace(r'\left','').replace(r'\right','')
    s = re.sub(r'\s+','', s)
    return s

def math_equivalent(pred, gold):
    if pred is None or gold is None: return False
    p, g = normalize_math(pred), normalize_math(gold)
    if p == g: return True
    try: return simplify(parse_latex(p) - parse_latex(g)) == 0
    except: pass
    try: return simplify(sympify(p) - sympify(g)) == 0
    except: pass
    return False

# ===== GSM8K =====
def gsm8k_load(n):
    ds = load_dataset('openai/gsm8k', 'main', split=f'test[:{n}]')
    return [(r['question'], re.search(r'####\s*(-?\d+(?:\.\d+)?)', r['answer']).group(1)) for r in ds]
def gsm8k_prompt(tok, q):
    messages = [{'role':'user','content':
        f"Solve step by step. End with: 'Final Answer: <number>'\n\n{q}"}]
    return apply_chat_template_safely(tok, messages)
def gsm8k_extract(text):
    cleaned = re.sub(r'\*+', '', text)
    patterns = [
        r'final answer[:\s]+\$?(-?[\d,]+(?:\.\d+)?)',
        r'answer is[:\s]+\$?(-?[\d,]+(?:\.\d+)?)',
        r'\\boxed\{[^}]*?(-?[\d,]+(?:\.\d+)?)[^}]*?\}',
        r'=\s*\$?(-?[\d,]+(?:\.\d+)?)\s*\.?\s*$',
        r'\$(-?[\d,]+(?:\.\d+)?)',
    ]
    for p in patterns:
        m = re.findall(p, cleaned, re.IGNORECASE | re.MULTILINE)
        if m:
            try: return float(m[-1].replace(',',''))
            except: continue
    return None
def gsm8k_correct(pred, gold):
    return pred is not None and abs(pred - float(gold)) < 1e-4

# ===== MATH-500 =====
def math500_load(n):
    ds = load_dataset('HuggingFaceH4/MATH-500', split=f'test[:{n}]')
    return [(r['problem'], r['answer']) for r in ds]
def math500_prompt(tok, q):
    messages = [{'role':'user','content':
        f"Solve step by step. Put final answer in \\boxed{{...}}.\n\n{q}"}]
    return apply_chat_template_safely(tok, messages)
math500_extract = extract_math_answer
math500_correct = math_equivalent

# ===== MMLU-Pro =====
def mmlu_pro_load(n):
    ds = load_dataset('TIGER-Lab/MMLU-Pro', split='test').shuffle(seed=SEED)
    n = min(n, len(ds))
    ds = ds.select(range(n))
    out = []
    for r in ds:
        opts = '\n'.join(f'{chr(65+i)}. {o}' for i, o in enumerate(r['options']))
        q = f"{r['question']}\n\nOptions:\n{opts}"
        # MMLU-Pro stores the gold answer as a letter in 'answer'; 'answer_index' also exists
        gold = r.get('answer') or (chr(65 + r['answer_index']) if 'answer_index' in r else 'A')
        out.append((q, gold))
    return out
def mmlu_pro_prompt(tok, q):
    messages = [{'role':'user','content':
        f"{q}\n\nReason step by step, then end with 'Answer: <letter>'."}]
    return apply_chat_template_safely(tok, messages)
def mmlu_pro_extract(text):
    cleaned = re.sub(r'\*+', '', text)
    patterns = [
        r'answer\s*[:=]\s*([A-J])\b',
        r'final answer\s*[:=]?\s*([A-J])\b',
        r'\\boxed\{([A-J])\}',
        r'\b([A-J])\b\s*\.?\s*$',
    ]
    for p in patterns:
        m = re.findall(p, cleaned, re.IGNORECASE | re.MULTILINE)
        if m: return m[-1].upper()
    return None
def mmlu_pro_correct(pred, gold):
    return pred is not None and pred.upper() == str(gold).upper()

# ===== TriviaQA =====
def triviaqa_load(n):
    ds = load_dataset('mandarjoshi/trivia_qa', 'rc.nocontext', split=f'validation[:{n}]')
    return [(r['question'], r['answer']['aliases']) for r in ds]
def triviaqa_prompt(tok, q):
    messages = [{'role':'user','content':
        f"Answer the following. End with 'Final Answer: <answer>'.\n\n{q}"}]
    return apply_chat_template_safely(tok, messages)
def triviaqa_extract(text):
    m = re.findall(r'Final Answer\s*:?\s*(.+)', text, flags=re.IGNORECASE)
    if m: return m[-1].strip().split('\n')[0].strip().rstrip('.,')
    m = re.findall(r'(?:the answer is|answer is|answer:)\s*(.+)', text, flags=re.IGNORECASE)
    if m: return m[-1].strip().split('\n')[0].strip().rstrip('.,')
    return None
def triviaqa_normalize(s):
    if s is None: return ''
    s = str(s).lower().strip()
    s = re.sub(r'[^a-z0-9\s]', '', s)
    return re.sub(r'\s+', ' ', s).strip()
def triviaqa_correct(pred, aliases):
    if pred is None: return False
    p = triviaqa_normalize(pred)
    return any(triviaqa_normalize(a) == p or triviaqa_normalize(a) in p or p in triviaqa_normalize(a)
               for a in aliases)

# ===== Code datasets — sandboxed test execution =====
def run_code_tests(code, test_code, timeout=10):
    full = code + '\n\n' + test_code
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'solution.py')
            with open(path, 'w') as f: f.write(full)
            result = subprocess.run(
                ['python', path], capture_output=True, timeout=timeout,
                cwd=tmpdir,  # confine to tempdir
            )
            return result.returncode == 0
    except (subprocess.TimeoutExpired, Exception):
        return False

def extract_code_block(text):
    m = re.findall(r'```(?:python)?\n?(.*?)```', text, re.DOTALL)
    return m[-1].strip() if m else text

def mbpp_test_load(n):
    ds = load_dataset('google-research-datasets/mbpp', 'sanitized', split='test')
    n = min(n, len(ds))
    ds = ds.select(range(n))
    return [(r['prompt'], (r['code'], r['test_list'])) for r in ds]
def mbpp_train_load(n):
    # 'mbpp_train' is a separate MBPP split used as within-family pair partner
    ds = load_dataset('google-research-datasets/mbpp', 'sanitized', split='train')
    n = min(n, len(ds))
    ds = ds.select(range(n))
    return [(r['prompt'], (r['code'], r['test_list'])) for r in ds]
def mbpp_prompt(tok, q):
    messages = [{'role':'user','content':
        f"{q}\n\nWrite the Python solution. Return only the code in a ```python``` block."}]
    return apply_chat_template_safely(tok, messages)
def mbpp_extract(text):
    return extract_code_block(text)
def mbpp_correct(pred, gold):
    if pred is None: return False
    _, test_list = gold
    return run_code_tests(pred, '\n'.join(test_list))

DATASET_FNS = {
    'gsm8k':      (gsm8k_load,      gsm8k_prompt,      gsm8k_extract,      gsm8k_correct),
    'math500':    (math500_load,    math500_prompt,    math500_extract,    math500_correct),
    'mmlu_pro':   (mmlu_pro_load,   mmlu_pro_prompt,   mmlu_pro_extract,   mmlu_pro_correct),
    'mbpp_test':  (mbpp_test_load,  mbpp_prompt,       mbpp_extract,       mbpp_correct),
    'mbpp_train': (mbpp_train_load, mbpp_prompt,       mbpp_extract,       mbpp_correct),
    'triviaqa':   (triviaqa_load,   triviaqa_prompt,   triviaqa_extract,   triviaqa_correct),
}

print(f'Registered {len(DATASET_FNS)} datasets')

"""## Cell 6 — Extractor self-tests"""

tests = [
    ('gsm8k', gsm8k_extract,     'Final Answer: 42', 42.0),
    ('gsm8k', gsm8k_extract,     'Therefore, the answer is **$1,596**.', 1596.0),
    ('gsm8k', gsm8k_extract,     'we get \\boxed{18}.', 18.0),
    ('math500', math500_extract, 'Therefore, \\boxed{\\frac{14}{3}}', r'\frac{14}{3}'),
    ('math500', math500_extract, 'Final Answer: 42', '42'),
    ('mmlu_pro', mmlu_pro_extract, 'The answer is C.', 'C'),
    ('mmlu_pro', mmlu_pro_extract, 'Answer: B', 'B'),
    ('triviaqa', triviaqa_extract, 'Final Answer: Paris', 'Paris'),
]
fails = 0
for name, fn, inp, expected in tests:
    got = fn(inp)
    ok = got == expected or (isinstance(got, float) and isinstance(expected, float) and abs(got-expected) < 1e-6)
    if not ok: fails += 1
    print(f'[{"OK  " if ok else "FAIL"}] {name}: {repr(inp[:60])} -> {got} (expected {expected})')
assert fails == 0, f'{fails} extractor self-tests failed'
print('All extractor tests passed.')

"""## Cell 7 — Generate-then-replay hidden state extraction (the v2 core)

**Pass 1 (generate):** produce text + logprobs/entropies. No hidden states stored, low memory.

**Pass 2 (replay):** single forward pass of `prompt + generated_text` with `output_hidden_states=True`. Extract activations at 4 positions, discard the buffer. Memory peaks at one sequence, not one-per-token.

**Position 3 is now true pre-answer.** We find the first occurrence of an answer-format marker (`Final Answer:`, `\boxed{`, `Answer:`) in the generated text and probe at the token just before it. If no marker is found, fall back to the penultimate generation token.
"""

# Markers indicating the model is about to commit to its final answer.
# Removed 'Therefore' (too common mid-CoT). We search for the LAST occurrence,
# since models often reconsider and re-state their answer near the end.
ANSWER_MARKERS = [
    r'Final Answer',
    r'final answer',
    r'\\boxed\{',
    r'Answer:',
    r'The answer is',
    r'answer is',
]

def find_pre_answer_char_pos(gen_text):
    """Return the character index of the LAST answer-format marker in gen_text.
    Returns None if no marker found.
    Uses 'last' rather than 'first' because models often restate answer near the end."""
    latest = None
    for marker in ANSWER_MARKERS:
        for m in re.finditer(marker, gen_text):
            if latest is None or m.start() > latest:
                latest = m.start()
    return latest  # may be None

def char_pos_to_token_pos(gen_text, char_pos, tokenizer):
    """Convert a character position in gen_text to a token offset within the generated tokens."""
    if char_pos is None or char_pos <= 0: return None
    prefix = gen_text[:char_pos]
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    return max(0, len(prefix_ids) - 1)  # token just before the marker

def generate_pass(model, tokenizer, prompt):
    """Pass 1: generate text + logprobs/entropies. No hidden states."""
    inputs = tokenizer(prompt, return_tensors='pt').to(DEVICE)
    prompt_len = inputs.input_ids.shape[1]
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
            output_hidden_states=False,  # critical: no hidden states
            pad_token_id=tokenizer.eos_token_id,
        )
    gen_ids = out.sequences[0][prompt_len:]
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

    logp_list, ent_list = [], []
    for s, tok in zip(out.scores, gen_ids):
        logp = torch.nn.functional.log_softmax(s[0].float(), dim=-1)
        logp_list.append(logp[tok].item())
        p = torch.exp(logp)
        ent_list.append(float(-(p * logp).sum().item()))

    return {
        'gen_ids':       gen_ids.cpu(),
        'gen_text':      gen_text,
        'prompt_len':    prompt_len,
        'mean_logprob':  float(np.mean(logp_list)) if logp_list else 0.0,
        'final_logprob': float(logp_list[-1]) if logp_list else 0.0,
        'mean_entropy':  float(np.mean(ent_list)) if ent_list else 0.0,
        'max_entropy':   float(np.max(ent_list)) if ent_list else 0.0,
        'n_gen_tokens':  len(gen_ids),
    }

def replay_extract(model, tokenizer, prompt, gen_ids, gen_text):
    """Pass 2: forward pass on full sequence, extract activations at 4 positions."""
    # Build full sequence: prompt tokens + generated tokens
    prompt_ids = tokenizer(prompt, return_tensors='pt').input_ids[0].to(DEVICE)
    full = torch.cat([prompt_ids, gen_ids.to(DEVICE)]).unsqueeze(0)
    prompt_len = len(prompt_ids)
    n_gen = len(gen_ids)

    with torch.no_grad():
        out = model(full, output_hidden_states=True, return_dict=True)

    # out.hidden_states is tuple of (num_layers + 1) tensors, each [1, seq_len, hidden_dim]
    # Index 0 = embedding output; we skip it.
    transformer_layers = out.hidden_states[1:]  # explicit: layer indices are transformer blocks
    num_layers = len(transformer_layers)
    hidden_dim = transformer_layers[0].shape[-1]

    # Compute the 4 positions
    # Position 1: end of question (last token of prompt)
    pos1_idx = prompt_len - 1
    # Position 2: middle of generation
    pos2_idx = prompt_len + max(0, n_gen // 2 - 1) if n_gen > 0 else pos1_idx
    # Position 3: true pre-answer — find marker in gen_text
    marker_char = find_pre_answer_char_pos(gen_text)
    if marker_char is not None and marker_char > 0:
        token_offset = char_pos_to_token_pos(gen_text, marker_char, tokenizer)
        if token_offset is not None and 0 <= token_offset < n_gen:
            pos3_idx = prompt_len + token_offset
        else:
            pos3_idx = prompt_len + max(0, n_gen - 2)
    else:
        pos3_idx = prompt_len + max(0, n_gen - 2)  # penultimate fallback
    # Position 4: last generated token
    pos4_idx = prompt_len + n_gen - 1 if n_gen > 0 else pos1_idx

    # Stack: [4 positions, num_layers, hidden_dim] as float16 to save space
    activations = np.zeros((4, num_layers, hidden_dim), dtype=np.float16)
    for p, idx in enumerate([pos1_idx, pos2_idx, pos3_idx, pos4_idx]):
        idx = min(max(0, idx), full.shape[1] - 1)
        for L in range(num_layers):
            activations[p, L, :] = transformer_layers[L][0, idx, :].cpu().float().numpy().astype(np.float16)

    # Free the buffer (note: cuda.empty_cache happens at checkpoint frequency, not per-example)
    del out, transformer_layers

    return {
        'activations':   activations,
        'pos3_used':     'marker' if marker_char is not None else 'penultimate',
    }

print('Generate-then-replay extraction defined.')
print('Layer indices = transformer block indices (embedding output excluded).')
print('Position 3 = true pre-answer via marker detection, with penultimate fallback.')

"""## Cell 8 — Cell runner with proper error handling

Failed examples are excluded entirely (saved to errors/) rather than poisoning the labels with zero vectors.
"""

def run_cell(model, tokenizer, model_short, dataset_name, n_samples):
    """Run inference for one (model, dataset). Returns valid activations and labels only."""
    load_fn, prompt_fn, extract_fn, correct_fn = DATASET_FNS[dataset_name]

    cell_dir  = RAW_DIR / 'activations' / model_short / dataset_name
    cell_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = cell_dir / 'checkpoint.npz'
    diag_path = RAW_DIR / 'diagnostics' / f'{model_short}__{dataset_name}.json'
    err_path  = RAW_DIR / 'errors' / f'{model_short}__{dataset_name}.json'
    diag_path.parent.mkdir(parents=True, exist_ok=True)
    err_path.parent.mkdir(parents=True, exist_ok=True)

    problems = load_fn(n_samples)

    # Resume
    if ckpt_path.exists() and diag_path.exists():
        ck = np.load(ckpt_path)
        all_acts = list(ck['activations'])
        all_labels = list(ck['labels'])
        all_lp = list(ck['logprobs'])
        all_orig_idx = list(ck['orig_indices'])
        with open(diag_path) as f: all_diag = json.load(f)
        with open(err_path) as f: all_err = json.load(f) if err_path.exists() else []
        last_processed = max([d['idx'] for d in all_diag] + [e['idx'] for e in all_err] + [-1])
        start = last_processed + 1
        print(f'  Resuming from problem {start}/{n_samples}')
    else:
        all_acts, all_labels, all_lp, all_orig_idx = [], [], [], []
        all_diag, all_err = [], []
        start = 0

    t_start = time.time()
    for i in range(start, n_samples):
        try:
            q, gold = problems[i]
            prompt = prompt_fn(tokenizer, q)

            gen_r = generate_pass(model, tokenizer, prompt)
            replay_r = replay_extract(model, tokenizer, prompt, gen_r['gen_ids'], gen_r['gen_text'])

            pred = extract_fn(gen_r['gen_text'])
            label = int(correct_fn(pred, gold))

            all_acts.append(replay_r['activations'])
            all_labels.append(label)
            all_lp.append([gen_r['mean_logprob'], gen_r['final_logprob'],
                           gen_r['mean_entropy'], gen_r['max_entropy']])
            all_orig_idx.append(i)
            all_diag.append({
                'idx': i, 'pred': str(pred)[:500], 'gold': str(gold)[:500],
                'label': label,
                'gen_text': gen_r['gen_text'],  # full generation, not truncated
                'n_tokens': gen_r['n_gen_tokens'],
                'pos3_used': replay_r['pos3_used'],
            })
        except Exception as e:
            err = {
                'idx': i,
                'error': str(e),
                'traceback': traceback.format_exc()[:1000],
            }
            all_err.append(err)
            print(f'  ERROR on problem {i}: {e}')
            notify(f'{model_short}/{dataset_name} problem {i} errored: {str(e)[:100]}',
                   priority='high')

        if (i + 1) % CHECKPOINT_EVERY == 0 or (i + 1) == n_samples:
            if all_acts:
                np.savez(ckpt_path,
                         activations=np.stack(all_acts),
                         labels=np.array(all_labels),
                         logprobs=np.array(all_lp),
                         orig_indices=np.array(all_orig_idx))
            with open(diag_path, 'w') as f: json.dump(all_diag, f, indent=2)
            with open(err_path, 'w') as f: json.dump(all_err, f, indent=2)
            torch.cuda.empty_cache()  # batched, not per-example, to avoid overhead
            elapsed = time.time() - t_start
            done_now = (i + 1) - start
            rate = done_now / elapsed if elapsed > 0 else 0
            eta = (n_samples - i - 1) / rate / 60 if rate > 0 else 0
            acc = sum(all_labels) / max(1, len(all_labels))
            print(f'  [{i+1:>4}/{n_samples}]  valid={len(all_labels)}  errors={len(all_err)}  '
                  f'acc={acc:.1%}  rate={rate:.2f} probs/s  ETA={eta:.1f} min')

    if not all_acts:
        return None, None, None, all_diag, all_err
    return np.stack(all_acts), np.array(all_labels), np.array(all_lp), all_diag, all_err

print('run_cell defined. Failed examples go to /raw/errors/, not labels.')

"""## Cell 9 — Probe training with shared-layer direction similarity"""

def best_layer_cv(H_pos, y, n_splits=5):
    """H_pos shape [N, layers, dim]. Returns (best_layer, best_auc, std, per_layer_means)."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    per_layer = []
    for layer in range(H_pos.shape[1]):
        X = H_pos[:, layer, :].astype(np.float32)
        aucs = []
        for tr, te in skf.split(X, y):
            if len(np.unique(y[te])) < 2: continue
            clf = LogisticRegression(max_iter=2000, C=1.0, class_weight='balanced')
            clf.fit(X[tr], y[tr])
            aucs.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:,1]))
        per_layer.append(float(np.mean(aucs)) if aucs else float('nan'))
    if all(np.isnan(per_layer)): return (-1, float('nan'), float('nan'), per_layer)
    best_layer = int(np.nanargmax(per_layer))
    return best_layer, per_layer[best_layer], 0.0, per_layer

def train_full_probe(H_pos, y, layer):
    X = H_pos[:, layer, :].astype(np.float32)
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight='balanced')
    clf.fit(X, y)
    return clf

def cross_auc(clf, H_tgt_pos, y_tgt, layer):
    X = H_tgt_pos[:, layer, :].astype(np.float32)
    if len(np.unique(y_tgt)) < 2: return float('nan')
    return float(roc_auc_score(y_tgt, clf.predict_proba(X)[:,1]))

def logprob_baselines(lp_arr, y):
    out = {}
    if len(np.unique(y)) < 2:
        return {n: float('nan') for n in ['mean_logp','final_logp','mean_ent','max_ent']}
    out['mean_logp']  = float(roc_auc_score(y, lp_arr[:, 0]))
    out['final_logp'] = float(roc_auc_score(y, lp_arr[:, 1]))
    out['mean_ent']   = float(roc_auc_score(y, -lp_arr[:, 2]))
    out['max_ent']    = float(roc_auc_score(y, -lp_arr[:, 3]))
    return out

def build_transfer_matrix(model_short, all_data, primary_position=2):
    """
    Returns dict containing: matrix, probes, baselines, best_layers, per_layer_aucs,
    shared_layer, shared_layer_probes, cosine_similarity (computed at shared_layer).
    """
    datasets = list(all_data.keys())

    # === Per-source best layer ===
    best_layers, probes, indomain, per_layer = {}, {}, {}, {}
    for src in datasets:
        H = all_data[src]['H']
        y = all_data[src]['y']
        if H is None or H.size == 0 or y.sum() < 5 or (1-y).sum() < 5:
            best_layers[src] = None; probes[src] = None
            indomain[src] = float('nan'); per_layer[src] = []
            continue
        H_p = H[:, primary_position, :, :]
        layer, auc, _, layer_means = best_layer_cv(H_p, y)
        best_layers[src] = layer
        indomain[src] = auc
        per_layer[src] = layer_means
        if layer >= 0:
            probes[src] = train_full_probe(H_p, y, layer)

    # === 6x6 transfer matrix ===
    matrix = {src: {} for src in datasets}
    for src in datasets:
        if probes[src] is None:
            for tgt in datasets: matrix[src][tgt] = float('nan')
            continue
        for tgt in datasets:
            if src == tgt:
                matrix[src][tgt] = indomain[src]
            else:
                if all_data[tgt]['H'] is None:
                    matrix[src][tgt] = float('nan')
                else:
                    H_tgt = all_data[tgt]['H'][:, primary_position, :, :]
                    matrix[src][tgt] = cross_auc(probes[src], H_tgt, all_data[tgt]['y'], best_layers[src])

    # === Baselines ===
    baselines = {ds: logprob_baselines(all_data[ds]['lp'], all_data[ds]['y'])
                 if all_data[ds]['lp'] is not None else {} for ds in datasets}

    # === Direction similarity at a shared layer ===
    # Pick the layer with best AVERAGE AUC across all sources, not each source's individual best.
    valid_per_layer = [v for v in per_layer.values() if v and len(v) > 0]
    if valid_per_layer:
        layer_means_across_sources = np.nanmean(np.stack(valid_per_layer), axis=0)
        shared_layer = int(np.nanargmax(layer_means_across_sources))
    else:
        shared_layer = None

    shared_probes = {}
    cosine = {src: {} for src in datasets}
    if shared_layer is not None:
        for src in datasets:
            H = all_data[src]['H']; y = all_data[src]['y']
            if H is None or H.size == 0 or y.sum() < 5 or (1-y).sum() < 5:
                shared_probes[src] = None
                continue
            H_p = H[:, primary_position, :, :]
            shared_probes[src] = train_full_probe(H_p, y, shared_layer)
        dirs = {}
        for src in datasets:
            if shared_probes.get(src) is not None:
                w = shared_probes[src].coef_[0]
                dirs[src] = w / (np.linalg.norm(w) + 1e-12)
        for a in datasets:
            for b in datasets:
                if a in dirs and b in dirs:
                    cosine[a][b] = float(np.dot(dirs[a], dirs[b]))
                else:
                    cosine[a][b] = float('nan')
    else:
        for a in datasets:
            for b in datasets:
                cosine[a][b] = float('nan')

    return {
        'matrix': matrix,
        'probes': probes,
        'baselines': baselines,
        'best_layers': best_layers,
        'per_layer_aucs': per_layer,
        'shared_layer': shared_layer,
        'shared_layer_probes': shared_probes,
        'cosine_similarity': cosine,
    }

print('Stage 3+4 functions defined. Direction similarity uses shared layer.')

"""## Cell 10 — SMOKE TEST (run this FIRST)


"""

# === Smoke test config ===
SMOKE_MODEL = MODELS[0]  # just use the first model (Qwen)
SMOKE_N = 10              # 10 problems per dataset

# Redirect output to smoke subdirectory so we don't pollute real results
SMOKE_RAW_DIR = PHASE4_DIR / 'raw' / 'smoke'
SMOKE_FINAL_DIR = PHASE4_DIR / 'final' / 'smoke'
for sub in ['activations', 'diagnostics', 'errors']:
    (SMOKE_RAW_DIR / sub).mkdir(parents=True, exist_ok=True)
SMOKE_FINAL_DIR.mkdir(parents=True, exist_ok=True)

# Temporarily monkey-patch RAW_DIR for the smoke test
_original_raw_dir = RAW_DIR
RAW_DIR = SMOKE_RAW_DIR

notify(f'SMOKE TEST starting: {SMOKE_MODEL["short"]} × {SMOKE_N} per dataset')
print(f'\n{"="*70}\nSMOKE TEST: {SMOKE_MODEL["name"]}  ({SMOKE_N} per dataset)\n{"="*70}\n')

smoke_model_name = SMOKE_MODEL['name']
smoke_short = SMOKE_MODEL['short']

try:
    print('Loading model...')
    tokenizer = AutoTokenizer.from_pretrained(smoke_model_name, token=HF_TOKEN, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        smoke_model_name, token=HF_TOKEN, torch_dtype=DTYPE, device_map='auto', trust_remote_code=True,
    )
    model.eval()
    print(f'  Loaded. Layers={model.config.num_hidden_layers}  Hidden_dim={model.config.hidden_size}')
except Exception as e:
    notify(f'SMOKE TEST: model load FAILED: {str(e)[:100]}', priority='high')
    raise

smoke_data = {}
smoke_summary = {}
for ds_name in DATASETS:
    print(f'\n--- smoke: {ds_name} (N={SMOKE_N}) ---')
    try:
        H, y, lp, diag, err = run_cell(model, tokenizer, smoke_short, ds_name, SMOKE_N)
        smoke_data[ds_name] = {'H': H, 'y': y, 'lp': lp}
        if y is not None and len(y) > 0:
            print(f'  Done. valid={len(y)} errors={len(err)} acc={y.mean():.1%}')
            smoke_summary[ds_name] = {
                'valid': int(len(y)), 'errors': len(err),
                'acc': float(y.mean()), 'correct': int(y.sum()),
                'incorrect': int((1-y).sum()),
            }
        else:
            print(f'  ZERO valid examples — investigate!')
            smoke_summary[ds_name] = {'valid': 0, 'errors': len(err)}
    except Exception as e:
        print(f'  SMOKE FAILED on {ds_name}: {e}')
        print(traceback.format_exc())
        smoke_data[ds_name] = {'H': None, 'y': None, 'lp': None}
        smoke_summary[ds_name] = {'error': str(e)}

# Free GPU
del model, tokenizer
torch.cuda.empty_cache()

# Save smoke summary
with open(SMOKE_FINAL_DIR / 'smoke_summary.json', 'w') as f:
    json.dump(smoke_summary, f, indent=2)

# Print results
print(f'\n\n{"="*70}\nSMOKE TEST SUMMARY\n{"="*70}')
print(f'{"dataset":<15} {"valid":>8} {"errors":>8} {"acc":>8} {"balance":>15}')
for ds, s in smoke_summary.items():
    if 'error' in s:
        print(f'  {ds:<13} FAILED: {s["error"][:50]}')
    elif s.get('valid', 0) == 0:
        print(f'  {ds:<13} ZERO VALID — investigate')
    else:
        bal = f'{s["correct"]}/{s["incorrect"]}'
        print(f'  {ds:<13} {s["valid"]:>8} {s["errors"]:>8} {s["acc"]:>7.1%} {bal:>15}')

# Optional: try building a transfer matrix from smoke data (will be noisy with N=10 but tests the code path)
print(f'\nAttempting smoke transfer matrix (will be noisy at N={SMOKE_N})...')
try:
    smoke_results = build_transfer_matrix(smoke_short, smoke_data)
    with open(SMOKE_FINAL_DIR / 'smoke_matrix.json', 'w') as f:
        json.dump({
            'matrix': smoke_results['matrix'],
            'baselines': smoke_results['baselines'],
            'best_layers': smoke_results['best_layers'],
            'shared_layer': smoke_results['shared_layer'],
        }, f, indent=2)
    print('  Matrix built successfully. Sample row (gsm8k -> *):')
    print(f'    {smoke_results["matrix"].get("gsm8k", {})}')
except Exception as e:
    print(f'  Transfer matrix build FAILED: {e}')
    print(traceback.format_exc())

# Restore real RAW_DIR
RAW_DIR = _original_raw_dir

notify('SMOKE TEST complete. Review summary before running full Cell 11.')
print(f'\n{"="*70}')
print('SMOKE TEST DONE.')
print(f'Outputs in: {SMOKE_FINAL_DIR}')
print('Review the summary above. If everything looks right, proceed to Cell 11.')
print('If anything looks wrong, debug here before committing to the full run.')
print(f'{"="*70}')

"""## Cell 11 — FULL EXPERIMENT main loop (the long-running one)


"""

for model_info in MODELS:
    model_name = model_info['name']
    short = model_info['short']

    # Skip if already done
    result_path = FINAL_DIR / 'results' / 'transfer_matrices' / f'{short}.json'
    if result_path.exists():
        print(f'Skipping {short} (already done at {result_path})')
        notify(f'Skipping {short} — results exist on disk')
        continue

    notify(f'Starting model: {short}', title='Phase 4 progress')
    print(f'\n{"="*70}\nMODEL: {model_name} ({short})\n{"="*70}\n')

    try:
        print('Loading model...')
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, token=HF_TOKEN, torch_dtype=DTYPE, device_map='auto', trust_remote_code=True,
        )
        model.eval()
        print(f'  Loaded. Layers={model.config.num_hidden_layers}  Hidden_dim={model.config.hidden_size}')
    except Exception as e:
        notify(f'FAILED to load {short}: {str(e)[:100]}', priority='high')
        print(f'Model load failed: {e}')
        continue

    model_data = {}
    for ds_name in DATASETS:
        print(f'\n--- {short} on {ds_name} (N={PER_DATASET_N[ds_name]}) ---')
        try:
            H, y, lp, diag, err = run_cell(model, tokenizer, short, ds_name, PER_DATASET_N[ds_name])
            model_data[ds_name] = {'H': H, 'y': y, 'lp': lp}
            if y is not None and len(y) > 0:
                print(f'  Done. valid={len(y)} (errors={len(err)})  acc={y.mean():.1%}  balance={y.sum()}/{(1-y).sum()}')
                notify(f'{short}: {ds_name} done, acc={y.mean():.1%}, errors={len(err)}')
            else:
                notify(f'{short}: {ds_name} produced no valid samples', priority='high')
        except Exception as e:
            print(f'  CELL FAILED: {e}\n{traceback.format_exc()}')
            notify(f'{short}: {ds_name} CRASHED — {str(e)[:100]}', priority='high')
            model_data[ds_name] = {'H': None, 'y': None, 'lp': None}

    # Free GPU
    del model, tokenizer
    torch.cuda.empty_cache()

    print(f'\n--- Stages 3+4 for {short} ---')
    try:
        results = build_transfer_matrix(short, model_data)

        # Save results JSON (without probes object)
        with open(result_path, 'w') as f:
            json.dump({
                'matrix': results['matrix'],
                'baselines': results['baselines'],
                'best_layers': results['best_layers'],
                'per_layer_aucs': results['per_layer_aucs'],
                'shared_layer': results['shared_layer'],
                'cosine_similarity': results['cosine_similarity'],
            }, f, indent=2)

        # Pickle probes
        with open(FINAL_DIR / 'results' / 'probes' / f'{short}_probes.pkl', 'wb') as f:
            pickle.dump({'best_layer_probes': results['probes'],
                         'shared_layer_probes': results['shared_layer_probes']}, f)

        notify(f'{short} complete. Shared layer: {results["shared_layer"]}',
               title='Model complete')
    except Exception as e:
        notify(f'{short} Stage 3+4 failed: {str(e)[:100]}', priority='high')
        print(f'Stage 3+4 failed: {e}\n{traceback.format_exc()}')

notify('All models attempted', title='Phase 4 generation complete')

"""## Cell 12 — Aggregate analysis (reads from disk, not memory)"""

# Load all model results from disk — safe to re-run after restart
ALL_RESULTS = {}
for m in MODELS:
    path = FINAL_DIR / 'results' / 'transfer_matrices' / f'{m["short"]}.json'
    if path.exists():
        with open(path) as f: ALL_RESULTS[m['short']] = json.load(f)
    else:
        print(f'Missing: {path}')

if not ALL_RESULTS:
    sys.exit('No model results found on disk.')

print(f'Loaded results for: {list(ALL_RESULTS.keys())}')

def matrix_to_array(matrix_dict, datasets):
    n = len(datasets)
    arr = np.full((n, n), np.nan)
    for i, src in enumerate(datasets):
        if src in matrix_dict:
            for j, tgt in enumerate(datasets):
                if tgt in matrix_dict[src]:
                    arr[i, j] = matrix_dict[src][tgt]
    return arr

matrices = [matrix_to_array(ALL_RESULTS[m['short']]['matrix'], DATASETS)
            for m in MODELS if m['short'] in ALL_RESULTS]
averaged = np.nanmean(np.stack(matrices), axis=0)

avg_dict = {DATASETS[i]: {DATASETS[j]: float(averaged[i,j]) for j in range(len(DATASETS))}
            for i in range(len(DATASETS))}
with open(FINAL_DIR / 'results' / 'transfer_matrices' / 'averaged.json', 'w') as f:
    json.dump(avg_dict, f, indent=2)

def plot_heatmap(arr, datasets, title, path, vmin=0.5, vmax=0.95, cmap='viridis'):
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(datasets))); ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.set_yticks(range(len(datasets))); ax.set_yticklabels(datasets)
    ax.set_xlabel('Target (evaluate)'); ax.set_ylabel('Source (train)')
    ax.set_title(title)
    for i in range(len(datasets)):
        for j in range(len(datasets)):
            if not np.isnan(arr[i,j]):
                color = 'white' if arr[i,j] < (vmax+vmin)/2 else 'black'
                ax.text(j, i, f'{arr[i,j]:.2f}', ha='center', va='center', color=color, fontsize=10)
    plt.colorbar(im, ax=ax, label='AUC')
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()

for m, arr in zip([m for m in MODELS if m['short'] in ALL_RESULTS], matrices):
    plot_heatmap(arr, DATASETS, f'Transfer matrix — {m["short"]}',
                 FINAL_DIR / 'figures' / f'transfer_{m["short"]}.png')
plot_heatmap(averaged, DATASETS, f'Averaged transfer matrix ({len(matrices)} models)',
             FINAL_DIR / 'figures' / 'transfer_averaged.png')

for m in MODELS:
    if m['short'] not in ALL_RESULTS: continue
    cos = ALL_RESULTS[m['short']]['cosine_similarity']
    arr = matrix_to_array(cos, DATASETS)
    sl = ALL_RESULTS[m['short']]['shared_layer']
    plot_heatmap(arr, DATASETS, f'Probe direction cosine (layer {sl}) — {m["short"]}',
                 FINAL_DIR / 'figures' / f'cosine_{m["short"]}.png',
                 vmin=-0.5, vmax=1.0, cmap='RdBu_r')

# Probe-vs-baseline gap — now computed for EVERY (source, target) pair,
# not just in-domain. For each cell, we compare the probe's AUC at evaluating
# the target against the target's best logprob baseline.
gap_summary = {}
gap_csv_rows = ['model,source,target,probe_auc,target_best_baseline,gap']
for m in MODELS:
    if m['short'] not in ALL_RESULTS: continue
    short = m['short']
    gaps = []
    for src in DATASETS:
        for tgt in DATASETS:
            probe = ALL_RESULTS[short]['matrix'].get(src, {}).get(tgt, float('nan'))
            bls = ALL_RESULTS[short]['baselines'].get(tgt, {})
            bl_vals = [v for v in bls.values() if not (isinstance(v, float) and np.isnan(v))]
            best_bl = max(bl_vals) if bl_vals else float('nan')
            gap = (probe - best_bl) if not (np.isnan(probe) or np.isnan(best_bl)) else float('nan')
            gaps.append({'source': src, 'target': tgt, 'probe': probe,
                         'target_best_baseline': best_bl, 'gap': gap})
            gap_csv_rows.append(f'{short},{src},{tgt},{probe:.4f},{best_bl:.4f},{gap:.4f}')
    gap_summary[short] = gaps
with open(FINAL_DIR / 'results' / 'baselines' / 'gap_summary.json', 'w') as f:
    json.dump(gap_summary, f, indent=2)
with open(FINAL_DIR / 'results' / 'baselines' / 'gap_summary.csv', 'w') as f:
    f.write('\n'.join(gap_csv_rows))

# CSV of averaged transfer matrix for easy Excel import
csv_lines = ['source,' + ','.join(DATASETS)]
for i, src in enumerate(DATASETS):
    csv_lines.append(f'{src},' + ','.join(f'{averaged[i,j]:.4f}' for j in range(len(DATASETS))))
with open(FINAL_DIR / 'results' / 'transfer_matrices' / 'averaged.csv', 'w') as f:
    f.write('\n'.join(csv_lines))

print(f'\nAveraged {len(DATASETS)}x{len(DATASETS)} matrix:')
print(f'  {"":12}' + ''.join(f'{ds[:10]:>11}' for ds in DATASETS))
for i, src in enumerate(DATASETS):
    print(f'  {src[:12]:12}' + ''.join(f'{averaged[i,j]:>11.3f}' for j in range(len(DATASETS))))
notify('Aggregate analysis complete')

"""## Cell 13 — PDF report"""

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

pdf_path = FINAL_DIR / 'phase4_report.pdf'
c = canvas.Canvas(str(pdf_path), pagesize=letter)
width, height = letter

c.setFont('Helvetica-Bold', 16)
c.drawString(50, height-70, 'Phase 4 — OOD Probe Transfer Experiment')
c.setFont('Helvetica', 10)
c.drawString(50, height-95, f'Generated: {datetime.now().isoformat()}')
c.drawString(50, height-110, f'Models: {[m["short"] for m in MODELS if m["short"] in ALL_RESULTS]}')
c.drawString(50, height-125, f'Datasets: {DATASETS}')

figs = sorted((FINAL_DIR / 'figures').glob('*.png'))
for fig in figs:
    c.showPage()
    c.setFont('Helvetica-Bold', 12)
    c.drawString(50, height-50, fig.stem)
    c.drawImage(ImageReader(str(fig)), 50, 100, width=width-100, height=height-200,
                preserveAspectRatio=True)

c.save()
print(f'PDF: {pdf_path}')
notify('PDF report generated')

"""## Cell 14 — Compress final outputs"""

subprocess.run(['tar', 'czf', '/workspace/phase4_final.tar.gz', '-C', '/workspace/phase4_outputs', 'final'])
size_mb = os.path.getsize('/workspace/phase4_final.tar.gz') / 1024**2
print(f'Archive: /workspace/phase4_final.tar.gz ({size_mb:.1f} MB)')

ip = os.environ.get('PUBLIC_IPADDR', '<your-vast-ip>')
port = os.environ.get('SSH_PORT', '<ssh-port>')

print(f'''
{"="*60}
PHASE 4 COMPLETE
{"="*60}

Step 1 — On YOUR LAPTOP terminal:
  scp -P {port} root@{ip}:/workspace/phase4_final.tar.gz ./

Step 2 — Verify on laptop:
  tar tzf phase4_final.tar.gz | head -30

Step 3 — Verify DELETE_AFTER_VERIFY checklist before deleting /raw/

Step 4 — TERMINATE instance from Vast.ai dashboard
         Confirm billing shows $0/hr
{"="*60}
''')
notify('Phase 4 done. Download and terminate.', title='Phase 4 COMPLETE', priority='high')
