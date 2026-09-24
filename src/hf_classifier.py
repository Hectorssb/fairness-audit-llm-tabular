"""
Few-shot classifier with local LLMs from HuggingFace.

Supported models:
    - meta-llama/Llama-3.1-8B-Instruct
    - Qwen/Qwen2.5-7B-Instruct
    - Qwen/Qwen2.5-14B-Instruct
    - google/gemma-4-E4B-it
    - google/gemma-4-31B-it
    - openai/gpt-oss-20b
    - Qwen/Qwen3-32B
    - microsoft/phi-4
"""

import os
import re
import time
import hashlib
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from data_loader import DATASET_CONFIG
from metrics import evaluate_all_sensitive, save_results
from prompts import DATASET_PROMPTS


# ---------------------------------------------------------------------------
# MODELS CONF
# ---------------------------------------------------------------------------

MODEL_CONFIGS = {
    "meta-llama/Llama-3.1-8B-Instruct": {
        "load_in_4bit": False,
        "load_in_8bit": False,
        "dtype": "float16",
        "desc": "Meta Llama 3.1 8B Instruct",
    },
    "Qwen/Qwen2.5-7B-Instruct": {
        "load_in_4bit": False,
        "load_in_8bit": False,
        "dtype": "float16",
        "desc": "Qwen 2.5 7B Instruct",
    },
    "Qwen/Qwen2.5-14B-Instruct": {
        "load_in_4bit": False,
        "load_in_8bit": False,
        "dtype": "float16",
        "desc": "Qwen 2.5 14B Instruct",
    },
    "google/gemma-4-E4B-it": {
        "load_in_4bit": False,
        "load_in_8bit": False,
        "dtype": "bfloat16",
        "desc": "Gemma 4 E4B Instruct",
    },
    "google/gemma-4-31B-it": {
        "load_in_4bit": True,
        "load_in_8bit": False,
        "dtype": "bfloat16",
        "desc": "Gemma 4 31B Instruct (nf4 4-bit)",
    },
    "openai/gpt-oss-20b": {
        "load_in_4bit": False,
        "load_in_8bit": False,
        "dtype": "bfloat16",
        "desc": "OpenAI GPT-OSS 20B (MoE)",
    },
    "Qwen/Qwen3-32B": {
        "load_in_4bit": True,
        "load_in_8bit": False,
        "dtype": "bfloat16",
        "desc": "Qwen 3 32B (nf4 4-bit)",
    },
    "microsoft/phi-4": {
        "load_in_4bit": False,
        "load_in_8bit": False,
        "dtype": "bfloat16",
        "desc": "Microsoft Phi-4 14B",
    },
}


BATCH_SIZES = {
    "meta-llama/Llama-3.1-8B-Instruct": {"fewshot": 24, "d4": 24},
    "Qwen/Qwen2.5-7B-Instruct":         {"fewshot": 24, "d4": 24},
    "google/gemma-4-E4B-it":            {"fewshot": 24, "d4": 24},
    "Qwen/Qwen2.5-14B-Instruct":        {"fewshot": 18, "d4": 18},
    "openai/gpt-oss-20b":               {"fewshot": 16, "d4": 4},
    "google/gemma-4-31B-it":            {"fewshot": 6,  "d4": 3},
    "Qwen/Qwen3-32B":                   {"fewshot": 12, "d4": 9},
    "microsoft/phi-4":                  {"fewshot": 18, "d4": 18},
}

# Applied to models absent from BATCH_SIZES, small enough to be safe untested.
DEFAULT_BATCH_SIZES = {"fewshot": 8, "d4": 4}


def _batch_sizes(model) -> dict:
    """Return the measured batch sizes for a loaded model.

    Args:
        model: Loaded model, matched by the id it was loaded from.

    Returns:
        dict: Keys ``fewshot`` and ``d4``.
    """
    model_id = getattr(model.config, "_name_or_path", "")
    for known_id, sizes in BATCH_SIZES.items():
        if model_id.lower().endswith(known_id.lower()) or model_id == known_id:
            return sizes
    return DEFAULT_BATCH_SIZES


# ---------------------------------------------------------------------------
# GENERATION INSTRUMENTATION
# ---------------------------------------------------------------------------

# Per-batch generation records accumulated by _generate_batch and flushed by
# save_generation_stats after each experiment pass.
GENERATION_LOG = []
_PARSE_FAILURES = 0

GPTOSS_MIN_TOKENS = int(os.environ.get("GPTOSS_MIN_TOKENS", 1024))
THINKING_MIN_TOKENS = int(os.environ.get("THINKING_MIN_TOKENS", 2048))


def _register_parse_failure():
    global _PARSE_FAILURES
    _PARSE_FAILURES += 1


def drain_generation_log() -> tuple:
    """Return and clear the accumulated records and parse-failure count."""
    global _PARSE_FAILURES
    records, failures = list(GENERATION_LOG), _PARSE_FAILURES
    GENERATION_LOG.clear()
    _PARSE_FAILURES = 0
    return records, failures


def save_generation_stats(output_dir, prefix: str, system_prompt: str = "", examples_df: pd.DataFrame = None):
    """Flush the per-batch generation records of one pass to a CSV.

    Each row carries batch size, input/output token counts and latency. The
    file also records SHA-256 hashes of the system prompt and the
    demonstrations, so every pass is traceable to its exact prompt
    configuration.

    Args:
        output_dir: Directory for the stats CSV. If None, records are discarded.
        prefix (str): Filename prefix, matching the pass result files.
        system_prompt (str): System prompt used in the pass.
        examples_df (pd.DataFrame, optional): Demonstrations used in the pass.
    """
    records, failures = drain_generation_log()
    if output_dir is None or not records:
        return
    stats = pd.DataFrame(records)
    stats["prompt_sha256"] = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()[:16]
    demos_repr = examples_df.to_csv(index=False) if examples_df is not None else ""
    stats["demos_sha256"] = hashlib.sha256(demos_repr.encode("utf-8")).hexdigest()[:16]
    stats["parse_failures_total"] = failures
    stats.to_csv(Path(output_dir) / f"{prefix}_generation_stats.csv", index=False)


# ---------------------------------------------------------------------------
# ROW SERIALIZATION
# ---------------------------------------------------------------------------

def serialize_row(row: pd.Series, target_col: str, label_map: dict = None) -> str:
    """Convert a dataset row to natural language text.

    Format: ``'feature1: value1, feature2: value2, ...'``
    If ``label_map`` is None, the label is omitted (for test instances).

    Args:
        row (pd.Series): DataFrame row.
        target_col (str): Name of the target column.
        label_map (dict, optional): Map {int_label: str_label} to include the
            label in the text. If None, the label is omitted.

    Returns:
        str: Row serialized as text.
    """
    parts = []
    for col, val in row.items():
        if col == target_col and label_map is None:
            continue
        if col == target_col and label_map is not None:
            parts.append(f"label: {label_map[int(val)]}")
        else:
            parts.append(f"{col}: {val}")
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# BALANCED FEW-SHOT EXAMPLE SELECTION
# ---------------------------------------------------------------------------

def select_fewshot_examples(df_train: pd.DataFrame, target_col: str, n_examples: int = 10, seed: int = 42) -> pd.DataFrame:
    """Select n_examples balanced examples (n/2 positive, n/2 negative).

    Replicates the strategy from Liu et al. (2024).

    Args:
        df_train (pd.DataFrame): Training DataFrame with target column.
        target_col (str): Name of the target column.
        n_examples (int): Total number of examples to select.
        seed (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: Balanced subset of df_train.
    """
    half_count = n_examples // 2
    pos = df_train[df_train[target_col] == 1].sample(
        min(half_count, (df_train[target_col] == 1).sum()), random_state=seed
    )
    neg = df_train[df_train[target_col] == 0].sample(
        min(half_count, (df_train[target_col] == 0).sum()), random_state=seed
    )
    return pd.concat([pos, neg]).sample(frac=1, random_state=seed)


def demo_seed_suffix(demo_seed: int) -> str:
    """Filename suffix identifying a non-default demonstration seed.
    """
    return "" if demo_seed == 42 else f"_seed{demo_seed}"


# ---------------------------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------------------------

def load_model(model_id: str):
    """Load tokenizer and model from HuggingFace.

    Supports three loading modes per MODEL_CONFIGS:
      - load_in_8bit=True  : BitsAndBytes int8 quantization
      - load_in_4bit=True  : BitsAndBytes nf4 quantization
      - neither            : full dtype (float16 or bfloat16)

    Args:
        model_id (str): Model identifier on HuggingFace Hub.

    Returns:
        tuple[AutoModelForCausalLM, AutoTokenizer]: (model, tokenizer).

    Raises:
        KeyError: If model_id is not in MODEL_CONFIGS.
    """
    model_config = MODEL_CONFIGS[model_id]
    print(f"[hf_classifier] Loading {model_id}")
    print(f"[hf_classifier]   Description: {model_config['desc']}")
    print(f"[hf_classifier]   8-bit: {model_config.get('load_in_8bit', False)} | "
          f"4-bit: {model_config['load_in_4bit']} | dtype: {model_config['dtype']}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.float16 if model_config["dtype"] == "float16" else torch.bfloat16

    if model_config["load_in_4bit"]:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto",
            max_memory={0: "44GiB"},
        )
    elif model_config.get("load_in_8bit", False):
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto",
            max_memory={0: "38GiB", "cpu": "48GiB"},
        )
    elif "gpt-oss" in model_id.lower():
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto",
            max_memory={0: "35GiB", "cpu": "48GiB"},
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map="auto",
        )

    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# GENERATION
# ---------------------------------------------------------------------------

def _is_gemma4(model) -> bool:
    model_id = getattr(model.config, "_name_or_path", "")
    return "gemma-4" in model_id.lower()


def _is_gptoss(model) -> bool:
    model_id = getattr(model.config, "_name_or_path", "")
    return "gpt-oss" in model_id.lower()


def _is_gemma4(model) -> bool:
    return "gemma-4" in getattr(model.config, "_name_or_path", "").lower()


def _reasons_natively(model) -> bool:
    """Check whether the model emits a <think> block unless told otherwise.

    Gemma-4 and Qwen3 both reason by default through their chat template. Such
    a model spends its budget on the reasoning block in every condition, not
    only in D4r, so it needs the thinking budget and an explicit
    `enable_thinking` in the template even when the condition disables it.
    """
    model_id = getattr(model.config, "_name_or_path", "").lower()
    return "gemma-4" in model_id or "qwen3" in model_id


def _build_messages(model, system_prompt: str, few_shot_pairs: list, user_content: str) -> list:
    """Build the message list for a single prediction.

    Gemma models do not support the 'system' role in their chat template.
    The system prompt is prepended to the first user message instead.

    Gemma-4 additionally supports a thinking mode that emits <think>...</think>
    tokens before the answer. For standard classification conditions (D1–D3,
    ZS, D4a/D4b) thinking is disabled via the chat template argument
    `enable_thinking=False`. D4r passes `enable_thinking=True` separately and
    does NOT use this function.

    Args:
        model: Loaded AutoModelForCausalLM.
        system_prompt (str): Instruction to set the task context.
        few_shot_pairs (list[dict]): Alternating user/assistant few-shot messages.
        user_content (str): The final user query to classify.

    Returns:
        list[dict]: Chat messages ready for apply_chat_template.
    """
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(few_shot_pairs)
    messages.append({"role": "user", "content": user_content})
    return messages


@torch.inference_mode()
def _generate(model, tokenizer, messages: list, max_new_tokens: int, temperature: float, enable_thinking: bool = False, open_answer_channel: bool = None) -> str:
    """Apply chat template, generate, and decode the new tokens.

    For Gemma-4 models the chat template accepts an `enable_thinking` kwarg
    that controls whether the model emits <think>...</think> reasoning tokens.
    Standard classification conditions pass enable_thinking=False (default).
    D4r passes enable_thinking=True.

    Args:
        model: Loaded AutoModelForCausalLM.
        tokenizer: Corresponding tokenizer.
        messages (list[dict]): Chat messages.
        max_new_tokens (int): Max tokens to generate.
        temperature (float): Sampling temperature (0 = greedy).
        enable_thinking (bool): Gemma-4 only — enable native thinking mode.

    Returns:
        str: Decoded generated text (new tokens only).
    """
    chat_template_kwargs = {"add_generation_prompt": True, "return_tensors": "pt"}
    if _reasons_natively(model):
        chat_template_kwargs["enable_thinking"] = enable_thinking

    if open_answer_channel is None:
        open_answer_channel = not enable_thinking
    input_ids = _encode_prompt(tokenizer, model, messages, chat_template_kwargs,
                               open_answer_channel=open_answer_channel).to(model.device)

    if _is_gptoss(model):
        effective_max = max(max_new_tokens, GPTOSS_MIN_TOKENS)
    elif enable_thinking or _reasons_natively(model):
        effective_max = max(max_new_tokens, THINKING_MIN_TOKENS)
    else:
        effective_max = max_new_tokens

    generation_kwargs = {
        "max_new_tokens": effective_max,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": _stop_token_ids(tokenizer, model, open_answer_channel),
    }
    if temperature > 0:
        generation_kwargs["temperature"] = temperature

    output_ids = model.generate(input_ids, **generation_kwargs)
    new_token_ids = output_ids[0][input_ids.shape[-1]:]
    return tokenizer.decode(new_token_ids, skip_special_tokens=True)


@torch.inference_mode()
def _generate_batch(model, tokenizer, messages_list: list, max_new_tokens: int, temperature: float, enable_thinking: bool = False, open_answer_channel: bool = None) -> list:
    """Batch inference: apply chat template to a list of conversations, generate in parallel.

    Sequences are left-padded so the generation aligns on the right.
    Returns a list of decoded response strings (new tokens only), one per input.

    Args:
        model: Loaded AutoModelForCausalLM.
        tokenizer: Corresponding tokenizer.
        messages_list (list[list[dict]]): One conversation per element.
        max_new_tokens (int): Max tokens to generate.
        temperature (float): Sampling temperature (0 = greedy).

    Returns:
        list[str]: Decoded responses, same length as messages_list.
    """
    if _is_gptoss(model):
        effective_max = max(max_new_tokens, GPTOSS_MIN_TOKENS)
    elif enable_thinking or _reasons_natively(model):
        effective_max = max(max_new_tokens, THINKING_MIN_TOKENS)
    else:
        effective_max = max_new_tokens

    # Tokenize each conversation individually
    chat_template_kwargs = {"add_generation_prompt": True, "return_tensors": "pt"}
    if _reasons_natively(model):
        chat_template_kwargs["enable_thinking"] = enable_thinking
    if open_answer_channel is None:
        open_answer_channel = not enable_thinking
    encoded = [
        _encode_prompt(tokenizer, model, msgs, chat_template_kwargs,
                       open_answer_channel=open_answer_channel)
        for msgs in messages_list
    ]

    # Left-pad to the same length
    max_len = max(e.shape[-1] for e in encoded)
    pad_id  = tokenizer.eos_token_id
    padded  = torch.full((len(encoded), max_len), pad_id, dtype=torch.long)
    attn    = torch.zeros((len(encoded), max_len), dtype=torch.long)
    for i, e in enumerate(encoded):
        seq_len = e.shape[-1]
        padded[i, max_len - seq_len:] = e[0]
        attn[i,   max_len - seq_len:] = 1

    input_ids      = padded.to(model.device)
    attention_mask = attn.to(model.device)
    input_lengths  = [e.shape[-1] for e in encoded]

    generation_kwargs = {
        "max_new_tokens": effective_max,
        "do_sample": temperature > 0,
        "pad_token_id": pad_id,
        "eos_token_id": _stop_token_ids(tokenizer, model, open_answer_channel),
        "attention_mask": attention_mask,
    }
    if temperature > 0:
        generation_kwargs["temperature"] = temperature

    start_time = time.perf_counter()
    output_ids = model.generate(input_ids, **generation_kwargs)
    latency_s = time.perf_counter() - start_time

    results = []
    output_tokens = 0
    for i, inp_len in enumerate(input_lengths):
        new_tokens = output_ids[i][max_len:]   # skip the padded input portion
        output_tokens += int((new_tokens != pad_id).sum())
        raw = tokenizer.decode(new_tokens, skip_special_tokens=True)
        results.append(raw)

    GENERATION_LOG.append({
        "n_sequences": len(encoded),
        "input_tokens": int(sum(input_lengths)),
        "output_tokens": output_tokens,
        "latency_s": round(latency_s, 3),
    })
    return results


def _length_sorted_order(tokenizer, texts: list) -> list:
    """Return instance indices sorted by ascending tokenized length.

    Grouping instances of similar length into the same batch minimises
    padding and lets whole batches finish generation together. Callers must
    map results back to the original order.

    Args:
        tokenizer: Tokenizer used to measure sequence lengths.
        texts (list[str]): The variable part of each instance's prompt.

    Returns:
        list[int]: Indices into texts, sorted by token length.
    """
    lengths = [len(tokenizer.encode(t, add_special_tokens=False)) for t in texts]
    return sorted(range(len(texts)), key=lengths.__getitem__)


# ---------------------------------------------------------------------------
# FEW-SHOT PREDICTION
# ---------------------------------------------------------------------------

def _stop_token_ids(tokenizer, model, open_answer_channel: bool = True) -> list:
    """Token ids that end an assistant turn.

    The model's generation config lists the ids its chat template closes a turn
    with. Gemma-4 ends every turn with ``<end_of_turn>`` and keeps only
    ``<eos>`` in the tokenizer, so stopping on the tokenizer's id alone lets it
    run on into a turn of its own making once it has answered.

    GPT-OSS closes each channel message with ``<|end|>`` and the whole turn with
    ``<|return|>``. When the prompt opened the final channel the answer is the
    only message, so either marker ends it. When the analysis channel is left
    open, ``<|end|>`` closes the reasoning before the answer has started, so the
    turn has to run on to ``<|return|>``.

    Args:
        tokenizer: Model tokenizer.
        model: Loaded model.
        open_answer_channel: Whether the prompt opened the final channel.

    Returns:
        list[int]: Ids to stop on, always including the tokenizer's EOS.
    """
    stops = [tokenizer.eos_token_id]
    config_eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if config_eos is not None:
        stops.extend(config_eos if isinstance(config_eos, (list, tuple)) else [config_eos])

    markers = ["<end_of_turn>"]
    if _is_gptoss(model):
        markers.append("<|return|>")
        if open_answer_channel:
            markers.append("<|end|>")
    for marker in markers:
        token_id = tokenizer.convert_tokens_to_ids(marker)
        if token_id is not None and token_id >= 0 and token_id != tokenizer.unk_token_id:
            stops.append(token_id)
    return list(dict.fromkeys(int(t) for t in stops if t is not None))


def _encode_prompt(tokenizer, model, messages: list, chat_template_kwargs: dict, open_answer_channel: bool = True):
    """Tokenize one conversation, opening the answer channel for GPT-OSS.

    The Harmony template ends the prompt at ``<|start|>assistant`` and lets the
    model pick a channel. On a classification prompt it picks ``analysis`` and
    tabulates the demonstrations feature by feature, exhausting the budget
    without ever answering. Appending the channel header the demonstrations
    themselves use makes it answer directly.

    Conditions that measure the model's reasoning pass
    ``open_answer_channel=False`` so the analysis channel stays available.

    Args:
        tokenizer: Model tokenizer.
        model: Loaded model, used to detect GPT-OSS.
        messages: One conversation.
        chat_template_kwargs: Extra arguments for ``apply_chat_template``.
        open_answer_channel: Whether to start the response in the final channel.

    Returns:
        torch.Tensor: Token ids, shape (1, seq_len).
    """
    if _is_gptoss(model) and open_answer_channel:
        prefix = "<|channel|>final<|message|>"
    elif _is_gemma4(model) and chat_template_kwargs.get("enable_thinking") and not open_answer_channel:
        prefix = "<|channel>thought\n"
    else:
        encoded = tokenizer.apply_chat_template(messages, **chat_template_kwargs)
        return encoded.input_ids if hasattr(encoded, "input_ids") else encoded

    kwargs = {k: v for k, v in chat_template_kwargs.items() if k != "return_tensors"}
    kwargs["tokenize"] = False
    prompt = tokenizer.apply_chat_template(messages, **kwargs) + prefix
    return tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids


def _strip_analysis_channel(response: str, labels=()) -> str:
    """Return the answer channel of a Harmony-formatted response.

    GPT-OSS emits its reasoning in an ``analysis`` channel before answering in
    a ``final`` one. Decoding the raw text yields the reasoning, whose mentions
    of the labels the parser would otherwise read as the answer. When the
    response never reaches the final channel the reasoning ran out without
    answering; its tail is kept only when it states one of ``labels`` as an
    answer, which is how a label asked for afterwards is appended.

    Args:
        response: Decoded model output.
        labels: Label words an answer may state.

    Returns:
        The final channel, or an empty string when it is absent. Responses
        without any channel marker are returned unchanged.
    """

    opens_analysis = response.lstrip().startswith("analysis")
    turn_marker = re.compile(r"<\|(?:channel|start|end|return|message)\|>"
                             r"|assistant(?:analysis|final|commentary)")
    if not (opens_analysis or turn_marker.search(response)):
        return response

    if not opens_analysis:
        return turn_marker.split(response)[0].strip()

    channel_open = re.compile(r"<\|channel\|>\s*final\s*(?:<\|message\|>)?"
                              r"|assistantfinal"
                              r"|(?:^|\n)\s*final<\|message\|>"
                              r"|(?:^|\n)\s*final(?=[>|:\s]|$)")
    match = channel_open.search(response)
    if match is None:
        tail = response.strip()[-160:]
        return tail if _answer_line(tail, labels) is not None else ""
    return response[match.end():].lstrip(">|: \n")


def _clean_line(line: str) -> str:
    return line.lower().strip(".,!?;: \"'*#_`()[]")


_ANSWER_CUE = re.compile(r"\b(?:answer|prediction|predict|label|conclusion|verdict|result|"
                         r"classif\w*|decision|output|final)\b")


def _answer_line(text: str, labels):
    """The label a response states as its answer, or None.

    A label glued to the end of a sentence, 'the risk is bad.bad' or
    "'good' or 'bad'.good", is the answer token a model appends to its last
    sentence and counts wherever it appears. Otherwise the last line counts
    when it is a label on its own or ends with one after a separator: 'bad',
    'Final answer: yes', 'the answer is good'. An earlier line counts only
    when it also announces an answer, so a closing remark after 'Final
    answer: yes' does not hide it, while a reasoning step that happens to end
    in a label word, 'savings are good', or a list header, 'Bad:', stays
    prose. A line naming more than one label, as in the echoed instruction
    "exactly one word: 'good' or 'bad'", states no answer.

    Args:
        text: Response text.
        labels: Label words to look for.

    Returns:
        The label found, or None.
    """
    lines = [cleaned for cleaned in (_clean_line(l) for l in text.split("\n")) if cleaned]
    for position, cleaned in enumerate(reversed(lines)):
        glued = [key for key in labels if re.search(rf"\.{re.escape(key)}\b", cleaned)]
        if glued:
            return glued[-1] if len(glued) == 1 else max(
                glued, key=lambda key: cleaned.rfind("." + key))
        named = [key for key in labels if re.search(rf"\b{re.escape(key)}\b", cleaned)]
        if len(named) != 1:
            continue
        key = named[0]
        states = (cleaned == key
                  or re.search(rf"(?:^|[\s:*'\"(\[])({re.escape(key)})$", cleaned))
        if states and (position == 0 or _ANSWER_CUE.search(cleaned)):
            return key
    return None


def _fallback_label(text: str, labels):
    """The label mentioned closest to the end of the last lines, or None.

    Used only when no line states an answer. The last three non-empty lines
    are read first, then the whole text; within a line the label whose last
    mention ends latest wins, so 'not yes but no' reads as 'no'.

    Args:
        text: Response text.
        labels: Label words to look for.

    Returns:
        The label found, or None.
    """
    def latest(fragment):
        best, best_end = None, -1
        for key in labels:
            for m in re.finditer(rf"\b{re.escape(key)}\b", fragment):
                if m.end() > best_end:
                    best, best_end = key, m.end()
        return best

    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    for line in reversed(lines[-3:]):
        key = latest(line.lower())
        if key is not None:
            return key
    return latest(text.lower())


def _extract_label(response: str, pred_map: dict):
    """The label a response commits to, or None when it states none.

    Args:
        response: Decoded model output.
        pred_map: Map {label: int}.

    Returns:
        int or None.
    """
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    response = _strip_analysis_channel(response, pred_map)
    if not response.strip():
        return None
    word = response.strip().lower().split()[0].strip(".,!?;:\"'")
    if word in pred_map:
        return pred_map[word]
    key = _answer_line(response, pred_map)
    return None if key is None else pred_map[key]


def _parse_response(response: str, pred_map: dict, tag: str = "[hf_classifier]") -> int:
    """Parse a single model response into a binary prediction.

    Strategy (handles both terse and verbose/CoT responses):
    1. Strip <think>...</think> blocks and the Harmony analysis channel.
    2. First word, or a line that states the answer.
    3. Label closest to the end of the last lines, then of the whole text.
    """
    if not response.strip():
        print(f"{tag} Empty response -> defaulting to 0")
        _register_parse_failure()
        return 0

    label = _extract_label(response, pred_map)
    if label is not None:
        return label

    text = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    if not text:
        print(f"{tag} Response was only thinking tokens -> defaulting to 0")
        _register_parse_failure()
        return 0
    text = _strip_analysis_channel(text, pred_map)
    if not text.strip():
        print(f"{tag} Response never reached its final channel -> defaulting to 0")
        _register_parse_failure()
        return 0

    key = _fallback_label(text, pred_map)
    if key is not None:
        return pred_map[key]

    print(f"{tag} Unrecognized response: '{response[:120]}' -> defaulting to 0")
    _register_parse_failure()
    return 0


def _finish_unanswered(model, tokenizer, batch_messages, responses, pred_map, tag="[hf_classifier]"):
    """Ask again for the label on responses that ran out mid-reasoning.

    A reasoning model can spend its whole budget on the trace and stop before
    stating a label, leaving nothing to score. Rather than recording the
    parser's default, the trace so far is fed back with a request for the label
    alone, so the answer still comes from the model's own reasoning.

    Args:
        model: Loaded model.
        tokenizer: Corresponding tokenizer.
        batch_messages: Conversations sent in this batch.
        responses: Their generated texts.
        pred_map: Map {label: int} for the dataset.
        tag: Log prefix.

    Returns:
        list[str]: Responses, with the unanswered ones extended by the label.
    """
    unanswered = [i for i, text in enumerate(responses)
                  if _extract_label(text, pred_map) is None]
    if not unanswered:
        return responses

    labels = " or ".join(f"'{k}'" for k in pred_map)
    follow_up = [
        batch_messages[i] + [
            {"role": "assistant", "content": responses[i]},
            {"role": "user", "content": f"Answer now with exactly one word: {labels}."},
        ]
        for i in unanswered
    ]
    print(f"  {tag} {len(unanswered)}/{len(responses)} responses ended without a "
          f"label; asking for it directly.")
    finals = _generate_batch(model, tokenizer, follow_up, max_new_tokens=8,
                             temperature=0.0)

    responses = list(responses)
    for i, final in zip(unanswered, finals):
        responses[i] = responses[i] + "\n" + final.strip()
    return responses


def _fewshot_batch_size(model) -> int:
    """Batch size for short few-shot generations (max_new_tokens=20).

    Override via the FEWSHOT_BATCH env var. See ``BATCH_SIZES``.
    """
    return int(os.environ.get("FEWSHOT_BATCH", _batch_sizes(model)["fewshot"]))


def predict_fewshot(model, tokenizer, X_test: pd.DataFrame, examples_df: pd.DataFrame, system_prompt: str, target_col: str, label_map: dict, pred_map: dict, max_new_tokens: int = 20, temperature: float = 0.0, verbose: bool = True, batch_size: int = 8) -> np.ndarray:
    """Few-shot prediction over X_test using the provided examples.

    Args:
        model: Loaded AutoModelForCausalLM.
        tokenizer: Corresponding tokenizer.
        X_test (pd.DataFrame): Test features (without target).
        examples_df (pd.DataFrame): Few-shot examples including target column.
        system_prompt (str): System message for the model.
        target_col (str): Name of the target column in examples_df.
        label_map (dict): Map {int_label: str_label} for serializing examples.
        pred_map (dict): Map {str_label: int_label} for parsing predictions.
        max_new_tokens (int): Max tokens to generate per instance.
        temperature (float): Sampling temperature (0 = greedy).
        verbose (bool): If True, shows a tqdm progress bar.
        batch_size (int): Number of instances to process in parallel.

    Returns:
        np.ndarray: Array of binary predictions.
    """
    few_shot_pairs = []
    for _, ex in examples_df.iterrows():
        few_shot_pairs.append({"role": "user",      "content": serialize_row(ex, target_col)})
        few_shot_pairs.append({"role": "assistant", "content": label_map[int(ex[target_col])]})

    rows = [row for _, row in X_test.iterrows()]
    user_contents = [serialize_row(row, target_col) for row in rows]
    order = _length_sorted_order(tokenizer, user_contents)
    predictions = [0] * len(rows)

    pbar = tqdm(total=len(rows), desc="[hf_classifier] Predicting") if verbose else None
    for i in range(0, len(order), batch_size):
        batch_idx = order[i:i + batch_size]
        batch_messages = [
            _build_messages(model, system_prompt, few_shot_pairs, user_contents[j])
            for j in batch_idx
        ]
        responses = _generate_batch(model, tokenizer, batch_messages, max_new_tokens, temperature)
        for j, resp in zip(batch_idx, responses):
            predictions[j] = _parse_response(resp, pred_map, "[hf_classifier]")
        if pbar:
            pbar.update(len(batch_idx))
    if pbar:
        pbar.close()

    return np.array(predictions)


# ---------------------------------------------------------------------------
# EXPERIMENT
# ---------------------------------------------------------------------------

def run_fewshot_experiment(df_train: pd.DataFrame, df_test: pd.DataFrame, dataset_name: str, data_condition: str, model, tokenizer, model_id: str, n_examples: int = 10, output_dir: Path = None, demo_seed: int = 42, test_tag: str = "D1") -> tuple:
    """Run the full few-shot experiment for a data condition.

    Args:
        df_train (pd.DataFrame): Training data with target column.
        df_test (pd.DataFrame): Test data with target column.
        dataset_name (str): Dataset name.
        data_condition (str): Data condition ('D1_original', 'D2_fair_causal', 'D3_resampled').
        model: Loaded AutoModelForCausalLM.
        tokenizer: Corresponding tokenizer.
        model_id (str): Model identifier (for result labeling).
        n_examples (int): Number of few-shot examples.
        output_dir (Path, optional): Directory to save results.
        demo_seed (int): Seed for demonstration sampling and order. Non-default
            values add a `_seed<n>` suffix to output filenames.
        test_tag (str): Canonical test split the predictions are evaluated on,
            "D1" or "D2". Values other than "D1" add a `_test<tag>` suffix to
            output filenames and are recorded in a `test_set` column.

    Returns:
        tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]: (y_true, y_pred, table1, table2).
    """
    cfg             = DATASET_CONFIG[dataset_name]
    target_col      = cfg["target"]
    sensitive_feats = cfg["protected"]
    privileged_vals = cfg["privileged"]
    prompts         = DATASET_PROMPTS[dataset_name]

    model_tag    = model_id.replace("/", "-").replace(".", "-")
    model_label  = f"LLM_FewShot_{model_tag}"

    examples_df = select_fewshot_examples(df_train, target_col, n_examples, seed=demo_seed)
    print(f"[hf_classifier] FewShot ready with {len(examples_df)} examples.")

    X_test = df_test[[c for c in df_test.columns if c != target_col]]
    y_test = df_test[target_col].values

    batch_size = _fewshot_batch_size(model)

    y_pred = predict_fewshot(
        model=model,
        tokenizer=tokenizer,
        X_test=X_test,
        examples_df=examples_df,
        system_prompt=prompts["system"],
        target_col=target_col,
        label_map=prompts["label_map"],
        pred_map=prompts["pred_map"],
        batch_size=batch_size,
    )

    t1, t2 = evaluate_all_sensitive(
        y_true=y_test,
        y_pred=y_pred,
        df=df_test,
        sensitive_features=sensitive_feats,
        privileged_values=privileged_vals,
        model_name=f"{model_label} ({data_condition})",
        dataset_name=dataset_name,
    )

    t1["test_set"] = test_tag
    t2["test_set"] = test_tag

    if output_dir:
        suffix = "" if test_tag == "D1" else f"_test{test_tag}"
        prefix = f"{dataset_name}_{data_condition}_{model_label}{suffix}{demo_seed_suffix(demo_seed)}"
        save_results(t1, t2, output_dir, prefix=prefix)
        save_generation_stats(output_dir, prefix, prompts["system"], examples_df)

        pred_df = df_test.copy()
        pred_df["y_pred"] = y_pred
        pred_df["y_true"] = y_test
        pred_df.insert(0, "instance_idx", range(len(pred_df)))
        pred_df["test_set"] = test_tag
        pred_df.to_csv(Path(output_dir) / f"{prefix}_predictions.csv", index=False)

    return y_test, y_pred, t1, t2
