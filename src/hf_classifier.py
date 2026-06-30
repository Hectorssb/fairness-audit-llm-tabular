"""
Few-shot classifier with local LLMs from HuggingFace.

Supported models:
    - meta-llama/Llama-3.1-8B-Instruct
    - Qwen/Qwen2.5-7B-Instruct
    - Qwen/Qwen2.5-14B-Instruct
    - google/gemma-4-E4B-it
    - google/gemma-4-31B-it
    - openai/gpt-oss-20b
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoProcessor,
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
}


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


# ---------------------------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------------------------

def load_model(model_id: str):
    """Load tokenizer and model from HuggingFace.

    Supports three loading modes per MODEL_CONFIGS:
      - load_in_8bit=True  : BitsAndBytes int8 quantization (e.g. Gemma-4-27B on A100 40GB)
      - load_in_4bit=True  : BitsAndBytes nf4 quantization
      - neither            : full dtype (float16 or bfloat16)

    Gemma-4 models require thinking mode disabled at tokenizer level to avoid
    <think>...</think> tokens being generated in standard conditions.

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

    # Gemma-4 is multimodal but we only use text — load the tokenizer directly
    # to avoid AutoProcessor pulling in Gemma4VideoProcessor (requires torchvision).
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

def _is_gemma4_id(model_id: str) -> bool:
    """Check by string model ID (used before the model is loaded)."""
    return "gemma-4" in model_id.lower()


def _is_gemma4(model) -> bool:
    model_id = getattr(model.config, "_name_or_path", "")
    return "gemma-4" in model_id.lower()


def _is_gptoss(model) -> bool:
    model_id = getattr(model.config, "_name_or_path", "")
    return "gpt-oss" in model_id.lower()


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
    # gpt-oss uses Harmony response format — "Reasoning: low" minimises
    # the chain-of-thought prefix so the answer fits within max_new_tokens.
    effective_system = (system_prompt + "\nReasoning: low") if _is_gptoss(model) else system_prompt
    messages = [{"role": "system", "content": effective_system}]
    messages.extend(few_shot_pairs)
    messages.append({"role": "user", "content": user_content})
    return messages


@torch.inference_mode()
def _generate(model, tokenizer, messages: list, max_new_tokens: int, temperature: float,
              enable_thinking: bool = False) -> str:
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
    if _is_gemma4(model):
        chat_template_kwargs["enable_thinking"] = enable_thinking

    inputs = tokenizer.apply_chat_template(messages, **chat_template_kwargs)
    if hasattr(inputs, "input_ids"):
        input_ids = inputs.input_ids.to(model.device)
    else:
        input_ids = inputs.to(model.device)

    # gpt-oss needs larger budget for its internal reasoning prefix.
    # Gemma-4 with thinking needs extra tokens for the <think> block.
    if _is_gptoss(model):
        effective_max = max(max_new_tokens, 512)
    elif _is_gemma4(model) and enable_thinking:
        effective_max = max(max_new_tokens, 1024)
    else:
        effective_max = max_new_tokens

    generation_kwargs = {
        "max_new_tokens": effective_max,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0:
        generation_kwargs["temperature"] = temperature

    output_ids = model.generate(input_ids, **generation_kwargs)
    new_token_ids = output_ids[0][input_ids.shape[-1]:]
    return tokenizer.decode(new_token_ids, skip_special_tokens=True)


@torch.inference_mode()
def _generate_batch(model, tokenizer, messages_list: list, max_new_tokens: int, temperature: float,
                    enable_thinking: bool = False) -> list:
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
        effective_max = max(max_new_tokens, 512)
    elif _is_gemma4(model) and enable_thinking:
        effective_max = max(max_new_tokens, 1024)
    else:
        effective_max = max_new_tokens

    # Tokenize each conversation individually (apply_chat_template is not batched)
    chat_template_kwargs = {"add_generation_prompt": True, "return_tensors": "pt"}
    if _is_gemma4(model):
        chat_template_kwargs["enable_thinking"] = enable_thinking
    encoded = [
        tokenizer.apply_chat_template(msgs, **chat_template_kwargs)
        for msgs in messages_list
    ]
    # Normalize BatchEncoding -> plain tensor
    encoded = [e.input_ids if hasattr(e, "input_ids") else e for e in encoded]

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
        "attention_mask": attention_mask,
    }
    if temperature > 0:
        generation_kwargs["temperature"] = temperature

    output_ids = model.generate(input_ids, **generation_kwargs)

    results = []
    for i, inp_len in enumerate(input_lengths):
        new_tokens = output_ids[i][max_len:]   # skip the padded input portion
        raw = tokenizer.decode(new_tokens, skip_special_tokens=True)
        results.append(raw)
    return results


# ---------------------------------------------------------------------------
# FEW-SHOT PREDICTION
# ---------------------------------------------------------------------------

def _parse_response(response: str, pred_map: dict, tag: str = "[hf_classifier]") -> int:
    """Parse a single model response into a binary prediction.

    Strategy (handles both terse and verbose/CoT responses):
    1. Strip Gemma-4 <think>...</think> blocks if present (thinking mode
       should be disabled for standard conditions, but strip as safeguard).
    2. First word — fast path for well-formatted responses.
    3. Last 3 non-empty lines — catches models that reason before answering.
    4. Last occurrence in full text — fallback for very long responses.
    """
    import re as _re
    if not response.strip():
        print(f"{tag} Empty response -> defaulting to 0")
        return 0

    # Strip any <think>...</think> block (Gemma-4 safeguard)
    response = _re.sub(r"<think>.*?</think>", "", response, flags=_re.DOTALL).strip()
    if not response.strip():
        print(f"{tag} Response was only thinking tokens -> defaulting to 0")
        return 0

    # 1. First word fast path
    word = response.strip().lower().split()[0].strip(".,!?;:\"'")
    if word in pred_map:
        return pred_map[word]

    # 2. Scan last 3 non-empty lines
    lines = [l.strip() for l in response.strip().split("\n") if l.strip()]
    for line in reversed(lines[-3:]):
        cleaned = line.lower().strip(".,!?;: \"'*#")
        for key in pred_map:
            if cleaned == key or cleaned.endswith(key):
                return pred_map[key]
            if _re.search(rf'\b{_re.escape(key)}\b', cleaned):
                return pred_map[key]

    # 3. Last occurrence anywhere in full text
    lower = response.lower()
    positions = {key: lower.rfind(key) for key in pred_map}
    best = max(positions, key=lambda k: positions[k])
    if positions[best] != -1:
        return pred_map[best]

    print(f"{tag} Unrecognized response: '{response[:120]}' -> defaulting to 0")
    return 0


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
    predictions = []

    pbar = tqdm(total=len(rows), desc="[hf_classifier] Predicting") if verbose else None
    for i in range(0, len(rows), batch_size):
        batch_rows = rows[i:i + batch_size]
        batch_messages = [
            _build_messages(model, system_prompt, few_shot_pairs, serialize_row(row, target_col))
            for row in batch_rows
        ]
        responses = _generate_batch(model, tokenizer, batch_messages, max_new_tokens, temperature)
        for resp in responses:
            predictions.append(_parse_response(resp, pred_map, "[hf_classifier]"))
        if pbar:
            pbar.update(len(batch_rows))
    if pbar:
        pbar.close()

    return np.array(predictions)


# ---------------------------------------------------------------------------
# EXPERIMENT
# ---------------------------------------------------------------------------

def run_fewshot_experiment(df_train: pd.DataFrame, df_test: pd.DataFrame, dataset_name: str, data_condition: str, model, tokenizer, model_id: str, n_examples: int = 10, output_dir: Path = None) -> tuple:
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

    examples_df = select_fewshot_examples(df_train, target_col, n_examples)
    print(f"[hf_classifier] FewShot ready with {len(examples_df)} examples.")

    X_test = df_test[[c for c in df_test.columns if c != target_col]]
    y_test = df_test[target_col].values

    is_large_model = sum(p.numel() for p in model.parameters()) > 15_000_000_000
    batch_size = 1 if is_large_model else 8

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

    if output_dir:
        prefix = f"{dataset_name}_{data_condition}_{model_label}"
        save_results(t1, t2, output_dir, prefix=prefix)

        pred_df = df_test.copy()
        pred_df["y_pred"] = y_pred
        pred_df["y_true"] = y_test
        pred_df.insert(0, "instance_idx", range(len(pred_df)))
        pred_df.to_csv(Path(output_dir) / f"{prefix}_predictions.csv", index=False)

    return y_test, y_pred, t1, t2
