"""Chain-of-Thought classifiers for local HuggingFace LLMs.

D4a — Zero-shot CoT: no examples; a "think step by step" trigger is appended
      to each user message.
D4b — Few-shot CoT: 10 examples sampled from df_train; the reasoning chain
      for each example is generated deterministically from d4b_reasoning_template.

Both conditions use a fairness-constrained system prompt that instructs the
model to ignore protected attributes (sex, race, age).
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from data_loader import DATASET_CONFIG
from metrics import evaluate_all_sensitive, save_results
from hf_classifier import serialize_row, _generate, _generate_batch, _is_gptoss, select_fewshot_examples
from prompts import D4_SYSTEM_PROMPTS, D4_CONFIG
from reasoning_generator import generate_reasoning


def _cot_batch_size(model) -> int:
    """Batch size for CoT generation (D4a/D4b).

    Smaller batch for >15B models to stay within VRAM (CoT traces are long).
    Override via the D4_BATCH env var if the GPU has more headroom.
    """
    import os
    is_large_model = sum(p.numel() for p in model.parameters()) > 15_000_000_000
    return int(os.environ.get("D4_BATCH", 2 if is_large_model else 8))


# ---------------------------------------------------------------------------
# COT RESPONSE PARSER
# ---------------------------------------------------------------------------

def parse_cot_response(response_text: str, pred_map: dict) -> tuple:
    """Extract the final prediction from a CoT response.

    Strategy: scan the last 3 non-empty lines for a valid label word.
    Fallback: last occurrence anywhere in the full text.

    Args:
        response_text (str): Full generated text (reasoning + answer).
        pred_map (dict): Map {str_label: int_label}.

    Returns:
        tuple[int, str]: (int_prediction, full_cot_text).
    """
    lines = [l.strip() for l in response_text.strip().split("\n") if l.strip()]

    for line in reversed(lines[-3:]):
        cleaned = line.lower().strip(".,!?;: \"'*#")
        for key in pred_map:
            if cleaned == key or cleaned.endswith(key):
                return pred_map[key], response_text
            if re.search(rf'\b{re.escape(key)}\b', cleaned):
                return pred_map[key], response_text

    # Fallback: last occurrence in full text
    positions = {key: response_text.lower().rfind(key) for key in pred_map}
    best = max(positions, key=lambda k: positions[k])
    if positions[best] != -1:
        return pred_map[best], response_text

    return 0, response_text


# ---------------------------------------------------------------------------
# D4a — ZERO-SHOT COT PREDICTION
# ---------------------------------------------------------------------------

def predict_zs_cot(model, tokenizer, X_test: pd.DataFrame, dataset_name: str, max_new_tokens: int = 512, temperature: float = 0.0, verbose: bool = True, batch_size: int = 8) -> tuple:
    """Zero-shot CoT prediction (D4a, Kojima et al. 2022).

    No few-shot examples. The trigger phrase is appended to each user message
    so the model generates its own step-by-step reasoning chain.

    Args:
        model: Loaded AutoModelForCausalLM.
        tokenizer: Corresponding tokenizer.
        X_test (pd.DataFrame): Test features (without target).
        dataset_name (str): Dataset name.
        max_new_tokens (int): Max tokens to generate (512 to allow full chain).
        temperature (float): Sampling temperature (0 = greedy).
        verbose (bool): Show tqdm progress bar.

    Returns:
        tuple[np.ndarray, list[dict]]:
            - Array of binary predictions.
            - CoT log: list of dicts with instance_idx, input, cot_output, prediction.
    """
    cfg        = D4_CONFIG[dataset_name]
    target_col = cfg["target"]
    pred_map   = cfg["pred_map"]
    system_msg = D4_SYSTEM_PROMPTS[dataset_name]
    cot_log    = []

    gptoss = _is_gptoss(model)
    effective_system = (system_msg + "\nReasoning: low") if gptoss else system_msg
    effective_max    = max(max_new_tokens, 256) if gptoss else max_new_tokens

    # Zero-shot CoT trigger appended to each user message
    ZS_COT_TRIGGER = "\nThink step by step, considering only non-protected features, then give your final answer on the last line."

    # Build all per-instance messages first, then generate in batches. Prompt
    # construction is identical to the one-at-a-time path; only generation is batched.
    row_texts = [serialize_row(row, target_col) + ZS_COT_TRIGGER for _, row in X_test.iterrows()]
    messages_all = [
        [{"role": "system", "content": effective_system},
         {"role": "user",   "content": rt}]
        for rt in row_texts
    ]

    predictions, cot_log = [], []
    pbar = tqdm(total=len(messages_all), desc="[hf_cot] D4a ZS-CoT") if verbose else None
    for start in range(0, len(messages_all), batch_size):
        batch_messages = messages_all[start:start + batch_size]
        responses = _generate_batch(model, tokenizer, batch_messages, effective_max, temperature)
        for offset, response in enumerate(responses):
            i = start + offset
            pred, cot_text = parse_cot_response(response, pred_map)
            cot_log.append({"instance_idx": i, "input": row_texts[i], "cot_output": cot_text, "prediction": pred})
            predictions.append(pred)
        if pbar:
            pbar.update(len(batch_messages))
    if pbar:
        pbar.close()

    return np.array(predictions), cot_log


# ---------------------------------------------------------------------------
# D4b — FEW-SHOT COT PREDICTION
# ---------------------------------------------------------------------------

def predict_fs_cot(model, tokenizer, X_test: pd.DataFrame, dataset_name: str,
                   df_train: pd.DataFrame, n_examples: int = 10, seed: int = 42,
                   max_new_tokens: int = 512, temperature: float = 0.0,
                   verbose: bool = True, batch_size: int = 8) -> tuple:
    """Few-shot CoT prediction (D4b, Wei et al. 2022) with dynamic demos.

    Samples n_examples balanced rows from df_train and generates the CoT
    reasoning chain for each using the deterministic template in
    d4b_reasoning_template.json.

    Args:
        model: Loaded AutoModelForCausalLM.
        tokenizer: Corresponding tokenizer.
        X_test: Test features (without target).
        dataset_name: Dataset name.
        df_train: Training split. Must contain the target column.
        n_examples: Number of demonstrations.
        seed: Seed used by select_fewshot_examples.
        max_new_tokens: Max tokens to generate.
        temperature: Sampling temperature (0 = greedy).
        verbose: Show tqdm progress bar.

    Returns:
        (predictions: np.ndarray, cot_log: list[dict]). Each log entry has
        instance_idx, input, cot_output, prediction.
    """
    cfg        = D4_CONFIG[dataset_name]
    target_col = cfg["target"]
    pred_map   = cfg["pred_map"]
    label_map  = cfg["label_map"]
    system_msg = D4_SYSTEM_PROMPTS[dataset_name]

    examples_df = select_fewshot_examples(df_train, target_col, n_examples, seed=seed)

    few_shot_pairs = []
    for _, ex in examples_df.iterrows():
        ex_input  = serialize_row(ex, target_col)
        ex_label  = label_map[int(ex[target_col])]
        ex_reason = generate_reasoning(ex, dataset_name, ex_label)
        few_shot_pairs.append({"role": "user",      "content": ex_input})
        few_shot_pairs.append({"role": "assistant", "content": ex_reason + "\n" + ex_label})

    gptoss = _is_gptoss(model)
    effective_system = (system_msg + "\nReasoning: low") if gptoss else system_msg
    effective_max    = max(max_new_tokens, 256) if gptoss else max_new_tokens

    # Build all per-instance messages first, then generate in batches. Prompt
    # construction is identical to the one-at-a-time path; only generation is batched.
    row_texts = [serialize_row(row, target_col) for _, row in X_test.iterrows()]
    messages_all = [
        [{"role": "system", "content": effective_system}]
        + few_shot_pairs
        + [{"role": "user", "content": rt}]
        for rt in row_texts
    ]

    predictions, cot_log = [], []
    pbar = tqdm(total=len(messages_all), desc="[hf_cot] D4b FS-CoT") if verbose else None
    for start in range(0, len(messages_all), batch_size):
        batch_messages = messages_all[start:start + batch_size]
        responses = _generate_batch(model, tokenizer, batch_messages, effective_max, temperature)
        for offset, response in enumerate(responses):
            i = start + offset
            pred, cot_text = parse_cot_response(response, pred_map)
            cot_log.append({"instance_idx": i, "input": row_texts[i], "cot_output": cot_text, "prediction": pred})
            predictions.append(pred)
        if pbar:
            pbar.update(len(batch_messages))
    if pbar:
        pbar.close()

    return np.array(predictions), cot_log


# ---------------------------------------------------------------------------
# EXPERIMENT BLOCKS
# ---------------------------------------------------------------------------

def run_zs_cot_experiment(df_test: pd.DataFrame, dataset_name: str, data_condition: str,
                          model, tokenizer, model_id: str, output_dir: Path = None,
                          test_set: str = None) -> tuple:
    """Run D4a (zero-shot CoT).

    When test_set is set ('D1' or 'D2'), output filenames get a `_test<tag>`
    suffix and CSVs carry a `test_set` column.

    Returns (y_true, y_pred, t1, t2, cot_log).
    """
    cfg             = DATASET_CONFIG[dataset_name]
    target_col      = cfg["target"]
    sensitive_feats = cfg["protected"]
    privileged_vals = cfg["privileged"]
    model_tag       = model_id.replace("/", "-").replace(".", "-")
    model_label     = f"LLM_ZSCoT_D4a_{model_tag}"

    X_test = df_test[[c for c in df_test.columns if c != target_col]]
    y_test = df_test[target_col].values

    y_pred, cot_log = predict_zs_cot(model=model, tokenizer=tokenizer, X_test=X_test, dataset_name=dataset_name, batch_size=_cot_batch_size(model))

    t1, t2 = evaluate_all_sensitive(
        y_true=y_test, y_pred=y_pred, df=df_test,
        sensitive_features=sensitive_feats, privileged_values=privileged_vals,
        model_name=f"{model_label} ({data_condition})", dataset_name=dataset_name,
    )
    if test_set is not None:
        t1 = t1.copy(); t1["test_set"] = test_set
        t2 = t2.copy(); t2["test_set"] = test_set

    if output_dir:
        suffix = f"_test{test_set}" if test_set is not None else ""
        prefix = f"{dataset_name}_{data_condition}_{model_label}{suffix}"
        save_results(t1, t2, output_dir, prefix=prefix)
        cot_df = pd.DataFrame(cot_log)
        if test_set is not None:
            cot_df["test_set"] = test_set
        cot_df.to_csv(Path(output_dir) / f"{prefix}_cot_log.csv", index=False)
        pred_df = df_test.copy()
        pred_df["y_pred"] = y_pred
        pred_df["y_true"] = y_test
        pred_df.insert(0, "instance_idx", range(len(pred_df)))
        if test_set is not None:
            pred_df["test_set"] = test_set
        pred_df.to_csv(Path(output_dir) / f"{prefix}_predictions.csv", index=False)
        print(f"[hf_cot] D4a results saved: {prefix}")

    return y_test, y_pred, t1, t2, cot_log


def run_fs_cot_experiment(df_train: pd.DataFrame, df_test: pd.DataFrame,
                          dataset_name: str, data_condition: str,
                          model, tokenizer, model_id: str,
                          output_dir: Path = None, test_set: str = None) -> tuple:
    """Run D4b (few-shot CoT) with dynamic demos sampled from df_train.

    When test_set is set ('D1' or 'D2'), output filenames get a `_test<tag>`
    suffix and CSVs carry a `test_set` column.

    Returns (y_true, y_pred, t1, t2, cot_log).
    """
    cfg             = DATASET_CONFIG[dataset_name]
    target_col      = cfg["target"]
    sensitive_feats = cfg["protected"]
    privileged_vals = cfg["privileged"]
    model_tag       = model_id.replace("/", "-").replace(".", "-")
    model_label     = f"LLM_FSCoT_D4b_{model_tag}"

    X_test = df_test[[c for c in df_test.columns if c != target_col]]
    y_test = df_test[target_col].values

    y_pred, cot_log = predict_fs_cot(
        model=model, tokenizer=tokenizer, X_test=X_test,
        dataset_name=dataset_name, df_train=df_train,
        batch_size=_cot_batch_size(model),
    )

    t1, t2 = evaluate_all_sensitive(
        y_true=y_test, y_pred=y_pred, df=df_test,
        sensitive_features=sensitive_feats, privileged_values=privileged_vals,
        model_name=f"{model_label} ({data_condition})", dataset_name=dataset_name,
    )
    if test_set is not None:
        t1 = t1.copy(); t1["test_set"] = test_set
        t2 = t2.copy(); t2["test_set"] = test_set

    if output_dir:
        suffix = f"_test{test_set}" if test_set is not None else ""
        prefix = f"{dataset_name}_{data_condition}_{model_label}{suffix}"
        save_results(t1, t2, output_dir, prefix=prefix)
        cot_df = pd.DataFrame(cot_log)
        if test_set is not None:
            cot_df["test_set"] = test_set
        cot_df.to_csv(Path(output_dir) / f"{prefix}_cot_log.csv", index=False)
        pred_df = df_test.copy()
        pred_df["y_pred"] = y_pred
        pred_df["y_true"] = y_test
        pred_df.insert(0, "instance_idx", range(len(pred_df)))
        if test_set is not None:
            pred_df["test_set"] = test_set
        pred_df.to_csv(Path(output_dir) / f"{prefix}_predictions.csv", index=False)
        print(f"[hf_cot] D4b results saved: {prefix}")

    return y_test, y_pred, t1, t2, cot_log

