"""Zero-shot classifier using local HuggingFace LLMs.

Conditions:
  ZS_D1 — original column names and domain prompt.
  ZS_D5 — generic column names (feat_00, feat_01, ...) and a generic prompt
          without domain mention.
"""

import random
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from data_loader import DATASET_CONFIG
from hf_classifier import DATASET_PROMPTS, serialize_row, _generate_batch, _parse_response, _is_gptoss
from metrics import evaluate_all_sensitive, save_results, print_results
from prompts import ZS_D5_SYSTEM_PROMPTS

SEED = 42

RESULTS_DIR = Path(__file__).parent.parent / "results"
DATA_DIR    = Path(__file__).parent.parent / "data"


# ---------------------------------------------------------------------------
# COLUMN PERMUTATION
# ---------------------------------------------------------------------------

def build_column_mapping(df: pd.DataFrame, target_col: str, seed: int = SEED) -> dict:
    """Mapping from columns: original_name -> feat_NN (permuted with fixed seed).

    The target is mapped to 'target'.

    Args:
        df (pd.DataFrame): DataFrame with columns to rename.
        target_col (str): Name of the target column.
        seed (int): Seed for the permutation.

    Returns:
        dict: Map {original_name: generic_name}.
    """
    rng = random.Random(seed)
    feature_cols = [c for c in df.columns if c != target_col]
    shuffled = feature_cols[:]
    rng.shuffle(shuffled)
    mapping = {orig: f"feat_{i:02d}" for i, orig in enumerate(shuffled)}
    mapping[target_col] = "target"
    return mapping


# ---------------------------------------------------------------------------
# ZERO-SHOT PREDICTION
# ---------------------------------------------------------------------------

def predict_zeroshot(model, tokenizer, X_test: pd.DataFrame, system_prompt: str, target_col: str, pred_map: dict, max_new_tokens: int = 20, temperature: float = 0.0, verbose: bool = True, batch_size: int = 8) -> np.ndarray:
    """Zero-shot prediction over X_test (no few-shot examples).

    Args:
        model: Loaded AutoModelForCausalLM.
        tokenizer: Corresponding tokenizer.
        X_test (pd.DataFrame): Test features (without target).
        system_prompt (str): System message for the model.
        target_col (str): Target column name (used by serialize_row to skip it).
        pred_map (dict): Map {str_label: int_label} for parsing predictions.
        max_new_tokens (int): Max tokens to generate per instance.
        temperature (float): Sampling temperature (0 = greedy).
        verbose (bool): If True, shows a tqdm progress bar.
        batch_size (int): Number of instances to process in parallel.

    Returns:
        np.ndarray: Array of binary predictions.
    """
    rows = [row for _, row in X_test.iterrows()]
    predictions = []
    gptoss = _is_gptoss(model)
    effective_system = (system_prompt + "\nReasoning: low") if gptoss else system_prompt
    effective_max    = max(max_new_tokens, 256) if gptoss else max_new_tokens

    pbar = tqdm(total=len(rows), desc="[hf_zeroshot] Predicting") if verbose else None
    for i in range(0, len(rows), batch_size):
        batch_rows = rows[i:i + batch_size]
        batch_messages = [
            [{"role": "system", "content": effective_system},
             {"role": "user",   "content": serialize_row(row, target_col)}]
            for row in batch_rows
        ]
        responses = _generate_batch(model, tokenizer, batch_messages, effective_max, temperature)
        for resp in responses:
            predictions.append(_parse_response(resp, pred_map, "[hf_zeroshot]"))
        if pbar:
            pbar.update(len(batch_rows))
    if pbar:
        pbar.close()

    return np.array(predictions)


# ---------------------------------------------------------------------------
# EXPERIMENT FUNCTIONS
# ---------------------------------------------------------------------------

def run_zs_d1(dataset_name: str, model, tokenizer, model_id: str, test_size: int = 500, output_dir: Path = None) -> tuple:
    """Zero-shot with original column names (ZS_D1).

    Measures the model's prior over tabular data with semantic context.

    Args:
        dataset_name (str): Dataset name.
        model: Loaded model.
        tokenizer: Corresponding tokenizer.
        model_id (str): Model identifier.
        test_size (int): Test set size.
        output_dir (Path, optional): Output directory for results.

    Returns:
        tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
            (y_true, y_pred, table1, table2), or (None, None, None, None) if
            data files do not exist.
    """
    cfg             = DATASET_CONFIG[dataset_name]
    target_col      = cfg["target"]
    sensitive_feats = cfg["protected"]
    privileged_vals = cfg["privileged"]
    prompts         = DATASET_PROMPTS[dataset_name]

    model_tag = model_id.replace("/", "-").replace(".", "-")

    d1_path = DATA_DIR / dataset_name / "D1_original.csv"
    if not d1_path.exists():
        print(f"[hf_zeroshot] Not found: {d1_path}")
        return None, None, None, None

    df = pd.read_csv(d1_path)
    _, df_test = train_test_split(df, test_size=test_size, random_state=SEED, stratify=df[target_col])
    print(f"[hf_zeroshot] ZS_D1 test set: {len(df_test)} instances")

    X_test = df_test[[c for c in df_test.columns if c != target_col]]
    y_test = df_test[target_col].values

    y_pred = predict_zeroshot(
        model=model,
        tokenizer=tokenizer,
        X_test=X_test,
        system_prompt=prompts["system"],
        target_col=target_col,
        pred_map=prompts["pred_map"],
    )

    t1, t2 = evaluate_all_sensitive(
        y_true=y_test,
        y_pred=y_pred,
        df=df_test,
        sensitive_features=sensitive_feats,
        privileged_values=privileged_vals,
        model_name=f"LLM_ZeroShot_{model_tag} (ZS_D1)",
        dataset_name=dataset_name,
    )
    print_results(t1, t2)

    if output_dir:
        results_out = Path(output_dir)
        results_out.mkdir(parents=True, exist_ok=True)
        prefix = f"{dataset_name}_ZS_D1_LLM_ZeroShot_{model_tag}"
        save_results(t1, t2, results_out, prefix=prefix)

        pred_df = df_test.copy()
        pred_df["y_pred"] = y_pred
        pred_df["y_true"] = y_test
        pred_df.insert(0, "instance_idx", range(len(pred_df)))
        pred_df.to_csv(results_out / f"{prefix}_predictions.csv", index=False)
        print(f"[hf_zeroshot] Predictions saved: {prefix}_predictions.csv")

    return y_test, y_pred, t1, t2


def run_zs_d5( dataset_name: str, model, tokenizer, model_id: str, test_size: int = 500, output_dir: Path = None) -> tuple:
    """Zero-shot with generic column names (ZS_D5).

    Measures the model's prior without activating pretraining memory.
    Uses the same column permutation as the D5 decontamination experiment.

    Args:
        dataset_name (str): Dataset name.
        model: Loaded model.
        tokenizer: Corresponding tokenizer.
        model_id (str): Model identifier.
        test_size (int): Test set size.
        output_dir (Path, optional): Output directory for results.

    Returns:
        tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
            (y_true, y_pred, table1, table2), or (None, None, None, None) if
            data files do not exist.
    """
    cfg             = DATASET_CONFIG[dataset_name]
    target_col      = cfg["target"]
    sensitive_feats = cfg["protected"]
    privileged_vals = cfg["privileged"]

    model_tag = model_id.replace("/", "-").replace(".", "-")

    d1_path = DATA_DIR / dataset_name / "D1_original.csv"
    if not d1_path.exists():
        print(f"[hf_zeroshot] Not found: {d1_path}")
        return None, None, None, None

    df = pd.read_csv(d1_path)
    _, df_test = train_test_split(df, test_size=test_size, random_state=SEED, stratify=df[target_col])

    col_map        = build_column_mapping(df, target_col, seed=SEED)
    anon_target    = col_map[target_col]
    df_test_anon   = df_test.rename(columns=col_map)

    anon_feats = [col_map[f] for f in sensitive_feats]
    anon_priv  = {col_map[f]: v for f, v in privileged_vals.items()}

    print(f"[hf_zeroshot] ZS_D5 test set: {len(df_test_anon)} instances")
    print(f"[hf_zeroshot] Target: '{target_col}' -> '{anon_target}'")

    X_test = df_test_anon[[c for c in df_test_anon.columns if c != anon_target]]
    y_test = df_test_anon[anon_target].values

    y_pred = predict_zeroshot(
        model=model,
        tokenizer=tokenizer,
        X_test=X_test,
        system_prompt=ZS_D5_SYSTEM_PROMPTS[dataset_name],
        target_col=anon_target,
        pred_map=DATASET_PROMPTS[dataset_name]["pred_map"],
    )

    t1, t2 = evaluate_all_sensitive(
        y_true=y_test,
        y_pred=y_pred,
        df=df_test_anon,
        sensitive_features=anon_feats,
        privileged_values=anon_priv,
        model_name=f"LLM_ZeroShot_{model_tag} (ZS_D5)",
        dataset_name=dataset_name,
    )

    # Restore original column names in result tables for readability
    reverse_map = {v: k for k, v in col_map.items()}
    t1["Feature"] = t1["Feature"].map(reverse_map).fillna(t1["Feature"])
    t2["Feature"] = t2["Feature"].map(reverse_map).fillna(t2["Feature"])
    print_results(t1, t2)

    if output_dir:
        results_out = Path(output_dir)
        results_out.mkdir(parents=True, exist_ok=True)
        prefix = f"{dataset_name}_ZS_D5_LLM_ZeroShot_{model_tag}"
        save_results(t1, t2, results_out, prefix=prefix)

        pred_df = df_test.copy()
        pred_df["y_pred"] = y_pred
        pred_df["y_true"] = y_test
        pred_df.insert(0, "instance_idx", range(len(pred_df)))
        pred_df.to_csv(results_out / f"{prefix}_predictions.csv", index=False)
        print(f"[hf_zeroshot] Predictions saved: {prefix}_predictions.csv")

    return y_test, y_pred, t1, t2
