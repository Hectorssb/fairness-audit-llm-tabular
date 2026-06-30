"""Classifier via Groq API with round-robin key rotation.

Mirrors the interface of hf_classifier.py. Supported models:
    - llama-3.3-70b-versatile
    - qwen/qwen3-32b           (supports native reasoning via reasoning_effort)

Conditions (data-level / ZS / D5 evaluated on test=D1; D4* on test=D1 and test=D2):
    D1/D2/D3                  — few-shot ICL with the named training distribution.
    D4a                       — zero-shot CoT (fair-CoT instruction, no demos).
    D4b_D1 / D4b_D2           — few-shot CoT, 10 demos with template reasoning.
    D4r_0 / D4r_D1 / D4r_D2   — native reasoning, 0 or 10 input+label demos
                                 (qwen/qwen3-32b only).
    ZS_D1                     — zero-shot with original column names.
    ZS_D5                     — zero-shot with anonymised column names.
    D5_decontam               — few-shot with anonymised column names.
"""

import os
import re
import time
import datetime
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from groq import Groq

from data_loader import DATASET_CONFIG
from metrics import evaluate_all_sensitive, save_results
from hf_classifier import serialize_row, select_fewshot_examples, _parse_response
from hf_cot_classifier import parse_cot_response
from hf_zeroshot_classifier import build_column_mapping
from prompts import (
    DATASET_PROMPTS,
    D4_SYSTEM_PROMPTS,
    D4_CONFIG,
    ZS_D5_SYSTEM_PROMPTS,
    D5_SYSTEM as _D5_SYSTEM,
)
from reasoning_generator import generate_reasoning


# ---------------------------------------------------------------------------
# KEY ROTATION
# ---------------------------------------------------------------------------

def _load_groq_clients() -> list:
    """Load all GROQ_API_KEY_N from environment and return a list of Groq clients."""
    clients = []
    for i in range(1, 15):
        key = os.getenv(f"GROQ_API_KEY_{i}", "").strip()
        if key:
            clients.append(Groq(api_key=key))
    if not clients:
        raise ValueError("No GROQ_API_KEY_N found in environment. Check your .env file.")
    print(f"[groq_classifier] {len(clients)} Groq API key(s) loaded.")
    return clients


# Global round-robin iterator — initialised once at import time
_GROQ_CLIENTS: list = []
_CLIENT_CYCLE = None
_CLIENT_IDX = -1  # 0-based index of the key returned by the most recent _get_client()


def _get_client() -> Groq:
    global _GROQ_CLIENTS, _CLIENT_CYCLE, _CLIENT_IDX
    if _CLIENT_CYCLE is None:
        _GROQ_CLIENTS = _load_groq_clients()
        _CLIENT_CYCLE = itertools.cycle(_GROQ_CLIENTS)
    _CLIENT_IDX = (_CLIENT_IDX + 1) % len(_GROQ_CLIENTS)
    return _GROQ_CLIENTS[_CLIENT_IDX]


def _num_clients() -> int:
    """Number of Groq API keys loaded (initialises the pool if needed)."""
    if _CLIENT_CYCLE is None:
        _get_client()
    return len(_GROQ_CLIENTS)


# ---------------------------------------------------------------------------
# RATE LIMIT HELPERS
# ---------------------------------------------------------------------------

def _is_daily_limit(err: str) -> bool:
    """Return True only if the error is a *daily* quota limit, not a per-minute burst.

    Groq returns HTTP 429 for both per-minute (RPM/TPM) throttling and the daily
    (RPD/TPD) quota. The message text is what distinguishes them, e.g.:
        "Rate limit reached ... Limit 14400, ... please try again in 2m30s."   (daily)
        "Rate limit reached ... Limit 6000, ... please try again in 1.2s."     (TPM burst)
    Only the daily case should trigger the sleep-until-midnight path. We match on
    explicit daily markers ('per day', 'the day', 'RPD'/'TPD', 'quota') and on the
    long retry windows Groq reports for daily limits (minutes/hours, never
    sub-minute). A bare 'exceeded'/'rate' with a sub-minute retry window is treated
    as a transient TPM/RPM burst and handled by the normal exponential backoff.
    """
    err_lower = err.lower()
    daily_markers = (
        "per day", "/day", "the day", "requests per day", "tokens per day",
        " rpd", " tpd", "daily", "quota",
    )
    if any(k in err_lower for k in daily_markers):
        return True
    # Fall back to the retry window Groq reports: daily limits say "try again in
    # Xm.../Xh...", TPM bursts say "try again in X.Xs". Treat a stated wait of >= 5
    # minutes as daily; anything shorter is a transient burst.
    m = re.search(r"try again in\s+(?:(\d+)h)?\s*(?:(\d+)m)?\s*([\d.]+)?s?", err_lower)
    if m and (m.group(1) or m.group(2)):
        hours = int(m.group(1)) if m.group(1) else 0
        mins = int(m.group(2)) if m.group(2) else 0
        if hours * 60 + mins >= 5:
            return True
    return False


def _sleep_until_midnight_utc():
    """Sleep until the next midnight UTC (when Groq daily limits reset)."""
    now = datetime.datetime.now(datetime.timezone.utc)
    tomorrow = (now + datetime.timedelta(days=1)).replace(
        hour=0, minute=0, second=5, microsecond=0
    )
    seconds = (tomorrow - now).total_seconds()
    print(
        f"[groq_classifier] Daily rate limit hit — sleeping {seconds/3600:.1f}h "
        f"until {tomorrow.strftime('%Y-%m-%d %H:%M:%S')} UTC"
    )
    time.sleep(seconds)


# ---------------------------------------------------------------------------
# CORE CALL WITH RETRY
# ---------------------------------------------------------------------------

def _call_groq(model_id: str, messages: list, max_tokens: int = 20, temperature: float = 0.0, retries: int = 10, backoff: float = 2.0) -> str:
    """Call Groq chat completions with automatic retry on rate-limit errors.

    Args:
        model_id: Groq model identifier (e.g. 'qwen/qwen3-32b').
        messages: List of {role, content} dicts.
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature (0 = greedy).
        retries: Number of retries on rate-limit (429) errors.
        backoff: Initial wait seconds; doubles each retry.

    Returns:
        str: The assistant's response text.
    """
    # Model-specific kwargs.
    # - qwen3: reasoning_effort="none" fully disables thinking tokens on the
    #   standard FewShot/ZeroShot path.
    extra_kwargs = {}
    if "qwen3" in model_id.lower():
        extra_kwargs["reasoning_effort"] = "none"

    wait = backoff
    attempt = 0
    # Count consecutive keys that report a *daily* quota limit. Only sleep until
    # midnight once every key in the pool has been tried and hit the daily wall —
    # a single key reporting it just means we should rotate to the next one.
    daily_hits = 0
    while True:
        try:
            client = _get_client()
            resp = client.chat.completions.create(
                model=model_id,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **extra_kwargs,
            )
            content = resp.choices[0].message.content or ""
            if content.strip():
                return content
            # Empty response despite 200 OK — likely TPM throttling or reasoning
            # budget exhausted. Use longer waits (30s base) to let the TPM window reset.
            daily_hits = 0
            if attempt < retries:
                tpm_wait = max(wait, 30)
                if attempt == 0:
                    print(f"[groq_classifier] Empty response, retrying (up to {retries}x, wait {tpm_wait:.0f}s)...")
                time.sleep(tpm_wait)
                wait *= 2
                attempt += 1
            else:
                print("[groq_classifier] Max retries exceeded, returning empty string.")
                return ""
        except Exception as e:
            err = str(e)
            if _is_daily_limit(err):
                daily_hits += 1
                n = _num_clients()
                if daily_hits >= n:
                    # Every key has hit its daily quota — wait for the reset.
                    print(f"[groq_classifier] All {n} keys hit the daily quota; sleeping until reset.")
                    _sleep_until_midnight_utc()
                    # Reset counters after sleeping through the reset.
                    daily_hits = 0
                    attempt = 0
                    wait = backoff
                else:
                    # Rotate to the next key immediately (no sleep).
                    print(f"[groq_classifier] Key #{_CLIENT_IDX + 1} hit daily quota, rotating to next key ({daily_hits}/{n}).")
                continue
            daily_hits = 0
            if "429" in err or "rate" in err.lower():
                if attempt < retries:
                    if attempt == 0:
                        print(f"[groq_classifier] Rate limit, backing off (up to {retries}x)...")
                    time.sleep(wait)
                    wait *= 2
                    attempt += 1
                else:
                    print("[groq_classifier] Max retries exceeded, returning empty string.")
                    return ""
            else:
                print(f"[groq_classifier] API error: {e}")
                time.sleep(2)
                attempt += 1
                if attempt >= retries:
                    return ""


# ---------------------------------------------------------------------------
# FEW-SHOT PREDICTION
# ---------------------------------------------------------------------------

def predict_fewshot_groq(model_id: str, X_test: pd.DataFrame, examples_df: pd.DataFrame, system_prompt: str, target_col: str, label_map: dict, pred_map: dict, max_new_tokens: int = 20, temperature: float = 0.0, verbose: bool = True) -> np.ndarray:
    """Few-shot prediction via Groq API.

    Args:
        model_id: Groq model identifier.
        X_test: Test features (without target).
        examples_df: Few-shot examples including target column.
        system_prompt: System message for the model.
        target_col: Name of the target column in examples_df.
        label_map: Map {int_label: str_label} for serializing examples.
        pred_map: Map {str_label: int_label} for parsing predictions.
        max_new_tokens: Max tokens to generate per instance.
        temperature: Sampling temperature.
        verbose: Show tqdm progress bar.

    Returns:
        np.ndarray: Array of binary predictions.
    """
    few_shot_pairs = []
    for _, ex in examples_df.iterrows():
        few_shot_pairs.append({"role": "user",      "content": serialize_row(ex, target_col)})
        few_shot_pairs.append({"role": "assistant", "content": label_map[int(ex[target_col])]})

    predictions = []
    rows = [row for _, row in X_test.iterrows()]
    iterator = tqdm(rows, desc="[groq_classifier] FewShot") if verbose else rows

    # gpt-oss models use internal reasoning tokens even without reasoning_effort,
    # which burns TPM quickly and causes empty responses. A small inter-request
    # sleep keeps us within the per-minute token budget.
    inter_request_sleep = 15.0 if "gpt-oss" in model_id.lower() else 0.0

    for row in iterator:
        messages = (
            [{"role": "system", "content": system_prompt}]
            + few_shot_pairs
            + [{"role": "user", "content": serialize_row(row, target_col)}]
        )
        response = _call_groq(model_id, messages, max_new_tokens, temperature)
        predictions.append(_parse_response(response, pred_map, "[groq_classifier]"))
        if inter_request_sleep:
            time.sleep(inter_request_sleep)

    return np.array(predictions)


# ---------------------------------------------------------------------------
# ZERO-SHOT PREDICTION
# ---------------------------------------------------------------------------

def predict_zeroshot_groq(model_id: str, X_test: pd.DataFrame, system_prompt: str, target_col: str, pred_map: dict, max_new_tokens: int = 20, temperature: float = 0.0, verbose: bool = True) -> np.ndarray:
    """Zero-shot prediction via Groq API.

    Args:
        model_id: Groq model identifier.
        X_test: Test features (without target).
        system_prompt: System message.
        target_col: Target column name (used by serialize_row to skip it).
        pred_map: Map {str_label: int_label}.
        max_new_tokens: Max tokens to generate.
        temperature: Sampling temperature.
        verbose: Show tqdm progress bar.

    Returns:
        np.ndarray: Array of binary predictions.
    """
    predictions = []
    rows = [row for _, row in X_test.iterrows()]
    iterator = tqdm(rows, desc="[groq_classifier] ZeroShot") if verbose else rows

    # gpt-oss models use internal reasoning tokens even without reasoning_effort,
    # which burns TPM quickly and causes empty responses. A small inter-request
    # sleep keeps us within the per-minute token budget.
    inter_request_sleep = 15.0 if "gpt-oss" in model_id.lower() else 0.0

    for row in iterator:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": serialize_row(row, target_col)},
        ]
        response = _call_groq(model_id, messages, max_new_tokens, temperature)
        predictions.append(_parse_response(response, pred_map, "[groq_classifier]"))
        if inter_request_sleep:
            time.sleep(inter_request_sleep)

    return np.array(predictions)


# ---------------------------------------------------------------------------
# D4a — ZERO-SHOT COT PREDICTION
# ---------------------------------------------------------------------------

ZS_COT_TRIGGER = "\nThink step by step, considering only non-protected features, then give your final answer on the last line."


def predict_zs_cot_groq(model_id: str, X_test: pd.DataFrame, dataset_name: str, max_new_tokens: int = 512, temperature: float = 0.0, verbose: bool = True) -> tuple:
    """Zero-shot CoT prediction via Groq API (D4a, Kojima et al. 2022).

    No few-shot examples. The CoT trigger is appended to each user message.

    Args:
        model_id: Groq model identifier.
        X_test: Test features (without target).
        dataset_name: Dataset name.
        max_new_tokens: Max tokens to generate.
        temperature: Sampling temperature.
        verbose: Show tqdm progress bar.

    Returns:
        tuple[np.ndarray, list[dict]]: (predictions, cot_log).
    """
    cfg        = D4_CONFIG[dataset_name]
    target_col = cfg["target"]
    pred_map   = cfg["pred_map"]
    system_msg = D4_SYSTEM_PROMPTS[dataset_name]

    predictions = []
    cot_log     = []
    rows = list(X_test.iterrows())
    iterator = tqdm(enumerate(rows), total=len(rows), desc="[groq_classifier] D4a ZS-CoT") if verbose else enumerate(rows)

    for i, (_, row) in iterator:
        row_text = serialize_row(row, target_col) + ZS_COT_TRIGGER
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": row_text},
        ]
        response = _call_groq(model_id, messages, max_new_tokens, temperature)
        pred, cot_text = parse_cot_response(response, pred_map)
        predictions.append(pred)
        cot_log.append({"instance_idx": i, "input": row_text, "cot_output": cot_text, "prediction": pred})

    return np.array(predictions), cot_log


# ---------------------------------------------------------------------------
# D4b — FEW-SHOT COT PREDICTION
# ---------------------------------------------------------------------------

def predict_fs_cot_groq(model_id: str, X_test: pd.DataFrame, dataset_name: str,
                        df_train: pd.DataFrame, n_examples: int = 10, seed: int = 42,
                        max_new_tokens: int = 512, temperature: float = 0.0,
                        verbose: bool = True) -> tuple:
    """Few-shot CoT prediction via Groq API.

    Samples ``n_examples`` balanced rows from ``df_train`` and generates the
    CoT reasoning chain for each using the deterministic template. See
    ``hf_cot_classifier.predict_fs_cot`` for the HF equivalent.

    Args:
        model_id: Groq model identifier.
        X_test: Test features (without target).
        dataset_name: Dataset name.
        df_train: Training split from which the 10 demos are sampled (D1 for
            D4b_D1, D2 for D4b_D2). Must include the target column.
        n_examples: Number of demonstrations (kept at 10).
        seed: Seed used by select_fewshot_examples (fixed at 42).
        max_new_tokens: Max tokens to generate.
        temperature: Sampling temperature.
        verbose: Show tqdm progress bar.

    Returns:
        tuple[np.ndarray, list[dict]]: (predictions, cot_log).
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

    predictions = []
    cot_log     = []
    rows = list(X_test.iterrows())
    iterator = tqdm(enumerate(rows), total=len(rows), desc="[groq_classifier] D4b FS-CoT") if verbose else enumerate(rows)

    for i, (_, row) in iterator:
        row_text = serialize_row(row, target_col)
        messages = (
            [{"role": "system", "content": system_msg}]
            + few_shot_pairs
            + [{"role": "user", "content": row_text}]
        )
        response = _call_groq(model_id, messages, max_new_tokens, temperature)
        pred, cot_text = parse_cot_response(response, pred_map)
        predictions.append(pred)
        cot_log.append({"instance_idx": i, "input": row_text, "cot_output": cot_text, "prediction": pred})

    return np.array(predictions), cot_log


# ---------------------------------------------------------------------------
# EXPERIMENT BLOCKS
# ---------------------------------------------------------------------------

def run_fewshot_groq(df_train: pd.DataFrame, df_test: pd.DataFrame, dataset_name: str, data_condition: str, model_id: str, n_examples: int = 10, output_dir: Path = None):
    cfg             = DATASET_CONFIG[dataset_name]
    target_col      = cfg["target"]
    sensitive_feats = cfg["protected"]
    privileged_vals = cfg["privileged"]
    prompts         = DATASET_PROMPTS[dataset_name]
    model_tag       = model_id.replace("/", "-").replace(".", "-")
    model_label     = f"LLM_FewShot_{model_tag}"

    examples_df = select_fewshot_examples(df_train, target_col, n_examples)
    X_test = df_test[[c for c in df_test.columns if c != target_col]]
    y_test = df_test[target_col].values

    y_pred = predict_fewshot_groq(
        model_id=model_id, X_test=X_test, examples_df=examples_df,
        system_prompt=prompts["system"], target_col=target_col,
        label_map=prompts["label_map"], pred_map=prompts["pred_map"],
    )

    t1, t2 = evaluate_all_sensitive(
        y_true=y_test, y_pred=y_pred, df=df_test,
        sensitive_features=sensitive_feats, privileged_values=privileged_vals,
        model_name=f"{model_label} ({data_condition})", dataset_name=dataset_name,
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


def run_zs_cot_groq(df_test: pd.DataFrame, dataset_name: str, data_condition: str, model_id: str,
                    output_dir: Path = None, test_set: str = None):
    """Run D4a (zero-shot CoT) via Groq API.

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

    y_pred, cot_log = predict_zs_cot_groq(model_id=model_id, X_test=X_test, dataset_name=dataset_name)

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
        print(f"[groq_classifier] D4a results saved: {prefix}")

    return y_test, y_pred, t1, t2, cot_log


def run_fs_cot_groq(df_train: pd.DataFrame, df_test: pd.DataFrame,
                    dataset_name: str, data_condition: str, model_id: str,
                    output_dir: Path = None, test_set: str = None):
    """Run D4b (few-shot CoT) via Groq API. Demos sampled from df_train.

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

    y_pred, cot_log = predict_fs_cot_groq(
        model_id=model_id, X_test=X_test,
        dataset_name=dataset_name, df_train=df_train,
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
        print(f"[groq_classifier] D4b results saved: {prefix}")

    return y_test, y_pred, t1, t2, cot_log


def run_zs_d1_groq(dataset_name: str, model_id: str, data_dir: Path, test_size: int = 500, seed: int = 42, output_dir: Path = None):
    cfg             = DATASET_CONFIG[dataset_name]
    target_col      = cfg["target"]
    sensitive_feats = cfg["protected"]
    privileged_vals = cfg["privileged"]
    prompts         = DATASET_PROMPTS[dataset_name]
    model_tag       = model_id.replace("/", "-").replace(".", "-")

    df = pd.read_csv(data_dir / dataset_name / "D1_original.csv")
    _, df_test = train_test_split(df, test_size=test_size, random_state=seed, stratify=df[target_col])
    df_test = df_test.reset_index(drop=True)

    X_test = df_test[[c for c in df_test.columns if c != target_col]]
    y_test = df_test[target_col].values

    y_pred = predict_zeroshot_groq(
        model_id=model_id, X_test=X_test,
        system_prompt=prompts["system"], target_col=target_col,
        pred_map=prompts["pred_map"],
    )

    t1, t2 = evaluate_all_sensitive(
        y_true=y_test, y_pred=y_pred, df=df_test,
        sensitive_features=sensitive_feats, privileged_values=privileged_vals,
        model_name=f"LLM_ZeroShot_{model_tag} (ZS_D1)", dataset_name=dataset_name,
    )

    if output_dir:
        prefix = f"{dataset_name}_ZS_D1_LLM_ZeroShot_{model_tag}"
        save_results(t1, t2, output_dir, prefix=prefix)
        pred_df = df_test.copy()
        pred_df["y_pred"] = y_pred
        pred_df["y_true"] = y_test
        pred_df.insert(0, "instance_idx", range(len(pred_df)))
        pred_df.to_csv(Path(output_dir) / f"{prefix}_predictions.csv", index=False)

    return y_test, y_pred, t1, t2


def run_zs_d5_groq(dataset_name: str, model_id: str, data_dir: Path, test_size: int = 500, seed: int = 42, output_dir: Path = None):
    cfg             = DATASET_CONFIG[dataset_name]
    target_col      = cfg["target"]
    sensitive_feats = cfg["protected"]
    privileged_vals = cfg["privileged"]
    model_tag       = model_id.replace("/", "-").replace(".", "-")

    df = pd.read_csv(data_dir / dataset_name / "D1_original.csv")
    _, df_test = train_test_split(df, test_size=test_size, random_state=seed, stratify=df[target_col])
    df_test = df_test.reset_index(drop=True)

    col_map      = build_column_mapping(df, target_col, seed=seed)
    anon_target  = col_map[target_col]
    df_test_anon = df_test.rename(columns=col_map)
    anon_feats   = [col_map[f] for f in sensitive_feats]
    anon_priv    = {col_map[f]: v for f, v in privileged_vals.items()}

    X_test = df_test_anon[[c for c in df_test_anon.columns if c != anon_target]]
    y_test = df_test_anon[anon_target].values

    y_pred = predict_zeroshot_groq(
        model_id=model_id, X_test=X_test,
        system_prompt=ZS_D5_SYSTEM_PROMPTS[dataset_name],
        target_col=anon_target,
        pred_map=DATASET_PROMPTS[dataset_name]["pred_map"],
    )

    t1, t2 = evaluate_all_sensitive(
        y_true=y_test, y_pred=y_pred, df=df_test_anon,
        sensitive_features=anon_feats, privileged_values=anon_priv,
        model_name=f"LLM_ZeroShot_{model_tag} (ZS_D5)", dataset_name=dataset_name,
    )

    # Restore original column names
    reverse_map = {v: k for k, v in col_map.items()}
    t1["Feature"] = t1["Feature"].map(reverse_map).fillna(t1["Feature"])
    t2["Feature"] = t2["Feature"].map(reverse_map).fillna(t2["Feature"])

    if output_dir:
        prefix = f"{dataset_name}_ZS_D5_LLM_ZeroShot_{model_tag}"
        save_results(t1, t2, output_dir, prefix=prefix)
        pred_df = df_test.copy()
        pred_df["y_pred"] = y_pred
        pred_df["y_true"] = y_test
        pred_df.insert(0, "instance_idx", range(len(pred_df)))
        pred_df.to_csv(Path(output_dir) / f"{prefix}_predictions.csv", index=False)

    return y_test, y_pred, t1, t2


# ---------------------------------------------------------------------------
# NATIVE REASONING — D4r
# ---------------------------------------------------------------------------

# Models that support native reasoning via Groq's reasoning_effort parameter
_REASONING_MODELS = {"qwen/qwen3-32b"}


def _supports_reasoning(model_id: str) -> bool:
    return model_id in _REASONING_MODELS


def _call_groq_reasoning(model_id: str, messages: list, max_tokens: int = 512, temperature: float = 1.0, reasoning_effort: str = "high", retries: int = 5, backoff: float = 2.0) -> tuple:
    """Call Groq with native reasoning enabled, returns (answer_text, reasoning_text).

    Model-specific reasoning parameters:
      - qwen/qwen3-32b:      reasoning_effort="default", reasoning_format="parsed"
                             (does not accept "high"; reasoning in msg.reasoning)

    Args:
        model_id: Groq model identifier (must support reasoning_effort).
        messages: List of {role, content} dicts.
        max_tokens: Maximum tokens for the final answer (not thinking).
        temperature: Sampling temperature (reasoning models often require > 0).
        reasoning_effort: ignored for qwen3 (always uses "default").
        retries: Number of retries on rate-limit errors.
        backoff: Initial wait in seconds; doubles each retry.

    Returns:
        tuple[str, str]: (answer_content, reasoning_content).
    """
    is_qwen3 = "qwen3" in model_id.lower()

    if is_qwen3:
        extra_kwargs = {
            "reasoning_effort": "default",
            "reasoning_format": "parsed",
        }
    else:
        extra_kwargs = {"reasoning_effort": reasoning_effort}

    wait = backoff
    attempt = 0
    daily_hits = 0  # see _call_groq: only sleep once every key hits the daily wall
    while True:
        try:
            client = _get_client()
            resp = client.chat.completions.create(
                model=model_id,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **extra_kwargs,
            )
            msg = resp.choices[0].message
            answer    = (msg.content or "").strip()
            reasoning = (getattr(msg, "reasoning", None) or "").strip()
            if answer:
                return answer, reasoning
            # Some reasoning models (e.g. qwen3-32b) put everything in reasoning
            # and leave content empty — use reasoning as the answer in that case
            if reasoning:
                return reasoning, reasoning
            # Truly empty — likely TPM throttle
            daily_hits = 0
            if attempt < retries:
                if attempt == 0:
                    print(f"[groq_classifier/D4r] Empty response, retrying (up to {retries}x)...")
                time.sleep(wait)
                wait *= 2
                attempt += 1
            else:
                print("[groq_classifier/D4r] Max retries exceeded.")
                return "", ""
        except Exception as e:
            err = str(e)
            if _is_daily_limit(err):
                daily_hits += 1
                n = _num_clients()
                if daily_hits >= n:
                    print(f"[groq_classifier/D4r] All {n} keys hit the daily quota; sleeping until reset.")
                    _sleep_until_midnight_utc()
                    daily_hits = 0
                    attempt = 0
                    wait = backoff
                else:
                    print(f"[groq_classifier/D4r] Key #{_CLIENT_IDX + 1} hit daily quota, rotating to next key ({daily_hits}/{n}).")
                continue
            daily_hits = 0
            if "429" in err or "rate" in err.lower():
                if attempt < retries:
                    if attempt == 0:
                        print(f"[groq_classifier/D4r] Rate limit, backing off (up to {retries}x)...")
                    time.sleep(wait)
                    wait *= 2
                    attempt += 1
                else:
                    print("[groq_classifier/D4r] Max retries exceeded.")
                    return "", ""
            else:
                print(f"[groq_classifier/D4r] API error: {e}")
                time.sleep(2)
                attempt += 1
                if attempt >= retries:
                    return "", ""


def predict_d4r_groq(model_id: str, X_test: pd.DataFrame, dataset_name: str,
                     df_train: pd.DataFrame = None, n_examples: int = 10,
                     seed: int = 42, reasoning_effort: str = "high",
                     max_new_tokens: int = 512, temperature: float = 1.0,
                     verbose: bool = True) -> tuple:
    """Native-reasoning prediction via Groq API (D4r).

    Uses the domain system prompt (DATASET_PROMPTS), not the fair-CoT instruction.
    df_train: None -> D4r_0 (no demos); D1 -> D4r_D1; D2 -> D4r_D2. Demos carry
    input + label only — no visible CoT chain.

    Args:
        model_id: Groq model identifier (must support reasoning_effort).
        X_test: Test features (without target).
        dataset_name: Dataset name.
        df_train: Training split for demos, or None for D4r_0.
        n_examples: Number of demos when df_train is provided.
        seed: Seed for select_fewshot_examples.
        reasoning_effort: 'low', 'medium', or 'high'.
        max_new_tokens: Max tokens for the final answer.
        temperature: Sampling temperature (must be > 0 for reasoning models).
        verbose: Show tqdm progress bar.

    Returns:
        (predictions: np.ndarray, reasoning_log: list[dict]). Each log entry has
        instance_idx, input, answer, reasoning, prediction.
    """
    cfg            = DATASET_CONFIG[dataset_name]
    target_col     = cfg["target"]
    prompts        = DATASET_PROMPTS[dataset_name]
    system_msg     = prompts["system"]
    pred_map       = prompts["pred_map"]
    label_map      = prompts["label_map"]

    few_shot_pairs = []
    if df_train is not None:
        examples_df = select_fewshot_examples(df_train, target_col, n_examples, seed=seed)
        for _, ex in examples_df.iterrows():
            ex_input = serialize_row(ex, target_col)
            ex_label = label_map[int(ex[target_col])]
            few_shot_pairs.append({"role": "user",      "content": ex_input})
            few_shot_pairs.append({"role": "assistant", "content": ex_label})

    predictions = []
    reasoning_log = []
    rows = list(X_test.iterrows())
    iterator = tqdm(enumerate(rows), total=len(rows), desc=f"[groq_classifier] D4r ({reasoning_effort})") if verbose else enumerate(rows)

    for i, (_, row) in iterator:
        row_text = serialize_row(row, target_col)
        messages = (
            [{"role": "system", "content": system_msg}]
            + few_shot_pairs
            + [{"role": "user", "content": row_text}]
        )
        answer, reasoning = _call_groq_reasoning(
            model_id, messages,
            max_tokens=max_new_tokens,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
        )
        pred = _parse_response(answer, pred_map, "[groq_classifier/D4r]")
        predictions.append(pred)
        reasoning_log.append({
            "instance_idx": i,
            "input":        row_text,
            "answer":       answer,
            "reasoning":    reasoning,
            "prediction":   pred,
        })

    return np.array(predictions), reasoning_log


def run_d4r_groq(df_test: pd.DataFrame, dataset_name: str, data_condition: str,
                 model_id: str, df_train: pd.DataFrame = None,
                 reasoning_effort: str = "high", output_dir: Path = None,
                 test_set: str = None):
    """Run D4r (native reasoning) for a data condition.

    data_condition selects the variant: 'D4r_0' (df_train=None), 'D4r_D1'
    (df_train=D1, 10 demos), 'D4r_D2' (df_train=D2, 10 demos). Saves a
    reasoning_log with the internal thinking text per instance.

    Skips gracefully if model_id does not support reasoning_effort.

    Args:
        df_test: Test data with target column.
        dataset_name: Dataset name.
        data_condition: 'D4r_0', 'D4r_D1', or 'D4r_D2'.
        model_id: Groq model identifier.
        df_train: Training split for demos, or None for D4r_0.
        reasoning_effort: 'low', 'medium', or 'high'.
        output_dir: Output directory for results CSVs.
        test_set: Test-set tag ('D1' or 'D2'). When provided, filenames get
            a `_test<tag>` suffix and CSVs carry a `test_set` column.

    Returns:
        tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame, list] or
        None if unsupported.
    """
    if not _supports_reasoning(model_id):
        print(f"[groq_classifier/D4r] Model '{model_id}' does not support native reasoning — skipping D4r.")
        return None

    cfg             = DATASET_CONFIG[dataset_name]
    target_col      = cfg["target"]
    sensitive_feats = cfg["protected"]
    privileged_vals = cfg["privileged"]
    model_tag       = model_id.replace("/", "-").replace(".", "-")
    model_label     = f"LLM_D4r_high_{model_tag}"

    X_test = df_test[[c for c in df_test.columns if c != target_col]]
    y_test = df_test[target_col].values

    y_pred, reasoning_log = predict_d4r_groq(
        model_id=model_id, X_test=X_test, dataset_name=dataset_name,
        df_train=df_train, reasoning_effort=reasoning_effort,
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
        log_df = pd.DataFrame(reasoning_log)
        if test_set is not None:
            log_df["test_set"] = test_set
        log_df.to_csv(Path(output_dir) / f"{prefix}_reasoning_log.csv", index=False)
        pred_df = df_test.copy()
        pred_df["y_pred"] = y_pred
        pred_df["y_true"] = y_test
        pred_df.insert(0, "instance_idx", range(len(pred_df)))
        if test_set is not None:
            pred_df["test_set"] = test_set
        pred_df.to_csv(Path(output_dir) / f"{prefix}_predictions.csv", index=False)
        print(f"[groq_classifier/D4r] Results saved: {prefix}")

    return y_test, y_pred, t1, t2, reasoning_log


def run_decontam_groq(df_train: pd.DataFrame, df_test: pd.DataFrame, dataset_name: str, model_id: str, n_examples: int = 10, seed: int = 42, output_dir: Path = None):
    cfg             = DATASET_CONFIG[dataset_name]
    target_col      = cfg["target"]
    sensitive_feats = cfg["protected"]
    privileged_vals = cfg["privileged"]
    prompts         = DATASET_PROMPTS[dataset_name]
    model_tag       = model_id.replace("/", "-").replace(".", "-")

    col_map      = build_column_mapping(df_test, target_col, seed=seed)
    anon_target  = col_map[target_col]
    df_test_anon = df_test.rename(columns=col_map)
    df_train_anon = df_train.rename(columns=col_map)
    prot_anon    = [col_map[f] for f in sensitive_feats]
    priv_anon    = {col_map[f]: v for f, v in privileged_vals.items()}

    examples    = select_fewshot_examples(df_train_anon, anon_target, n_examples, seed=seed)
    X_test_anon = df_test_anon[[c for c in df_test_anon.columns if c != anon_target]]
    y_test      = df_test_anon[anon_target].values

    y_pred = predict_fewshot_groq(
        model_id=model_id, X_test=X_test_anon, examples_df=examples,
        system_prompt=_D5_SYSTEM[dataset_name], target_col=anon_target,
        label_map=prompts["label_map"], pred_map=prompts["pred_map"],
    )

    t1, t2 = evaluate_all_sensitive(
        y_true=y_test, y_pred=y_pred, df=df_test_anon,
        sensitive_features=prot_anon, privileged_values=priv_anon,
        model_name=f"LLM_FewShot_{model_tag} (D5_decontam)", dataset_name=dataset_name,
    )

    reverse_map = {v: k for k, v in col_map.items()}
    t1["Feature"] = t1["Feature"].map(reverse_map).fillna(t1["Feature"])
    t2["Feature"] = t2["Feature"].map(reverse_map).fillna(t2["Feature"])

    if output_dir:
        prefix = f"{dataset_name}_D5_decontam_LLM_FewShot_{model_tag}"
        save_results(t1, t2, output_dir, prefix=prefix)
        pred_df = df_test.copy()
        pred_df["y_pred"] = y_pred
        pred_df["y_true"] = y_test
        pred_df.insert(0, "instance_idx", range(len(pred_df)))
        pred_df.to_csv(Path(output_dir) / f"{prefix}_predictions.csv", index=False)

    return y_test, y_pred, t1, t2
