"""
Computes performance and fairness metrics identical to those in
González-Sendino et al. (2024) — Tables 1 and 2 of the base paper.

Performance metrics:
  A   — Accuracy
  TPR — True Positive Rate
  FPR — False Positive Rate
  FNR — False Negative Rate
  PPP — Predicted as Positive (Proportion)

Fairness metrics:
  EOD — Equal Opportunity Difference  (ideal = 0)
  DI  — Disparate Impact              (ideal = 1)
  SPD — Statistical Parity Difference (ideal = 0)
  OD  — Odds Difference               (ideal = 0)
"""

import numpy as np
import pandas as pd
from typing import Union
from pathlib import Path


# ---------------------------------------------------------------------------
# BASE METRICS PER GROUP
# ---------------------------------------------------------------------------

def group_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute performance metrics for a subgroup.

    Args:
        y_true (np.ndarray): Ground-truth labels (0/1).
        y_pred (np.ndarray): Model predictions (0/1).

    Returns:
        dict: Dictionary with keys A, TPR, FPR, FNR, PPP.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    total_samples = len(y_true)

    true_positives = np.sum((y_pred == 1) & (y_true == 1))
    true_negatives = np.sum((y_pred == 0) & (y_true == 0))
    false_positives = np.sum((y_pred == 1) & (y_true == 0))
    false_negatives = np.sum((y_pred == 0) & (y_true == 1))

    accuracy = (true_positives + true_negatives) / total_samples if total_samples > 0 else 0.0
    true_positive_rate = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0   # Recall / Sensitivity
    false_positive_rate = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0.0
    false_negative_rate = false_negatives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    predicted_positive_proportion = (true_positives + false_positives) / total_samples  if total_samples > 0 else 0.0            # Predicted Positive Proportion

    return {"A": round(accuracy, 4), "TPR": round(true_positive_rate, 4),
            "FPR": round(false_positive_rate, 4), "FNR": round(false_negative_rate, 4), "PPP": round(predicted_positive_proportion, 4)}


# ---------------------------------------------------------------------------
# FAIRNESS METRICS
# ---------------------------------------------------------------------------

def equal_opportunity_difference(y_true: np.ndarray, y_pred: np.ndarray, sensitive: np.ndarray, privileged_value: int) -> float:
    """Compute the Equal Opportunity Difference (EOD).

    EOD = TPR_underprivileged - TPR_privileged.
    Ideal = 0. Acceptable range [-0.1, 0.1].

    Args:
        y_true (np.ndarray): Ground-truth labels.
        y_pred (np.ndarray): Predictions.
        sensitive (np.ndarray): Protected attribute values.
        privileged_value (int): Value of the privileged group.

    Returns:
        float: EOD rounded to 4 decimal places.
    """
    is_privileged   = sensitive == privileged_value
    is_unprivileged = sensitive != privileged_value

    tpr_priv   = group_metrics(y_true[is_privileged],   y_pred[is_privileged])["TPR"]
    tpr_unpriv = group_metrics(y_true[is_unprivileged], y_pred[is_unprivileged])["TPR"]

    return round(tpr_unpriv - tpr_priv, 4)


def disparate_impact(y_true: np.ndarray, y_pred: np.ndarray, sensitive: np.ndarray, privileged_value: int) -> float:
    """Compute the Disparate Impact (DI).

    DI = PPP_underprivileged / PPP_privileged.
    Ideal = 1. Acceptable range [0.8, 1.2].

    Args:
        y_true (np.ndarray): Ground-truth labels.
        y_pred (np.ndarray): Predictions.
        sensitive (np.ndarray): Protected attribute values.
        privileged_value (int): Value of the privileged group.

    Returns:
        float: DI rounded to 4 decimal places, or inf if PPP_privileged == 0.
    """
    is_privileged   = sensitive == privileged_value
    is_unprivileged = sensitive != privileged_value

    ppp_priv   = group_metrics(y_true[is_privileged],   y_pred[is_privileged])["PPP"]
    ppp_unpriv = group_metrics(y_true[is_unprivileged], y_pred[is_unprivileged])["PPP"]

    if ppp_priv == 0:
        return float("inf")
    return round(ppp_unpriv / ppp_priv, 4)


def statistical_parity_difference(y_true: np.ndarray, y_pred: np.ndarray, sensitive: np.ndarray, privileged_value: int) -> float:
    """Compute the Statistical Parity Difference (SPD).

    SPD = PPP_underprivileged - PPP_privileged.
    Ideal = 0. Acceptable range [-0.1, 0.1].

    Args:
        y_true (np.ndarray): Ground-truth labels.
        y_pred (np.ndarray): Predictions.
        sensitive (np.ndarray): Protected attribute values.
        privileged_value (int): Value of the privileged group.

    Returns:
        float: SPD rounded to 4 decimal places.
    """
    is_privileged   = sensitive == privileged_value
    is_unprivileged = sensitive != privileged_value

    ppp_priv   = group_metrics(y_true[is_privileged],   y_pred[is_privileged])["PPP"]
    ppp_unpriv = group_metrics(y_true[is_unprivileged], y_pred[is_unprivileged])["PPP"]

    return round(ppp_unpriv - ppp_priv, 4)


def odds_difference(y_true: np.ndarray, y_pred: np.ndarray, sensitive: np.ndarray, privileged_value: int) -> float:
    """Compute the Odds Difference (OD).

    OD = (FPR_underprivileged - FPR_privileged) + (TPR_underprivileged - TPR_privileged).
    Ideal = 0. Acceptable range [-0.1, 0.1].

    Args:
        y_true (np.ndarray): Ground-truth labels.
        y_pred (np.ndarray): Predictions.
        sensitive (np.ndarray): Protected attribute values.
        privileged_value (int): Value of the privileged group.

    Returns:
        float: OD rounded to 4 decimal places.
    """
    is_privileged   = sensitive == privileged_value
    is_unprivileged = sensitive != privileged_value

    m_priv   = group_metrics(y_true[is_privileged],   y_pred[is_privileged])
    m_unpriv = group_metrics(y_true[is_unprivileged], y_pred[is_unprivileged])

    fpr_diff = m_unpriv["FPR"] - m_priv["FPR"]
    tpr_diff = m_unpriv["TPR"] - m_priv["TPR"]

    return round(fpr_diff + tpr_diff, 4)


# ---------------------------------------------------------------------------
# FULL EVALUATION
# ---------------------------------------------------------------------------

def evaluate(y_true: np.ndarray, y_pred: np.ndarray, sensitive: np.ndarray, privileged_value: int, sensitive_name: str = "feature", model_name: str = "Model", dataset_name: str = "Dataset") -> tuple:
    """Evaluate performance and fairness, returning two DataFrames.

    Produces output with format identical to Table 1 + Table 2 of
    González-Sendino et al. (2024).

    Args:
        y_true (np.ndarray): Ground-truth labels.
        y_pred (np.ndarray): Model predictions.
        sensitive (np.ndarray): Protected attribute values.
        privileged_value (int): Value of the privileged group.
        sensitive_name (str): Name of the protected attribute.
        model_name (str): Model/algorithm name.
        dataset_name (str): Dataset name.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            - table_performance: per-group performance (Algorithm, Dataset, Feature, Group, A, TPR, FPR, FNR, PPP).
            - table_fairness: global fairness metrics (Algorithm, Dataset, Feature, EOD, DI, SPD, OD).
    """
    is_privileged   = sensitive == privileged_value
    is_unprivileged = sensitive != privileged_value

    privileged_group_metrics   = group_metrics(y_true[is_privileged],   y_pred[is_privileged])
    underprivileged_group_metrics = group_metrics(y_true[is_unprivileged], y_pred[is_unprivileged])

    eod = equal_opportunity_difference(y_true, y_pred, sensitive, privileged_value)
    di  = disparate_impact(y_true, y_pred, sensitive, privileged_value)
    spd = statistical_parity_difference(y_true, y_pred, sensitive, privileged_value)
    od  = odds_difference(y_true, y_pred, sensitive, privileged_value)

    # Table 1 — per-group performance
    performance_rows = [
        {
            "Algorithm": model_name, "Dataset": dataset_name,
            "Feature": sensitive_name, "Group": "Privilege",
            **privileged_group_metrics
        },
        {
            "Algorithm": model_name, "Dataset": dataset_name,
            "Feature": sensitive_name, "Group": "Underprivileged",
            **underprivileged_group_metrics
        },
    ]
    performance_table = pd.DataFrame(performance_rows)

    # Table 2 — global fairness
    fairness_table = pd.DataFrame([{
        "Algorithm": model_name, "Dataset": dataset_name,
        "Feature": sensitive_name,
        "EOD": eod, "DI": di, "SPD": spd, "OD": od
    }])

    return performance_table, fairness_table


def evaluate_all_sensitive(y_true: np.ndarray, y_pred: np.ndarray, df: pd.DataFrame, sensitive_features: list, privileged_values: dict, model_name: str = "Model", dataset_name: str = "Dataset") -> tuple:
    """Evaluate all protected attributes for a dataset.

    Args:
        y_true (np.ndarray): Ground-truth labels.
        y_pred (np.ndarray): Model predictions.
        df (pd.DataFrame): DataFrame with protected attribute columns.
        sensitive_features (list): List of protected attribute names.
        privileged_values (dict): Dictionary {feature: privileged_value}.
        model_name (str): Model/algorithm name.
        dataset_name (str): Dataset name.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (full_table_performance, full_table_fairness).
    """
    performance_tables, fairness_tables = [], []

    for feat in sensitive_features:
        if feat not in df.columns:
            print(f"[metrics] Warning: '{feat}' not found in df, skipping.")
            continue
        sensitive = df[feat].values
        privileged_value  = privileged_values[feat]

        performance_table, fairness_table = evaluate(
            y_true=y_true,
            y_pred=y_pred,
            sensitive=sensitive,
            privileged_value=privileged_value,
            sensitive_name=feat,
            model_name=model_name,
            dataset_name=dataset_name,
        )
        performance_tables.append(performance_table)
        fairness_tables.append(fairness_table)

    return pd.concat(performance_tables, ignore_index=True), pd.concat(fairness_tables, ignore_index=True)


def print_results(table_performance: pd.DataFrame, table_fairness: pd.DataFrame):
    """Print results in human-readable format.

    Args:
        table_performance (pd.DataFrame): Per-group performance table.
        table_fairness (pd.DataFrame): Fairness metrics table.
    """
    print("\n=== TABLE 1 — Per-group performance ===")
    print(table_performance.to_string(index=False))
    print("\n=== TABLE 2 — Fairness metrics ===")
    print(table_fairness.to_string(index=False))
    print()


def save_results(table_performance: pd.DataFrame, table_fairness: pd.DataFrame, output_dir, prefix: str = "results"):
    """Save Table Performance and Table Fairness as CSV files.

    Args:
        table_performance (pd.DataFrame): Per-group performance table.
        table_fairness (pd.DataFrame): Fairness metrics table.
        output_dir (str or Path): Output directory.
        prefix (str): Filename prefix.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    performance_csv_path = output_path / f"{prefix}_table_performance.csv"
    fairness_csv_path    = output_path / f"{prefix}_table_fairness.csv"

    table_performance.to_csv(performance_csv_path, index=False)
    table_fairness.to_csv(fairness_csv_path, index=False)
    print(f"[metrics] Saved: {performance_csv_path}")
    print(f"[metrics] Saved: {fairness_csv_path}")


# ---------------------------------------------------------------------------
# CONSOLIDATED SUMMARY OF ALL EXPERIMENTS
# ---------------------------------------------------------------------------

def consolidate_results(results_dir) -> tuple:
    """Read all results CSVs and consolidate them into two DataFrames.

    Useful for generating the final paper tables with all models
    and conditions in a single DataFrame.

    Args:
        results_dir (str or Path): Root results directory.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (consolidated_table_performance, consolidated_table_fairness)
            or (None, None) if no files are found.
    """
    results_dir = Path(results_dir)

    performance_files = sorted(results_dir.rglob("*table_performance.csv"))
    fairness_files    = sorted(results_dir.rglob("*table_fairness.csv"))

    if not performance_files:
        print("[metrics] No results found.")
        return None, None

    consolidated_performance = pd.concat([pd.read_csv(f) for f in performance_files], ignore_index=True)
    consolidated_fairness = pd.concat([pd.read_csv(f) for f in fairness_files], ignore_index=True)

    return consolidated_performance, consolidated_fairness


