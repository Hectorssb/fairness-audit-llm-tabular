"""Retrainable-paradigm baselines: XGBoost over the canonical splits.

XGBoost is trained on D1_train/D2_train/D3_train and evaluated on the same
D1_test and D2_test splits used by the LLM conditions, covering the three
standard mitigation families:

  - Pre-processing : D2 (FLAI) and D3 (resampling) as training data.
  - In-processing  : ExponentiatedGradient (fairlearn) with an equalized-odds
                     constraint on D1_train.
  - Post-processing: per-group decision threshold fitted on train to equalise
                     TPR and FPR.

Run standalone (after the data step of `main.py`):
    python -m xgboost_baselines

Output goes to `results/{dataset}/baselines/`.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from data_loader import DATASET_CONFIG
from metrics import evaluate_all_sensitive, save_results, print_results

from xgboost import XGBClassifier
from fairlearn.reductions import ExponentiatedGradient, EqualizedOdds

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PACKAGE_ROOT / "data"
RESULTS_DIR  = PACKAGE_ROOT / "results"
SEED = 42

TRAIN_FILES = {
    "D1_original":    "D1_train.csv",
    "D2_fair_causal": "D2_train.csv",
    "D3_resampled":   "D3_train.csv",
}


N_JOBS = 8


def make_clf(seed):
    return XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        random_state=seed, eval_metric="logloss", verbosity=0,
        n_jobs=N_JOBS
    )


# ---- FEATURE ENCODING (categories fixed on the training split) ----

def fit_encoders(df_train: pd.DataFrame, feature_cols: list) -> dict:
    """Build a {column: {category: integer}} map for non-numeric training columns.

    Args:
        df_train: Training split.
        feature_cols: Columns used as predictors.

    Returns:
        Mapping from column name to its category-to-integer dictionary.
    """
    encoders = {}
    for col in feature_cols:
        if not pd.api.types.is_numeric_dtype(df_train[col]):
            categories = sorted(df_train[col].astype(str).unique())
            encoders[col] = {cat: i for i, cat in enumerate(categories)}
    return encoders


def encode_features(df: pd.DataFrame, feature_cols: list, encoders: dict) -> np.ndarray:
    """Apply the encoders. Categories unseen in training are encoded as -1.

    Args:
        df: Split to encode.
        feature_cols: Columns used as predictors.
        encoders: Output of `fit_encoders`.

    Returns:
        Float feature matrix.
    """
    df = df[feature_cols].copy()
    for col, mapping in encoders.items():
        df[col] = df[col].astype(str).map(mapping).fillna(-1).astype(int)
    return df.values.astype(float)


# ---- POST-PROCESSING THRESHOLD MITIGATION ----
# The per-group decision threshold is fitted on the training split only.
# The test split never takes part in the search.

def find_threshold_equal_odds(y_prob, y_true, sensitive, target=None):
    """Search the per-group threshold that equalises TPR and FPR across groups.

    Args:
        y_prob: Predicted positive-class probabilities.
        y_true: Ground-truth labels.
        sensitive: Protected-attribute values, aligned with `y_prob`.
        target: (TPR, FPR) pair to match. Defaults to the mean across groups at
            threshold 0.5.

    Returns:
        Mapping from group value to its decision threshold.
    """
    groups = np.unique(sensitive)

    def _rates(y_t, y_p, thr):
        y_pred = (y_p >= thr).astype(int)
        pos, neg = y_t.sum(), (y_t == 0).sum()
        tpr = ((y_pred == 1) & (y_t == 1)).sum() / pos if pos > 0 else 0.0
        fpr = ((y_pred == 1) & (y_t == 0)).sum() / neg if neg > 0 else 0.0
        return tpr, fpr

    if target is None:
        rates = [_rates(y_true[sensitive == g], y_prob[sensitive == g], 0.5)
                 for g in groups]
        target = (np.mean([r[0] for r in rates]), np.mean([r[1] for r in rates]))

    opt_thresholds = {}
    for g in groups:
        mask = sensitive == g
        y_t, y_p = y_true[mask], y_prob[mask]
        if y_t.sum() == 0:
            opt_thresholds[g] = 0.5
            continue
        best_thr, best_diff = 0.5, float("inf")
        for thr in np.linspace(0.01, 0.99, 200):
            tpr_g, fpr_g = _rates(y_t, y_p, thr)
            diff = abs(tpr_g - target[0]) + abs(fpr_g - target[1])
            if diff < best_diff:
                best_diff = diff
                best_thr = thr
        opt_thresholds[g] = best_thr

    return opt_thresholds


def predict_with_thresholds(y_prob, sensitive, thresholds):
    """Apply the per-group thresholds to the predicted probabilities.

    Args:
        y_prob: Predicted positive-class probabilities.
        sensitive: Protected-attribute values, aligned with `y_prob`.
        thresholds: Output of `find_threshold_equal_odds`.

    Returns:
        Binary prediction array.
    """
    y_pred = np.zeros(len(y_prob), dtype=int)
    for g, thr in thresholds.items():
        mask = sensitive == g
        y_pred[mask] = (y_prob[mask] >= thr).astype(int)
    return y_pred


# ---- PER-DATASET EXPERIMENT ----

def _evaluate_and_save(y_test, y_pred, df_test, sensitive_feats, privileged_vals, model_name, dataset_name, results_out, prefix, test_tag, y_prob=None):
    t1, t2 = evaluate_all_sensitive(
        y_true=y_test, y_pred=y_pred, df=df_test,
        sensitive_features=sensitive_feats, privileged_values=privileged_vals,
        model_name=model_name, dataset_name=dataset_name,
    )
    t1["test_set"] = test_tag
    t2["test_set"] = test_tag
    print_results(t1, t2)
    save_results(t1, t2, results_out, prefix=prefix)

    pred_log = df_test.copy()
    pred_log["y_pred"] = y_pred
    pred_log["y_true"] = y_test
    if y_prob is not None:
        pred_log["y_prob"] = y_prob
    pred_log.insert(0, "instance_idx", range(len(pred_log)))
    pred_log["test_set"] = test_tag
    pred_log.to_csv(results_out / f"{prefix}_predictions.csv", index=False)
    return t1, t2


def _expected_prefixes(dataset_name: str, sensitive_feats: list) -> list:
    """Filename prefixes one dataset produces, across every baseline."""
    prefixes = []
    for test_tag in ("D1", "D2"):
        for cond_name in TRAIN_FILES:
            prefixes.append(f"{dataset_name}_{cond_name}_XGBoost_test{test_tag}")
        for feat in sensitive_feats:
            prefixes.append(f"{dataset_name}_D2_XGBoost_Mitigated_{feat}_test{test_tag}")
            prefixes.append(f"{dataset_name}_D1_XGBoost_InProcessing_{feat}_test{test_tag}")
    return prefixes


def run_xgboost_validation(dataset_name: str, data_dir: Path = DATA_DIR, results_dir: Path = RESULTS_DIR, seed: int = SEED):
    """Run every XGBoost baseline for one dataset over the canonical splits.

    Args:
        dataset_name: One of the keys of `DATASET_CONFIG`.
        data_dir: Directory holding the canonical split CSVs.
        results_dir: Root results directory.
        seed: Random state for the classifiers.
    """
    print(f"\n{'='*65}")
    print(f"  XGBOOST BASELINES | {dataset_name.upper()}")
    print(f"{'='*65}")

    cfg             = DATASET_CONFIG[dataset_name]
    target_col      = cfg["target"]
    sensitive_feats = cfg["protected"]
    privileged_vals = cfg["privileged"]

    results_out = results_dir / dataset_name / "baselines"
    expected = _expected_prefixes(dataset_name, sensitive_feats)
    done = [p for p in expected
            if (results_out / f"{p}_table_performance.csv").exists()]
    if len(done) == len(expected):
        print(f"  [SKIP] Baselines already complete ({len(done)} passes).")
        return

    required = ["D1_train.csv", "D1_test.csv", "D2_test.csv"]
    missing = [f for f in required if not (data_dir / dataset_name / f).exists()]
    if missing:
        print(f"  [SKIP] Canonical splits missing ({', '.join(missing)}); "
              f"run the data generation step first.")
        return

    results_out.mkdir(parents=True, exist_ok=True)

    df_d1_train = pd.read_csv(data_dir / dataset_name / "D1_train.csv")
    test_sets = {
        "D1": pd.read_csv(data_dir / dataset_name / "D1_test.csv"),
        "D2": pd.read_csv(data_dir / dataset_name / "D2_test.csv"),
    }
    feature_cols = [c for c in df_d1_train.columns if c != target_col]
    encoders = fit_encoders(df_d1_train, feature_cols)
    print(f"  Canonical test: {len(test_sets['D1'])} (D1) / {len(test_sets['D2'])} (D2) instances")

    all_t2 = []

    for cond_name, train_file in TRAIN_FILES.items():
        data_path = data_dir / dataset_name / train_file
        if not data_path.exists():
            print(f"  [SKIP] Missing: {data_path.name}")
            continue

        df_train = pd.read_csv(data_path)
        X_train = encode_features(df_train, feature_cols, encoders)
        y_train = df_train[target_col].values

        print(f"\n  [{cond_name}] Train={len(X_train)}")
        clf = make_clf(seed)
        clf.fit(X_train, y_train)
        prob_train = clf.predict_proba(X_train)[:, 1]

        for test_tag, df_test in test_sets.items():
            X_test = encode_features(df_test, feature_cols, encoders)
            y_test = df_test[target_col].values
            y_pred = clf.predict(X_test)
            y_prob = clf.predict_proba(X_test)[:, 1]

            prefix = f"{dataset_name}_{cond_name}_XGBoost_test{test_tag}"
            _, t2 = _evaluate_and_save(
                y_test, y_pred, df_test, sensitive_feats, privileged_vals,
                f"XGBoost ({cond_name})", dataset_name, results_out, prefix,
                test_tag, y_prob=y_prob,
            )
            all_t2.append(t2)

            if cond_name == "D2_fair_causal":
                for feat in sensitive_feats:
                    thresholds = find_threshold_equal_odds(
                        prob_train, y_train, df_train[feat].values
                    )
                    y_pred_mit = predict_with_thresholds(y_prob, df_test[feat].values, thresholds)
                    pos_rate = y_pred_mit.mean()
                    print(f"    Mitigated {feat}: per-group thresholds = {thresholds} "
                          f"| positive rate = {pos_rate:.3f}")
                    if pos_rate <= 0.01 or pos_rate >= 0.99:
                        print(f"    WARNING: the mitigated classifier predicts a "
                              f"single class ({pos_rate:.3f}); whatever parity it "
                              f"reaches is apparent, not fairness.")

                    prefix_mit = f"{dataset_name}_D2_XGBoost_Mitigated_{feat}_test{test_tag}"
                    _, t2m = _evaluate_and_save(
                        y_test, y_pred_mit, df_test, [feat],
                        {feat: privileged_vals[feat]},
                        f"XGBoost Mitigated D2 ({feat})", dataset_name,
                        results_out, prefix_mit, test_tag,
                    )
                    all_t2.append(t2m)

    for feat in sensitive_feats:
        print(f"\n  [D1_original + in-processing ({feat})]")
        X_train = encode_features(df_d1_train, feature_cols, encoders)
        y_train = df_d1_train[target_col].values
        mitigator = ExponentiatedGradient(make_clf(seed), constraints=EqualizedOdds())
        mitigator.fit(X_train, y_train, sensitive_features=df_d1_train[feat].values)

        for test_tag, df_test in test_sets.items():
            X_test = encode_features(df_test, feature_cols, encoders)
            y_test = df_test[target_col].values
            y_pred_inp = mitigator.predict(X_test)

            prefix_inp = f"{dataset_name}_D1_XGBoost_InProcessing_{feat}_test{test_tag}"
            _, t2i = _evaluate_and_save(
                y_test, y_pred_inp, df_test, [feat],
                {feat: privileged_vals[feat]},
                f"XGBoost InProcessing D1 ({feat})", dataset_name,
                results_out, prefix_inp, test_tag,
            )
            all_t2.append(t2i)

    print(f"\n{'='*65}")
    print(f"  EOD SUMMARY — {dataset_name.upper()}")
    print(f"{'='*65}")
    if all_t2:
        summary = pd.concat(all_t2, ignore_index=True)
        print(summary[["Algorithm", "Feature", "test_set", "EOD", "DI", "SPD"]].to_string(index=False))


# ---- ENTRY POINT ----

def run_all_baselines(datasets: list, data_dir: Path = DATA_DIR, results_dir: Path = RESULTS_DIR, seed: int = SEED):
    """Run the XGBoost baselines for every dataset.

    Args:
        datasets: Dataset names to process.
        data_dir: Directory holding the canonical split CSVs.
        results_dir: Root results directory.
        seed: Random state for the classifiers.
    """
    print("\n" + "="*70)
    print("  XGBOOST BASELINES — retrainable paradigm, canonical splits")
    print("="*70)
    print(f"Data: {data_dir}")

    for dataset_name in datasets:
        try:
            run_xgboost_validation(dataset_name, data_dir, results_dir, seed)
        except Exception as e:
            print(f"[ERROR] {dataset_name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*70)
    print("  BASELINES COMPLETED")
    print("="*70)


if __name__ == "__main__":
    run_all_baselines(["german", "adult", "compas"])
