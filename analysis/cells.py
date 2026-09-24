"""Single source of truth for the quantities the analysis reports.

Loads results/ into one frame with one row per (condition, model, dataset,
protected attribute, test set), carrying the fairness metrics, the per-group
rates, the collapse flag and the per-pass accuracy, plus the prediction arrays
needed to resample a cell. analysis.ipynb reads from here, so a number cannot
be computed two ways. This module only loads and scores: it writes no file and
renders no table.

Conventions:
  - Collapse is read off `insufficient_support`, never recomputed.
  - Single-pass files carry no test_set column; the split is read from the
    filename (`_testD1`/`_testD2`), defaulting to D1.
  - Accuracy of a pass is the mean over protected attributes of the mean over
    its two group rows. A condition's accuracy is the mean over models.
"""

import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

MODEL_LABEL = {
    "meta-llama-Llama-3-1-8B-Instruct": "Llama-3.1-8B",
    "microsoft-phi-4": "Phi-4",
    "Qwen-Qwen2-5-7B-Instruct": "Qwen2.5-7B",
    "Qwen-Qwen2-5-14B-Instruct": "Qwen2.5-14B",
    "Qwen-Qwen3-32B": "Qwen3-32B",
    "openai-gpt-oss-20b": "GPT-OSS-20B",
    "google-gemma-4-E4B-it": "Gemma-4-E4B",
    "google-gemma-4-31B-it": "Gemma-4-31B",
}
MODEL_ABBR = {"Llama-3.1-8B": "L8", "Qwen2.5-7B": "Q7", "Qwen2.5-14B": "Q14",
              "GPT-OSS-20B": "G4", "Gemma-4-E4B": "Ge4", "Gemma-4-31B": "G31",
              "Phi-4": "P4", "Qwen3-32B": "Q32"}
MODEL_ORDER = ["Llama-3.1-8B", "Qwen2.5-7B", "Qwen2.5-14B", "GPT-OSS-20B",
               "Gemma-4-E4B", "Gemma-4-31B", "Phi-4", "Qwen3-32B"]

PROTECTED = {"adult": ["sex", "race"], "compas": ["sex", "race"], "german": ["sex", "age"]}

# Display order and labels of the conditions.
CONDITIONS = ["D1_original", "D2_fair_causal", "D3_resampled",
              "D4a", "D4b_D1", "D4b_D2", "D4r_0", "D4r_D1", "D4r_D2", "D6_fairprompt",
              "D5_decontam", "ZS_D1", "ZS_D5"]
COND_LABEL = {"D1_original": "D1", "D2_fair_causal": "D2", "D3_resampled": "D3",
              "D4a": "D4a", "D4b_D1": "D4b\\_D1", "D4b_D2": "D4b\\_D2",
              "D4r_0": "D4r\\_0", "D4r_D1": "D4r\\_D1", "D4r_D2": "D4r\\_D2",
              "D6_fairprompt": "D6", "D5_decontam": "D5", "ZS_D1": "ZS\\_D1", "ZS_D5": "ZS\\_D5"}
# Conditions reported on both canonical test splits.
DUAL_TEST = ["D4a", "D4b_D1", "D4b_D2", "D4r_0", "D4r_D1", "D4r_D2", "D6_fairprompt", "D5_decontam",
             "D1_original", "D2_fair_causal", "D3_resampled", "ZS_D1", "ZS_D5"]

EOD_BAND = 0.1
DI_BAND = (0.8, 1.2)
SUPPORT_MARGIN = 0.01
COLLAPSE_FPR_MARGIN = 0.1


def _test_of(path: str) -> str:
    match = re.search(r"_test(D[12])_", path)
    return match.group(1) if match else "D1"


def _condition_of(algorithm: str) -> str:
    match = re.search(r"\(([^)]*)\)\s*$", str(algorithm))
    return match.group(1) if match else str(algorithm)


def _pass_frames(kind: str) -> pd.DataFrame:
    frames = []
    for path in glob.glob(str(RESULTS / "*" / "*" / f"*_table_{kind}.csv")):
        tag = Path(path).parent.name
        if tag not in MODEL_LABEL:
            continue
        frame = pd.read_csv(path)
        frame["model"] = MODEL_LABEL[tag]
        frame["tag"] = tag
        frame["dataset"] = Path(path).parents[1].name
        frame["test"] = _test_of(path)
        frame["cond"] = frame["Algorithm"].map(_condition_of)
        frame["prefix"] = Path(path).name[: -len(f"_table_{kind}.csv")]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _require_current_schema(fair: pd.DataFrame, perf: pd.DataFrame) -> None:
    """Fail early, and legibly, on result files written by an earlier layout.
    """
    missing = [name for name, frame, column in
               (("insufficient_support", fair, "insufficient_support"), ("N", perf, "N"))
               if column not in frame.columns]
    if missing:
        raise RuntimeError(
            "results/ was written by an earlier version of the pipeline and lacks "
            f"{', '.join(missing)}. The collapse filter cannot be applied to it. "
            "Unzip the results.zip shipped with this release, or re-run main.py.")


def load_cells() -> pd.DataFrame:
    """One row per cell with metrics, group rates, collapse flag and accuracy.

    Returns:
        pd.DataFrame indexed by (cond, model, dataset, feature, test).
    """
    fair = _pass_frames("fairness")
    perf = _pass_frames("performance")
    _require_current_schema(fair, perf)
    key = ["cond", "model", "dataset", "Feature", "test"]

    rates = perf.pivot_table(index=key, columns="Group",
                             values=["TPR", "FPR", "PPP", "A", "N"], aggfunc="first")
    rates.columns = [f"{metric}_{'priv' if group == 'Privilege' else 'unpriv'}"
                     for metric, group in rates.columns]
    cells = fair.set_index(key).join(rates, how="left")

    cells["collapsed"] = cells["insufficient_support"].fillna(False).astype(bool)
    cells["eod_ok"] = cells["EOD"].abs() <= EOD_BAND
    cells["di_ok"] = np.isfinite(cells["DI"]) & (cells["DI"] >= DI_BAND[0]) & (cells["DI"] <= DI_BAND[1])
    cells["raw_gf"] = cells["eod_ok"] & cells["di_ok"]
    cells["gf"] = cells["raw_gf"] & ~cells["collapsed"]
    cells["yellow"] = cells["eod_ok"] & ~cells["di_ok"]
    cells["acc"] = (cells["A_priv"] + cells["A_unpriv"]) / 2
    cells = cells.reset_index().rename(columns={"Feature": "feature"})
    cells["abbr"] = cells["model"].map(MODEL_ABBR)
    return cells.set_index(["cond", "model", "dataset", "feature", "test"]).sort_index()


def load_predictions(conds=None) -> dict:
    """Prediction arrays for resampling, grouped by (dataset, test).

    Returns:
        dict: (dataset, test) -> {"y_true", "sensitive": {feature: array},
            "privileged": {feature: int}, "preds": {(cond, model): array}}.
    """
    groups = {}
    for path in glob.glob(str(RESULTS / "*" / "*" / "*_predictions.csv")):
        tag = Path(path).parent.name
        if tag not in MODEL_LABEL:
            continue
        dataset = Path(path).parents[1].name
        name = Path(path).name
        cond = re.sub(rf"^{dataset}_", "", name).split("_LLM_")[0]
        if conds is not None and cond not in conds:
            continue
        test = _test_of(name)
        frame = pd.read_csv(path).sort_values("instance_idx")
        group = groups.setdefault((dataset, test), {"y_true": None, "sensitive": {}, "privileged": {}, "preds": {}})
        y_true = frame["y_true"].to_numpy()
        if group["y_true"] is None:
            group["y_true"] = y_true
            for feature in PROTECTED[dataset]:
                group["sensitive"][feature] = frame[feature].to_numpy()
                group["privileged"][feature] = 1
        else:
            assert np.array_equal(group["y_true"], y_true), f"{path}: y_true differs within {dataset}/{test}"
        group["preds"][(cond, MODEL_LABEL[tag])] = frame["y_pred"].to_numpy()
    return groups


def genuine_matrix(y_true, sensitive, privileged, preds, idx=None, fpr_margin=COLLAPSE_FPR_MARGIN, support_margin=SUPPORT_MARGIN) -> tuple:
    """Gates of every prediction vector at once, optionally on a resample.

    Mirrors metrics.py: rates rounded to four decimals, EOD unmeasured when a
    group has no positive instance, DI unmeasured or infinite when the
    privileged group has no positive prediction, and a group degenerate when
    its predicted-positive rate sits at either extreme or its rates saturate.

    Args:
        y_true: Labels of the test set.
        sensitive: Protected attribute of the test set.
        privileged: Value of the privileged group.
        preds: Matrix (n_cells, n_instances) of predictions.
        idx: Resample indices, or None for the sample itself.
        fpr_margin: Width of the saturation band on FPR.
        support_margin: Margin of the predicted-positive extreme.

    Returns:
        tuple of boolean arrays (eod_in, raw_gf, collapsed, genuine).
    """
    if idx is not None:
        y_true, sensitive, preds = y_true[idx], sensitive[idx], preds[:, idx]
    out = {}
    for name, mask in (("priv", sensitive == privileged), ("unpriv", sensitive != privileged)):
        y = y_true[mask]
        p = preds[:, mask]
        pos, neg = (y == 1), (y == 0)
        tp = (p[:, pos] == 1).sum(1)
        fp = (p[:, neg] == 1).sum(1)
        n_pos, n_neg, n = pos.sum(), neg.sum(), mask.sum()
        tpr = np.round(tp / n_pos, 4) if n_pos else np.full(len(p), np.nan)
        fpr = np.round(fp / n_neg, 4) if n_neg else np.full(len(p), np.nan)
        ppp = np.round((tp + fp) / n, 4) if n else np.full(len(p), np.nan)
        out[name] = (tpr, fpr, ppp)
    (tpr_p, fpr_p, ppp_p), (tpr_u, fpr_u, ppp_u) = out["priv"], out["unpriv"]

    eod = np.round(tpr_u - tpr_p, 4)
    with np.errstate(divide="ignore", invalid="ignore"):
        di = np.where(ppp_p == 0, np.where(ppp_u == 0, np.nan, np.inf), np.round(ppp_u / ppp_p, 4))

    def degenerate(tpr, fpr, ppp):
        extreme = (ppp <= support_margin) | (ppp >= 1 - support_margin)
        saturated = ((tpr == 1.0) & (fpr >= 1 - fpr_margin)) | ((tpr == 0.0) & (fpr <= fpr_margin))
        return extreme | np.nan_to_num(saturated, nan=False)

    unmeasured = np.isnan(eod) | np.isnan(di) | np.isinf(di)
    collapsed = unmeasured | degenerate(tpr_p, fpr_p, ppp_p) | degenerate(tpr_u, fpr_u, ppp_u)
    eod_in = ~np.isnan(eod) & (np.abs(eod) <= EOD_BAND)
    di_in = np.isfinite(di) & (di >= DI_BAND[0]) & (di <= DI_BAND[1])
    raw = eod_in & di_in
    return eod_in, raw, collapsed, raw & ~collapsed


def load_baselines() -> pd.DataFrame:
    """XGBoost baselines: one row per (baseline, dataset, feature, test)."""
    rows = []
    for path in glob.glob(str(RESULTS / "*" / "baselines" / "*_table_fairness.csv")):
        dataset = Path(path).parents[1].name
        fair = pd.read_csv(path)
        perf = pd.read_csv(path.replace("_table_fairness", "_table_performance"))
        test = _test_of(Path(path).name)
        name = Path(path).name[: -len("_table_fairness.csv")]
        stem = re.sub(rf"^{dataset}_", "", name).replace(f"_test{test}", "")
        acc = perf.groupby("Feature")["A"].mean()
        for _, row in fair.iterrows():
            rows.append({"baseline": stem, "dataset": dataset, "feature": row["Feature"], "test": test,
                         "EOD": row["EOD"], "DI": row["DI"], "SPD": row["SPD"],
                         "collapsed": bool(row.get("insufficient_support", False)),
                         "acc": acc[row["Feature"]]})
    frame = pd.DataFrame(rows)
    frame["eod_ok"] = frame["EOD"].abs() <= EOD_BAND
    frame["di_ok"] = np.isfinite(frame["DI"]) & (frame["DI"] >= DI_BAND[0]) & (frame["DI"] <= DI_BAND[1])
    frame["gf"] = frame["eod_ok"] & frame["di_ok"] & ~frame["collapsed"]
    return frame


def pct(n: int, d: int) -> int:
    return int(round(100 * n / d)) if d else 0
