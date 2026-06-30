"""Deterministic reasoning chain generator for D4b.

Given a row sampled from a dataset's training split and the corresponding
binary label, this module produces a chain-of-thought reasoning string by
applying the rules declared in `prompts/d4b_reasoning_template.json`.

Properties (audited in test_reasoning_generator.py):
1. Never references protected attributes (sex, race, age in German Credit).
2. Fully deterministic: same row + same template + same label always produce
   the same string. No RNG, no auxiliary LLM call.
3. Missing or unrecognised feature values fall back to neutral default clauses
   without crashing.
4. The conclusion explicitly declares the label (Wei et al. 2022 CoT pattern).
"""

from __future__ import annotations

import math
import pandas as pd
from typing import Any

from prompts import D4B_REASONING_TEMPLATE


# ---------------------------------------------------------------------------
# CORE: generate_reasoning(row, dataset_name, label_str) -> str
# ---------------------------------------------------------------------------


def generate_reasoning(row: pd.Series, dataset_name: str, label_str: str) -> str:
    """Return the deterministic CoT reasoning string for a single row.

    Args:
        row (pd.Series): A row from df_train (with all columns of the dataset,
            including protected attributes which are ignored by the template).
        dataset_name (str): One of 'adult', 'compas', 'german'. Used to look up
            the template rules.
        label_str (str): The dataset-native positive/negative label as it
            should appear in the final line of the assistant turn (e.g. 'yes',
            'no', 'good', 'bad').

    Returns:
        str: Multi-line reasoning chain ending in the conclusion line. Does NOT
            include the label by itself on a separate line — the caller appends
            it (the format used in predict_fs_cot).
    """
    template = D4B_REASONING_TEMPLATE[dataset_name]
    features = template["features_to_mention"]
    step_fmt = template["step_format"]
    topic_labels = template["topic_labels"]
    rules = template["rules"]

    steps: list[str] = []
    step_num = 1
    for feat_key in features:
        rule = rules[feat_key]
        topic = topic_labels.get(feat_key, feat_key)
        clause = _apply_rule(rule, feat_key, row, dataset_name, template)
        steps.append(step_fmt.format(n=step_num, topic=topic, clause=clause))
        step_num += 1

    conclusion = template["conclusion_template"].format(n=step_num, label=label_str)
    steps.append(conclusion)
    return "\n".join(steps)


# ---------------------------------------------------------------------------
# RULE DISPATCHERS
# ---------------------------------------------------------------------------


def _apply_rule(rule: dict, feat_key: str, row: pd.Series,
                dataset_name: str, template: dict) -> str:
    rule_type = rule["type"]
    if rule_type == "categorical":
        return _apply_categorical(rule, feat_key, row)
    if rule_type == "numeric_range":
        return _apply_numeric_range(rule, feat_key, row)
    if rule_type == "composite":
        return _apply_composite(rule, feat_key, row, template)
    return rule.get("default_clause", f"{feat_key} could not be assessed")


def _apply_categorical(rule: dict, feat_key: str, row: pd.Series) -> str:
    """Match the row's value of feat_key against one of the listed categories."""
    raw = row.get(feat_key, None) if isinstance(row, pd.Series) else row.get(feat_key, None)
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return rule.get("default_clause", f"{feat_key} is not reported")
    val = str(raw).strip()
    for cat_name, cat in rule["categories"].items():
        if val in cat["values"]:
            return cat["clause"].replace("{value}", val)
    return rule.get("default_clause", f"{feat_key} is not reported in a recognised category")


def _apply_numeric_range(rule: dict, feat_key: str, row: pd.Series) -> str:
    """Match the row's numeric value of feat_key against one of the ranges.

    Each range entry may use any subset of:
        min_inclusive, min_exclusive, max_inclusive, max_exclusive.
    The first range that satisfies all of its declared bounds fires.
    """
    raw = row.get(feat_key, None)
    if raw is None:
        return rule.get("default_clause", f"{feat_key} is not reported")
    try:
        v = float(raw)
        if math.isnan(v):
            return rule.get("default_clause", f"{feat_key} is not reported")
    except (TypeError, ValueError):
        return rule.get("default_clause", f"{feat_key} is not reported")

    for rng in rule["ranges"]:
        if "min_inclusive" in rng and not (v >= rng["min_inclusive"]):
            continue
        if "min_exclusive" in rng and not (v > rng["min_exclusive"]):
            continue
        if "max_inclusive" in rng and not (v <= rng["max_inclusive"]):
            continue
        if "max_exclusive" in rng and not (v < rng["max_exclusive"]):
            continue
        return rng["clause"].replace("{value}", _fmt_numeric(v))
    return rule.get("default_clause", f"{feat_key} not matched against any range")


def _apply_composite(rule: dict, feat_key: str, row: pd.Series, template: dict) -> str:
    """Composite reasoning, currently only used for 'capital_activity' in adult.

    Reads the conditions list from the template's <feat_key>_logic block and
    fires the first matching one.
    """
    logic_key = f"{feat_key}_logic"
    logic = template.get(logic_key, None)
    if logic is None:
        return rule.get("default_clause", f"{feat_key} could not be assessed")

    for cond in logic["conditions"]:
        if _composite_matches(cond, row):
            return _composite_clause(cond, row)
    return rule.get("default_clause", f"{feat_key} could not be assessed")


def _composite_matches(cond: dict, row: pd.Series) -> bool:
    """Check whether all numeric bounds declared in cond are satisfied."""
    for key, threshold in cond.items():
        if key == "clause":
            continue
        # key is like "capital_gain_min_exclusive", "capital_loss_max_inclusive"
        # Parse: <feature>_<bound_kind>
        parts = key.rsplit("_", 2)
        if len(parts) != 3:
            return False
        feat = parts[0].replace("_", "-")
        bound = f"{parts[1]}_{parts[2]}"
        raw = row.get(feat, None)
        if raw is None:
            return False
        try:
            v = float(raw)
        except (TypeError, ValueError):
            return False
        if math.isnan(v):
            return False
        if bound == "min_inclusive" and not (v >= threshold):
            return False
        if bound == "min_exclusive" and not (v > threshold):
            return False
        if bound == "max_inclusive" and not (v <= threshold):
            return False
        if bound == "max_exclusive" and not (v < threshold):
            return False
    return True


def _composite_clause(cond: dict, row: pd.Series) -> str:
    """Substitute {capital-gain} and {capital-loss} placeholders in cond['clause']."""
    s = cond["clause"]
    for feat in ("capital-gain", "capital-loss"):
        if "{" + feat + "}" in s:
            v = row.get(feat, "0")
            try:
                vnum = float(v)
                s = s.replace("{" + feat + "}", _fmt_numeric(vnum))
            except (TypeError, ValueError):
                s = s.replace("{" + feat + "}", str(v))
    return s


def _fmt_numeric(v: float) -> str:
    """Format a numeric value as integer when it's a whole number, else as float."""
    if v == int(v):
        return str(int(v))
    return f"{v:.2f}"


# ---------------------------------------------------------------------------
# PROTECTED-ATTRIBUTE AUDIT (used by tests)
# ---------------------------------------------------------------------------

# Columns the reasoning is NEVER allowed to reference textually for each
# dataset. Matches the 'protected' field of DATASET_CONFIG in data_loader.py.
PROTECTED_KEYS = {
    "adult":  {"sex", "race"},
    "compas": {"sex", "race"},
    "german": {"sex", "age"},
}


def reasoning_references_protected(reasoning: str, dataset_name: str) -> set[str]:
    """Return the set of protected keys that textually appear in the reasoning.

    The empty set means the reasoning is protected-attribute-free (the
    desired outcome under the fair CoT design).
    """
    found = set()
    text = reasoning.lower()
    for key in PROTECTED_KEYS.get(dataset_name, set()):
        if key.lower() in text:
            found.add(key)
    return found
