"""
Central prompt registry for the experiments.

All prompts and few-shot examples are stored as JSON files in this directory
so they can be edited independently of the Python code (ablation studies,
prompt engineering, etc.).

Exported names (imported by classifiers):
    DATASET_PROMPTS         — system prompts + label/pred maps for standard ICL
    D4_SYSTEM_PROMPTS       — fairness-constrained system prompts for D4a/D4b
    D4_CONFIG               — label/pred maps and target column for CoT conditions
    D4B_REASONING_TEMPLATE  — deterministic reasoning rules for D4b
    ZS_D5_SYSTEM_PROMPTS    — generic system prompts for zero-shot decontam (ZS_D5)
    D5_SYSTEM               — generic system prompts for few-shot decontam (D5)
    D6_SYSTEM               — explicit debiasing-instruction prompts for D6
"""

import json
from pathlib import Path

_DIR = Path(__file__).parent


def _load(filename: str) -> dict:
    with open(_DIR / filename, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Standard ICL prompts
# ---------------------------------------------------------------------------

_dp = _load("dataset_prompts.json")

# Convert string keys in label_map back to int (JSON keys are always strings)
DATASET_PROMPTS = {
    dataset: {
        **cfg,
        "label_map": {int(k): v for k, v in cfg["label_map"].items()},
    }
    for dataset, cfg in _dp.items()
}

# ---------------------------------------------------------------------------
# CoT prompts and config (D4a, D4b, D4r)
# ---------------------------------------------------------------------------

D4_SYSTEM_PROMPTS: dict = _load("d4_system_prompts.json")

D4_CONFIG = {
    "german": {"label_map": {1: "good", 0: "bad"}, "pred_map": {"good": 1, "bad": 0}, "target": "label"},
    "adult":  {"label_map": {1: "yes",  0: "no"},  "pred_map": {"yes":  1, "no":  0}, "target": "income"},
    "compas": {"label_map": {1: "yes",  0: "no"},  "pred_map": {"yes":  1, "no":  0}, "target": "two_year_recid"},
}

D4B_REASONING_TEMPLATE: dict = _load("d4b_reasoning_template.json")

# ---------------------------------------------------------------------------
# Decontamination / zero-shot prompts
# ---------------------------------------------------------------------------

ZS_D5_SYSTEM_PROMPTS: dict = _load("zs_d5_system_prompts.json")

D5_SYSTEM: dict = _load("d5_system_prompts.json")

# ---------------------------------------------------------------------------
# Fair-prompt condition (D6)
# ---------------------------------------------------------------------------

D6_SYSTEM: dict = _load("d6_system_prompts.json")
