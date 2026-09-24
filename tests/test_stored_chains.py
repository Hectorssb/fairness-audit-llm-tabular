"""The stored reasoning traces must match the design of the conditions.

D4b builds its demonstration chains from a per-dataset template that walks only
non-protected features, and D6 answers directly without eliciting a chain at
all. Both are properties of the files under results/, so they are checked
against what was actually generated rather than against the template alone.

"""

import glob
import re
from pathlib import Path

import pandas as pd
import pytest

RESULTS = Path(__file__).resolve().parents[1] / "results"
PROTECTED = {"adult": ["sex", "race"], "compas": ["sex", "race"], "german": ["sex", "age"]}
TERMS = {
    "sex": r"\bsex\b|\bmale\b|\bfemale\b|\bgender\b|\bwoman\b|\bwomen\b",
    "race": r"\brace\b|\bracial\b|\bcaucasian\b|\bafrican[- ]american\b|\bhispanic\b",
    "age": r"\bage\b|\byears old\b|\belderly\b|\byounger\b|\bolder\b",
}
# A trace that names an attribute in order to set it aside, or that echoes the
# serialised row back, is not reasoning over it.
SETTING_ASIDE = r"must not use|not use|ignore|exclud|disregard|avoid|without using"
ECHOES_ROW = r"sex:\s*\d|age:\s*\d|race:\s*\d"
# Chains the model wrote itself may still reach for a protected attribute.
# Roughly one in 24000 does; the bound keeps that a rarity, not a silent norm.
MAX_SELF_INITIATED = 0.001


def _chain_files(pattern):
    return sorted(glob.glob(str(RESULTS / "*" / "*" / pattern)))


pytestmark = pytest.mark.skipif(not RESULTS.exists(), reason="results/ is not unzipped")


def test_d6_elicits_no_chain():
    assert not _chain_files("*D6*_cot_log.csv"), \
        "D6 answers directly, so it must not have produced chain files"
    assert not _chain_files("*D6*_reasoning_log.csv")


def test_d4b_chains_do_not_reason_over_protected_attributes():
    total, offending = 0, []
    for path in _chain_files("*_D4b_*_cot_log.csv"):
        dataset = Path(path).parents[1].name
        for text in pd.read_csv(path)["cot_output"].astype(str):
            total += 1
            lowered = text.lower()
            for attr in PROTECTED[dataset]:
                if not re.search(TERMS[attr], lowered):
                    continue
                if re.search(SETTING_ASIDE, lowered) or re.search(ECHOES_ROW, lowered):
                    continue
                offending.append((Path(path).name, attr, text[:160]))
    assert total, "no D4b chain files were read"
    rate = len(offending) / total
    assert rate <= MAX_SELF_INITIATED, (
        f"{len(offending)} of {total} D4b chains ({rate:.2%}) reason over a protected "
        f"attribute, above the {MAX_SELF_INITIATED:.1%} bound, e.g. {offending[0]}")
