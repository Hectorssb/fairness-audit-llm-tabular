"""Tests for the D4b deterministic reasoning: template-data coherence."""

import itertools
import re

import pandas as pd

from prompts import D4B_REASONING_TEMPLATE
from reasoning_generator import generate_reasoning, reasoning_references_protected

GERMAN_DEFAULT_CLAUSES = [
    rule["default_clause"]
    for rule in D4B_REASONING_TEMPLATE["german"]["rules"].values()
]
COMPAS_DEFAULT_CLAUSES = [
    rule["default_clause"]
    for rule in D4B_REASONING_TEMPLATE["compas"]["rules"].values()
]


def test_german_reasoning_never_falls_to_default_clause():
    credit_histories = ["A30", "A31", "A32", "A33", "A34"]
    savings          = ["A61", "A62", "A63", "A64", "A65"]
    employments      = ["A71", "A72", "A73", "A74", "A75"]
    for ch, sv, em in itertools.product(credit_histories, savings, employments):
        row = pd.Series({
            "checking_status": "A11", "duration": 24, "credit_history": ch,
            "purpose": "A43", "credit_amount": 3000, "savings": sv,
            "employment": em, "installment_rate": 2, "sex": 1,
            "other_debtors": "A101", "property": "A121", "age": 1,
            "housing": "A152", "job": "A173", "label": 1,
        })
        reasoning = generate_reasoning(row, "german", "good")
        for clause in GERMAN_DEFAULT_CLAUSES:
            assert clause not in reasoning, (
                f"default clause fired for credit_history={ch}, savings={sv}, "
                f"employment={em}: '{clause}'"
            )


def test_compas_reasoning_never_falls_to_default_clause():
    for priors, degree, age in itertools.product([0, 2, 4, 8], [0, 1], [20, 30, 40, 50]):
        row = pd.Series({
            "sex": 1, "race": 0, "age": age, "c_charge_degree": degree,
            "priors_count": priors, "two_year_recid": 1,
        })
        reasoning = generate_reasoning(row, "compas", "yes")
        for clause in COMPAS_DEFAULT_CLAUSES:
            assert clause not in reasoning


def test_compas_template_reasons_only_over_allowed_features():
    template = D4B_REASONING_TEMPLATE["compas"]
    allowed = {"priors_count", "c_charge_degree", "age"}
    assert set(template["features_to_mention"]) == allowed
    assert set(template["rules"].keys()) == allowed


def test_german_template_features_exist_in_corrected_schema():
    from data_loader import GERMAN_COLUMN_MAP
    schema = set(GERMAN_COLUMN_MAP.values())
    for feat in D4B_REASONING_TEMPLATE["german"]["features_to_mention"]:
        assert feat in schema, f"template mentions '{feat}' absent from the loader schema"

PROTECTED = {"german": ["sex", "age"], "compas": ["sex", "race"]}
PROTECTED_TERMS = {
    "sex": [r"\bsex\b", r"\bmale\b", r"\bfemale\b", r"\bman\b", r"\bwoman\b", r"\bgender\b"],
    "age": [r"\bage\b", r"\byears old\b", r"\byounger\b", r"\bolder\b", r"\belderly\b"],
    "race": [r"\brace\b", r"\bracial\b", r"\bcaucasian\b", r"\bafrican\b", r"\bhispanic\b",
             r"\bwhite\b", r"\bblack\b"],
}


def _mentions_protected(text, dataset):
    """Protected attribute named by a generated chain, or None."""
    lowered = text.lower()
    for attr in PROTECTED[dataset]:
        for term in PROTECTED_TERMS[attr]:
            if re.search(term, lowered):
                return attr, term
    return None


def test_german_chains_never_mention_a_protected_attribute():
    # age is protected in German, so no chain may reason over it either.
    for ch, sv, em in itertools.product(["A30", "A34"], ["A61", "A65"], ["A71", "A75"]):
        for sex, age in itertools.product([0, 1], [0, 1]):
            row = pd.Series({
                "checking_status": "A11", "duration": 24, "credit_history": ch,
                "purpose": "A43", "credit_amount": 3000, "savings": sv,
                "employment": em, "installment_rate": 2, "sex": sex,
                "other_debtors": "A101", "property": "A121", "age": age,
                "housing": "A152", "job": "A173", "label": 1,
            })
            for outcome in ("good", "bad"):
                chain = generate_reasoning(row, "german", outcome)
                hit = _mentions_protected(chain, "german")
                assert hit is None, f"german chain mentions {hit}: {chain}"


def test_compas_chains_never_mention_a_protected_attribute():
    for priors, degree, age in itertools.product([0, 8], [0, 1], [20, 50]):
        for sex, race in itertools.product([0, 1], [0, 1]):
            row = pd.Series({"sex": sex, "race": race, "age": age,
                             "c_charge_degree": degree, "priors_count": priors,
                             "two_year_recid": 1})
            for outcome in ("yes", "no"):
                chain = generate_reasoning(row, "compas", outcome)
                hit = _mentions_protected(chain, "compas")
                assert hit is None, f"compas chain mentions {hit}: {chain}"


def test_templates_reason_only_over_unprotected_features():
    for dataset in ("german", "compas"):
        mentioned = set(D4B_REASONING_TEMPLATE[dataset]["features_to_mention"])
        assert not mentioned & set(PROTECTED[dataset]), \
            f"{dataset} template reasons over a protected attribute"


def test_generated_chains_pass_the_protected_attribute_audit():
    """The module's own audit helper agrees with the checks above.

    reasoning_references_protected is the textual audit the generator exposes;
    running the generated chains through it keeps the helper honest against the
    templates rather than leaving it to be verified by eye.
    """
    cases = [
        ("german", "good", pd.Series({
            "checking_status": "A11", "duration": 24, "credit_history": "A34",
            "purpose": "A43", "credit_amount": 3000, "savings": "A63",
            "employment": "A73", "installment_rate": 2, "sex": 1,
            "other_debtors": "A101", "property": "A121", "age": 1,
            "housing": "A152", "job": "A173", "label": 1})),
        ("compas", "yes", pd.Series({
            "sex": 1, "race": 0, "age": 35, "c_charge_degree": 1,
            "priors_count": 4, "two_year_recid": 1})),
    ]
    for dataset, outcome, row in cases:
        chain = generate_reasoning(row, dataset, outcome)
        found = reasoning_references_protected(chain, dataset)
        assert found == set(), f"{dataset} chain references {found}: {chain}"
