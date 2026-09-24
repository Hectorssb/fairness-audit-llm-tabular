"""Tests for the dataset loaders: German column semantics and COMPAS features."""

import json
from pathlib import Path

import pandas as pd
import pytest

ucimlrepo = pytest.importorskip("ucimlrepo")

import data_loader
from data_loader import GERMAN_COLUMN_MAP

PROMPTS_DIR = Path(__file__).parent.parent / "src" / "prompts"

# Official UCI Statlog (German Credit Data) codebook, id=144:
# https://archive.ics.uci.edu/dataset/144
OFFICIAL_GERMAN_ATTRIBUTES = {
    "Attribute1":  "checking_status",   # Status of existing checking account (A11-A14)
    "Attribute2":  "duration",          # Duration in months
    "Attribute3":  "credit_history",    # Credit history (A30-A34)
    "Attribute4":  "purpose",           # Purpose (A40-A410)
    "Attribute5":  "credit_amount",     # Credit amount
    "Attribute6":  "savings",           # Savings account/bonds (A61-A65)
    "Attribute7":  "employment",        # Present employment since (A71-A75)
    "Attribute8":  "installment_rate",  # Installment rate (% of disposable income)
    "Attribute9":  "sex",               # Personal status and sex (A91-A95)
    "Attribute10": "other_debtors",     # Other debtors / guarantors (A101-A103)
    "Attribute12": "property",          # Property (A121-A124)
    "Attribute13": "age",               # Age in years
    "Attribute15": "housing",           # Housing (A151-A153)
    "Attribute17": "job",               # Job (A171-A174)
}


def test_german_column_map_matches_official_codebook():
    for attr, name in OFFICIAL_GERMAN_ATTRIBUTES.items():
        assert GERMAN_COLUMN_MAP.get(attr) == name, (
            f"{attr} must map to '{name}' per the UCI codebook, "
            f"got '{GERMAN_COLUMN_MAP.get(attr)}'"
        )


def test_german_column_map_excludes_telephone_and_foreign_worker():
    assert "Attribute19" not in GERMAN_COLUMN_MAP  # telephone
    assert "Attribute20" not in GERMAN_COLUMN_MAP  # foreign worker
    assert "housing" not in {GERMAN_COLUMN_MAP.get("Attribute19"), GERMAN_COLUMN_MAP.get("Attribute20")}


def test_compas_loader_excludes_score_text(tmp_path, monkeypatch):
    raw = pd.DataFrame({
        "sex": ["Male", "Female"] * 10,
        "race": ["African-American", "Caucasian"] * 10,
        "age": list(range(20, 40)),
        "c_charge_degree": ["F", "M"] * 10,
        "priors_count": list(range(20)),
        "score_text": ["Low", "High"] * 10,
        "days_b_screening_arrest": [0] * 20,
        "is_recid": [0, 1] * 10,
        "two_year_recid": [0, 1] * 10,
    })
    monkeypatch.setattr(data_loader, "DATA_DIR", tmp_path)
    monkeypatch.setattr(pd, "read_csv", lambda *a, **k: raw.copy())

    df = data_loader.load_compas(save=False)
    assert "score_text" not in df.columns


def test_stale_cache_is_ignored(tmp_path):
    cache = tmp_path / "german.csv"
    stale = [c for c in data_loader.GERMAN_COLUMNS if c != "checking_status"]
    pd.DataFrame(columns=stale + ["credit_history_detail"]).to_csv(cache, index=False)

    assert data_loader._read_cache(cache, data_loader.GERMAN_COLUMNS, "German") is None


def test_cache_with_expected_columns_is_used(tmp_path):
    cache = tmp_path / "compas.csv"
    pd.DataFrame(columns=data_loader.COMPAS_COLUMNS).to_csv(cache, index=False)

    cached = data_loader._read_cache(cache, data_loader.COMPAS_COLUMNS, "COMPAS")
    assert cached is not None
    assert list(cached.columns) == data_loader.COMPAS_COLUMNS


def test_score_text_absent_from_all_prompt_files():
    for path in PROMPTS_DIR.glob("*.json"):
        content = json.loads(path.read_text(encoding="utf-8"))
        assert "score_text" not in json.dumps(content), f"score_text found in {path.name}"
