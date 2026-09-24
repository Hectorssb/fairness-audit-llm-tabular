"""Tests for the canonical split protocol"""

import numpy as np
import pandas as pd
import pytest

import fair_data


def _toy_df(n=60):
    rng = np.random.RandomState(0)
    return pd.DataFrame({
        "row_id": range(n),
        "sex": rng.randint(0, 2, n),
        "feature": rng.randint(0, 5, n),
        "y": np.tile([0, 1], n // 2),
    })


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    df = _toy_df()
    captured = {}

    def fake_fair(df, target_col, sensitive_features, n_samples, **kwargs):
        captured["flai_input"] = df.copy()
        return df.sample(n_samples, replace=True, random_state=0).reset_index(drop=True)

    def fake_resample(df, sensitive_feature, privileged_value, strategy, save_path):
        captured["resample_input"] = df.copy()
        df.to_csv(save_path, index=False)
        return df

    monkeypatch.setattr(fair_data, "generate_fair_data_causal", fake_fair)
    monkeypatch.setattr(fair_data, "generate_resampled_data", fake_resample)

    result = fair_data.prepare_all_datasets(
        df=df, dataset_name="toy", target_col="y",
        sensitive_features=["sex"], privileged_values={"sex": 1},
        output_dir=tmp_path, test_size=20, seed=42,
    )
    return df, result, captured


def test_split_is_disjoint_and_complete(prepared):
    df, result, _ = prepared
    train_ids = set(result["D1_train"]["row_id"])
    test_ids  = set(result["D1_test"]["row_id"])
    assert train_ids & test_ids == set()
    assert train_ids | test_ids == set(df["row_id"])
    assert len(test_ids) == 20


def test_flai_only_sees_train(prepared):
    _, result, captured = prepared
    flai_ids  = set(captured["flai_input"]["row_id"])
    train_ids = set(result["D1_train"]["row_id"])
    test_ids  = set(result["D1_test"]["row_id"])
    assert flai_ids == train_ids
    assert flai_ids & test_ids == set()


def test_d3_generated_from_train_only(prepared):
    _, result, captured = prepared
    resample_ids = set(captured["resample_input"]["row_id"])
    train_ids    = set(result["D1_train"]["row_id"])
    assert resample_ids == train_ids


def test_d2_split_sizes_match_canonical_split(prepared):
    _, result, _ = prepared
    assert len(result["D2_train"]) == len(result["D1_train"])
    assert len(result["D2_test"])  == len(result["D1_test"])
