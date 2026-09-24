"""Tests for demonstration sampling: reproducibility and class balance.

Every condition draws ten demonstrations, five per class, and must draw the
same ten given the same seed, since otherwise a rerun would not reproduce the
stored predictions.
"""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from hf_classifier import select_fewshot_examples, demo_seed_suffix


def _toy_train(n=100):
    rng = np.random.RandomState(0)
    return pd.DataFrame({
        "row_id": range(n),
        "feature": rng.randint(0, 5, n),
        "y": np.tile([0, 1], n // 2),
    })


def test_the_default_seed_leaves_filenames_unsuffixed():
    # The shipped results carry no _seed tag, so 42 must produce none.
    assert demo_seed_suffix(42) == ""
    assert demo_seed_suffix(43) == "_seed43"


def test_sampling_is_reproducible():
    df = _toy_train()
    first = select_fewshot_examples(df, "y", 10, seed=42)
    second = select_fewshot_examples(df, "y", 10, seed=42)
    assert list(first["row_id"]) == list(second["row_id"])


def test_demonstrations_are_balanced_five_and_five():
    examples = select_fewshot_examples(_toy_train(), "y", 10, seed=42)
    assert len(examples) == 10
    assert examples["y"].sum() == 5
