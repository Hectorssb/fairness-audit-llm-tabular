"""Tests for length bucketing: ordering and restoration of instance order."""

import pytest

torch = pytest.importorskip("torch")

from hf_classifier import _length_sorted_order


class _FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()


def test_order_is_a_permutation_sorted_by_length():
    texts = ["a b c", "a", "a b c d e", "a b", "a b c d"]
    order = _length_sorted_order(_FakeTokenizer(), texts)
    assert sorted(order) == list(range(len(texts)))
    lengths = [len(texts[i].split()) for i in order]
    assert lengths == sorted(lengths)


def test_batched_processing_restores_original_order():
    texts = [f"{'x ' * (i % 7)}row{i}" for i in range(23)]
    order = _length_sorted_order(_FakeTokenizer(), texts)
    predictions = [None] * len(texts)
    batch_size = 4
    for start in range(0, len(order), batch_size):
        batch_idx = order[start:start + batch_size]
        responses = [texts[j] for j in batch_idx]
        for j, resp in zip(batch_idx, responses):
            predictions[j] = resp
    assert predictions == texts
