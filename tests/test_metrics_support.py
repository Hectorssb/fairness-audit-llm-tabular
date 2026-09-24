"""Tests for the support flag and the DI interval."""

import numpy as np

from metrics import disparate_impact, disparate_impact_interval, evaluate


def _cell(n_priv, pos_priv, n_unpriv, pos_unpriv):
    """Build one cell with the given positive predictions per group."""
    sensitive = np.array([1] * n_priv + [0] * n_unpriv)
    y_pred = np.array([1] * pos_priv + [0] * (n_priv - pos_priv)
                      + [1] * pos_unpriv + [0] * (n_unpriv - pos_unpriv))
    y_true = np.resize([0, 1], len(y_pred))
    return y_true, y_pred, sensitive


def _flag(*args):
    _, fairness = evaluate(*_cell(*args), 1)
    return bool(fairness["insufficient_support"].iloc[0])


def test_a_group_without_positives_is_unmeasured_either_way():
    """DI rounds to 0.0 one way and inf the other; both are degenerate."""
    assert _flag(180, 30, 100, 0) is True
    assert _flag(180, 0, 100, 60) is True


def test_a_group_predicted_positive_throughout_is_unmeasured():
    """The mirror case: DI lands on 1.0 and would read as perfect parity."""
    assert _flag(250, 250, 250, 250) is True
    assert _flag(250, 249, 250, 249) is True
    assert _flag(250, 250, 250, 125) is True


def test_a_group_that_separates_nothing_is_unmeasured():
    """Rates saturate while the predicted-positive rate stays mid-range.

    Taken from a real pass: every true positive passes and 92% of negatives
    do too, leaving PPP at 0.94 — inside the margin, so only the rates show it.
    """
    sensitive = np.array([1] * 100 + [0] * 100)
    y_true = np.array([1] * 20 + [0] * 80 + [1] * 50 + [0] * 50)
    y_pred = np.array([1] * 94 + [0] * 6 + [1] * 50 + [0] * 50)
    _, fairness = evaluate(y_true, y_pred, sensitive, 1)
    assert bool(fairness["insufficient_support"].iloc[0]) is True


def test_measured_disparity_keeps_its_support():
    assert _flag(250, 60, 250, 55) is False


def test_interval_widens_when_a_group_is_thin():
    """A ratio resting on one positive prediction carries no precision."""
    thin_low, thin_high = disparate_impact_interval(*_cell(180, 1, 100, 60), 1)
    wide_low, wide_high = disparate_impact_interval(*_cell(250, 60, 250, 55), 1)

    assert thin_high - thin_low > 100
    assert wide_high - wide_low < 1


def test_interval_is_undefined_without_positives_in_a_group():
    low, high = disparate_impact_interval(*_cell(180, 30, 100, 0), 1)
    assert np.isnan(low) and np.isnan(high)


def test_interval_brackets_the_point_estimate():
    y_true, y_pred, sensitive = _cell(250, 60, 250, 55)
    di = disparate_impact(y_true, y_pred, sensitive, 1)
    low, high = disparate_impact_interval(y_true, y_pred, sensitive, 1)
    assert low < di < high
