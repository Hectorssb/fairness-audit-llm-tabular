"""Tests for the metrics module: NA handling on impossible denominators."""

import numpy as np

from metrics import (
    group_metrics,
    disparate_impact,
    equal_opportunity_difference,
    evaluate,
)


def test_group_metrics_nan_on_empty_denominators():
    # No positive instances: TPR and FNR are unmeasured, not zero
    m = group_metrics(np.array([0, 0, 0]), np.array([0, 1, 0]))
    assert np.isnan(m["TPR"]) and np.isnan(m["FNR"])
    assert not np.isnan(m["FPR"]) and not np.isnan(m["A"])
    assert m["N"] == 3

    # Empty group: everything unmeasured
    m = group_metrics(np.array([]), np.array([]))
    assert all(np.isnan(m[k]) for k in ("A", "TPR", "FPR", "FNR", "PPP"))
    assert m["N"] == 0


def test_disparate_impact_inf_vs_nan():
    sensitive = np.array([1, 1, 0, 0])
    y_true    = np.array([1, 0, 1, 0])

    # Privileged PPP = 0, underprivileged PPP > 0 -> maximal disparity (inf)
    assert np.isinf(disparate_impact(y_true, np.array([0, 0, 1, 1]), sensitive, 1))

    # Both PPP = 0 -> 0/0, unmeasured (NaN)
    assert np.isnan(disparate_impact(y_true, np.array([0, 0, 0, 0]), sensitive, 1))

    # Normal case
    assert disparate_impact(y_true, np.array([1, 0, 1, 0]), sensitive, 1) == 1.0


def test_eod_nan_propagates_to_flag():
    # Privileged group has no positive instances -> TPR_priv NaN -> EOD NaN
    sensitive = np.array([1, 1, 0, 0])
    y_true    = np.array([0, 0, 1, 0])
    y_pred    = np.array([0, 1, 1, 0])
    assert np.isnan(equal_opportunity_difference(y_true, y_pred, sensitive, 1))

    _, fairness = evaluate(y_true, y_pred, sensitive, 1, "sex", "M", "toy")
    assert bool(fairness["insufficient_support"].iloc[0]) is True


def test_evaluate_flag_false_with_full_support():
    sensitive = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    y_true    = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    y_pred    = np.array([1, 0, 0, 1, 1, 1, 1, 0])
    performance, fairness = evaluate(y_true, y_pred, sensitive, 1, "sex", "M", "toy")
    assert bool(fairness["insufficient_support"].iloc[0]) is False
    assert set(performance["N"]) == {4}
