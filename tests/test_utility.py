"""Sanity checks on the vendored official utility metric."""
import numpy as np

from sepsis_utility import compute_prediction_utility, normalized_utility


def _septic(n=40, first_label=20):
    labels = np.zeros(n, dtype=int)
    labels[first_label:] = 1
    return labels


def test_never_predicting_scores_zero():
    labels = [_septic(), np.zeros(30, dtype=int)]
    preds = [np.zeros_like(y) for y in labels]
    assert normalized_utility(labels, preds) == 0.0


def test_optimal_predictions_score_one():
    labels = [_septic(), np.zeros(30, dtype=int)]
    preds = [labels[0].copy(), np.zeros(30, dtype=int)]
    # Optimal = predict from 12h before onset (onset = first label + 6) to +3h.
    onset = 20 + 6
    preds[0][:] = 0
    preds[0][onset - 12: onset + 3 + 1] = 1
    assert np.isclose(normalized_utility(labels, preds), 1.0)


def test_false_alarms_are_penalized():
    labels = np.zeros(30, dtype=int)
    assert np.isclose(compute_prediction_utility(labels, np.ones(30, dtype=int)), -0.05 * 30)


def test_alerting_too_early_earns_nothing():
    """Alerts long before the window score as false positives, not credit."""
    labels = _septic(n=60, first_label=40)  # onset at row 46
    early = np.zeros(60, dtype=int)
    early[5] = 1                            # 41h before onset
    assert compute_prediction_utility(labels, early) < 0
