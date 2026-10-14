"""
Official PhysioNet/CinC 2019 utility metric.

`compute_prediction_utility` is taken verbatim from the challenge's
evaluate_sepsis_score.py (physionetchallenges/evaluation-2019, BSD-licensed)
so scores here are computed the same way the Challenge scored entries.
`normalized_utility` replicates the cohort normalization from that script but
operates on in-memory per-patient arrays instead of directories of files.
"""
import numpy as np


def compute_prediction_utility(labels, predictions, dt_early=-12,
                               dt_optimal=-6, dt_late=3.0, max_u_tp=1,
                               min_u_fn=-2, u_fp=-0.05, u_tn=0):
    labels = np.asarray(labels, dtype=int)
    predictions = np.asarray(predictions, dtype=int)

    if np.any(labels):
        is_septic = True
        t_sepsis = np.argmax(labels) - dt_optimal
    else:
        is_septic = False
        t_sepsis = float("inf")

    n = len(labels)
    m_1 = float(max_u_tp) / float(dt_optimal - dt_early)
    b_1 = -m_1 * dt_early
    m_2 = float(-max_u_tp) / float(dt_late - dt_optimal)
    b_2 = -m_2 * dt_late
    m_3 = float(min_u_fn) / float(dt_late - dt_optimal)
    b_3 = -m_3 * dt_optimal

    u = np.zeros(n)
    for t in range(n):
        if t <= t_sepsis + dt_late:
            if is_septic and predictions[t]:
                if t <= t_sepsis + dt_optimal:
                    u[t] = max(m_1 * (t - t_sepsis) + b_1, u_fp)
                elif t <= t_sepsis + dt_late:
                    u[t] = m_2 * (t - t_sepsis) + b_2
            elif not is_septic and predictions[t]:
                u[t] = u_fp
            elif is_septic and not predictions[t]:
                if t <= t_sepsis + dt_optimal:
                    u[t] = 0
                elif t <= t_sepsis + dt_late:
                    u[t] = m_3 * (t - t_sepsis) + b_3
            elif not is_septic and not predictions[t]:
                u[t] = u_tn
    return float(np.sum(u))


def normalized_utility(cohort_labels, cohort_predictions,
                       dt_early=-12, dt_optimal=-6, dt_late=3.0):
    """Normalized observed utility over a cohort.

    cohort_labels / cohort_predictions: lists of per-patient 1D arrays
    (binary), one array per patient, in hourly order. Returns a score where
    1.0 is the best achievable and 0.0 is never predicting positive.
    """
    obs = best = inaction = 0.0
    for labels, preds in zip(cohort_labels, cohort_predictions):
        labels = np.asarray(labels, dtype=int)
        num_rows = len(labels)
        best_pred = np.zeros(num_rows, dtype=int)
        if np.any(labels):
            t_sepsis = int(np.argmax(labels)) - dt_optimal
            lo = int(max(0, t_sepsis + dt_early))
            hi = int(min(t_sepsis + dt_late + 1, num_rows))
            best_pred[lo:hi] = 1
        obs += compute_prediction_utility(labels, preds, dt_early, dt_optimal, dt_late)
        best += compute_prediction_utility(labels, best_pred, dt_early, dt_optimal, dt_late)
        inaction += compute_prediction_utility(labels, np.zeros(num_rows, dtype=int),
                                               dt_early, dt_optimal, dt_late)
    if best == inaction:
        return 0.0
    return (obs - inaction) / (best - inaction)
