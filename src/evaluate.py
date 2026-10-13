"""
Evaluate the model the way the challenge and a clinician would read it.

Reports:
  - Official normalized utility (the challenge's headline metric).
  - AUROC / AUPRC for discrimination.
  - Early-warning lead time: for septic patients detected inside the useful
    window, how many hours BEFORE clinical onset did the first in-window alert
    fire? This is the number a clinician cares about.

The useful window is the one the utility metric rewards: from 12h before
clinical onset to 3h after. An alert earlier than that is not "very early
detection"; it is an alarm that fired before anything predictive could be
seen (often at admission) and the metric treats it as a false positive.
Those are counted separately so they cannot inflate the lead time.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from config import LABEL, LABEL_SHIFT_HOURS
from sepsis_utility import normalized_utility

ARTIFACT_DIR = Path("artifacts")
# Utility-metric window relative to clinical onset (hours), see sepsis_utility.
WINDOW_EARLY, WINDOW_LATE = -12, 3


def evaluate():
    meta = pd.read_csv(ARTIFACT_DIR / "test_predictions.csv")
    thr = json.loads((ARTIFACT_DIR / "metrics.json").read_text())["threshold"]

    labels_list, preds_list, lead_times = [], [], []
    n_septic = n_detected = n_early_alarm = n_nonseptic = n_nonseptic_alarm = 0

    for _, g in meta.groupby("patient_id", sort=False):
        g = g.sort_values("ICULOS").reset_index(drop=True)
        labels = g[LABEL].to_numpy()
        proba = g["proba"].to_numpy()
        preds = (proba >= thr).astype(int)
        labels_list.append(labels)
        preds_list.append(preds)

        if labels.any():
            n_septic += 1
            # clinical onset = first shifted-label hour + shift (row index, the
            # same time axis the utility metric uses)
            onset_hour = int(np.argmax(labels)) + LABEL_SHIFT_HOURS
            lo = max(0, onset_hour + WINDOW_EARLY)
            hi = onset_hour + WINDOW_LATE + 1
            if preds[:lo].any():
                n_early_alarm += 1
            in_window = np.flatnonzero(preds[lo:hi])
            if len(in_window):
                n_detected += 1
                # hours before clinical onset that the first in-window alert fired
                lead_times.append(onset_hour - (lo + int(in_window[0])))
        else:
            n_nonseptic += 1
            n_nonseptic_alarm += int(preds.any())

    util = normalized_utility(labels_list, preds_list)
    y = meta[LABEL].to_numpy()
    p = meta["proba"].to_numpy()

    lead = np.array(lead_times, dtype=float)
    report = {
        "normalized_utility": round(util, 4),
        "auroc": round(float(roc_auc_score(y, p)), 4),
        "auprc": round(float(average_precision_score(y, p)), 4),
        "threshold": thr,
        "septic_patients": n_septic,
        "septic_detected_in_window": n_detected,
        "detection_rate": round(n_detected / n_septic, 3) if n_septic else None,
        "median_lead_time_hours": round(float(np.median(lead)), 1) if len(lead) else None,
        "septic_with_alarm_before_window": n_early_alarm,
        "nonseptic_patients": n_nonseptic,
        "nonseptic_patients_alarmed": n_nonseptic_alarm,
        "nonseptic_alarm_rate": (round(n_nonseptic_alarm / n_nonseptic, 3)
                                 if n_nonseptic else None),
    }
    print(json.dumps(report, indent=2))
    if len(lead):
        print(f"\nPlain-English: of {n_septic} septic patients, {n_detected} were "
              f"flagged between 12h before and 3h after clinical onset, a median "
              f"of {np.median(lead):.0f}h before onset. {n_early_alarm} septic "
              f"patients also had an alarm earlier than that window, and "
              f"{n_nonseptic_alarm}/{n_nonseptic} non-septic patients were "
              f"alarmed at least once.")
    (ARTIFACT_DIR / "clinical_report.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    evaluate()
