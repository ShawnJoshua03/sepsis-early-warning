"""
Train the hourly sepsis-risk model.

Key correctness choices:
  - Split by PATIENT, never by row. Hours from one patient must not appear in
    both train and test, or the model leaks and the score is fiction.
  - Tune the alert threshold to maximize the official normalized utility on
    OUT-OF-FOLD predictions for the training patients (patient-grouped CV),
    then apply it unchanged to test. Tuning on the final model's in-sample
    predictions would pick a threshold for overconfident scores that don't
    look like what the model produces on new patients. Utility, not accuracy
    or F1, is what the challenge rewards.
"""
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from config import LABEL
from features import build_features
from load import load_psv_dir
from sepsis_utility import normalized_utility

ARTIFACT_DIR = Path("artifacts")
CV_FOLDS = 3


def _sequences(meta, proba, threshold):
    """Group into per-patient hourly (labels, binary-preds) arrays for utility."""
    df = meta.copy()
    df["proba"] = proba
    labels_list, preds_list = [], []
    for _, g in df.groupby("patient_id", sort=False):
        g = g.sort_values("ICULOS")
        labels_list.append(g[LABEL].to_numpy())
        preds_list.append((g["proba"].to_numpy() >= threshold).astype(int))
    return labels_list, preds_list


def _best_threshold(meta, proba):
    grid = np.round(np.arange(0.01, 0.96, 0.01), 3)
    best_t, best_u = 0.5, -1e9
    for t in grid:
        labels_list, preds_list = _sequences(meta, proba, t)
        u = normalized_utility(labels_list, preds_list)
        if u > best_u:
            best_u, best_t = u, t
    return float(best_t), float(best_u)


def _fit(X, y):
    """Class-balanced gradient boosting (sepsis hours are ~2% of rows)."""
    w = np.where(y == 1, (y == 0).sum() / max((y == 1).sum(), 1), 1.0)
    clf = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_depth=6,
        l2_regularization=1.0, random_state=42)
    clf.fit(X, y, sample_weight=w)
    return clf


def _oof_proba(X, y, groups, n_splits=CV_FOLDS):
    """Out-of-fold probabilities: each training patient is scored by a model
    that never saw them, so the scores resemble those on unseen patients."""
    oof = np.zeros(len(y))
    for k, (fit_idx, val_idx) in enumerate(
            GroupKFold(n_splits=n_splits).split(X, y, groups), start=1):
        clf = _fit(X.iloc[fit_idx], y.iloc[fit_idx])
        oof[val_idx] = clf.predict_proba(X.iloc[val_idx])[:, 1]
        print(f"  fold {k}/{n_splits} done")
    return oof


def train(data_dir="data/sample_psv"):
    ARTIFACT_DIR.mkdir(exist_ok=True)
    long_df = load_psv_dir(data_dir)
    X, y, meta = build_features(long_df)

    groups = meta["patient_id"].values
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    tr_idx, te_idx = next(splitter.split(X, y, groups))

    X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
    y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]
    meta_tr, meta_te = meta.iloc[tr_idx], meta.iloc[te_idx]

    print(f"Tuning threshold on {CV_FOLDS}-fold out-of-fold predictions...")
    p_oof = _oof_proba(X_tr, y_tr, meta_tr["patient_id"].values)
    thr, u_oof = _best_threshold(meta_tr, p_oof)

    clf = _fit(X_tr, y_tr)  # final model on all training patients
    p_tr = clf.predict_proba(X_tr)[:, 1]
    p_te = clf.predict_proba(X_te)[:, 1]

    labels_tr, preds_tr = _sequences(meta_tr, p_tr, thr)
    u_tr = normalized_utility(labels_tr, preds_tr)
    labels_te, preds_te = _sequences(meta_te, p_te, thr)
    u_te = normalized_utility(labels_te, preds_te)

    metrics = {
        "test_auroc": round(float(roc_auc_score(y_te, p_te)), 4),
        "test_auprc": round(float(average_precision_score(y_te, p_te)), 4),
        "threshold": thr,
        "cv_utility": round(u_oof, 4),          # what threshold tuning saw
        "train_utility_in_sample": round(u_tr, 4),  # overfit reference only
        "test_normalized_utility": round(u_te, 4),
        "n_train_patients": int(meta_tr["patient_id"].nunique()),
        "n_test_patients": int(meta_te["patient_id"].nunique()),
    }
    print(json.dumps(metrics, indent=2))

    joblib.dump({"model": clf, "threshold": thr,
                 "columns": list(X.columns)}, ARTIFACT_DIR / "model.joblib")
    meta_te = meta_te.copy()
    meta_te["proba"] = p_te
    meta_te.to_csv(ARTIFACT_DIR / "test_predictions.csv", index=False)
    (ARTIFACT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved model + test predictions to {ARTIFACT_DIR}/")
    return metrics


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/sample_psv",
                    help="Dir of .psv files (real: training_setA/setB)")
    train(ap.parse_args().data)
