"""Feature engineering must be causal and keep X / y / meta aligned."""
import numpy as np
import pandas as pd
import pytest

from config import LABEL
from features import build_features


def _patient_with_stay(long_df, min_hours=20):
    sizes = long_df.groupby("patient_id").size()
    pid = sizes[sizes >= min_hours].index[0]
    return long_df[long_df["patient_id"] == pid].sort_values("ICULOS")


@pytest.mark.parametrize("cut", [1, 5, 12])
def test_features_are_causal(sample_long_df, cut):
    """Appending future hours must not change any feature for earlier hours.
    If it did, the model would be peeking at data it can't have at the bedside."""
    patient = _patient_with_stay(sample_long_df)
    X_full, _, _ = build_features(patient)
    X_prefix, _, _ = build_features(patient.iloc[:cut])
    pd.testing.assert_frame_equal(
        X_prefix.reset_index(drop=True),
        X_full.iloc[:cut].reset_index(drop=True))


def test_future_values_do_not_leak(sample_long_df):
    """Perturbing hour t+1 onward leaves hours <= t identical."""
    patient = _patient_with_stay(sample_long_df).copy()
    t = 10
    perturbed = patient.copy()
    perturbed.iloc[t + 1:, perturbed.columns.get_loc("HR")] = 250.0
    perturbed.iloc[t + 1:, perturbed.columns.get_loc("Lactate")] = 15.0
    X_a, _, _ = build_features(patient)
    X_b, _, _ = build_features(perturbed)
    pd.testing.assert_frame_equal(X_a.iloc[: t + 1], X_b.iloc[: t + 1])


def test_unlabeled_rows_keep_alignment(sample_long_df):
    df = sample_long_df.copy()
    drop_idx = df.index[::7]
    df.loc[drop_idx, LABEL] = np.nan
    X, y, meta = build_features(df)
    assert len(X) == len(y) == len(meta) == df[LABEL].notna().sum()
    assert (meta[LABEL].to_numpy() == y.to_numpy()).all()
    assert (meta["ICULOS"].to_numpy() == X["ICULOS"].to_numpy()).all()


def test_no_nans_in_model_input(sample_long_df):
    X, _, _ = build_features(sample_long_df)
    assert not X.isna().any().any()
