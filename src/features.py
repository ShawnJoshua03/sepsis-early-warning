"""
Causal time-series feature engineering for hourly sepsis prediction.

Every feature at hour t uses only data available at or before t: forward-fills
and rolling/expanding windows, nothing that looks ahead. tests/test_features.py
checks this.

Per channel (vitals + labs):
  - forward-filled last observed value (carry the most recent measurement)
  - a "measured this hour" indicator (missingness is informative in ICU data)
  - hours since the channel was last measured
For vitals additionally:
  - rolling mean / min / max over the last ROLL_WINDOW hours
  - short-term slope (current minus value ROLL_WINDOW hours ago)
Plus ICULOS, demographics, and a running count of total lab draws.
"""
import numpy as np
import pandas as pd

from config import CHANNELS, DEMOGRAPHICS, LABEL, LABS, ROLL_WINDOW, VITALS


def _per_patient(g):
    g = g.sort_values("ICULOS").reset_index(drop=True)
    cols = {}

    for c in CHANNELS:
        col = g[c]
        measured = col.notna().astype(int)
        ff = col.ffill()
        cols[f"{c}_ff"] = ff
        cols[f"{c}_measured"] = measured
        since = np.full(len(g), np.nan)
        last = np.nan
        for i, m in enumerate(measured.to_numpy()):
            last = 0 if m else (last + 1 if not np.isnan(last) else np.nan)
            since[i] = last
        cols[f"{c}_since"] = since

    for v in VITALS:
        ff = cols[f"{v}_ff"]
        cols[f"{v}_roll_mean"] = ff.rolling(ROLL_WINDOW, min_periods=1).mean()
        cols[f"{v}_roll_min"] = ff.rolling(ROLL_WINDOW, min_periods=1).min()
        cols[f"{v}_roll_max"] = ff.rolling(ROLL_WINDOW, min_periods=1).max()
        cols[f"{v}_slope"] = ff - ff.shift(ROLL_WINDOW)

    for d in DEMOGRAPHICS:
        cols[d] = g[d].values
    cols["lab_draws_so_far"] = (
        g[LABS].notna().any(axis=1).astype(int).cumsum().values)
    cols["patient_id"] = g["patient_id"].values
    cols["ICULOS"] = g["ICULOS"].values
    cols[LABEL] = g[LABEL].values

    return pd.DataFrame(cols, index=g.index)


def build_features(long_df):
    """Return (X, y, meta) where meta carries patient_id, ICULOS for grouping
    and utility scoring. X is numeric and model-ready."""
    # Ensure numeric dtype: uploaded CSVs and streamed JSON can arrive with
    # object/NA columns that break rolling windows. Coerce once here.
    long_df = long_df.copy()
    for c in CHANNELS + DEMOGRAPHICS:
        if c in long_df.columns:
            long_df[c] = pd.to_numeric(long_df[c], errors="coerce")

    parts = [
        _per_patient(g)
        for _, g in long_df.groupby("patient_id", sort=False)
    ]
    feat = pd.concat(parts, ignore_index=True)

    # Drop unlabeled rows BEFORE building meta so X, y and meta stay row-aligned.
    feat = feat[feat[LABEL].notna()].reset_index(drop=True)
    meta = feat[["patient_id", "ICULOS", LABEL]].copy()
    y = feat[LABEL].astype(int)
    X = feat.drop(columns=["patient_id", LABEL])
    # Any remaining NaNs (channels never measured for a patient) -> fill with
    # a sentinel; the "_measured" flags let the model know it was imputed.
    X = X.fillna(-1.0)
    return X, y, meta

