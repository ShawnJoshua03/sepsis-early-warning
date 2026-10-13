"""Load a directory of per-patient .psv files into one long DataFrame."""
import glob
import os

import pandas as pd

from config import ALL_COLUMNS


def load_psv_dir(path):
    """Read every .psv in `path`; return a long frame with a patient_id column.

    Works for both the synthetic sample and the real challenge folders
    (training_setA / training_setB), which have identical schemas.
    """
    files = sorted(glob.glob(os.path.join(path, "*.psv")))
    if not files:
        raise FileNotFoundError(f"No .psv files in {path}")
    frames = []
    for f in files:
        pid = os.path.splitext(os.path.basename(f))[0]
        try:
            df = pd.read_csv(f, sep="|")
        except Exception:
            continue
        if df.empty or df["SepsisLabel"].isna().all():
            continue
        for c in ALL_COLUMNS:
            if c not in df.columns:
                df[c] = pd.NA
        df["patient_id"] = pid
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    return out


if __name__ == "__main__":
    df = load_psv_dir("data/sample_psv")
    print("rows:", len(df), "patients:", df["patient_id"].nunique())
    print("sepsis-positive hours:", int(df["SepsisLabel"].sum()))
