"""Load a directory of per-patient .psv files into one long DataFrame."""
import glob
import os

import pandas as pd

from config import ALL_COLUMNS, LABEL


def load_psv_dir(path):
    """Read every .psv in `path`; return a long frame with a patient_id column.

    Fails if any file is unreadable or has no labels, so a corrupt download
    can't quietly shrink the training set.
    """
    files = sorted(glob.glob(os.path.join(path, "*.psv")))
    if not files:
        raise FileNotFoundError(f"No .psv files in {path}")
    frames, skipped = [], []
    for f in files:
        try:
            df = pd.read_csv(f, sep="|")
        except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError):
            skipped.append(f)
            continue
        if LABEL not in df.columns or df[LABEL].isna().all():
            skipped.append(f)
            continue
        for c in ALL_COLUMNS:
            if c not in df.columns:
                df[c] = pd.NA
        df["patient_id"] = os.path.splitext(os.path.basename(f))[0]
        frames.append(df)
    if skipped:
        raise ValueError(f"{len(skipped)} of {len(files)} files in {path} are unreadable "
                         f"or unlabeled, e.g. {skipped[:3]}")
    return pd.concat(frames, ignore_index=True)
