"""
Export a handful of HELD-OUT test patients for the web demo.

Picks real patients from the test split (never seen in training) covering the
cases worth showing: septic and caught in the useful window, septic but
missed, non-septic and quiet, and a non-septic false alarm. Raw hourly rows are
written to src/static/demo_patients.json for the demo page to replay through
POST /score.

Challenge data is CC BY 4.0 (PhysioNet/CinC 2019); see NOTICE.md.

Usage: python scripts/export_demo.py --data training_setA
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from config import ALL_COLUMNS, LABEL, LABEL_SHIFT_HOURS

ARTIFACT_DIR = Path("artifacts")
OUT = Path("src/static/demo_patients.json")
WINDOW_EARLY, WINDOW_LATE = -12, 3


def _summarize(g, thr):
    g = g.sort_values("ICULOS").reset_index(drop=True)
    labels = g[LABEL].to_numpy()
    preds = g["proba"].to_numpy() >= thr
    s = {"hours": len(g), "septic": bool(labels.any()), "any_alarm": bool(preds.any())}
    if s["septic"]:
        onset = int(np.argmax(labels)) + LABEL_SHIFT_HOURS
        lo, hi = max(0, onset + WINDOW_EARLY), onset + WINDOW_LATE + 1
        s["early_alarm"] = bool(preds[:lo].any())
        s["caught"] = bool(preds[lo:hi].any())
        s["onset_row"] = onset
    return s


def _pick(summ, want, n, rng):
    ids = [pid for pid, s in summ.items() if want(s)]
    rng.shuffle(ids)
    return ids[:n]


def main(data_dir):
    thr = json.loads((ARTIFACT_DIR / "metrics.json").read_text())["threshold"]
    preds = pd.read_csv(ARTIFACT_DIR / "test_predictions.csv")
    summ = {pid: _summarize(g, thr) for pid, g in preds.groupby("patient_id")}

    rng = np.random.default_rng(7)

    def fits(s):
        return 24 <= s["hours"] <= 96

    picks = [
        ("Septic, caught in time", _pick(summ, lambda s: fits(s) and s["septic"]
         and s["caught"] and not s["early_alarm"] and s["onset_row"] >= 18, 3, rng)),
        ("Septic, not caught in window", _pick(summ, lambda s: fits(s) and s["septic"]
         and not s["caught"], 1, rng)),
        ("Not septic, no alarm", _pick(summ, lambda s: fits(s) and not s["septic"]
         and not s["any_alarm"], 2, rng)),
        ("Not septic, false alarm", _pick(summ, lambda s: fits(s) and not s["septic"]
         and s["any_alarm"], 1, rng)),
    ]

    patients = []
    for kind, ids in picks:
        for pid in ids:
            df = pd.read_csv(os.path.join(data_dir, f"{pid}.psv"), sep="|")
            df = df[[c for c in ALL_COLUMNS if c in df.columns]]
            records = [{k: (int(v) if k == "ICULOS" else float(v))
                        for k, v in row.items() if pd.notna(v)}
                       for row in df.to_dict("records")]
            labels = df[LABEL].fillna(0).to_numpy()
            onset_iculos = (int(df["ICULOS"].iloc[int(np.argmax(labels))])
                            + LABEL_SHIFT_HOURS if labels.any() else None)
            patients.append({"id": pid, "kind": kind,
                             "onset_iculos": onset_iculos, "records": records})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(patients, separators=(",", ":")))
    print(f"Wrote {len(patients)} patients to {OUT}:")
    for p in patients:
        print(f"  {p['id']}  {p['kind']:<26} {len(p['records'])}h  "
              f"onset={p['onset_iculos']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="training_setA")
    main(ap.parse_args().data)
