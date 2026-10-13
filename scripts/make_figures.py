"""
README figures from the held-out test predictions in artifacts/.

  docs/utility_vs_threshold.png  test utility across alert thresholds, with
                                 the threshold chosen by cross-validation
  docs/example_patient.png       one septic test patient's hourly risk

Usage: python scripts/make_figures.py   (after train.py)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import LABEL, LABEL_SHIFT_HOURS
from sepsis_utility import normalized_utility

ARTIFACT_DIR, DOCS = Path("artifacts"), Path("docs")
DEMO = Path("src/static/demo_patients.json")
BLUE, RED, GREEN = "#2457c5", "#b4232c", "#1f8a4c"

plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150})


def utility_curve(preds, thr):
    groups = [g.sort_values("ICULOS") for _, g in preds.groupby("patient_id")]
    labels = [g[LABEL].to_numpy() for g in groups]
    proba = [g["proba"].to_numpy() for g in groups]
    grid = np.round(np.arange(0.05, 0.81, 0.01), 2)
    util = [normalized_utility(labels, [(p >= t).astype(int) for p in proba]) for t in grid]

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.plot(grid, util, color=BLUE, lw=2)
    ax.axvline(thr, color=RED, ls="--", lw=1.3)
    u_thr = util[int(np.argmin(np.abs(grid - thr)))]
    ax.annotate(f"chosen by CV: {thr:.2f}\ntest utility {u_thr:.3f}",
                (thr, u_thr), xytext=(12, -36), textcoords="offset points", fontsize=9)
    ax.set(xlabel="alert threshold", ylabel="normalized utility (test)",
           title="Alert threshold vs. challenge utility")
    ax.grid(axis="y", alpha=.3)
    fig.tight_layout()
    fig.savefig(DOCS / "utility_vs_threshold.png")


def example_patient(preds, thr):
    demo = json.loads(DEMO.read_text())
    p = next(d for d in demo if d["kind"].startswith("Septic, caught"))
    g = preds[preds["patient_id"] == p["id"]].sort_values("ICULOS")
    onset = p["onset_iculos"]

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.axvspan(onset - 12, onset + 3, color=GREEN, alpha=.12, label="useful window (−12h to +3h)")
    ax.plot(g["ICULOS"], g["proba"], color=BLUE, lw=2, label="hourly risk")
    ax.axhline(thr, color=RED, ls="--", lw=1.3, label="alert threshold")
    ax.axvline(onset, color=GREEN, lw=2, label="clinical onset")
    first = g[g["proba"] >= thr]["ICULOS"].min()
    ax.plot(first, g.set_index("ICULOS").loc[first, "proba"], "o", color=RED)
    ax.set(xlabel="hours in ICU", ylabel="risk", ylim=(0, 1),
           title=f"Held-out patient {p['id']}: alert {onset - first}h before onset")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(DOCS / "example_patient.png")


if __name__ == "__main__":
    DOCS.mkdir(exist_ok=True)
    thr = json.loads((ARTIFACT_DIR / "metrics.json").read_text())["threshold"]
    preds = pd.read_csv(ARTIFACT_DIR / "test_predictions.csv")
    utility_curve(preds, thr)
    example_patient(preds, thr)
    print("Wrote docs/utility_vs_threshold.png and docs/example_patient.png")
