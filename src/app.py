"""
Streaming sepsis-risk scorer.

POST a patient's hourly records (as they'd arrive from the monitor/EHR) and get
back the per-hour risk trajectory plus the first hour an alert fires. This
mirrors how an early-warning model actually runs at the bedside: one score per
hour, using only data up to that hour.
"""
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, create_model, field_validator

from config import ALL_COLUMNS, CHANNELS, DEMOGRAPHICS, LABEL
from features import build_features

ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = ROOT / "artifacts"
STATIC_DIR = ROOT / "src" / "static"
app = FastAPI(title="Sepsis Early Warning", version="1.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
_bundle = None


@app.get("/", include_in_schema=False)
def demo_page():
    return FileResponse(STATIC_DIR / "index.html")


def _load():
    global _bundle
    if _bundle is None:
        _bundle = joblib.load(ARTIFACT_DIR / "model.joblib")
    return _bundle


MAX_HOURS = 2000  # far beyond any real ICU stay in the challenge data

# One hourly row in the challenge schema. Every measurement is optional
# (missing values are the norm hour to hour), but ICULOS is the time index and
# is required. Unknown fields are rejected so a typo like "hr" for "HR" fails
# loudly instead of silently scoring the patient as "never measured".
HourlyRecord = create_model(
    "HourlyRecord",
    __config__=ConfigDict(extra="forbid"),
    **{c: (float | None, None)
       for c in CHANNELS + DEMOGRAPHICS if c != "ICULOS"},
    ICULOS=(int, Field(ge=1)),
    # Accepted so raw .psv rows can be posted as-is; ignored for inference.
    **{LABEL: (float | None, None)},
)


class PatientStream(BaseModel):
    # Hourly rows for ONE patient (any order; scored in ICULOS order).
    records: list[HourlyRecord] = Field(min_length=1, max_length=MAX_HOURS)

    @field_validator("records")
    @classmethod
    def _unique_hours(cls, records):
        hours = [r.ICULOS for r in records]
        if len(hours) != len(set(hours)):
            raise ValueError("duplicate ICULOS values; send one row per hour")
        return records


@app.get("/health")
def health():
    _load()
    return {"status": "ok"}


@app.post("/score")
def score(stream: PatientStream):
    bundle = _load()
    model, thr, columns = bundle["model"], bundle["threshold"], bundle["columns"]

    df = pd.DataFrame([r.model_dump() for r in stream.records])
    for c in ALL_COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA
    df[LABEL] = 0            # dummy; ignored for inference
    df["patient_id"] = "stream"

    X, _, meta = build_features(df)
    X = X.reindex(columns=columns, fill_value=-1.0)
    proba = model.predict_proba(X)[:, 1]

    alerts = [i for i, p in enumerate(proba) if p >= thr]
    first = alerts[0] if alerts else None
    return {
        "hours_scored": len(proba),
        "hourly_risk": [round(float(p), 4) for p in proba],
        "threshold": thr,
        "first_alert_hour": (int(meta.iloc[first]["ICULOS"]) if first is not None else None),
        "recommendation": (
            "Sepsis risk threshold crossed; initiate bundle assessment"
            if first is not None else "No alert; continue monitoring"),
    }
