"""API contract: scores valid streams, rejects malformed ones, serves the demo."""
import json

import pytest
from fastapi.testclient import TestClient

from app import STATIC_DIR, app

client = TestClient(app)


@pytest.fixture(scope="module")
def demo_patient():
    return json.loads((STATIC_DIR / "demo_patients.json").read_text())[0]


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_score_demo_patient(demo_patient):
    r = client.post("/score", json={"records": demo_patient["records"]})
    assert r.status_code == 200
    body = r.json()
    assert body["hours_scored"] == len(demo_patient["records"])
    assert all(0.0 <= p <= 1.0 for p in body["hourly_risk"])


def test_score_is_order_independent(demo_patient):
    recs = demo_patient["records"]
    a = client.post("/score", json={"records": recs}).json()
    b = client.post("/score", json={"records": recs[::-1]}).json()
    assert a["hourly_risk"] == b["hourly_risk"]


@pytest.mark.parametrize("records, why", [
    ([], "empty"),
    ([{"HR": 90}], "missing ICULOS"),
    ([{"ICULOS": 1, "hr": 90}], "unknown field"),
    ([{"ICULOS": 1}, {"ICULOS": 1}], "duplicate hour"),
    ([{"ICULOS": 0}], "ICULOS < 1"),
    ([{"ICULOS": 1, "HR": "fast"}], "non-numeric"),
])
def test_rejects_bad_input(records, why):
    assert client.post("/score", json={"records": records}).status_code == 422, why


def test_demo_page_served():
    r = client.get("/")
    assert r.status_code == 200 and "Sepsis Early Warning" in r.text
    assert client.get("/static/demo_patients.json").status_code == 200
