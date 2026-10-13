"""
Build the static GitHub Pages demo into site/.

Each demo patient is scored through the real FastAPI /score endpoint (same
validation, features and model as the live service) and the response is
embedded next to the patient, so the static page shows exactly what the API
returns without needing a server.

Usage: python scripts/build_pages.py
"""
import json
import shutil
from pathlib import Path

from fastapi.testclient import TestClient

from app import STATIC_DIR, app

SITE = Path("site")


def main():
    client = TestClient(app)
    patients = json.loads((STATIC_DIR / "demo_patients.json").read_text())
    for p in patients:
        r = client.post("/score", json={"records": p["records"]})
        r.raise_for_status()
        p["result"] = r.json()

    if SITE.exists():
        shutil.rmtree(SITE)
    (SITE / "static").mkdir(parents=True)
    shutil.copy(STATIC_DIR / "index.html", SITE / "index.html")
    (SITE / "static" / "demo_patients.json").write_text(
        json.dumps(patients, separators=(",", ":")))
    (SITE / ".nojekyll").touch()
    for p in patients:
        print(f"  {p['id']}  {p['kind']:<26} first alert: {p['result']['first_alert_hour']}")
    print(f"Built {SITE}/ with {len(patients)} precomputed patients")


if __name__ == "__main__":
    main()
