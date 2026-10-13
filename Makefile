PY ?= python
export PYTHONPATH := src

.PHONY: setup run test lint pages sample

setup:            ## install pinned runtime + dev dependencies
	$(PY) -m pip install -r requirements-dev.txt

run:              ## demo + API at http://localhost:8000
	$(PY) -m uvicorn app:app --app-dir src --reload

test:             ## causality, metric and API tests
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check .

pages:            ## build the static GitHub Pages demo into site/
	$(PY) scripts/build_pages.py

sample:           ## synthetic patients in the real schema (no download needed)
	$(PY) data/make_sample.py
