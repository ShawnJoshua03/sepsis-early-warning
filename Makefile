PY ?= python
export PYTHONPATH := src

.PHONY: setup run test lint pages sample

setup:
	$(PY) -m pip install -r requirements-dev.txt

run:
	$(PY) -m uvicorn app:app --app-dir src --reload

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check .

pages:
	$(PY) scripts/build_pages.py

sample:
	$(PY) data/make_sample.py
