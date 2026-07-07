.PHONY: install models run smoke test clean

VENV := .venv
PY := $(VENV)/bin/python
PYTHONPATH := src

install:
	python3.11 -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements-mac.txt

models:
	$(PY) scripts/download_models.py

run:
	PYTHONPATH=$(PYTHONPATH) $(PY) -m uvicorn realtime_ai.server:app --host 0.0.0.0 --port 18080 --reload

smoke:
	$(PY) scripts/smoke_test.py

test:
	$(PY) scripts/test_client.py

clean:
	rm -rf $(VENV) src/**/__pycache__
