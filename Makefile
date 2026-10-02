PY=.venv/bin/python
.PHONY: setup test lint ingest api ui eval compare
setup:
	python3.11 -m venv .venv && $(PY) -m pip install -r requirements.txt -r requirements-local.txt
test:
	$(PY) -m pytest -q
lint:
	.venv/bin/ruff check . && .venv/bin/ruff format --check .
ingest:
	$(PY) -m scripts.ingest --config $(CONFIG)
api:
	BANKLENS_CONFIG=$(CONFIG) .venv/bin/uvicorn app.main:app --port 8000
ui:
	.venv/bin/streamlit run ui/streamlit_app.py
eval:
	$(PY) -m eval.run_eval --config $(CONFIG)
compare:
	$(PY) -m eval.compare
