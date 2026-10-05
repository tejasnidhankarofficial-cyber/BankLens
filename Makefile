PY=.venv/bin/python
.PHONY: setup test lint ingest api ui eval compare demo stop
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

# Demo: start the app in Docker, wait until it is ready, load the reranker, open the browser.
demo:
	docker compose up -d
	@until curl -sf localhost:8000/health >/dev/null; do sleep 2; done
	@curl -s -m 300 -X POST localhost:8000/ask -H 'content-type: application/json' -d '{"question":"What was Citigroup net income for 2025?"}' -o /dev/null
	@echo "BankLens is ready: http://localhost:8501"
	@open http://localhost:8501
stop:
	docker compose down
