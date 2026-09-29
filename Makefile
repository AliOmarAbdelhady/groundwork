.PHONY: install ingest index eval eval-answers report test lint serve frontend dev download-llm clean

PY := .venv/bin/python
export PYTHONPATH := backend

install:
	uv sync
	cd frontend && npm install

ingest:
	$(PY) -m app.cli ingest

index:
	$(PY) -m app.cli index

eval:
	$(PY) -m app.cli eval

eval-answers:
	$(PY) -m app.cli eval --answers

report:
	$(PY) -m app.cli report

download-llm:
	$(PY) -m app.cli download-llm

serve:
	$(PY) -m app.cli serve

frontend:
	cd frontend && npm run build

dev:
	@echo "terminal 1: make serve   |   terminal 2: cd frontend && npm run dev"

# NOTE: integration tests open the embedded Qdrant directly — stop the API
# server first (embedded mode takes an exclusive storage lock).
test:
	$(PY) -m pytest backend/tests -q

lint:
	$(PY) -m ruff check backend

clean:
	rm -rf data/artifacts/qdrant data/artifacts/bm25 data/processed
