FROM python:3.12-slim AS base

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock* ./
RUN pip install --no-cache-dir uv && uv venv && uv pip install -p .venv \
    fastapi uvicorn sse-starlette pydantic pydantic-settings qdrant-client \
    fastembed bm25s PyStemmer huggingface-hub httpx lxml pypdf structlog PyYAML numpy tiktoken
ENV PATH="/app/.venv/bin:$PATH"

COPY backend backend
ENV PYTHONPATH=/app/backend

EXPOSE 8000
CMD ["python", "-m", "app.cli", "serve", "--host", "0.0.0.0", "--port", "8000"]
