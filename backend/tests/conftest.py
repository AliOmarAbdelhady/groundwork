"""Integration tests.

Run against the real indexes when present (marked `integration`), so CI can
run unit-only quickly and full integration locally after `make ingest index`.

Requires: data/processed/*.jsonl + data/artifacts (built by the pipeline).
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.config import Settings  # noqa: E402


def _indexes_ready() -> bool:
    s = Settings()
    return (
        (s.processed_dir / "chunks.jsonl").exists()
        and (s.artifacts_dir / "bm25" / "vocab.index.json").exists()
    )


pytestmark = pytest.mark.skipif(
    not _indexes_ready(), reason="indexes not built — run `make ingest index`"
)


@pytest.fixture(scope="session")
def pipeline():
    from app.rag.retrieve import RetrievalPipeline

    return RetrievalPipeline(Settings())


@pytest.fixture(scope="session")
def settings():
    return Settings()
