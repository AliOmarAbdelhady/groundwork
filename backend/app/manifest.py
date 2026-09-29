"""Corpus + index version manifests.

Every index build stamps a manifest recording exactly which sources (with
content hashes) went into the corpus, which parser/chunker/embedding
versions shaped it, and when it was built. The corpus version travels with
every answer (SSE meta event, /api/config) so any response can be traced
back to the exact evidence snapshot it was produced from.
"""

from __future__ import annotations

import json

from .config import Settings, get_settings


def load_corpus_manifest(settings: Settings | None = None) -> dict:
    settings = settings or get_settings()
    path = settings.processed_dir / "manifest.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def load_index_manifest(settings: Settings | None = None) -> dict:
    settings = settings or get_settings()
    path = settings.artifacts_dir / "manifest.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def corpus_version(settings: Settings | None = None) -> str:
    return load_index_manifest(settings).get("corpus_version") or "unknown"
