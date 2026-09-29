"""Ingestion orchestration: download → parse → chunk → JSONL artifacts."""

from __future__ import annotations

import json
import time

from ..config import Settings, get_settings
from ..obs.logging import get_logger
from ..schemas import Chunk, Section
from . import ecfr, pdfs
from .chunker import chunk_section, count_tokens

log = get_logger(__name__)


def run_ingest(settings: Settings | None = None, force: bool = False) -> dict:
    settings = settings or get_settings()
    t0 = time.perf_counter()

    xml_path, issue_date = ecfr.download_title_xml(settings, force=force)
    sections: list[Section] = list(ecfr.parse_ecfr(xml_path, settings))
    log.info("ecfr_parsed", sections=len(sections))

    docs = pdfs.download_pdfs(settings, force=force)
    sections.extend(pdfs.parse_pdfs(settings, docs))
    log.info("pdfs_parsed", docs=len(docs), total_sections=len(sections))

    chunks: list[Chunk] = []
    for sec in sections:
        chunks.extend(chunk_section(sec, settings))

    sec_path = settings.processed_dir / "sections.jsonl"
    chunk_path = settings.processed_dir / "chunks.jsonl"
    with open(sec_path, "w") as f:
        for s in sections:
            f.write(s.model_dump_json() + "\n")
    with open(chunk_path, "w") as f:
        for c in chunks:
            f.write(c.model_dump_json() + "\n")

    by_part: dict[str, int] = {}
    for s in sections:
        by_part[s.part] = by_part.get(s.part, 0) + 1

    stats = {
        "issue_date": issue_date,
        "sections": len(sections),
        "chunks": len(chunks),
        "tokens": sum(count_tokens(c.text) for c in chunks),
        "sections_by_part": dict(sorted(by_part.items())),
        "duration_s": round(time.perf_counter() - t0, 1),
        "sections_file": str(sec_path),
        "chunks_file": str(chunk_path),
    }
    (settings.processed_dir / "ingest_stats.json").write_text(json.dumps(stats, indent=2))
    log.info("ingest_complete", **stats)
    return stats


if __name__ == "__main__":
    print(json.dumps(run_ingest(), indent=2))
