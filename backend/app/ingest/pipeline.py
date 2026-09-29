"""Ingestion orchestration: download → parse → chunk → JSONL artifacts.

Every run stamps ``data/processed/manifest.json`` — an immutable record of
the exact corpus snapshot (source URLs + content hashes, eCFR issue date,
parser/chunker versions) that downstream index builds and every answer's
``corpus_version`` refer back to.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import UTC, datetime

from ..config import Settings, get_settings
from ..obs.logging import get_logger
from ..schemas import Chunk, Section
from . import ecfr, pdfs
from .chunker import CHUNKER_VERSION, chunk_section, count_tokens

log = get_logger(__name__)

PARSER_VERSION = 2


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _write_manifest(
    settings: Settings,
    issue_date: str,
    sections: list[Section],
    chunks: list[Chunk],
    docs: list[dict],
) -> str:
    h = hashlib.sha1()
    for s in sections:
        h.update(s.section_id.encode())
        h.update(s.text.encode())
    short = h.hexdigest()[:8]

    sources: list[dict] = [
        {
            "id": f"ecfr-title-{settings.ecfr_title}",
            "url": settings.ecfr_full_url.format(date=issue_date, title=settings.ecfr_title),
            "sha256": _sha256(settings.raw_dir / f"title{settings.ecfr_title}.xml"),
            "issue_date": issue_date,
        }
    ]
    for doc in docs:
        p = settings.raw_dir / "osha_pdfs" / f"{doc['slug']}.pdf"
        if p.exists():
            sources.append({"id": doc["slug"], "url": doc["url"], "sha256": _sha256(p)})

    manifest = {
        "corpus_version": f"ecfr-{issue_date}-{short}",
        "ecfr_issue_date": issue_date,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "parser_version": PARSER_VERSION,
        "chunker_version": CHUNKER_VERSION,
        "sections": len(sections),
        "chunks": len(chunks),
        "sources": sources,
    }
    (settings.processed_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest["corpus_version"]


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

    corpus_version = _write_manifest(settings, issue_date, sections, chunks, docs)

    by_part: dict[str, int] = {}
    for s in sections:
        by_part[s.part] = by_part.get(s.part, 0) + 1

    stats = {
        "issue_date": issue_date,
        "corpus_version": corpus_version,
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
