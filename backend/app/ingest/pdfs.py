"""OSHA publication PDF ingestion.

Downloads the curated, URL-verified manifest (``pdf_manifest.yaml``) and
extracts text page-by-page with pypdf, grouping pages into parent Sections.
osha.gov rejects default client UA strings, so a browser-like UA is required.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

import httpx
import yaml
from pypdf import PdfReader

from ..config import Settings
from ..obs.logging import get_logger
from ..schemas import Section

log = get_logger(__name__)

_PAGES_PER_SECTION = 3
_NOISE = re.compile(r"^(?:\s*\d+\s*|OSHA\s*\d{3,4}[A-Za-z0-9\- ]*|www\.osha\.gov\s*"
                    r"|1-800-321-OSHA.*|Occupational Safety and Health Administration\s*)$",
                    re.IGNORECASE)


def _clean_page(txt: str) -> str:
    lines = []
    for ln in txt.splitlines():
        s = ln.strip()
        if not s or _NOISE.match(s):
            continue
        lines.append(s)
    return " ".join(lines)


def download_pdfs(settings: Settings, force: bool = False) -> list[dict]:
    """Fetch manifest PDFs into data/raw/osha_pdfs; returns manifest entries."""
    manifest_path = settings.pdf_manifest
    manifest = yaml.safe_load(manifest_path.read_text())["documents"]
    out_dir = settings.raw_dir / "osha_pdfs"
    out_dir.mkdir(parents=True, exist_ok=True)

    ok: list[dict] = []
    with httpx.Client(
        headers={"User-Agent": settings.ecfr_user_agent},
        timeout=settings.http_timeout,
        follow_redirects=True,
    ) as client:
        for doc in manifest:
            dest = out_dir / f"{doc['slug']}.pdf"
            if dest.exists() and dest.stat().st_size > 1000 and not force:
                ok.append(doc)
                continue
            try:
                r = client.get(doc["url"])
                r.raise_for_status()
                if not r.content.startswith(b"%PDF"):
                    log.warning("pdf_not_a_pdf", slug=doc["slug"])
                    continue
                dest.write_bytes(r.content)
                ok.append(doc)
                log.info("pdf_downloaded", slug=doc["slug"], kb=round(len(r.content) / 1024))
            except Exception as e:  # noqa: BLE001 — ingestion must skip-and-continue
                log.warning("pdf_failed", slug=doc["slug"], error=str(e))
    return ok


def parse_pdfs(settings: Settings, docs: list[dict]) -> Iterator[Section]:
    """Yield one Section per ~3-page group per document."""
    pdf_dir = settings.raw_dir / "osha_pdfs"
    for doc in docs:
        path = pdf_dir / f"{doc['slug']}.pdf"
        if not path.exists():
            continue
        try:
            reader = PdfReader(str(path))
            pages = [_clean_page(p.extract_text() or "") for p in reader.pages]
        except Exception as e:  # noqa: BLE001
            log.warning("pdf_parse_failed", slug=doc["slug"], error=str(e))
            continue

        pages = [p for p in pages if p]
        if not pages:
            log.warning("pdf_empty_text", slug=doc["slug"])
            continue

        url = doc["url"]
        doc_no = doc["slug"].removeprefix("OSHA")  # OSHA2254 -> 2254, FS3529 -> FS3529
        for i in range(0, len(pages), _PAGES_PER_SECTION):
            group = pages[i : i + _PAGES_PER_SECTION]
            first, last = i + 1, i + len(group)
            span = f"pp. {first}-{last}" if len(group) > 1 else f"p. {first}"
            yield Section(
                section_id=f"OSHA {doc_no} ({span})",
                source="osha_pdf",
                part=f"OSHA-{doc['kind'].upper()}",
                parent_path=f"OSHA Publications › {doc['title']}",
                heading=doc["title"],
                text="\n".join(group),
                url=url,
            )
