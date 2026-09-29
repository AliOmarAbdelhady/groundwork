"""eCFR (Electronic Code of Federal Regulations) ingestion.

Downloads the full Title 29 XML from the official eCFR versioner API and
stream-parses it into ``Section`` models for OSHA chapter XVII (parts
1900–2099), preserving the statutory hierarchy: part → subpart → subject
group → section / appendix.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

import httpx
from lxml import etree

from ..config import Settings
from ..obs.logging import get_logger
from ..schemas import Section

log = get_logger(__name__)

_SKIP_TAGS = {"HEAD", "EDNOTE", "CITA", "AUTH", "SOURCE", "FPNOTE", "PRTPAGE"}
_PARA_TAGS = {"P", "FP", "PSPACE"}
_FIRST_INT = re.compile(r"\d+")
_SEC_NO = re.compile(r"^§?\s*[\dA-Z]+[.\dA-Z]*\.?\s*")


def latest_issue_date(settings: Settings, client: httpx.Client) -> str:
    """Resolve the most recent issue date known to the eCFR API for the title."""
    r = client.get(settings.ecfr_titles_url)
    r.raise_for_status()
    for t in r.json()["titles"]:
        if t["number"] == settings.ecfr_title:
            return t["up_to_date_as_of"]
    raise RuntimeError(f"title {settings.ecfr_title} not found in eCFR titles.json")


def download_title_xml(settings: Settings, force: bool = False) -> tuple[Path, str]:
    """Download (or reuse cached) full title XML. Returns (path, issue_date)."""
    dest = settings.raw_dir / f"title{settings.ecfr_title}.xml"
    with httpx.Client(
        headers={
            "Accept-Encoding": "gzip",
            "User-Agent": settings.ecfr_user_agent,
        },
        timeout=settings.http_timeout,
        follow_redirects=True,
    ) as client:
        date = latest_issue_date(settings, client)
        if dest.exists() and dest.stat().st_size > 1_000_000 and not force:
            log.info("ecfr_xml_cached", path=str(dest))
            return dest, date
        url = settings.ecfr_full_url.format(date=date, title=settings.ecfr_title)
        log.info("ecfr_downloading", url=url, issue_date=date)
        with client.stream("GET", url) as resp:
            resp.raise_for_status()
            tmp = dest.with_suffix(".part")
            with open(tmp, "wb") as fh:
                for chunk in resp.iter_bytes(1 << 20):
                    fh.write(chunk)
            tmp.rename(dest)
    return dest, date


def _table_rows(table: etree._Element) -> list[str]:
    rows: list[str] = []
    for tr in table.iter("TR"):
        cells = []
        for cell in tr:
            tag = etree.QName(cell).localname
            if tag in {"TH", "TD"}:
                txt = " ".join("".join(cell.itertext()).split())
                if txt:
                    cells.append(txt)
        if cells:
            rows.append(" | ".join(cells))
    return rows


def _blocks(el: etree._Element) -> list[str]:
    """Extract ordered text blocks (paragraphs + table rows) from a container."""
    out: list[str] = []
    for child in el:
        tag = etree.QName(child).localname
        if tag in _SKIP_TAGS:
            continue
        # Appendices (DIV9) are emitted as their own sections — never fold them
        # into the enclosing section's text.
        if tag == "DIV9" and child.get("TYPE") == "APPENDIX":
            continue
        if tag in _PARA_TAGS:
            txt = " ".join("".join(child.itertext()).split())
            if txt:
                out.append(txt)
        elif tag == "GPOTABLE":
            out.extend(_table_rows(child))
        else:
            out.extend(_blocks(child))
    return out


def _in_part_range(n_attr: str, lo: int, hi: int) -> bool:
    m = _FIRST_INT.match(n_attr)
    return bool(m and lo <= int(m.group()) <= hi)


def parse_ecfr(
    xml_path: Path, settings: Settings
) -> Iterator[Section]:
    """Stream-parse the title XML, yielding Sections for OSHA parts only."""
    part = part_head = subpart = subjgrp = ""
    used_ids: set[str] = set()
    ctx = etree.iterparse(str(xml_path), events=("start", "end"), huge_tree=True)

    for event, el in ctx:
        tag = etree.QName(el).localname
        if event == "start":
            if tag == "DIV5" and el.get("TYPE") == "PART":
                n = el.get("N", "")
                if _in_part_range(n, settings.ecfr_parts_min, settings.ecfr_parts_max):
                    part = _FIRST_INT.search(n).group()  # type: ignore[union-attr]
                    part_head = (el.findtext("HEAD") or "").strip()
                    subpart = subjgrp = ""
                else:
                    part = ""  # outside OSHA chapter: skip subtree
            elif tag == "DIV6" and el.get("TYPE") == "SUBPART" and part:
                subpart = (el.findtext("HEAD") or el.get("N") or "").strip()
                subjgrp = ""
            elif tag == "DIV7" and el.get("TYPE") == "SUBJGRP" and part:
                subjgrp = (el.findtext("HEAD") or "").strip()
        else:  # end
            is_unit = tag in {"DIV8", "DIV9"} and part and el.get("TYPE") in {
                "SECTION", "APPENDIX"
            }
            if is_unit:
                sec = _section_from_div8(el, part, part_head, subpart, subjgrp, used_ids)
                if sec is not None:
                    yield sec
                el.clear()
                while el.getprevious() is not None:
                    del el.getparent()[0]
            elif tag == "DIV5":
                part = ""
            elif tag == "DIV6":
                subpart = ""
            elif tag == "DIV7":
                subjgrp = ""


def _section_from_div8(
    el: etree._Element,
    part: str,
    part_head: str,
    subpart: str,
    subjgrp: str,
    used_ids: set[str],
) -> Section | None:
    head = (el.findtext("HEAD") or "").strip()
    blocks = _blocks(el)
    text = "\n".join(blocks).strip()
    if len(text) < 120 or "[Reserved]" in head:
        return None

    typ = el.get("TYPE")
    n_attr = el.get("N", "")

    path_bits = [f"29 CFR {part}"]
    if subpart:
        path_bits.append(subpart)
    if subjgrp:
        path_bits.append(subjgrp)
    parent_path = " › ".join(path_bits)

    if typ == "SECTION":
        sec_no = n_attr  # e.g. "1910.132", "1926.501"
        section_id = f"29 CFR {sec_no}"
        heading = _SEC_NO.sub("", head).strip() or head
        url = f"https://www.ecfr.gov/current/title-29/section-{sec_no}"
    else:  # APPENDIX
        section_id = f"29 CFR {part} App {n_attr}"
        if section_id in used_ids and subpart:
            section_id = f"29 CFR {part} {subpart.split('—')[0].strip()} App {n_attr}"
        heading = head
        url = f"https://www.ecfr.gov/current/title-29/part-{part}"

    if section_id in used_ids:  # final dedup guard
        section_id = f"{section_id} ({parent_path})"
    used_ids.add(section_id)

    return Section(
        section_id=section_id,
        source="ecfr",
        part=part,
        parent_path=parent_path,
        heading=heading[:300],
        text=text,
        url=url,
    )
