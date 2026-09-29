"""Citation extraction + exact evidence validation.

Answers cite sources inline as ``[29 CFR 1910.132(b)]`` or
``[OSHA 3151 (pp. 4-6)]``. This module extracts every citation from a
generated answer and validates it against the evidence the model actually
saw: a citation is verified only when

1. its section was part of the final prompt context (not merely retrieved),
2. any cited subsection chain — e.g. ``(b)(1)(ii)`` — exists as markers in
   that section's statutory text, and
3. for OSHA publications, any cited page range overlaps the page groups
   that were included in the context.

Root-section presence alone is never treated as verification: a fabricated
``29 CFR 1910.132(z)`` fails the chain check and is reported as
hallucinated.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# 29 CFR 1910.132(b) | §1910.134 | 1910 subpart... | OSHA 3151 (pp. 4-6) | OSHA FS3529
_CFR = re.compile(r"\b29\s*(?:CFR|C\.F\.R\.)?\s*§?\s*(\d{4}\.\d+[a-z0-9]*(?:\([0-9a-zA-Z]+\))*)", re.I)
_BARE = re.compile(r"§\s*(\d{4}\.\d+[a-z0-9]*(?:\([0-9a-zA-Z]+\))*)")
_NUM = re.compile(r"^\s*(\d{4}\.\d+[a-z0-9]*(?:\([0-9a-zA-Z]+\))*)\s*$")
_OSHA_DOC = re.compile(
    r"\bOSHA[- ]([A-Z0-9\-]{2,20})\s*"
    r"(?:\((?:pp?\.|pages?)\s*(\d+)\s*(?:[-–]\s*(\d+))?\))?",
    re.I,
)
_SUBSEC = re.compile(r"\([0-9a-zA-Z]+\)$")
_MARKERS = re.compile(r"\(([0-9a-zA-Z]+)\)")


@dataclass
class CitationRef:
    """One citation occurrence, decomposed for exact validation."""

    raw: str                 # as written in the answer, e.g. "29 CFR 1910.132(b)(1)"
    root: str                # normalized section root, e.g. "29 CFR 1910.132"
    subsections: list[str] = field(default_factory=list)  # ["b", "1"]
    doc: str = ""            # OSHA publication number, e.g. "3151" ("" for CFR)
    pages: tuple[int, int] | None = None  # cited page range for OSHA docs


def _split_subsections(num: str) -> tuple[str, list[str]]:
    body = num
    subs: list[str] = []
    while (m := _SUBSEC.search(body)):
        marker = m.group(0)
        subs.insert(0, marker[1:-1])
        body = body[: -len(marker)]
    return body, subs


def _root(num: str) -> str:
    body, _ = _split_subsections(num)
    return f"29 CFR {body}"


def normalize_section_id(ref: str) -> str:
    """Root a citation at its statutory section: 1910.132(b)(1) -> 29 CFR 1910.132."""
    ref = ref.strip()
    m = _CFR.search(ref)
    if m:
        return _root(m.group(1))
    m = _BARE.search(ref)
    if m:
        return _root(m.group(1))
    m = _OSHA_DOC.search(ref)
    if m:
        return f"OSHA {m.group(1).upper()}"
    m = _NUM.match(ref)
    if m:
        return _root(m.group(1))
    return ref


def parse_citation_refs(text: str) -> list[CitationRef]:
    """Return every distinct citation in ``text`` decomposed into a CitationRef."""
    refs: list[CitationRef] = []
    seen: set[tuple] = set()

    def add(ref: CitationRef) -> None:
        key = (ref.root, tuple(ref.subsections), ref.pages)
        if key not in seen:
            seen.add(key)
            refs.append(ref)

    for m in _CFR.finditer(text):
        body, subs = _split_subsections(m.group(1))
        add(CitationRef(raw=m.group(0), root=f"29 CFR {body}", subsections=subs))
    for m in _BARE.finditer(text):
        if _CFR.search(m.group(0)):
            continue  # already captured with the 29 CFR prefix
        body, subs = _split_subsections(m.group(1))
        add(CitationRef(raw=m.group(0), root=f"29 CFR {body}", subsections=subs))
    for m in _OSHA_DOC.finditer(text):
        doc = m.group(1).upper()
        pages: tuple[int, int] | None = None
        if m.group(2):
            lo = int(m.group(2))
            hi = int(m.group(3)) if m.group(3) else lo
            pages = (lo, hi)
        add(CitationRef(raw=m.group(0), root=f"OSHA {doc}", doc=doc, pages=pages))
    return refs


def extract_citations(text: str) -> list[str]:
    """Return normalized section roots for every citation found in ``text``."""
    found: list[str] = []
    for ref in parse_citation_refs(text):
        if ref.root not in found:
            found.append(ref.root)
    return found


def subsection_chain_exists(section_text: str, markers: list[str]) -> bool:
    """True when the marker chain — e.g. ["b", "1", "ii"] — appears in order
    in ``section_text`` as ``(b) ... (1) ... (ii)``.

    The scan is monotonic (each marker must appear after the previous one),
    so ``(b)(1)`` only matches a ``(1)`` that sits inside paragraph ``(b)``.
    Matching is case-insensitive: eCFR uses ``(a)/(1)/(i)/(A)`` level styles.
    """
    pos = 0
    for marker in markers:
        pat = re.compile(r"\(\s*" + re.escape(marker) + r"\s*\)", re.I)
        m = pat.search(section_text, pos)
        if not m:
            return False
        pos = m.end()
    return True


def _page_range_of_section_id(section_id: str) -> tuple[int, int] | None:
    m = _OSHA_DOC.search(section_id)
    if not m or not m.group(2):
        return None
    lo = int(m.group(2))
    hi = int(m.group(3)) if m.group(3) else lo
    return (lo, hi)


def validate_citations(
    answer: str,
    context_sections: list[str],
    section_texts: dict[str, str] | None = None,
) -> dict:
    """Validate answer citations against the evidence in the final prompt context.

    ``context_sections`` are the section ids whose text was actually packed
    into the model's context; ``section_texts`` maps those ids to the text
    bodies the model saw. When ``section_texts`` is omitted, only
    section-level presence is checked (backwards-compatible fallback).
    """
    # root -> [included section ids]
    by_root: dict[str, list[str]] = {}
    for sid in context_sections:
        by_root.setdefault(normalize_section_id(sid), []).append(sid)

    detail: list[dict] = []
    valid_roots: list[str] = []
    bad_roots: list[str] = []
    cited_roots: list[str] = []

    for ref in parse_citation_refs(answer):
        if ref.root not in cited_roots:
            cited_roots.append(ref.root)
        included = by_root.get(ref.root, [])
        ok = bool(included)
        reason = ""
        if not ok:
            reason = "section_not_in_context"
        elif section_texts is not None and ref.subsections:
            text = "\n".join(section_texts.get(sid, "") for sid in included)
            if not subsection_chain_exists(text, ref.subsections):
                chain = "(" + ")(".join(ref.subsections) + ")"
                ok = False
                reason = f"subsection {chain} not found in cited section"
        elif section_texts is not None and ref.pages and ref.doc:
            overlap = any(
                (rng := _page_range_of_section_id(sid))
                and not (ref.pages[1] < rng[0] or ref.pages[0] > rng[1])
                for sid in included
            )
            if not overlap:
                ok = False
                reason = f"cited pages {ref.pages[0]}-{ref.pages[1]} not in context"

        if ok:
            if ref.root not in valid_roots:
                valid_roots.append(ref.root)
        elif ref.root not in bad_roots:
            bad_roots.append(ref.root)
        detail.append(
            {
                "citation": ref.raw,
                "root": ref.root,
                "subsections": ref.subsections,
                "valid": ok,
                **({"reason": reason} if reason else {}),
            }
        )

    return {
        "cited": cited_roots,
        "valid": valid_roots,
        "hallucinated": bad_roots,
        "n_cited": len(cited_roots),
        "n_valid": len(valid_roots),
        "n_hallucinated": len(bad_roots),
        "detail": detail,
    }
