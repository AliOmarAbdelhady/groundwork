"""Citation extraction + validation.

Answers cite sources inline as ``[29 CFR 1910.132(b)]`` or
``[OSHA 3151 (pp. 4-6)]``. This module extracts every citation from a
generated answer and validates it against the sections actually retrieved —
the basis for the citation precision/recall eval metrics and the UI's
"verified citation" badges.
"""

from __future__ import annotations

import re

# 29 CFR 1910.132(b) | §1910.134 | 1910 subpart... | OSHA 3151 (pp. 4-6) | OSHA FS3529
_CFR = re.compile(r"\b29\s*(?:CFR|C\.F\.R\.)?\s*§?\s*(\d{4}\.\d+[a-z0-9]*(?:\([0-9a-zA-Z]+\))*)", re.I)
_BARE = re.compile(r"§\s*(\d{4}\.\d+[a-z0-9]*(?:\([0-9a-zA-Z]+\))*)")
_NUM = re.compile(r"^\s*(\d{4}\.\d+[a-z0-9]*(?:\([0-9a-zA-Z]+\))*)\s*$")
_OSHA_DOC = re.compile(r"\bOSHA[- ]([A-Z0-9\-]{2,20})\s*(?:\((?:pp?\.|pages?)\s*[\d–-]+\))?", re.I)
_SUBSEC = re.compile(r"\([0-9a-zA-Z]+\)$")


def _root(num: str) -> str:
    while _SUBSEC.search(num):
        num = _SUBSEC.sub("", num)
    return f"29 CFR {num}"


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


def extract_citations(text: str) -> list[str]:
    """Return normalized section roots for every citation found in ``text``."""
    found: list[str] = []
    for rx in (_CFR, _BARE, _OSHA_DOC):
        for m in rx.finditer(text):
            norm = normalize_section_id(m.group(0))
            if norm not in found:
                found.append(norm)
    return found


def validate_citations(
    answer: str, retrieved_section_ids: list[str]
) -> dict:
    """Check answer citations against the retrieved context.

    Returns counts + the offending ids — used by eval and the UI badges.
    """
    retrieved_roots = [normalize_section_id(s) for s in retrieved_section_ids]
    cited = extract_citations(answer)
    valid = [c for c in cited if c in retrieved_roots]
    hallucinated = [c for c in cited if c not in retrieved_roots]
    return {
        "cited": cited,
        "valid": valid,
        "hallucinated": hallucinated,
        "n_cited": len(cited),
        "n_valid": len(valid),
        "n_hallucinated": len(hallucinated),
    }
