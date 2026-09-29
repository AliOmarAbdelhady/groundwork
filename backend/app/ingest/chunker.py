"""Hierarchical (parent-child) chunking.

Parents are whole ``Section`` texts (what the LLM sees and the UI shows).
Children are overlapping sentence-window passages ~``child_target_tokens``
tokens long — the units that get embedded and searched. The trailing overlap
between consecutive children is budgeted in tokens with the real tokenizer
(``child_overlap_tokens``), not fixed to a sentence count. Small-to-big
retrieval maps child hits back to parents at query time.
"""

from __future__ import annotations

import re

import tiktoken

from ..config import Settings
from ..schemas import Chunk, Section

_SENT_SPLIT = re.compile(r"(?<=[.;:!?])\s+(?=[A-Z(§])")

_encoder = tiktoken.get_encoding("cl100k_base")

CHUNKER_VERSION = 2


def count_tokens(text: str) -> int:
    return len(_encoder.encode(text, disallowed_special=()))


def sentences(text: str) -> list[str]:
    sents = [s.strip() for s in _SENT_SPLIT.split(text) if s and s.strip()]
    return sents or ([text.strip()] if text.strip() else [])


def _overlap_tail(emitted: list[str], overlap_budget: int) -> list[str]:
    """Trailing sentences of the previous chunk covering ``overlap_budget``
    tokens — as close to the configured overlap as sentence boundaries allow.
    """
    tail: list[str] = []
    used = 0
    for sent in reversed(emitted):
        n = count_tokens(sent)
        if used + n > overlap_budget and tail:
            break
        tail.insert(0, sent)
        used += n
        if used >= overlap_budget:
            break
    return tail


def chunk_section(section: Section, settings: Settings) -> list[Chunk]:
    """Split one section into overlapping child chunks (>=1 chunk per section)."""
    target = settings.child_target_tokens
    overlap = max(0, settings.child_overlap_tokens)
    header = f"{section.section_id} — {section.heading}"

    def make(idx: int, body: str) -> Chunk:
        return Chunk(
            chunk_id=f"{section.section_id}#{idx}",
            section_id=section.section_id,
            chunk_idx=idx,
            heading=section.heading,
            text=f"{header}\n{body}",
            source=section.source,
            part=section.part,
            parent_path=section.parent_path,
            url=section.url,
        )

    # Short sections (incl. quickcards/posters) become a single child.
    if count_tokens(section.text) <= target + settings.child_min_tokens:
        return [make(0, section.text)]

    sents = sentences(section.text)
    chunks: list[Chunk] = []
    buf: list[str] = []
    buf_tokens = 0
    emitted: list[str] = []  # sentences of the previous chunk, for overlap

    def flush() -> None:
        nonlocal buf, buf_tokens, emitted
        if not buf:
            return
        window = _overlap_tail(emitted, overlap) + buf if emitted else buf
        chunks.append(make(len(chunks), " ".join(window)))
        emitted = list(buf)
        buf, buf_tokens = [], 0

    for sent in sents:
        n = count_tokens(sent)
        if n > target * 2:  # pathological single sentence — hard split on commas
            for part in re.split(r"(?<=[,;])\s+", sent):
                pn = count_tokens(part)
                if buf and buf_tokens + pn > target:
                    flush()
                buf.append(part)
                buf_tokens += pn
            continue
        if buf and buf_tokens + n > target:
            flush()
        buf.append(sent)
        buf_tokens += n
    flush()
    return chunks
