"""Unit tests: chunker, citations, metrics — fast, no model/network deps."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.ingest.chunker import chunk_section, count_tokens  # noqa: E402
from app.rag.citations import (  # noqa: E402
    extract_citations,
    normalize_section_id,
    validate_citations,
)
from app.schemas import Section  # noqa: E402


class _MiniSettings:
    child_target_tokens = 40
    child_overlap_tokens = 10
    child_min_tokens = 8
    context_budget_tokens = 1000


def make_section(text: str, sid: str = "29 CFR 1910.999") -> Section:
    return Section(
        section_id=sid,
        source="ecfr",
        part="1910",
        parent_path="29 CFR 1910 › Subpart Z",
        heading="Test section",
        text=text,
        url="https://example.com",
    )


class TestChunker:
    def test_short_section_single_chunk(self):
        sec = make_section("Wear gloves. " * 5)
        chunks = chunk_section(sec, _MiniSettings())  # type: ignore[arg-type]
        assert len(chunks) == 1
        assert chunks[0].chunk_id == f"{sec.section_id}#0"

    def test_long_section_split_with_metadata(self):
        sentences = [f"Sentence number {i} about protective equipment requirements." for i in range(60)]
        sec = make_section(" ".join(sentences))
        chunks = chunk_section(sec, _MiniSettings())  # type: ignore[arg-type]
        assert len(chunks) > 1
        assert all(c.section_id == sec.section_id for c in chunks)
        assert all(c.part == "1910" for c in chunks)
        # chunk ids form an ordered sequence
        idx = [c.chunk_idx for c in chunks]
        assert idx == sorted(idx)
        # every chunk carries the citation header
        assert all(c.text.startswith(sec.section_id) for c in chunks)

    def test_token_count_matches_tiktoken(self):
        assert count_tokens("hello world") == 2


class TestCitations:
    def test_normalize_roots_subsections(self):
        assert normalize_section_id("29 CFR 1910.132(b)(1)") == "29 CFR 1910.132"
        assert normalize_section_id("§ 1910.134(d)(2)(ii)") == "29 CFR 1910.134"
        assert normalize_section_id("1910.1000") == "29 CFR 1910.1000"

    def test_normalize_osha_docs(self):
        assert normalize_section_id("OSHA 3151 (pp. 4-6)") == "OSHA 3151"
        assert normalize_section_id("OSHA FS-INSPECTIONS (p. 1)") == "OSHA FS-INSPECTIONS"

    def test_extract_from_markdown_answer(self):
        answer = (
            "Employers must pay for PPE [29 CFR 1910.132(b)]. Training is also "
            "required per § 1910.132(f). See also [OSHA 3151 (pp. 4-6)]."
        )
        assert extract_citations(answer) == [
            "29 CFR 1910.132",
            "OSHA 3151",
        ]

    def test_validate_flags_hallucinated(self):
        info = validate_citations(
            "Per [29 CFR 1910.999] and [29 CFR 1910.132(b)] ...",
            ["29 CFR 1910.132"],
        )
        assert info["n_valid"] == 1
        assert info["hallucinated"] == ["29 CFR 1910.999"]


class TestMetrics:
    def test_perfect_ranking(self):
        from app.eval.metrics import retrieval_metrics

        m = retrieval_metrics(
            ["29 CFR 1910.132", "29 CFR 1910.133"], ["1910.132(b)"]
        )
        assert m["hit@1"] == 1.0 and m["mrr@10"] == 1.0 and m["ndcg@10"] == 1.0

    def test_partial_ranking(self):
        from app.eval.metrics import retrieval_metrics

        m = retrieval_metrics(
            ["29 CFR 1910.1", "29 CFR 1926.501", "29 CFR 1910.132"],
            ["29 CFR 1910.132"],
        )
        assert m["hit@1"] == 0.0
        assert m["hit@3"] == 1.0
        assert abs(m["mrr@10"] - 1 / 3) < 1e-9
        assert 0 < m["ndcg@10"] < 1

    def test_miss(self):
        from app.eval.metrics import retrieval_metrics

        m = retrieval_metrics(["29 CFR 1910.1"], ["29 CFR 1926.501"])
        assert all(m[k] == 0.0 for k in ("hit@1", "hit@5", "mrr@10", "ndcg@10"))
