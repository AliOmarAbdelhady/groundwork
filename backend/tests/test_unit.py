"""Unit tests: chunker, citations, metrics, context packing — fast, no model/network deps."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.ingest.chunker import _overlap_tail, chunk_section, count_tokens, sentences  # noqa: E402
from app.rag.citations import (  # noqa: E402
    extract_citations,
    normalize_section_id,
    parse_citation_refs,
    subsection_chain_exists,
    validate_citations,
)
from app.rag.retrieve import _pack_body  # noqa: E402
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

    def test_parse_refs_keep_subsection_chain(self):
        refs = parse_citation_refs("see [29 CFR 1910.132(b)(1)] and [OSHA 3151 (pp. 4-6)]")
        by_root = {r.root: r for r in refs}
        assert by_root["29 CFR 1910.132"].subsections == ["b", "1"]
        assert by_root["OSHA 3151"].pages == (4, 6)

    def test_validate_flags_hallucinated(self):
        info = validate_citations(
            "Per [29 CFR 1910.999] and [29 CFR 1910.132(b)] ...",
            ["29 CFR 1910.132"],
        )
        assert info["n_valid"] == 1
        assert info["hallucinated"] == ["29 CFR 1910.999"]

    def test_validate_checks_subsection_exists_in_source(self):
        """A fabricated subsection on a retrieved section root is invalid."""
        text = "(a) General. Protective equipment is required. (b) Selection. " \
               "Employers shall select PPE. (h) Payment. Employers must pay for PPE."
        ok = validate_citations(
            "Employers pay [29 CFR 1910.132(h)].", ["29 CFR 1910.132"],
            {"29 CFR 1910.132": text},
        )
        bad = validate_citations(
            "Employers pay [29 CFR 1910.132(z)].", ["29 CFR 1910.132"],
            {"29 CFR 1910.132": text},
        )
        assert ok["n_valid"] == 1 and ok["n_hallucinated"] == 0
        assert bad["n_hallucinated"] == 1
        assert bad["detail"][0]["valid"] is False

    def test_validate_only_against_prompt_context(self):
        """Retrieved-but-not-packed sections must not count as context."""
        info = validate_citations(
            "Per [29 CFR 1910.134(d)].", ["29 CFR 1910.134"],
            {"29 CFR 1910.146": "(d) something else"},
        )
        assert info["n_hallucinated"] == 1

    def test_validate_osha_pages_overlap(self):
        texts = {"OSHA 3151 (pp. 5-7)": "booklet text page five"}
        ok = validate_citations("See [OSHA 3151 (p. 6)].", ["OSHA 3151 (pp. 5-7)"], texts)
        bad = validate_citations("See [OSHA 3151 (p. 30)].", ["OSHA 3151 (pp. 5-7)"], texts)
        assert ok["n_valid"] == 1
        assert bad["n_hallucinated"] == 1

    def test_subsection_chain_monotonic(self):
        text = "(b) paragraph text with item (1) inside it. (c) later paragraph (1) again."
        assert subsection_chain_exists(text, ["b", "1"])
        # (1) under (c) must not validate a (b)(1) claim
        assert not subsection_chain_exists("(c) something (1) inside.", ["b", "1"])


class TestOverlap:
    def test_overlap_tail_meets_token_budget(self):
        sents = ["First sentence here.", "Second one follows.", "Third keeps going.",
                 "Fourth adds more.", "Fifth and final."]
        tail = _overlap_tail(sents, overlap_budget=10)
        assert 0 < count_tokens(" ".join(tail)) <= 30  # near budget, sentence-bounded
        assert sents[-1] in tail

    def test_chunks_carry_overlap(self):
        sentences_txt = " ".join(
            f"Sentence number {i} about protective equipment requirements." for i in range(60)
        )
        sec = make_section(sentences_txt)
        chunks = chunk_section(sec, _MiniSettings())  # type: ignore[arg-type]
        assert len(chunks) > 2
        # compare bodies (chunk text starts with a header line that would
        # merge into the first sentence and defeat exact matching)
        bodies = [c.text.split("\n", 1)[1] for c in chunks]
        shared = 0
        for a, b in zip(bodies, bodies[1:], strict=False):
            shared += len(set(sentences(a)) & set(sentences(b)))
        assert shared > 0


class TestContextPacking:
    def test_never_cuts_mid_sentence(self):
        body = "\n".join(
            f"Paragraph {i} with a complete sentence about exposure limits and rules." for i in range(40)
        )
        packed, truncated = _pack_body(body, budget=60)
        assert truncated
        assert count_tokens(packed) <= 60
        # every emitted line is a whole sentence or paragraph, never a fragment
        for line in packed.split("\n"):
            if line.strip() and not line.endswith("[...]"):
                assert line.rstrip().endswith((".", "!", "?", ":")), line[-40:]

    def test_small_body_untouched(self):
        body = "One short paragraph."
        packed, truncated = _pack_body(body, budget=100)
        assert packed == body and not truncated


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
