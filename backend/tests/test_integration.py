"""End-to-end retrieval + chat integration tests (require built indexes)."""

from __future__ import annotations

from app.schemas import RetrievalConfig


def test_hybrid_retrieval_finds_ppe_section(pipeline):
    result = pipeline.retrieve(
        "Who pays for personal protective equipment?",
        RetrievalConfig(rerank_enabled=False, final_top_k=5),
    )
    top5 = [h.section_id for h in result.hits[:5]]
    # Either the general-industry or the construction PPE-payment rule is a hit
    assert any(("1910.132" in s) or ("1926.95" in s) for s in top5), top5
    assert result.grounding_score > 0.4


def test_construction_fall_protection(pipeline):
    result = pipeline.retrieve(
        "At what height is fall protection required in construction?",
        RetrievalConfig(rerank_enabled=False, final_top_k=5),
    )
    top5 = [h.section_id for h in result.hits[:5]]
    # eCFR duty-to-have-fall-protection or the OSHA fall-protection booklet
    assert any(("1926.501" in s) or ("OSHA 3146" in s) for s in top5), top5


def test_parent_expansion_fills_parent_text(pipeline):
    result = pipeline.retrieve(
        "lockout tagout energy control program",
        RetrievalConfig(final_top_k=3),
    )
    assert result.hits
    assert any(h.parent_text for h in result.hits)


def test_context_respects_budget(pipeline, settings):
    result = pipeline.retrieve(
        "respirator medical evaluation requirements",
        RetrievalConfig(final_top_k=6),
    )
    from app.ingest.chunker import count_tokens

    assert count_tokens(result.context) <= settings.context_budget_tokens + 400


def test_ood_guard_fires_on_gibberish(pipeline, settings):
    result = pipeline.retrieve(
        "chicken soup recipe with noodles and celery for dinner tonight",
        RetrievalConfig(grounding_threshold=settings.grounding_threshold),
    )
    assert result.refused  # should be below threshold
    assert result.grounding_score < settings.grounding_threshold


def test_golden_hit5_floor(pipeline, settings):
    """CI regression gate: default hybrid config must clear the hit@5 floor."""
    from app.eval.harness import load_golden
    from app.eval.metrics import retrieval_metrics

    golden = [g for g in load_golden() if not g.adversarial]
    cfg = RetrievalConfig(dense_top_k=30, sparse_top_k=30, rerank_enabled=True,
                          final_top_k=10)
    hits = []
    for item in golden:
        result = pipeline.retrieve(item.question, cfg)
        m = retrieval_metrics([h.section_id for h in result.hits], item.expected_sections)
        hits.append(m["hit@5"])
    hit5 = sum(hits) / len(hits)
    assert hit5 >= settings.eval_hit5_floor, f"hit@5={hit5:.3f} < {settings.eval_hit5_floor}"
