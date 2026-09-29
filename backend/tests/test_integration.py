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


def test_exact_citation_leg_resolves_section(pipeline):
    """A query naming a regulation must surface that exact section on top."""
    result = pipeline.retrieve(
        "What does 29 CFR 1910.132 say about payment for protective equipment?",
        RetrievalConfig(rerank_enabled=False, final_top_k=5),
    )
    assert result.exact_citation
    assert "29 CFR 1910.132" in [h.section_id for h in result.hits[:3]]


def test_sparse_only_hits_keep_metadata(pipeline):
    """Lexical-only retrieval must still produce fully hydrated hits."""
    result = pipeline.retrieve(
        "permissible exposure limit benzene",
        RetrievalConfig(dense_top_k=0, sparse_top_k=10, rerank_enabled=False, final_top_k=5),
    )
    assert result.hits
    for h in result.hits:
        assert h.heading, f"hit {h.chunk_id} has no heading"
        assert h.text, f"hit {h.chunk_id} has no text"
        assert h.source, f"hit {h.chunk_id} has no source"


def test_golden_regression_gates(pipeline, settings):
    """CI regression gate: the default hybrid+rerank config must clear ALL
    floors simultaneously — hit@1, hit@5, nDCG, adversarial refusal, and
    golden answered rate (a single hit@5 floor can hide large regressions)."""
    from app.eval.harness import load_golden
    from app.eval.metrics import retrieval_metrics

    golden = load_golden()
    cfg = RetrievalConfig(dense_top_k=30, sparse_top_k=30, rerank_enabled=True,
                          final_top_k=10)
    per_query, refusals, answered = [], [], []
    for item in golden:
        result = pipeline.retrieve(item.question, cfg)
        if item.adversarial:
            refusals.append(result.refused)
        else:
            per_query.append(
                retrieval_metrics([h.section_id for h in result.hits], item.expected_sections)
            )
            answered.append(not result.refused)

    hit1 = sum(m["hit@1"] for m in per_query) / len(per_query)
    hit5 = sum(m["hit@5"] for m in per_query) / len(per_query)
    ndcg = sum(m["ndcg@10"] for m in per_query) / len(per_query)
    refusal = sum(refusals) / len(refusals)
    answered_rate = sum(answered) / len(answered)

    assert hit1 >= settings.eval_hit1_floor, f"hit@1={hit1:.3f}"
    assert hit5 >= settings.eval_hit5_floor, f"hit@5={hit5:.3f}"
    assert ndcg >= settings.eval_ndcg_floor, f"ndcg@10={ndcg:.3f}"
    assert refusal >= settings.eval_adversarial_refusal_floor, f"adversarial refusal={refusal:.3f}"
    assert answered_rate >= settings.eval_answered_floor, f"answered={answered_rate:.3f}"
