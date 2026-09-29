"""Deterministic eval metrics (no LLM judge in the critical path)."""

from __future__ import annotations

import math

from ..rag.citations import normalize_section_id


def root_sections(ids: list[str]) -> list[str]:
    """Normalize + dedupe, preserving order."""
    seen: set[str] = set()
    out: list[str] = []
    for i in ids:
        r = normalize_section_id(i)
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def retrieval_metrics(
    ranked: list[str], expected: list[str], k: int = 10
) -> dict[str, float]:
    """Section-level hit@{1,3,5,10}, MRR@10, nDCG@10 for one query."""
    ranked = root_sections(ranked)[:k]
    expected = set(root_sections(expected))

    first_rel: int | None = None
    dcg = 0.0
    for rank, sec in enumerate(ranked, start=1):
        if sec in expected:
            first_rel = first_rel or rank
            dcg += 1.0 / math.log2(rank + 1)

    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, min(len(expected), k) + 1))
    return {
        **{f"hit@{kk}": float(any(sec in expected for sec in ranked[:kk]))
           for kk in (1, 3, 5, 10)},
        "mrr@10": (1.0 / first_rel) if first_rel else 0.0,
        "ndcg@10": (dcg / idcg) if idcg > 0 else 0.0,
    }


def aggregate(per_query: list[dict[str, float]]) -> dict[str, float]:
    if not per_query:
        return {}
    keys = per_query[0].keys()
    return {k: round(sum(p[k] for p in per_query) / len(per_query), 4) for k in keys}


def _cos(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb + 1e-9)


def answer_metrics(
    cited_info: dict,
    answer_vec: list[float],
    context_vec: list[float],
    question_vec: list[float],
) -> dict[str, float]:
    """Citation quality + embedding-based groundedness/relevance."""
    precision = (
        cited_info["n_valid"] / cited_info["n_cited"] if cited_info["n_cited"] else 0.0
    )
    return {
        "citation_precision": round(precision, 4),
        "citation_rate": 1.0 if cited_info["n_cited"] else 0.0,
        "hallucinated_citations": float(cited_info["n_hallucinated"]),
        "groundedness": round(_cos(answer_vec, context_vec), 4),
        "relevance": round(_cos(answer_vec, question_vec), 4),
    }
