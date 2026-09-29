"""Domain + API schemas (pydantic v2)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# --------------------------------------------------------------------- domain
class Section(BaseModel):
    """One statutory section (eCFR) or one PDF document block (OSHA pubs)."""

    section_id: str  # "29 CFR 1910.132" | "OSHA-DOC-3120"
    source: Literal["ecfr", "osha_pdf"]
    part: str  # "1910" | "OSHA-PUB"
    parent_path: str  # "29 CFR 1910 › Subpart I" | doc series name
    heading: str
    text: str
    url: str = ""


class Chunk(BaseModel):
    """Searchable child window; expands to its parent Section for generation."""

    chunk_id: str  # "<section_id>#<idx>"
    section_id: str
    chunk_idx: int
    heading: str
    text: str
    source: Literal["ecfr", "osha_pdf"]
    part: str
    parent_path: str
    url: str = ""


# ------------------------------------------------------------------ retrieval
class RetrievalConfig(BaseModel):
    """Per-request retrieval knobs (all optional; server defaults apply)."""

    dense_top_k: int | None = None
    sparse_top_k: int | None = None
    rrf_k: int | None = None
    rerank_enabled: bool | None = None
    final_top_k: int | None = None
    grounding_threshold: float | None = None


class RetrievedHit(BaseModel):
    chunk_id: str
    section_id: str
    heading: str
    source: str
    part: str
    parent_path: str
    url: str = ""
    text: str  # child text (short excerpt)
    parent_text: str = ""  # set when parent expansion is applied
    dense_score: float | None = None
    sparse_score: float | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None
    final_score: float | None = None
    rank: int = 0


class StageTiming(BaseModel):
    stage: str
    ms: float


# ------------------------------------------------------------------ chat API
class ChatRequest(BaseModel):
    # Short greetings ("hi") pass validation; the grounding guard refuses
    # them gracefully instead of surfacing a hard 422 to the user.
    query: str = Field(min_length=2, max_length=1000)
    history: list[dict[str, str]] = Field(default_factory=list)  # [{role, content}]
    config: RetrievalConfig = Field(default_factory=RetrievalConfig)


class SourceRef(BaseModel):
    """A validated citation the answer actually relies on."""

    section_id: str
    heading: str
    source: str
    url: str
    excerpt: str
    score: float


class TraceInfo(BaseModel):
    query_id: str
    grounding_score: float
    refused: bool
    refusal_reason: str | None = None
    timings_ms: list[StageTiming] = Field(default_factory=list)
    retrieval: list[RetrievedHit] = Field(default_factory=list)


class SearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=1000)
    top_k: int = 10
    rerank_enabled: bool = False


# ---------------------------------------------------------------- eval models
class GoldenItem(BaseModel):
    id: str
    question: str
    expected_sections: list[str]  # ground-truth section ids, e.g. "29 CFR 1910.132(b)"
    source_type: Literal["ecfr", "osha_pdf"] = "ecfr"
    adversarial: bool = False  # out-of-domain / must-refuse


class EvalHit(BaseModel):
    config_name: str
    retrieval: dict[str, float]  # hit@1..hit@10, mrr@10, ndcg@10
    answers: dict[str, float] | None = None  # citation precision/recall, groundedness...
    refusal_rate_adversarial: float | None = None
    answered_rate_golden: float | None = None
    n_items: int = 0
    duration_s: float = 0.0
    per_item: list[dict[str, Any]] = Field(default_factory=list)


class EvalReport(BaseModel):
    created_at: str
    runs: list[EvalHit]


# ----------------------------------------------------------------- trace store
class TraceRecord(BaseModel):
    query_id: str
    ts: str
    query: str
    refused: bool
    answer: str
    latency_ms: float
    first_token_ms: float | None = None
    grounding_score: float
    cited_sections: list[str] = Field(default_factory=list)
    retrieval: list[dict[str, Any]] = Field(default_factory=list)
    timings: list[dict[str, Any]] = Field(default_factory=list)


class StatsSummary(BaseModel):
    total_traces: int
    refusal_rate: float
    p50_latency_ms: float
    p95_latency_ms: float
    avg_first_token_ms: float | None = None
    top_cited: list[dict[str, Any]] = Field(default_factory=list)
