"""Chat orchestration: retrieval → guard → streaming generation → trace.

Single entry point used by the API layer, the eval harness, and the CLI so
every consumer gets identical behavior and observability.
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterator

from ..config import Settings, get_settings
from ..manifest import load_index_manifest
from ..obs.logging import get_logger
from ..obs.traces import TraceStore
from ..schemas import (
    ChatRequest,
    RetrievedHit,
    SourceRef,
)
from . import citations as cite
from .generate import REFUSAL_ANSWER, build_messages, get_llm
from .retrieve import RetrievalPipeline, RetrievalResult

log = get_logger(__name__)

# Greetings / small talk / meta questions about the assistant itself. These
# bypass the grounding-refusal (they carry no regulatory content to ground)
# and are answered by the LLM under a strict conversation-only instruction.
_GREETING = re.compile(
    r"^(hi|hey|hello|yo|sup|howdy|thanks|thank you|thx|ty|ok|okay|cool|nice|great|"
    r"good (morning|afternoon|evening|day))"
    r"(\s+(there|everyone|everybody|all|friend|friends|team|guys|sir|ma'am|am|man))?"
    r"[\s!.,?]*$",
    re.I,
)
_META = re.compile(
    r"\b(who are you|what are you|what(?:'s| is) your name|what can you do|"
    r"how do you work|what do you know|can you help|help me understand you|"
    r"what is groundwork)\b",
    re.I,
)


def is_conversational(query: str) -> bool:
    q = query.strip()
    return bool(_GREETING.match(q) or _META.search(q))


class ChatOrchestrator:
    def __init__(
        self,
        settings: Settings | None = None,
        pipeline: RetrievalPipeline | None = None,
        traces: TraceStore | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.pipeline = pipeline or RetrievalPipeline(self.settings)
        self.traces = traces or TraceStore(self.settings.traces_db)
        self.corpus_version = (
            load_index_manifest(self.settings).get("corpus_version") or "unknown"
        )

    # ------------------------------------------------------------------ chat
    def stream_chat(
        self, req: ChatRequest
    ) -> Iterator[dict]:
        """Yield SSE-serializable event dicts:
        sources -> token* -> done | refused | error.
        """
        t0 = time.perf_counter()
        query_id = self.traces.new_query_id()
        try:
            result = self.pipeline.retrieve(req.query, req.config)
        except Exception as e:  # noqa: BLE001 — surface as SSE error, keep serving
            log.error("retrieval_failed", error=str(e))
            yield {"type": "error", "detail": f"retrieval failed: {e}"}
            return

        yield {
            "type": "meta",
            "query_id": query_id,
            "grounding_score": result.grounding_score,
            "lexical_overlap": result.lexical_overlap,
            "exact_citation": result.exact_citation,
            "rerank_agreement": result.rerank_agreement,
            "corpus_version": self.corpus_version,
            "timings": result.timings_ms,
        }
        yield {"type": "sources", "sources": [self._source(h) for h in result.hits]}

        conversational = is_conversational(req.query)
        if result.refused and not conversational:
            if result.lexical_overlap < self.settings.ood_min_overlap:
                reason = "out_of_domain"
            elif (
                result.rerank_agreement is not None
                and result.rerank_agreement < self.settings.guard_rerank_floor
            ):
                reason = "weak_evidence"
            else:
                reason = "low_grounding"
            yield {"type": "refused", "reason": reason,
                   "grounding_score": result.grounding_score,
                   "lexical_overlap": result.lexical_overlap}
            self._record(query_id, req, result, REFUSAL_ANSWER,
                         (time.perf_counter() - t0) * 1000, None)
            return

        messages = build_messages(
            req.query,
            context="" if conversational else result.context,
            history=req.history,
            conversational=conversational,
        )
        settings = self.settings
        llm = get_llm()

        answer_parts: list[str] = []
        first_token_ms: float | None = None
        t_gen = time.perf_counter()
        try:
            for tok in llm.stream(
                messages,
                max_tokens=settings.llm_max_tokens,
                temperature=settings.llm_temperature,
            ):
                if first_token_ms is None:
                    first_token_ms = (time.perf_counter() - t_gen) * 1000
                answer_parts.append(tok)
                yield {"type": "token", "text": tok}
        except Exception as e:  # noqa: BLE001
            log.error("generation_failed", error=str(e))
            yield {"type": "error", "detail": f"generation failed: {e}"}
            return

        answer = "".join(answer_parts)
        # Validate against the evidence the model actually saw: sections that
        # survived context packing, with their subsection structure checked.
        # Conversational turns carry no regulatory claims — skip validation
        # so prose like "an OSHA question" is never counted as a citation.
        if conversational:
            validation: dict = {
                "cited": [], "valid": [], "hallucinated": [],
                "n_cited": 0, "n_valid": 0, "n_hallucinated": 0, "detail": [],
            }
        else:
            validation = cite.validate_citations(
                answer, result.context_sections, result.context_section_texts
            )
        total_ms = (time.perf_counter() - t0) * 1000
        done_ev: dict = {
            "type": "done",
            "answer": answer,
            "citations": validation,
            "first_token_ms": round(first_token_ms or 0.0, 1),
            "total_ms": round(total_ms, 1),
        }
        if validation["n_hallucinated"]:
            # Fail visible: unverified citations are surfaced, never hidden.
            done_ev["warning"] = (
                f"{validation['n_hallucinated']} citation(s) could not be "
                "verified against the retrieved sections"
            )
        yield done_ev
        self._record(query_id, req, result, answer, total_ms, first_token_ms)

    # ---------------------------------------------------------------- search
    def search(self, query: str, top_k: int = 10, rerank: bool = False):
        from ..schemas import RetrievalConfig

        result = self.pipeline.retrieve(
            query,
            RetrievalConfig(final_top_k=top_k, rerank_enabled=rerank),
        )
        return result

    # -------------------------------------------------------------- internals
    @staticmethod
    def _source(h: RetrievedHit) -> SourceRef:
        return SourceRef(
            section_id=h.section_id,
            heading=h.heading,
            source=h.source,
            url=h.url,
            excerpt=(h.parent_text or h.text)[:1600],
            score=h.final_score or 0.0,
        )

    def _record(
        self,
        query_id: str,
        req: ChatRequest,
        result: RetrievalResult,
        answer: str,
        total_ms: float,
        first_token_ms: float | None,
    ) -> None:
        try:
            cited = cite.extract_citations(answer)
            self.traces.record(
                query_id=query_id,
                query=req.query,
                refused=result.refused,
                answer=answer,
                latency_ms=round(total_ms, 1),
                first_token_ms=round(first_token_ms, 1) if first_token_ms else None,
                grounding_score=result.grounding_score,
                cited_sections=cited,
                retrieval=[h.model_dump() for h in result.hits],
                timings=result.timings_ms,
            )
        except Exception as e:  # noqa: BLE001 — tracing must never break serving
            log.warning("trace_record_failed", error=str(e))
