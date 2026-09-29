"""Hybrid retrieval pipeline.

dense (bge) + sparse (BM25) → RRF fusion → cross-encoder rerank →
small-to-big parent expansion → grounding guard → budgeted context assembly.

Every stage records timings and score provenance so the UI/eval can show
*why* a given section was selected.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

from ..config import Settings, get_settings
from ..index.embedder import EmbeddingService, Reranker
from ..index.sparse import SparseIndex
from ..index.vector_store import VectorStore
from ..ingest.chunker import count_tokens
from ..obs.logging import get_logger
from ..schemas import RetrievalConfig, RetrievedHit, Section

log = get_logger(__name__)

_STOP = set(
    "a an the is are was were be been being do does did what which who whom whose "
    "when where why how of in on at to for from with by under over about into "
    "i me my we our you your he she it they them as and or if then than so not "
    "can could should would may might must will shall".split()
)
_WORD = re.compile(r"[a-z0-9]+")

_CONTENT_WORDS_CACHE: dict[str, list[str]] = {}


def content_words(text: str) -> list[str]:
    if text not in _CONTENT_WORDS_CACHE:
        _CONTENT_WORDS_CACHE[text] = [w for w in _WORD.findall(text.lower()) if w not in _STOP]
    return _CONTENT_WORDS_CACHE[text]


@dataclass
class EffectiveConfig:
    dense_top_k: int
    sparse_top_k: int
    rrf_k: int
    rrf_dense_weight: float
    rrf_sparse_weight: float
    rerank_enabled: bool
    final_top_k: int
    grounding_threshold: float
    context_budget_tokens: int

    @classmethod
    def resolve(
        cls, settings: Settings, override: RetrievalConfig | None = None
    ) -> EffectiveConfig:
        o = override or RetrievalConfig()
        get = lambda f: getattr(o, f) if getattr(o, f) is not None else getattr(settings, f)  # noqa: E731
        return cls(
            dense_top_k=get("dense_top_k"),
            sparse_top_k=get("sparse_top_k"),
            rrf_k=get("rrf_k"),
            rrf_dense_weight=settings.rrf_dense_weight,
            rrf_sparse_weight=settings.rrf_sparse_weight,
            rerank_enabled=get("rerank_enabled"),
            final_top_k=get("final_top_k"),
            grounding_threshold=get("grounding_threshold"),
            context_budget_tokens=settings.context_budget_tokens,
        )


@dataclass
class RetrievalResult:
    hits: list[RetrievedHit]          # final children (rank-ordered)
    grounding_score: float
    context: str                      # assembled parent context for the LLM
    context_sections: list[str]       # section_ids actually included
    refused: bool
    timings_ms: list[dict]


class RetrievalPipeline:
    def __init__(
        self,
        settings: Settings | None = None,
        embedder: EmbeddingService | None = None,
        store: VectorStore | None = None,
        sparse: SparseIndex | None = None,
        reranker: Reranker | None = None,
        sections: dict[str, Section] | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.embedder = embedder or EmbeddingService(self.settings)
        self.store = store or VectorStore(self.settings)
        self.sparse = sparse or self._load_sparse()
        self.reranker = reranker
        self.sections = sections if sections is not None else self._load_sections()

    def _load_sparse(self) -> SparseIndex:
        s = SparseIndex(self.settings)
        if not s.load():
            raise RuntimeError(
                "BM25 index not found — run `python -m app.index.build` first"
            )
        return s

    def _load_sections(self) -> dict[str, Section]:
        path = self.settings.processed_dir / "sections.jsonl"
        return {
            s.section_id: s
            for s in (
                Section.model_validate_json(line)
                for line in path.read_text().splitlines()
                if line
            )
        }

    # ------------------------------------------------------------------ public
    def retrieve(
        self, query: str, override: RetrievalConfig | None = None
    ) -> RetrievalResult:
        cfg = EffectiveConfig.resolve(self.settings, override)
        timings: list[dict] = []

        t = time.perf_counter()
        qvec = self.embedder.embed_queries([query])[0]
        timings.append({"stage": "embed_query", "ms": round((time.perf_counter() - t) * 1e3, 1)})

        t = time.perf_counter()
        dense = self.store.search(qvec.tolist(), limit=cfg.dense_top_k)
        timings.append({"stage": "dense_search", "ms": round((time.perf_counter() - t) * 1e3, 1)})

        t = time.perf_counter()
        sparse = self.sparse.search(query, top_k=cfg.sparse_top_k)
        timings.append({"stage": "sparse_search", "ms": round((time.perf_counter() - t) * 1e3, 1)})

        # ---------------- weighted RRF fusion ----------------
        fused: dict[str, dict] = {}
        for rank, (_point_id, score, payload) in enumerate(dense):
            cid = payload["chunk_id"]
            fused.setdefault(cid, {"payload": payload, "dense": None, "sparse": None, "rrf": 0.0})
            fused[cid]["dense"] = float(score)
            fused[cid]["rrf"] += cfg.rrf_dense_weight / (cfg.rrf_k + rank + 1)
        for rank, (cid, score) in enumerate(sparse):
            fused.setdefault(cid, {"payload": None, "dense": None, "sparse": None, "rrf": 0.0})
            fused[cid]["sparse"] = float(score)
            fused[cid]["rrf"] += cfg.rrf_sparse_weight / (cfg.rrf_k + rank + 1)
        candidates = sorted(fused.items(), key=lambda kv: -kv[1]["rrf"])
        timings.append({"stage": "fuse", "ms": 0.1})

        # ---------------- cross-encoder rerank ----------------
        # Candidates are truncated for scoring: cross-encoders rank the
        # head of a passage well, and short inputs keep CPU latency low.
        rerank_scores: dict[str, float] = {}
        if cfg.rerank_enabled and self.reranker is None:
            self.reranker = Reranker(self.settings)
        if cfg.rerank_enabled and candidates:
            t = time.perf_counter()
            top = candidates[: max(cfg.final_top_k * 3, 16)]
            texts = [
                (self._payload_text(cid, meta) or cid)[:400] for cid, meta in top
            ]
            scores = self.reranker.rerank(query, texts)
            rerank_scores = dict(
                ((cid, s) for (cid, _), s in zip(top, scores, strict=True))
            )
            candidates.sort(key=lambda kv: -(rerank_scores.get(kv[0], -1e9) + kv[1]["rrf"] * 10))
            timings.append(
                {"stage": "rerank", "ms": round((time.perf_counter() - t) * 1e3, 1)}
            )

        # ---------------- assemble hits + parent expansion ----------------
        t = time.perf_counter()
        hits: list[RetrievedHit] = []
        for rank, (cid, meta) in enumerate(candidates[: cfg.final_top_k]):
            p = meta["payload"] or {}
            sec = self.sections.get(p.get("section_id", ""))
            hits.append(
                RetrievedHit(
                    chunk_id=cid,
                    section_id=p.get("section_id", cid.split("#")[0]),
                    heading=p.get("heading", ""),
                    source=p.get("source", ""),
                    part=p.get("part", ""),
                    parent_path=p.get("parent_path", ""),
                    url=p.get("url", ""),
                    text=p.get("text", ""),
                    dense_score=meta["dense"],
                    sparse_score=meta["sparse"],
                    rrf_score=round(meta["rrf"], 6),
                    rerank_score=(
                        round(rerank_scores[cid], 4) if cid in rerank_scores else None
                    ),
                    final_score=round(rerank_scores.get(cid, meta["rrf"] * 10), 4),
                    rank=rank + 1,
                    parent_text=sec.text if sec else "",
                )
            )
        grounding = self._grounding_score(query, hits)
        context, included = self._assemble_context(hits, cfg)
        refused = grounding < cfg.grounding_threshold
        timings.append(
            {"stage": "assemble", "ms": round((time.perf_counter() - t) * 1e3, 1)}
        )

        return RetrievalResult(
            hits=hits,
            grounding_score=round(grounding, 4),
            context=context,
            context_sections=included,
            refused=refused,
            timings_ms=timings,
        )

    # ----------------------------------------------------------------- helpers
    def _payload_text(self, cid: str, meta: dict) -> str:
        if meta["payload"]:
            return meta["payload"].get("text", "")
        # sparse-only hit: text lives in qdrant payload; fall back to section store
        sec_id = cid.split("#")[0]
        sec = self.sections.get(sec_id)
        return sec.text[:1200] if sec else cid

    def _grounding_score(self, query: str, hits: list[RetrievedHit]) -> float:
        """0..1 blend of dense similarity and lexical coverage.

        Deliberately independent of the reranker: the guard must behave
        identically whether or not reranking is enabled (eval A/B relies on
        it, and serving must refuse cheaply before any model call).
        """
        if not hits:
            return 0.0
        max_cos = max((h.dense_score or 0.0) for h in hits)
        q_words = set(content_words(query))
        top_words: set[str] = set()
        for h in hits[:3]:
            top_words.update(content_words((h.parent_text or h.text)[:4000]))
        overlap = (len(q_words & top_words) / len(q_words)) if q_words else 0.0
        return float(0.55 * max(0.0, max_cos) + 0.45 * overlap)

    def _assemble_context(
        self, hits: list[RetrievedHit], cfg: EffectiveConfig
    ) -> tuple[str, list[str]]:
        """Small-to-big: dedupe parents of child hits, pack under token budget."""
        blocks: list[str] = []
        included: list[str] = []
        seen: set[str] = set()
        budget = cfg.context_budget_tokens
        for h in hits:
            if h.section_id in seen:
                continue
            seen.add(h.section_id)
            sec = self.sections.get(h.section_id)
            body = sec.text if sec else h.parent_text or h.text
            header = (
                f"### [{h.section_id}] {h.heading}\n"
                f"(path: {h.parent_path}; source: {h.source})\n"
            )
            block = header + body
            n = count_tokens(block)
            if n > budget:
                keep = body[: int(len(body) * budget / n)]
                block = header + keep + "\n[...truncated...]"
                n = count_tokens(block)
            if budget - n < 60:
                break
            budget -= n
            blocks.append(block)
            included.append(h.section_id)
        return "\n\n---\n\n".join(blocks), included
