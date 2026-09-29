"""Hybrid retrieval pipeline.

dense (bge) + sparse (BM25) → weighted RRF fusion (+ exact-citation
resolution) → cross-encoder rerank → hierarchy-aware MMR diversification →
small-to-big parent expansion → grounding guard → budgeted, boundary-safe
context assembly.

Design notes:

* **Canonical evidence store.** Every candidate — dense, sparse, or exact —
  is hydrated from the on-disk chunk store (``chunks.jsonl``), so no hit can
  reach reranking, the prompt, or the UI with missing text/metadata.
* **Exact-citation leg.** A query that names a regulation (``1910.132``,
  ``§ 1926.501(b)``) resolves directly against the corpus and dominates the
  fused ranking — identifiers are not paraphrased, so lexical/semantic legs
  should not outrank a literal match.
* **Reranker independence of the guard.** The grounding score deliberately
  excludes reranker output so the guard behaves identically across the
  eval A/B configs and can refuse before any expensive call.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass

import numpy as np

from ..config import Settings, get_settings
from ..index.embedder import EmbeddingService, Reranker
from ..index.sparse import SparseIndex
from ..index.vector_store import VectorStore
from ..ingest.chunker import count_tokens, sentences
from ..obs.logging import get_logger
from ..schemas import RetrievalConfig, RetrievedHit, Section
from .citations import parse_citation_refs

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
    rerank_top_n: int
    guard_rerank_floor: float
    final_top_k: int
    grounding_threshold: float
    ood_min_overlap: float
    mmr_enabled: bool
    mmr_lambda: float
    max_children_per_section: int
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
            rerank_top_n=settings.rerank_top_n,
            guard_rerank_floor=settings.guard_rerank_floor,
            final_top_k=get("final_top_k"),
            grounding_threshold=get("grounding_threshold"),
            ood_min_overlap=settings.ood_min_overlap,
            mmr_enabled=settings.mmr_enabled,
            mmr_lambda=settings.mmr_lambda,
            max_children_per_section=settings.max_children_per_section,
            context_budget_tokens=settings.context_budget_tokens,
        )


@dataclass
class RetrievalResult:
    hits: list[RetrievedHit]          # final children (rank-ordered)
    grounding_score: float
    lexical_overlap: float            # query-term coverage of top evidence
    exact_citation: bool              # query named a section present in the corpus
    rerank_agreement: float | None    # best cross-encoder score (None if rerank off)
    context: str                      # assembled parent context for the LLM
    context_sections: list[str]       # section_ids actually included in the prompt
    context_section_texts: dict[str, str]  # the exact bodies the model saw
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
        # Canonical evidence store: cid -> chunk record; cid -> dense point id.
        self._chunks: dict[str, dict] = {}
        self._cid_to_pid: dict[str, int] = {}
        self._load_chunk_store()

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

    def _load_chunk_store(self) -> None:
        path = self.settings.processed_dir / "chunks.jsonl"
        for pid, line in enumerate(path.read_text().splitlines()):
            if not line:
                continue
            c = json.loads(line)
            self._chunks[c["chunk_id"]] = c
            self._cid_to_pid[c["chunk_id"]] = pid

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

        # ---------------- exact-citation resolution ----------------
        # A query naming a regulation resolves straight to its chunks; those
        # get a dominant fusion boost so identifiers always outrank paraphrase.
        exact_sections = {
            ref.root
            for ref in parse_citation_refs(query)
            if any(
                sid == ref.root or sid.startswith(ref.root + " ")
                for sid in self.sections
            )
        }

        # ---------------- weighted RRF fusion ----------------
        fused: dict[str, dict] = {}
        for rank, (_point_id, score, payload) in enumerate(dense):
            cid = payload["chunk_id"]
            if cid not in self._chunks:  # stale index vs. fresh corpus guard
                continue
            fused.setdefault(cid, {"dense": None, "sparse": None, "rrf": 0.0, "exact": False})
            fused[cid]["dense"] = float(score)
            fused[cid]["rrf"] += cfg.rrf_dense_weight / (cfg.rrf_k + rank + 1)
        for rank, (cid, score) in enumerate(sparse):
            if cid not in self._chunks:
                continue
            fused.setdefault(cid, {"dense": None, "sparse": None, "rrf": 0.0, "exact": False})
            fused[cid]["sparse"] = float(score)
            fused[cid]["rrf"] += cfg.rrf_sparse_weight / (cfg.rrf_k + rank + 1)
        if exact_sections:
            for cid, rec in fused.items():
                sec = self._chunks[cid]["section_id"]
                if sec in exact_sections or sec.split(" (")[0] in exact_sections:
                    rec["rrf"] += 1.0  # dominates any RRF term (max ~ 1/61)
                    rec["exact"] = True
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
            pool = max(cfg.rerank_top_n, cfg.final_top_k)
            top = candidates[:pool]
            texts = [self._chunks[cid]["text"][:400] for cid, _ in top]
            scores = self.reranker.rerank(query, texts)
            rerank_scores = dict(
                ((cid, s) for (cid, _), s in zip(top, scores, strict=True))
            )
            # exact-citation hits keep their guaranteed lead
            candidates.sort(
                key=lambda kv: -(
                    (1e6 if kv[1]["exact"] else 0)
                    + rerank_scores.get(kv[0], -1e9)
                    + kv[1]["rrf"] * 10
                )
            )
            timings.append(
                {"stage": "rerank", "ms": round((time.perf_counter() - t) * 1e3, 1)}
            )

        # ---------------- hierarchy-aware diversification (MMR) ----------------
        t = time.perf_counter()
        ordered = [cid for cid, _ in candidates]
        if cfg.mmr_enabled:
            selected = self._diversify(ordered, qvec, cfg, rerank_scores)
        else:
            selected = self._cap_per_section(ordered, cfg)
        timings.append(
            {"stage": "diversify", "ms": round((time.perf_counter() - t) * 1e3, 1)}
        )

        # ---------------- assemble hits + parent expansion ----------------
        t = time.perf_counter()
        hits: list[RetrievedHit] = []
        for rank, cid in enumerate(selected[: cfg.final_top_k]):
            meta = fused[cid]
            c = self._chunks[cid]
            sec = self.sections.get(c["section_id"])
            hits.append(
                RetrievedHit(
                    chunk_id=cid,
                    section_id=c["section_id"],
                    heading=c["heading"],
                    source=c["source"],
                    part=c["part"],
                    parent_path=c["parent_path"],
                    url=c["url"],
                    text=c["text"],
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
        grounding, overlap = self._grounding_score(query, hits, bool(exact_sections))
        context, included, included_texts = self._assemble_context(hits, cfg)
        rerank_agreement = (
            max(rerank_scores.values()) if rerank_scores else None
        )
        refused = grounding < cfg.grounding_threshold or (
            overlap < cfg.ood_min_overlap and not exact_sections
        )
        # Reranker agreement: with reranking enabled, a query whose best
        # candidate the cross-encoder scores below zero is treated as
        # unanswerable — this separates near-domain vocabulary (wage law,
        # workers' comp, building code) from genuine OSHA questions far
        # better than embedding similarity alone.
        if (
            not refused
            and rerank_agreement is not None
            and rerank_agreement < cfg.guard_rerank_floor
            and not exact_sections
        ):
            refused = True
        timings.append(
            {"stage": "assemble", "ms": round((time.perf_counter() - t) * 1e3, 1)}
        )

        return RetrievalResult(
            hits=hits,
            grounding_score=round(grounding, 4),
            lexical_overlap=round(overlap, 4),
            exact_citation=bool(exact_sections),
            rerank_agreement=(
                round(rerank_agreement, 4) if rerank_agreement is not None else None
            ),
            context=context,
            context_sections=included,
            context_section_texts=included_texts,
            refused=refused,
            timings_ms=timings,
        )

    # ----------------------------------------------------------------- helpers
    def _cap_per_section(self, ordered: list[str], cfg: EffectiveConfig) -> list[str]:
        """No-MMR fallback: enforce only the per-section child cap."""
        selected: list[str] = []
        sec_count: dict[str, int] = {}
        for cid in ordered:
            if len(selected) >= max(cfg.final_top_k * 3, cfg.rerank_top_n):
                break
            sec = self._chunks[cid]["section_id"]
            if sec_count.get(sec, 0) >= cfg.max_children_per_section:
                continue
            sec_count[sec] = sec_count.get(sec, 0) + 1
            selected.append(cid)
        return selected

    def _diversify(
        self,
        ordered: list[str],
        qvec: np.ndarray,
        cfg: EffectiveConfig,
        rerank_scores: dict[str, float] | None = None,
    ) -> list[str]:
        """Greedy MMR with a per-section child cap.

        The relevance term is the RANKER's opinion (cross-encoder score when
        reranking ran, fused RRF otherwise), min-max normalized over the
        pool — so diversity reorders the tail without demoting the ranker's
        top pick. Vectors come from the dense store (stored L2-normalized,
        so dot product is cosine similarity) and only measure redundancy.
        """
        pool = [cid for cid in ordered if cid in self._cid_to_pid][
            : max(cfg.rerank_top_n, cfg.final_top_k)
        ]
        if not pool:
            return []
        pids = [self._cid_to_pid[cid] for cid in pool]
        vectors = self.store.vectors_for(pids)
        vmap = {cid: v for cid, v in zip(pool, vectors, strict=True) if v is not None}
        if not vmap:
            return pool[: cfg.final_top_k]

        # ranker relevance, normalized to [0, 1] within the pool
        if rerank_scores:
            rel_raw = {
                cid: rerank_scores.get(cid, min(rerank_scores.values()))
                for cid in vmap
            }
        else:
            n = len(pool)
            rel_raw = {cid: 1.0 - i / max(n - 1, 1) for i, cid in enumerate(pool)}
        lo, hi = min(rel_raw.values()), max(rel_raw.values())
        rel = {
            cid: (r - lo) / (hi - lo) if hi > lo else 1.0
            for cid, r in rel_raw.items()
        }

        selected: list[str] = []
        sel_vecs: list[np.ndarray] = []
        sec_count: dict[str, int] = {}
        want = max(cfg.final_top_k, 1)
        remaining = list(vmap)

        while len(selected) < want and remaining:
            best_cid, best_score = None, -np.inf
            for cid in remaining:
                sec = self._chunks[cid]["section_id"]
                if sec_count.get(sec, 0) >= cfg.max_children_per_section:
                    continue
                score = rel[cid]
                if sel_vecs:
                    redundancy = max(float(np.dot(vmap[cid], w)) for w in sel_vecs)
                    score = cfg.mmr_lambda * rel[cid] - (1 - cfg.mmr_lambda) * redundancy
                if score > best_score:
                    best_cid, best_score = cid, score
            if best_cid is None:  # every remaining candidate hit the section cap
                break
            selected.append(best_cid)
            sel_vecs.append(vmap[best_cid])
            sec = self._chunks[best_cid]["section_id"]
            sec_count[sec] = sec_count.get(sec, 0) + 1
            remaining.remove(best_cid)

        # fill any shortfall bypassing the cap (better redundant evidence than none)
        for cid in pool:
            if len(selected) >= want:
                break
            if cid not in selected:
                selected.append(cid)
        return selected

    def _grounding_score(
        self, query: str, hits: list[RetrievedHit], exact_citation: bool
    ) -> tuple[float, float]:
        """(score, lexical_overlap) — deliberately reranker-independent.

        The guard must behave identically whether or not reranking is enabled
        (eval A/B relies on it, and serving must refuse cheaply before any
        model call). Signals: max dense cosine, query-term coverage of the
        top evidence, and exact-citation resolution.
        """
        if not hits:
            return 0.0, 0.0
        max_cos = max((h.dense_score or 0.0) for h in hits)
        q_words = set(content_words(query))
        top_words: set[str] = set()
        for h in hits[:3]:
            top_words.update(content_words((h.parent_text or h.text)[:4000]))
        overlap = (len(q_words & top_words) / len(q_words)) if q_words else 0.0
        score = float(
            0.50 * max(0.0, max_cos)
            + 0.40 * overlap
            + 0.10 * float(exact_citation)
        )
        return score, overlap

    def _assemble_context(
        self, hits: list[RetrievedHit], cfg: EffectiveConfig
    ) -> tuple[str, list[str], dict[str, str]]:
        """Small-to-big: dedupe parents of child hits, pack under token budget.

        Packing is boundary-aware: whole paragraphs first, then whole
        sentences; legal prose is never cut mid-sentence. Whatever fit is
        recorded per section so citation validation can check the exact text
        the model saw.
        """
        blocks: list[str] = []
        included: list[str] = []
        included_texts: dict[str, str] = {}
        seen: set[str] = set()
        budget = cfg.context_budget_tokens
        for h in hits:
            if h.section_id in seen:
                continue
            seen.add(h.section_id)
            sec = self.sections.get(h.section_id)
            body = sec.text if sec else (h.parent_text or h.text)
            header = (
                f"### [{h.section_id}] {h.heading}\n"
                f"(path: {h.parent_path}; source: {h.source})\n"
            )
            body_budget = budget - count_tokens(header)
            if body_budget < 60:
                break
            packed, truncated = _pack_body(body, body_budget)
            tail = (
                "\n[...remainder of this section omitted to fit the context budget...]"
                if truncated
                else ""
            )
            block = header + packed + tail
            budget -= count_tokens(block)
            blocks.append(block)
            included.append(h.section_id)
            included_texts[h.section_id] = packed
        return "\n\n---\n\n".join(blocks), included, included_texts


def _pack_body(body: str, budget: int) -> tuple[str, bool]:
    """Fit ``body`` under ``budget`` tokens without slicing sentences.

    Returns (packed_text, truncated). Paragraphs (newline-separated) are kept
    whole while they fit; an oversized paragraph degrades to whole sentences;
    only a single pathological sentence is ever hard-cut.
    """
    if count_tokens(body) <= budget:
        return body, False

    kept: list[str] = []
    used = 0
    for para in body.split("\n"):
        pn = count_tokens(para)
        if used + pn <= budget:
            kept.append(para)
            used += pn
            continue
        for sent in sentences(para):
            sn = count_tokens(sent)
            if used + sn > budget:
                if not kept and used == 0 and sn > budget:
                    keep_chars = int(len(sent) * budget / sn)
                    kept.append(sent[:keep_chars] + " [...]")  # pathological only
                return "\n".join(kept), True
            kept.append(sent)
            used += sn
        return "\n".join(kept), True
    return "\n".join(kept), True
