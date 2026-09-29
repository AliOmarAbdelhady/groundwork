"""BM25 sparse index (bm25s) over child chunks, with persisted artifacts."""

from __future__ import annotations

import json

import bm25s
import Stemmer  # provided by bm25s[full]-equivalent extra (PyStemmer)

from ..config import Settings, get_settings
from ..schemas import Chunk


class SparseIndex:
    """Build / persist / query the BM25 index. Row i ⇄ chunk_ids[i]."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.dir = self.settings.artifacts_dir / "bm25"
        self.retriever: bm25s.BM25 | None = None
        self.chunk_ids: list[str] = []
        self._stemmer = Stemmer.Stemmer("english")

    def _tokenize(self, corpus: list[str]):
        return bm25s.tokenize(
            corpus, stopwords="en", stemmer=self._stemmer, show_progress=False
        )

    def build(self, chunks: list[Chunk]) -> None:
        corpus = [c.text for c in chunks]
        self.chunk_ids = [c.chunk_id for c in chunks]
        tok = self._tokenize(corpus)
        self.retriever = bm25s.BM25(method="lucene", k1=self.settings.bm25_k1,
                                    b=self.settings.bm25_b)
        self.retriever.index(tok, show_progress=False)
        self.save()

    def save(self) -> None:
        assert self.retriever is not None
        self.dir.mkdir(parents=True, exist_ok=True)
        self.retriever.save(str(self.dir), corpus=self.chunk_ids)
        (self.dir / "chunk_ids.json").write_text(json.dumps(self.chunk_ids))

    def load(self) -> bool:
        if not (self.dir / "vocab.index.json").exists():
            return False
        self.retriever = bm25s.BM25.load(str(self.dir), load_corpus=True, mmap=True)
        raw = (self.dir / "chunk_ids.json").read_text()
        self.chunk_ids = json.loads(raw)
        return True

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """Return [(chunk_id, bm25_score)] ordered by score."""
        if top_k <= 0:  # A/B eval configs disable one leg entirely
            return []
        assert self.retriever is not None, "call load() or build() first"
        q = bm25s.tokenize([query], stopwords="en", stemmer=self._stemmer,
                           show_progress=False)
        results, scores = self.retriever.retrieve(q, k=top_k, show_progress=False)
        out: list[tuple[str, float]] = []
        for cid, s in zip(results[0], scores[0], strict=False):
            if cid is None:
                continue
            if isinstance(cid, dict):  # bm25s corpus-store entries: {id, text}
                cid = cid.get("text", "")
            cid = str(cid)
            if cid.endswith(".txt"):  # legacy corpus-store naming
                cid = cid[:-4]
            out.append((cid, float(s)))
        return out
