"""Index build orchestration: chunks.jsonl → Qdrant (dense) + BM25 (sparse).

Embeddings are checkpointed to a memory-mapped .npy file with a progress
sidecar, so an interrupted build (crash, reboot, OOM) resumes from the last
completed batch instead of re-embedding the whole corpus. The Qdrant upsert
and BM25 phases always rebuild (they are fast).
"""

from __future__ import annotations

import hashlib
import json
import time

import numpy as np

from ..config import Settings, get_settings
from ..obs.logging import get_logger
from ..schemas import Chunk
from .embedder import EmbeddingService
from .sparse import SparseIndex
from .vector_store import VectorStore

log = get_logger(__name__)


def load_chunks(settings: Settings) -> list[Chunk]:
    path = settings.processed_dir / "chunks.jsonl"
    return [Chunk.model_validate_json(line) for line in path.read_text().splitlines() if line]


def _corpus_fingerprint(chunks: list[Chunk]) -> str:
    h = hashlib.sha1()
    for c in chunks:
        h.update(c.chunk_id.encode())
        h.update(len(c.text).to_bytes(8, "little"))
        h.update(c.text.encode())
    return h.hexdigest()[:16]


def embed_with_resume(
    chunks: list[Chunk], embedder: EmbeddingService, settings: Settings
) -> np.ndarray:
    """Embed all chunk texts, resuming from the on-disk checkpoint."""
    dim = embedder.dim
    emb_dir = settings.artifacts_dir / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)
    arr_path = emb_dir / "child_embeddings.npy"
    meta_path = emb_dir / "progress.json"
    fp = _corpus_fingerprint(chunks)
    batch = 512

    meta = {"fingerprint": fp, "done": 0}
    if meta_path.exists():
        try:
            saved = json.loads(meta_path.read_text())
            if saved.get("fingerprint") == fp:
                meta = saved
        except json.JSONDecodeError:
            pass

    done = min(meta["done"], len(chunks))
    mmap = np.lib.format.open_memmap(
        arr_path, mode="w+" if done == 0 else "r+", shape=(len(chunks), dim), dtype=np.float32
    )
    if done == 0:
        mmap[:] = 0

    t0 = time.perf_counter()
    log.info("embed_resume", done=done, total=len(chunks))
    while done < len(chunks):
        lo, hi = done, min(done + batch, len(chunks))
        vecs = embedder.embed_documents([c.text for c in chunks[lo:hi]])
        mmap[lo:hi] = vecs
        mmap.flush()
        done = hi
        meta_path.write_text(json.dumps({"fingerprint": fp, "done": done}))
        rate = done / max(time.perf_counter() - t0 + 1e-9, 1e-9) if done else 0
        log.info("embed_progress", done=done, total=len(chunks), per_s=round(rate, 1))
    return np.asarray(mmap)


def run_index(settings: Settings | None = None) -> dict:
    settings = settings or get_settings()
    t0 = time.perf_counter()
    chunks = load_chunks(settings)
    log.info("index_start", chunks=len(chunks))

    embedder = EmbeddingService(settings)
    vectors = embed_with_resume(chunks, embedder, settings)

    store = VectorStore(settings)
    store.recreate(dim=embedder.dim)
    step = 512
    for i in range(0, len(chunks), step):
        pts = [
            store.point_from_chunk(i + j, chunks[i + j], vectors[i + j].tolist())
            for j in range(min(step, len(chunks) - i))
        ]
        store.upsert(pts)
    log.info("qdrant_upserted", points=store.count())

    sparse = SparseIndex(settings)
    sparse.build(chunks)

    stats = {
        "chunks": len(chunks),
        "embedding_model": settings.embed_model,
        "dim": embedder.dim,
        "qdrant_collection": settings.qdrant_collection,
        "qdrant_points": store.count(),
        "bm25_dir": str(sparse.dir),
        "duration_s": round(time.perf_counter() - t0, 1),
    }
    (settings.processed_dir / "index_stats.json").write_text(json.dumps(stats, indent=2))
    log.info("index_complete", **stats)
    return stats


if __name__ == "__main__":
    print(json.dumps(run_index(), indent=2))
