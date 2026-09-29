"""Index build orchestration: chunks.jsonl → Qdrant (dense) + BM25 (sparse).

Embeddings are checkpointed per child chunk: the .npy matrix is stored next
to a hash list of the exact chunk texts it was computed from. An interrupted
build (crash, reboot, OOM) resumes by re-embedding only the rows whose text
hash changed or is missing — a metadata-only corpus refresh costs seconds,
not minutes. The Qdrant upsert and BM25 phases always rebuild (they are fast).

Every build stamps ``data/artifacts/manifest.json`` binding the index to the
corpus manifest (sources, hashes, versions) it was built from.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import UTC, datetime

import numpy as np

from ..config import Settings, get_settings
from ..manifest import load_corpus_manifest
from ..obs.logging import get_logger
from ..schemas import Chunk
from .embedder import EmbeddingService
from .sparse import SparseIndex
from .vector_store import VectorStore

log = get_logger(__name__)


def load_chunks(settings: Settings) -> list[Chunk]:
    path = settings.processed_dir / "chunks.jsonl"
    return [Chunk.model_validate_json(line) for line in path.read_text().splitlines() if line]


def _chunk_hash(chunk: Chunk) -> str:
    h = hashlib.sha1()
    h.update(chunk.chunk_id.encode())
    h.update(chunk.text.encode())
    return h.hexdigest()


def _prefix_done(new_hashes: list[str], reusable: set[int], embedded: list[int]) -> list[str]:
    """Hash list with blanks for rows not yet (re)embedded — the resume map."""
    known = reusable | set(embedded)
    return [h if i in known else "" for i, h in enumerate(new_hashes)]


def embed_with_resume(
    chunks: list[Chunk], embedder: EmbeddingService, settings: Settings
) -> np.ndarray:
    """Embed all chunk texts, re-embedding only rows whose text changed."""
    dim = embedder.dim
    emb_dir = settings.artifacts_dir / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)
    arr_path = emb_dir / "child_embeddings.npy"
    hashes_path = emb_dir / "chunk_hashes.json"
    want = (len(chunks), dim)

    new_hashes = [_chunk_hash(c) for c in chunks]
    old_hashes: list[str] = []
    if hashes_path.exists():
        try:
            old_hashes = json.loads(hashes_path.read_text())
        except json.JSONDecodeError:
            old_hashes = []

    def _open_or_recreate() -> np.memmap:
        """Open r+ when the array already matches the target shape; otherwise
        allocate fresh and carry over the old rows (the hash list then decides
        which carried rows still need re-embedding)."""
        if arr_path.exists():
            old = np.load(arr_path, mmap_mode="r")
            if old.shape == want and old.dtype == np.float32:
                del old
                return np.lib.format.open_memmap(arr_path, mode="r+", shape=want, dtype=np.float32)
            carried = np.array(old[: min(old.shape[0], want[0])])
            del old
            mmap = np.lib.format.open_memmap(arr_path, mode="w+", shape=want, dtype=np.float32)
            mmap[:] = 0
            mmap[: carried.shape[0]] = carried
            mmap.flush()
            return mmap
        mmap = np.lib.format.open_memmap(arr_path, mode="w+", shape=want, dtype=np.float32)
        mmap[:] = 0
        return mmap

    mmap = _open_or_recreate()

    shape_matches = arr_path.exists() and np.load(arr_path, mmap_mode="r").shape == want
    if not shape_matches:
        old_hashes = []
    reusable = {
        i
        for i in range(min(len(old_hashes), len(chunks)))
        if old_hashes[i] == new_hashes[i]
    }
    needs = [i for i in range(len(chunks)) if i not in reusable]
    log.info("embed_plan", total=len(chunks), reusable=len(reusable), to_embed=len(needs))

    batch = 512
    t0 = time.perf_counter()
    done = 0
    for lo in range(0, len(needs), batch):
        rows = needs[lo : lo + batch]
        vecs = embedder.embed_documents([chunks[i].text for i in rows])
        for r, v in zip(rows, vecs, strict=True):
            mmap[r] = v
        mmap.flush()
        done += len(rows)
        # checkpoint: hashes of rows known good so far (blank = not yet done)
        hashes_path.write_text(json.dumps(_prefix_done(new_hashes, reusable, needs[: lo + len(rows)])))
        rate = done / max(time.perf_counter() - t0 + 1e-9, 1e-9)
        log.info("embed_progress", done=done, total=len(needs), per_s=round(rate, 1))
    hashes_path.write_text(json.dumps(new_hashes))
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

    corpus = load_corpus_manifest(settings)
    manifest = {
        **corpus,
        "corpus_version": corpus.get("corpus_version", "unknown"),
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "embedding_model": settings.embed_model,
        "embedding_dim": embedder.dim,
        "qdrant_collection": settings.qdrant_collection,
        "qdrant_points": store.count(),
        "bm25_docs": len(sparse.chunk_ids),
        "build_duration_s": round(time.perf_counter() - t0, 1),
    }
    (settings.artifacts_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    stats = {
        **{k: manifest[k] for k in (
            "corpus_version", "qdrant_points", "bm25_docs", "embedding_model",
        )},
        "chunks": len(chunks),
        "dim": embedder.dim,
        "bm25_dir": str(sparse.dir),
        "duration_s": round(time.perf_counter() - t0, 1),
    }
    (settings.processed_dir / "index_stats.json").write_text(json.dumps(stats, indent=2))
    log.info("index_complete", **stats)
    return stats


if __name__ == "__main__":
    print(json.dumps(run_index(), indent=2))
