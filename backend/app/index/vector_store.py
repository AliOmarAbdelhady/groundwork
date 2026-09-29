"""Qdrant vector store: embedded local mode by default, server mode via env."""

from __future__ import annotations

from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from ..config import Settings, get_settings
from ..schemas import Chunk


class VectorStore:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        if self.settings.qdrant_url:
            self.client = QdrantClient(url=self.settings.qdrant_url, timeout=60)
            self._local_path: Path | None = None
        else:
            self._local_path = self.settings.qdrant_local_path or (
                self.settings.artifacts_dir / "qdrant"
            )
            self.client = QdrantClient(path=str(self._local_path))

    def recreate(self, dim: int) -> None:
        if self.client.collection_exists(self.settings.qdrant_collection):
            self.client.delete_collection(self.settings.qdrant_collection)
        self.client.create_collection(
            collection_name=self.settings.qdrant_collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

    def upsert(self, points: list[PointStruct]) -> None:
        self.client.upsert(collection_name=self.settings.qdrant_collection, points=points)

    def search(self, vector: list[float], limit: int) -> list[tuple[int, float, dict]]:
        """Return [(point_id, cosine_score, payload)] ordered by score."""
        if limit <= 0:  # A/B eval configs disable one leg entirely
            return []
        res = self.client.query_points(
            collection_name=self.settings.qdrant_collection,
            query=vector,
            limit=limit,
            with_payload=True,
        )
        return [(p.id, p.score, p.payload or {}) for p in res.points]

    def count(self) -> int:
        res = self.client.count(self.settings.qdrant_collection, exact=True)
        return res.count

    @staticmethod
    def point_from_chunk(idx: int, chunk: Chunk, vector: list[float]) -> PointStruct:
        return PointStruct(
            id=idx,
            vector=vector,
            payload={
                "chunk_id": chunk.chunk_id,
                "section_id": chunk.section_id,
                "heading": chunk.heading,
                "source": chunk.source,
                "part": chunk.part,
                "parent_path": chunk.parent_path,
                "url": chunk.url,
                "text": chunk.text,
            },
        )
