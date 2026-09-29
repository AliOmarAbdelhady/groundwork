"""Dense embedding + cross-encoder reranker services.

The embedder runs BAAI/bge-small-en-v1.5 (fp32 ONNX) directly on
onnxruntime with explicit CLS pooling + L2 normalization and dynamic
batch padding. A raw-ORT path was chosen over fastembed's default
int8-quantized model because the int8 QDQ kernels are ~2x slower than
fp32 on this host's CPU (measured 4.9 vs 9.0 chunks/s); on standard
cloud CPUs the difference disappears and either path works.

bge-*-en-v1.5 queries should carry the instruction prefix
"Represent this sentence for searching relevant passages:" — handled
in :meth:`EmbeddingService.embed_queries`.
"""

from __future__ import annotations

import threading
from functools import lru_cache

import numpy as np

from ..config import Settings, get_settings

_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class EmbeddingService:
    MODEL_FILE = "onnx/model.onnx"
    MAX_LEN = 512
    DIM = 384

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        # settings.embed_model is the single source of truth for the dense leg
        self.model_repo = self.settings.embed_model
        self._lock = threading.Lock()
        self._session = None
        self._tokenizer = None

    def _ensure_loaded(self) -> None:
        if self._session is not None:
            return
        with self._lock:
            if self._session is not None:
                return
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download
            from tokenizers import Tokenizer

            model_path = hf_hub_download(
                repo_id=self.model_repo,
                filename=self.MODEL_FILE,
                local_dir=str(self.settings.models_dir / "embed"),
            )
            tok_path = hf_hub_download(
                repo_id=self.model_repo,
                filename="tokenizer.json",
                local_dir=str(self.settings.models_dir / "embed"),
            )
            so = ort.SessionOptions()
            so.intra_op_num_threads = min(8, self.settings.llm_n_threads or 8)
            so.inter_op_num_threads = 1
            self._session = ort.InferenceSession(
                model_path, sess_options=so, providers=["CPUExecutionProvider"]
            )
            self._tokenizer = Tokenizer.from_file(tok_path)

    @property
    def dim(self) -> int:
        return self.DIM

    def _encode(self, texts: list[str]) -> np.ndarray:
        self._ensure_loaded()
        assert self._tokenizer is not None
        enc = self._tokenizer.encode_batch(texts)
        rows = []
        for e in enc:
            ids = e.ids[: self.MAX_LEN]
            rows.append(
                (
                    ids,
                    [1] * len(ids),
                    e.type_ids[: self.MAX_LEN],
                )
            )
        batch_max = max(len(r[0]) for r in rows)
        ids = np.zeros((len(rows), batch_max), dtype=np.int64)
        mask = np.zeros((len(rows), batch_max), dtype=np.int64)
        tids = np.zeros((len(rows), batch_max), dtype=np.int64)
        for i, (a, m, t) in enumerate(rows):
            ids[i, : len(a)] = a
            mask[i, : len(m)] = m
            tids[i, : len(t)] = t
        out = self._session.run(
            None,
            {"input_ids": ids, "attention_mask": mask, "token_type_ids": tids},
        )[0]
        cls = out[:, 0, :]  # CLS pooling (BAAI bge convention)
        cls = cls / (np.linalg.norm(cls, axis=1, keepdims=True) + 1e-12)
        return cls.astype(np.float32)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        bs = self.settings.embed_batch_size
        chunks = [self._encode(texts[i : i + bs]) for i in range(0, len(texts), bs)]
        return np.vstack(chunks) if len(chunks) > 1 else chunks[0]

    def embed_queries(self, queries: list[str]) -> np.ndarray:
        return self._encode([_QUERY_PREFIX + q for q in queries])


class Reranker:
    """ms-marco MiniLM cross-encoder via fastembed (ONNX, CPU-friendly)."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        cache = self.settings.models_dir / "fastembed"
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self._model = TextCrossEncoder(
            model_name=self.settings.rerank_model, cache_dir=str(cache)
        )

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        return [float(s) for s in self._model.rerank(query, documents)]


@lru_cache
def get_embedder() -> EmbeddingService:
    return EmbeddingService()


@lru_cache
def get_reranker() -> Reranker:
    return Reranker()
