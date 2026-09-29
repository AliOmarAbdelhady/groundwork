"""Central configuration for GroundWork.

All knobs are overridable via environment variables with the ``GW_`` prefix,
e.g. ``GW_RERANK_ENABLED=false`` or ``GW_LLM_PROVIDER=openai_compatible``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GW_", env_file=".env", extra="ignore")

    # ------------------------------------------------------------------ paths
    project_root: Path = Path(__file__).resolve().parents[2]

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def artifacts_dir(self) -> Path:
        return self.data_dir / "artifacts"

    @property
    def models_dir(self) -> Path:
        return self.project_root / "models"

    @property
    def traces_db(self) -> Path:
        return self.data_dir / "traces.db"

    # -------------------------------------------------------------- ingestion
    ecfr_title: int = 29
    ecfr_parts_min: int = 1900  # OSHA chapter XVII span
    ecfr_parts_max: int = 2099
    ecfr_titles_url: str = "https://www.ecfr.gov/api/versioner/v1/titles.json"
    ecfr_full_url: str = (
        "https://www.ecfr.gov/api/versioner/v1/full/{date}/title-{title}.xml"
    )
    ecfr_user_agent: str = (
        "Mozilla/5.0 (X11; Linux x86_64) groundwork-rag/0.1 (research project)"
    )
    pdf_manifest: Path = Path(__file__).resolve().parent / "ingest" / "pdf_manifest.yaml"
    http_timeout: float = 120.0
    http_retries: int = 3

    # --------------------------------------------------------------- chunking
    child_target_tokens: int = 220   # target size of searchable child chunks
    child_overlap_tokens: int = 48   # token budget of trailing overlap between children
    child_min_tokens: int = 32       # shorter sections stay single-chunk
    context_budget_tokens: int = 2400  # max tokens of parent context given to the LLM

    # -------------------------------------------------------------- embedding
    embed_model: str = "Xenova/bge-small-en-v1.5"  # fp32 ONNX port of bge-small-en-v1.5
    embed_batch_size: int = 64
    embed_cache_dir: Path | None = None  # defaults to <models_dir>/fastembed

    # ------------------------------------------------------------------ qdrant
    qdrant_collection: str = "groundwork_sections_children"
    qdrant_local_path: Path | None = None  # defaults to <artifacts_dir>/qdrant
    qdrant_url: str | None = None  # if set, use server instead of embedded mode

    # ------------------------------------------------------------------- bm25
    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    # -------------------------------------------------------------- retrieval
    dense_top_k: int = 30
    sparse_top_k: int = 30
    rrf_k: int = 60
    rrf_dense_weight: float = 0.65   # weighted RRF: dense leg (tuned on golden set)
    rrf_sparse_weight: float = 0.35  # weighted RRF: BM25 leg
    rerank_enabled: bool = True
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    rerank_top_n: int = 24           # cross-encoder candidate pool size
    final_top_k: int = 6
    # diversity: MMR over child vectors after rerank, with a per-section cap so
    # one long section cannot crowd out independent evidence
    mmr_enabled: bool = True
    mmr_lambda: float = 0.7
    max_children_per_section: int = 2
    grounding_threshold: float = 0.52  # below -> refuse (no LLM call; calibrated on golden set)
    ood_min_overlap: float = 0.06      # hard lexical floor: below this a query is
                                       # out-of-domain unless it cites an exact section
    guard_rerank_floor: float = 0.0    # with reranking on, the best candidate must
                                       # clear this cross-encoder score or the query
                                       # is refused (near-domain discriminator;
                                       # calibrated: adversarial max -1.4, golden
                                       # min +1.5 on the golden set)

    # --------------------------------------------------------------- generation
    llm_provider: str = "local"  # local | openai_compatible
    gguf_repo_id: str = "Qwen/Qwen2.5-3B-Instruct-GGUF"
    gguf_filename: str = "qwen2.5-3b-instruct-q4_k_m.gguf"
    llm_n_ctx: int = 6144
    llm_n_threads: int = 4
    llm_temperature: float = 0.2
    llm_max_tokens: int = 768
    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"

    # -------------------------------------------------------------------- api
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    # ------------------------------------------------------------------- eval
    # CI regression floors (multi-gate: a config passes only if ALL hold).
    # Anchored to measured hybrid+rerank performance on the 90-item golden
    # set (hit@1 0.557 / hit@5 0.914 / nDCG 0.765 / adversarial refusal 0.90 /
    # answered 1.00) minus a small safety margin.
    eval_hit1_floor: float = 0.50
    eval_hit5_floor: float = 0.88
    eval_ndcg_floor: float = 0.72
    eval_adversarial_refusal_floor: float = 0.85
    eval_answered_floor: float = 0.98


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    for p in (s.data_dir, s.raw_dir, s.processed_dir, s.artifacts_dir, s.models_dir):
        p.mkdir(parents=True, exist_ok=True)
    return s
