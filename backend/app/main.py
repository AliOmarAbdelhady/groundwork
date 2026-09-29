"""FastAPI application: chat SSE, search, eval, observability endpoints."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from .config import get_settings
from .index.build import run_index
from .ingest.pipeline import run_ingest
from .obs.logging import configure_logging, get_logger
from .rag.orchestrator import ChatOrchestrator
from .schemas import ChatRequest, SearchRequest

configure_logging()
log = get_logger(__name__)
settings = get_settings()

_orch: ChatOrchestrator | None = None
_eval_lock = asyncio.Lock()


def get_orchestrator() -> ChatOrchestrator:
    global _orch
    if _orch is None:
        log.info("orchestrator_init")
        _orch = ChatOrchestrator(settings)
    return _orch


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("api_starting", provider=settings.llm_provider)
    yield
    log.info("api_stopped")


app = FastAPI(
    title="GroundWork API",
    description="Production-grade hybrid RAG over U.S. OSHA regulations (29 CFR) "
    "and OSHA publications.",
    version="0.1.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "llm_provider": settings.llm_provider,
        "indexes": _index_status(),
    }


def _index_status() -> dict:
    try:
        from .index.sparse import SparseIndex
        from .index.vector_store import VectorStore

        store = VectorStore(settings)
        sparse = SparseIndex(settings)
        loaded = sparse.load()
        points = store.count() if store.client.collection_exists(settings.qdrant_collection) else 0
        return {"qdrant_points": points, "bm25_ready": loaded,
                "bm25_docs": len(sparse.chunk_ids) if loaded else 0}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def _sse(ev: dict) -> str:
    return "data: " + json.dumps(
        ev,
        ensure_ascii=False,
        default=lambda o: o.model_dump() if hasattr(o, "model_dump") else str(o),
    ) + "\n\n"


@app.post("/api/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    orch = get_orchestrator()

    async def gen():
        loop = asyncio.get_running_loop()
        try:
            q: asyncio.Queue = asyncio.Queue()

            def produce():
                try:
                    for ev in orch.stream_chat(req):
                        q.put_nowait(ev)
                finally:
                    q.put_nowait(None)

            loop.run_in_executor(None, produce)
            while True:
                ev = await q.get()
                if ev is None:
                    break
                yield _sse(ev)
        except Exception as e:  # noqa: BLE001
            yield _sse({"type": "error", "detail": str(e)})

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/search")
def search(req: SearchRequest) -> dict:
    orch = get_orchestrator()
    result = orch.search(req.query, top_k=req.top_k, rerank=req.rerank_enabled)
    return {
        "query": req.query,
        "grounding_score": result.grounding_score,
        "refused": result.refused,
        "timings_ms": result.timings_ms,
        "hits": [h.model_dump() for h in result.hits],
    }


@app.get("/api/collections")
def collections() -> dict:
    return _index_status()


@app.get("/api/traces")
def traces(limit: int = 50) -> list[dict]:
    return get_orchestrator().traces.recent(limit=limit)


@app.get("/api/stats")
def stats() -> dict:
    return get_orchestrator().traces.stats().model_dump()


@app.post("/api/eval/run")
async def eval_run(with_answers: bool = False) -> dict:
    from .eval.harness import DEFAULT_CONFIGS, run_eval

    if _eval_lock.locked():
        raise HTTPException(409, "an eval run is already in progress")
    async with _eval_lock:
        loop = asyncio.get_running_loop()
        report = await loop.run_in_executor(
            None, lambda: run_eval(DEFAULT_CONFIGS, with_answers=with_answers)
        )
    return report.model_dump(exclude={"per_item"})


@app.get("/api/eval/latest")
def eval_latest() -> dict:
    from .eval.harness import load_latest_report

    rep = load_latest_report()
    if rep is None:
        raise HTTPException(404, "no eval report yet — POST /api/eval/run first")
    return rep.model_dump(exclude={"per_item"})


@app.get("/api/eval/latest/detail")
def eval_latest_detail() -> dict:
    from .eval.harness import load_latest_report

    rep = load_latest_report()
    if rep is None:
        raise HTTPException(404, "no eval report yet")
    return rep.model_dump()


@app.get("/api/config")
def get_config() -> dict:
    return {
        "retrieval_defaults": {
            "dense_top_k": settings.dense_top_k,
            "sparse_top_k": settings.sparse_top_k,
            "rrf_k": settings.rrf_k,
            "rerank_enabled": settings.rerank_enabled,
            "final_top_k": settings.final_top_k,
            "grounding_threshold": settings.grounding_threshold,
        },
        "models": {
            "embed": settings.embed_model,
            "rerank": settings.rerank_model,
            "llm_provider": settings.llm_provider,
            "gguf": f"{settings.gguf_repo_id}/{settings.gguf_filename}"
            if settings.llm_provider == "local"
            else settings.openai_model,
        },
    }


@app.post("/api/ingest")
def ingest(force: bool = False) -> dict:
    """One-shot refresh of data + indexes (dev convenience)."""
    stats_i = run_ingest(settings, force=force)
    stats_x = run_index(settings)
    return {"ingest": stats_i, "index": stats_x}


@app.exception_handler(Exception)
async def unhandled(request, exc):  # noqa: ANN001
    log.error("unhandled_error", path=request.url.path, error=str(exc))
    return JSONResponse(status_code=500, content={"detail": "internal error"})


# ---------------------------------------------------------------------------
# Single-port production serving: the built SPA (frontend/dist) is served by
# this API process, so deployment is one process on one port. Registered last
# so /api, /health and OpenAPI routes keep priority.
# ---------------------------------------------------------------------------
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

_DIST = settings.project_root / "frontend" / "dist"
if (_DIST / "index.html").exists():
    app.mount("/assets", StaticFiles(directory=_DIST / "assets"), name="assets")

    @app.get("/{spa_path:path}", include_in_schema=False)
    def spa(spa_path: str):
        target = _DIST / spa_path
        if spa_path and target.is_file():
            return FileResponse(target)
        return FileResponse(_DIST / "index.html")
else:  # pragma: no cover - dev mode uses the vite server
    log.info("spa_not_built", hint="run `cd frontend && npm run build` for single-port serving")
