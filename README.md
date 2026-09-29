# GroundWork 🦺

**A production-grade, hybrid RAG system over U.S. workplace-safety regulations (29 CFR / OSHA) — with verifiable citations, grounded refusal, a full evaluation harness, and live observability.**

GroundWork answers real questions that workers, foremen, and safety managers ask —
*"At what height do construction workers need fall protection?"*, *"Who pays for
PPE?"*, *"How fast must a fatality be reported to OSHA?"* — and answers them from
the **official eCFR text plus OSHA publications**, citing the exact statutory unit
(`29 CFR 1910.132(b)`) for every claim. Questions outside the corpus are
**refused, not guessed**. Everything — embeddings, reranking, and generation —
runs **locally on CPU**.

> ⚠️ Not legal advice. GroundWork surfaces the official regulatory text and
> links to the authoritative source on ecfr.gov / osha.gov.

---

## Why this isn't another "chat with PDFs" demo

| Capability | GroundWork |
|---|---|
| **Corpus** | 1,869 sections / 13,562 chunks / ~3.9M tokens from the official **eCFR versioner API** (Title 29, OSHA chapter XVII) + 29 **URL-verified OSHA publications** — public-domain, refreshable with one command |
| **Chunking** | Hierarchical **parent/child (small-to-big)**: token-budgeted sentence-window children are embedded; whole sections are what the LLM sees and the UI displays |
| **Retrieval** | **Hybrid dense (bge-small ONNX) + BM25**, **RRF fusion**, **cross-encoder reranking** — every score exposed per hit |
| **Trust** | Inline citations are **parsed and validated against the retrieved set**; unverifiable citations are flagged in the UI and counted in eval |
| **Refusal** | A grounding guard (dense-similarity + lexical coverage + rerank confidence) makes off-domain questions fail fast **without calling the LLM** |
| **Evaluation** | 54-item golden set (46 in-domain + 8 adversarial must-refuse), deterministic metrics — hit@k, MRR, nDCG, citation precision, groundedness — **A/B'd across 4 retrieval configs** with a CI regression floor |
| **Observability** | structlog JSON logs + SQLite trace store → live Insights dashboard (p50/p95 latency, TTFT, refusal rate, most-cited sections) |
| **Ops** | FastAPI + SSE streaming, React/TypeScript/Tailwind UI, Makefile, Dockerfile, docker-compose, GitHub Actions CI |

## Measured results (CPU-only, golden set of 64 questions)

Retrieval quality at section level — full per-item report in
[`docs/EVALUATION.md`](docs/EVALUATION.md):

| Config | hit@5 | hit@10 | MRR@10 | nDCG@10 | Golden answered | Adversarial refusal |
|---|---|---|---|---|---|---|
| Dense only (bge-small) | 0.946 | 0.964 | 0.703 | 0.765 | 100% | 88% |
| BM25 only | 0.696 | 0.732 | 0.511 | 0.552 | 0%* | 100% |
| Hybrid (weighted RRF) | 0.875 | 0.946 | 0.693 | 0.736 | 100% | 75% |
| **Hybrid + cross-encoder rerank (default)** | **0.929** | **0.964** | **0.712** | **0.766** | **100%** | **88%** |

\* BM25-only lacks dense-similarity signal, so the grounding guard (correctly)
refuses everything in that configuration — the guard is designed to distrust
retrieval without semantic confidence.

Notes from the harness: (1) the OSHA publication booklets often rank top-1 for
plain-language questions while the statutory section lands in top-5 — exactly
the cross-source behavior a worker-facing assistant should have; (2) the one
remaining adversarial miss ("minimum wage in California") retrieves Davis-Bacon
prevailing-wage sections — related vocabulary, wrong domain, and documented as
a known edge case.

## Architecture

```
            OFFLINE PIPELINE                                   ONLINE SERVING
────────────────────────────────────                 ─────────────────────────────────
 eCFR API (title 29 XML)   OSHA.gov PDFs              query
        │                       │                        │
        ▼                       ▼                        ▼
  streaming lxml          pypdf + noise            embed (bge-small, ONNX)
        │                   filter                    + BM25 (bm25s)
        └─────────┬─────────────┘                       │
                ▼                                      ▼
        canonical Section model                   RRF fusion
                ▼                                      │
     hierarchical chunker                       cross-encoder rerank
   (children ~220 tok, parents                         │
    = whole sections, citation                        ▼
    metadata on every chunk)            small-to-big parent expansion
                │                                      │
       ┌────────┴────────┐                       grounding guard ──► refuse
       ▼                 ▼                       (no LLM call)     (SSE)
  Qdrant (dense)    bm25s (sparse)                     │
  + payloads        + id map                           ▼
                                            LLM: Qwen2.5-3B (llama.cpp) or
                                                 any OpenAI-compatible endpoint
                                                  │  streamed tokens (SSE)
                                                  ▼
                                    citation validation + trace store (SQLite)
                                                  │
                       FastAPI (/api/chat, /api/search, /api/eval, /api/traces)
                                                  │
                              React UI — Chat · Explore · Evaluation · Insights
```

**New to RAG?** Start with [`docs/RAG-GUIDE.md`](docs/RAG-GUIDE.md) — the full pipeline, every knob, and evaluation explained from zero, mapped to this codebase.

Full design in [`docs/PLAN.md`](docs/PLAN.md); requirements in
[`docs/SPEC.md`](docs/SPEC.md); corpus provenance in
[`docs/DATA_CARD.md`](docs/DATA_CARD.md).

## Quickstart

```bash
# 0) prerequisites: python 3.11–3.12 + uv, node 20+
uv venv && uv pip install -p .venv .        # or: make install
cd frontend && npm install && npm run build && cd ..

# 1) data + indexes (downloads ~28MB eCFR XML + ~30 OSHA PDFs, embeds 13.5k chunks)
make ingest          # ~45 s
make index           # minutes on CPU; crash-resumable

# 2) configure generation — Groq (free tier) or any OpenAI-compatible endpoint
cp .env.example .env
#   GW_LLM_PROVIDER=openai_compatible
#   GW_OPENAI_BASE_URL=https://api.groq.com/openai/v1
#   GW_OPENAI_API_KEY=gsk_...
#   GW_OPENAI_MODEL=openai/gpt-oss-120b

# 3) run — ONE process serves app + API on :8000 (measured: ~1 s to first token)
make serve                                 # http://127.0.0.1:8000

# 4) evaluate + publish the report
make eval && make report
```

Dev mode with hot reload: `cd frontend && npm run dev` (UI on :5173,
proxying to the API). Deployment options (VPS, Docker Compose,
Railway/Render/Fly): [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md).

### Fully-offline option (local llama.cpp)
No API key? Run generation locally with Qwen2.5-3B (Q4_K_M, ~2.1 GB):
`make download-llm` and set `GW_LLM_PROVIDER=local`. Expect ~5–15 tok/s on
normal CPUs.

## Repo layout

```
backend/
  app/
    config.py            all knobs (GW_* env vars, .env)
    schemas.py           pydantic domain + API models
    ingest/              eCFR + PDF download/parse, hierarchical chunker, pipeline
    index/               fastembed embedder, Qdrant store, bm25s index, builder
    rag/                 hybrid retrieval, generation, citations, orchestrator
    eval/                golden set, metrics, harness, report
    obs/                 structlog + SQLite traces
    main.py cli.py       FastAPI app / CLI (ingest|index|eval|serve|report)
  tests/                 unit (no deps) + integration (real indexes, hit@5 gate)
frontend/                Vite + React 18 + TS + Tailwind v4 app
docs/                    SPEC, PLAN, TASKS, DATA_CARD, EVALUATION
docker/                  Dockerfile; docker-compose.yml at repo root
```

## API surface

`GET /health` · `POST /api/chat` (SSE stream: `meta → sources → token* → done|refused`)
· `POST /api/search` · `GET /api/collections` · `GET /api/traces` · `GET /api/stats`
· `POST /api/eval/run` · `GET /api/eval/latest` · `GET /api/config` — full OpenAPI
spec at `/docs` when serving.

## Configuration highlights

| Knob | Default | Notes |
|---|---|---|
| `GW_GROUNDING_THRESHOLD` | 0.52 | below → refuse without LLM call |
| `GW_RERANK_ENABLED` | true | cross-encoder over fused candidates |
| `GW_DENSE_TOP_K` / `GW_SPARSE_TOP_K` | 30 / 30 | candidate depth per leg |
| `GW_FINAL_TOP_K` | 6 | sections packed into the LLM context |
| `GW_CONTEXT_BUDGET_TOKENS` | 3500 | small-to-big context budget |
| `GW_QDRANT_URL` | — | set to switch from embedded Qdrant to a server |

## Testing & CI

```bash
make test    # unit (fast) + integration (needs `make ingest index`;
             # stop `make serve` first — embedded Qdrant takes an exclusive lock)
make lint    # ruff
```

GitHub Actions (`.github/workflows/ci.yml`): lint → unit → **ingest+index on the
real public data** → integration (including the `hit@5 ≥ 0.70` regression floor)
→ golden-set eval.

## License & data

Code: MIT. Data: U.S. public domain (eCFR + OSHA publications). This project is
not affiliated with or endorsed by OSHA/DOL.
