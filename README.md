# GroundWork

Hybrid retrieval-augmented generation (RAG) system over U.S. workplace-safety
regulations — 29 CFR (OSHA) from the official eCFR API plus a curated set of
OSHA publications — with verifiable citations, grounded refusal, a
deterministic evaluation harness, and live query observability.

Workers, foremen, and safety managers ask questions like *"At what height do
construction workers need fall protection?"* or *"Who pays for PPE?"*.
GroundWork answers from the official regulatory text, cites the exact
statutory unit for every claim (`29 CFR 1910.132(b)`), refuses questions the
corpus cannot support, and links every answer back to the authoritative
source on ecfr.gov / osha.gov.

> Not legal advice. GroundWork surfaces the official text; it does not
> interpret the law for specific situations.

## How it works

```
OFFLINE                                              ONLINE
------                                               ------
eCFR Title 29 XML          OSHA publication PDFs     question
        |                        |                       |
  streaming lxml           pypdf + noise           embed (bge-small, ONNX)
        |                   filtering                + BM25 (bm25s)
        └──────────┬────────────┘                       |
                   ▼                                    ▼
          canonical Section model                    weighted RRF fusion
                   ▼                                    |
     hierarchical parent/child                    cross-encoder rerank
     chunking (children ~220 tok,                     |
     parents = whole sections)                        ▼
                   |                          parent expansion + token budget
        ┌──────────┴──────────┐                        |
        ▼                     ▼                  grounding guard ─► refuse
   Qdrant (dense)        bm25s (sparse)          (no LLM call)     (SSE)
                                                   ▼
                                    LLM (Groq / any OpenAI-compatible endpoint
                                     / local llama.cpp), streamed over SSE
                                                   ▼
                              citation validation + SQLite trace store
                                                   |
                          FastAPI (chat, search, eval, traces, stats)
                                                   |
                     React UI — Chat · Explore · Evaluation · Insights
```

Design notes:

- **Hierarchical small-to-big retrieval.** Children are token-budgeted
  sentence-window chunks that get embedded and searched; hits expand to
  whole parent sections for generation. Search precision without losing
  statutory context.
- **Hybrid retrieval.** Dense embeddings catch paraphrases; BM25 catches
  exact identifiers (§ numbers, chemical names). Weighted reciprocal-rank
  fusion merges both, and a cross-encoder reranks the fused candidates.
- **Grounded refusal.** A grounding score (dense similarity + lexical
  coverage, calibrated on the evaluation set) gates every query; below
  threshold the system refuses without calling the LLM.
- **Citation validation.** Every citation in a generated answer is checked
  against the sections actually retrieved; unverified citations are flagged
  in the UI and counted in evaluation.
- **Provider-agnostic generation.** Default is any OpenAI-compatible
  endpoint (Groq, OpenAI, vLLM, …); a fully local llama.cpp mode is available
  for offline deployments.

## Evaluation

Golden set of 64 questions (56 in-domain with ground-truth sections, 8
adversarial must-refuse), run across four retrieval configurations. All
metrics are deterministic and computed at section level.

| Configuration | hit@5 | hit@10 | MRR@10 | nDCG@10 | Golden answered | Adversarial refused |
|---|---|---|---|---|---|---|
| Dense only | 0.946 | 0.964 | 0.703 | 0.765 | 100% | 88% |
| BM25 only | 0.696 | 0.732 | 0.511 | 0.552 | — | 100% |
| Hybrid (weighted RRF) | 0.875 | 0.946 | 0.693 | 0.736 | 100% | 75% |
| **Hybrid + rerank (default)** | **0.929** | **0.964** | **0.712** | **0.766** | **100%** | **88%** |

An integration test enforces a hit@5 regression floor (≥ 0.70) so retrieval
quality cannot silently degrade. The one adversarial miss ("minimum wage in
California") retrieves Davis–Bacon prevailing-wage sections — adjacent
vocabulary in the wrong domain — and is a known limitation of the lexical
component of the guard.

## Quickstart

Requirements: Python 3.11–3.12 with [uv](https://docs.astral.sh/uv/),
Node.js 20+.

```bash
uv venv && uv pip install -p .venv .
cd frontend && npm install && npm run build && cd ..

make ingest    # eCFR + OSHA PDFs -> sections/chunks (~45 s)
make index     # Qdrant + BM25 indexes (resumable if interrupted)

cp .env.example .env    # configure generation, see below

make serve     # app + API on one port: http://127.0.0.1:8000
```

Frontend development with hot reload: `cd frontend && npm run dev`
(UI on :5173, proxying the API).

### Generation configuration

Any OpenAI-compatible endpoint works. Groq example (`.env`):

```dotenv
GW_LLM_PROVIDER=openai_compatible
GW_OPENAI_BASE_URL=https://api.groq.com/openai/v1
GW_OPENAI_API_KEY=gsk_...
GW_OPENAI_MODEL=openai/gpt-oss-120b
```

Fully offline alternative: `make download-llm` and `GW_LLM_PROVIDER=local`
(Qwen2.5-3B via llama.cpp).

### Deployment

Single process serves the SPA and the API on one port. Docker Compose
(standalone Qdrant) is included. On a PaaS, the start command is
`PYTHONPATH=backend python -m app.cli serve --host 0.0.0.0 --port $PORT`
with the `GW_*` variables set and a persistent disk mounted at `data/`.

## API

`GET /health` · `POST /api/chat` (SSE: `meta → sources → token* →
done | refused`) · `POST /api/search` · `GET /api/collections` ·
`GET /api/traces` · `GET /api/stats` · `POST /api/eval/run` ·
`GET /api/eval/latest` · `GET /api/config`. OpenAPI spec at `/docs`.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `GW_LLM_PROVIDER` | `local` | `local` (llama.cpp) or `openai_compatible` |
| `GW_OPENAI_BASE_URL` / `_API_KEY` / `_MODEL` | — | hosted endpoint settings |
| `GW_DENSE_TOP_K` / `GW_SPARSE_TOP_K` | 30 / 30 | candidate depth per retrieval leg |
| `GW_RRF_K` | 60 | reciprocal-rank fusion smoothing |
| `GW_RERANK_ENABLED` | true | cross-encoder reranking |
| `GW_FINAL_TOP_K` | 6 | sections packed into the LLM context |
| `GW_CONTEXT_BUDGET_TOKENS` | 2400 | context assembly budget |
| `GW_GROUNDING_THRESHOLD` | 0.52 | below → refuse without an LLM call |
| `GW_QDRANT_URL` | — | embedded local mode unless set |

## Project structure

```
backend/app/
  ingest/    eCFR + PDF acquisition and parsing, hierarchical chunker
  index/     embedding service, Qdrant store, BM25 index, index builder
  rag/       hybrid retrieval, reranking, generation, citations, orchestration
  eval/      golden set, metrics, harness, report rendering
  obs/       structured logging, SQLite trace store
  main.py    FastAPI application (also serves the built SPA)
  cli.py     ingest | index | eval | report | serve | download-llm
backend/tests/  unit + integration suites (retrieval regression gate)
frontend/    React 18 + TypeScript + Tailwind CSS application
```

## Testing

```bash
make test    # unit + integration (integration requires built indexes;
             # stop the API server first — embedded Qdrant takes an
             # exclusive storage lock)
make lint    # ruff
```

Continuous integration (`.github/workflows/ci.yml`): lint, unit tests, a
full ingest + index run against the live public data sources, the
integration suite with the retrieval regression floor, and the golden-set
evaluation.

## Data

- eCFR versioner API — Title 29, OSHA chapter (parts 1900–2099), U.S. public
  domain. Snapshot: 2026-09-24 issue; 1,465 sections ingested.
- OSHA publications (osha.gov) — 29 booklets, fact sheets, and quick cards,
  URL-verified; U.S. public domain.

Re-run `make ingest && make index` to refresh to the latest eCFR issue.

## License

MIT. Not affiliated with or endorsed by OSHA or the U.S. Department of Labor.
