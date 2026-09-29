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
          canonical Section model        weighted RRF fusion + exact-citation
          + corpus manifest (hashes)     resolution (queries naming a section)
                   ▼                                    |
     hierarchical parent/child                    cross-encoder rerank
     chunking (children ~220 tok,                     |
     parents = whole sections)                        ▼
                   |                          MMR diversity + per-section cap
        ┌──────────┴──────────┐                        |
        ▼                     ▼                        ▼
   Qdrant (dense)        bm25s (sparse)     parent expansion + token budget
                                              (boundary-safe packing)
                                                   |
                                            grounding guard ─► refuse
                                            (no LLM call)     (SSE)
                                                   ▼
                                    LLM (Groq / any OpenAI-compatible endpoint
                                     / local llama.cpp), streamed over SSE
                                                   ▼
                     exact citation validation (subsection chain must exist
                     in the prompt context) + SQLite trace store
                                                   |
                          FastAPI (chat, search, eval, traces, stats)
                                                   |
                     React UI — Chat · Explore · Evaluation · Insights
```

Design notes:

- **Hierarchical small-to-big retrieval.** Children are token-budgeted
  sentence-window chunks (trailing overlap measured with the real tokenizer)
  that get embedded and searched; hits expand to whole parent sections for
  generation. Search precision without losing statutory context.
- **Hybrid retrieval with exact-citation resolution.** Dense embeddings catch
  paraphrases; BM25 catches exact identifiers (§ numbers, chemical names).
  Weighted reciprocal-rank fusion merges both; a query that names a
  regulation (`1910.132`, `§ 1926.501(b)`) resolves directly against the
  corpus and outranks both legs — identifiers are not paraphrased.
- **Diversity-aware selection.** After cross-encoder reranking, greedy MMR
  over the child vectors (with a per-section cap) picks the final evidence,
  so one long section cannot crowd out independent rules a multi-part
  question needs. Every candidate is hydrated from a canonical chunk store —
  no hit can reach the prompt or the UI with missing text or metadata.
- **Grounded refusal.** A grounding score (dense similarity + lexical
  coverage + exact-citation match, calibrated on the evaluation set) gates
  every query; below threshold — or below a hard lexical-overlap floor for
  out-of-domain vocabulary — the system refuses without calling the LLM.
- **Exact citation validation.** A citation is verified only when its section
  was included in the final prompt context *and* the cited subsection chain
  (e.g. `(b)(1)(ii)`) actually exists in that section's statutory text. A
  fabricated `1910.132(z)` fails this check, is flagged in the UI, and is
  counted in evaluation. For OSHA publications, cited page ranges must
  overlap the page groups the model actually saw.
- **Corpus provenance.** Every ingest stamps a version manifest (source
  URLs, content hashes, eCFR issue date, parser/chunker versions); the
  corpus version travels with every answer and is exposed on `/health` and
  `/api/config`.
- **Injection-resistant prompting.** Retrieved text is fenced as source data
  and the system prompt instructs the model to ignore any directive found
  inside it.
- **Provider-agnostic generation.** Default is any OpenAI-compatible
  endpoint (Groq, OpenAI, vLLM, …); a fully local llama.cpp mode is available
  for offline deployments.

## Evaluation

Golden set of 90 questions (70 in-domain with ground-truth sections —
including exact-subsection labels for the decisive clause — and 20
adversarial must-refuse items, among them near-domain traps from adjacent
legal domains: wage law, workers' compensation, EPA/DOT/FAA rules, MSHA).
All metrics are deterministic and computed at section level.

| Configuration | hit@1 | hit@5 | hit@10 | MRR@10 | nDCG@10 | Subsection evidence | Golden answered | Adversarial refused |
|---|---|---|---|---|---|---|---|---|
| Dense only (bge-small) | 0.514 | 0.929 | 0.986 | 0.681 | 0.753 | 82% | 100% | 75% |
| BM25 only | 0.429 | 0.786 | 0.829 | 0.580 | 0.620 | 91% | refused by design* | 100% |
| Hybrid (weighted RRF) | 0.571 | 0.886 | 0.957 | 0.707 | 0.762 | 91% | 100% | 65% |
| **Hybrid + rerank (default)** | **0.557** | **0.914** | **0.986** | **0.713** | **0.765** | **91%** | **100%** | **90%** |

\* The grounding guard reads dense similarity; with the dense leg disabled
every query is refused — the A/B number is not meaningful for BM25-only.

The refusal guard combines three signals: dense similarity + lexical
coverage (threshold 0.52, calibrated on the score distribution of this set),
a hard lexical-overlap floor for out-of-domain vocabulary, and — when
reranking is enabled — cross-encoder agreement (the best candidate must
score above 0). That last signal is what separates near-domain traps from
genuine OSHA questions: on the golden set the adversarial items' best
cross-encoder scores top out at −1.4 while the weakest in-domain item
reaches +1.5.

An integration test enforces a multi-metric regression gate (hit@1 ≥ 0.50,
hit@5 ≥ 0.88, nDCG@10 ≥ 0.72, adversarial refusal ≥ 0.85, golden answered
≥ 0.98 — all simultaneously) so retrieval or guard quality cannot silently
degrade. Known limits: "building-code fire sprinkler" and "MSHA miner
training" questions retrieve genuinely similar OSHA fire-safety and
training-requirements material and can pass the guard — adjacent-domain
vocabulary the corpus legitimately contains.

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

For a long-running self-hosted deployment on a Linux box, a pair of systemd
user units works well — one for the API (`Restart=on-failure`,
`Environment=PYTHONPATH=backend`), and optionally one for a tunnel or
reverse proxy in front of it. Enable lingering (`loginctl enable-linger
$USER`) so the services survive reboots without an active login session.
Generation secrets stay in `.env` or the service's environment — never in
the repository.

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
| `GW_RERANK_ENABLED` / `GW_RERANK_TOP_N` | true / 24 | cross-encoder reranking and pool size |
| `GW_FINAL_TOP_K` | 6 | evidence units packed into the LLM context |
| `GW_MMR_ENABLED` / `GW_MMR_LAMBDA` | true / 0.7 | diversity selection (0 = pure relevance) |
| `GW_MAX_CHILDREN_PER_SECTION` | 2 | per-section cap in diversity selection |
| `GW_CONTEXT_BUDGET_TOKENS` | 2400 | context assembly budget (boundary-safe packing) |
| `GW_GROUNDING_THRESHOLD` | 0.52 | below → refuse without an LLM call |
| `GW_OOD_MIN_OVERLAP` | 0.06 | hard lexical floor for out-of-domain refusal |
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
  domain. Snapshot: 2026-09-25 issue; 1,465 sections ingested.
- OSHA publications (osha.gov) — 30 booklets, fact sheets, quick cards, and
  posters, URL-verified; U.S. public domain. Page citations keep the original
  PDF page numbers.

Re-run `make ingest && make index` to refresh to the latest eCFR issue; the
ingest stamps a corpus manifest (`data/processed/manifest.json`) recording
source URLs, content hashes, and parser/chunker versions, and every answer
reports the `corpus_version` it was produced from.

## License

MIT. Not affiliated with or endorsed by OSHA or the U.S. Department of Labor.
