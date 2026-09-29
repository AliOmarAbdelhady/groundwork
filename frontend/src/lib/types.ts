export interface SourceRef {
  section_id: string;
  heading: string;
  source: string;
  url: string;
  excerpt: string;
  score: number;
}

export interface StageTiming {
  stage: string;
  ms: number;
}

export interface CitationDetail {
  citation: string;
  root: string;
  subsections: string[];
  valid: boolean;
  reason?: string;
}

export type ChatEvent =
  | {
      type: "meta";
      query_id: string;
      grounding_score: number;
      lexical_overlap?: number;
      exact_citation?: boolean;
      rerank_agreement?: number | null;
      corpus_version?: string;
      timings: StageTiming[];
    }
  | { type: "sources"; sources: SourceRef[] }
  | { type: "token"; text: string }
  | { type: "refused"; reason: string; grounding_score: number; lexical_overlap?: number }
  | {
      type: "done";
      answer: string;
      citations: {
        cited: string[];
        valid: string[];
        hallucinated: string[];
        n_cited: number;
        n_valid: number;
        n_hallucinated: number;
        detail?: CitationDetail[];
      };
      warning?: string;
      first_token_ms: number;
      total_ms: number;
    }
  | { type: "error"; detail: string };

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  sources?: SourceRef[];
  meta?: {
    grounding_score: number;
    timings: StageTiming[];
    refused?: boolean;
    refusal_reason?: string;
    exact_citation?: boolean;
    corpus_version?: string;
    citations?: ChatEvent extends never ? never : { valid: string[]; hallucinated: string[]; n_valid: number; n_hallucinated: number; n_cited: number; cited: string[]; detail?: CitationDetail[] };
    warning?: string;
    first_token_ms?: number;
    total_ms?: number;
  };
  streaming?: boolean;
}

export interface SearchHit {
  chunk_id: string;
  section_id: string;
  heading: string;
  source: string;
  part: string;
  parent_path: string;
  url: string;
  text: string;
  dense_score: number | null;
  sparse_score: number | null;
  rrf_score: number | null;
  rerank_score: number | null;
  final_score: number | null;
  rank: number;
}

export interface SearchResponse {
  query: string;
  grounding_score: number;
  refused: boolean;
  timings_ms: StageTiming[];
  hits: SearchHit[];
}

export interface EvalRun {
  config_name: string;
  retrieval: Record<string, number>;
  answers: Record<string, number> | null;
  subsection_evidence: number | null;
  refusal_rate_adversarial: number | null;
  answered_rate_golden: number | null;
  n_items: number;
  duration_s: number;
  per_item: EvalItem[];
}

export interface EvalItem {
  id: string;
  question?: string;
  adversarial?: boolean;
  refused?: boolean;
  grounding_score?: number;
  expected?: string[];
  retrieved?: string[];
  "hit@5"?: number;
  "mrr@10"?: number;
  answer?: string;
  citations?: string[];
  hallucinated?: string[];
  top_retrieved?: string[];
}

export interface EvalReport {
  created_at: string;
  runs: EvalRun[];
}

export interface TraceRecord {
  query_id: string;
  ts: string;
  query: string;
  refused: number;
  answer: string;
  latency_ms: number;
  first_token_ms: number | null;
  grounding_score: number;
  cited_sections: string[];
  timings: StageTiming[];
}

export interface StatsSummary {
  total_traces: number;
  refusal_rate: number;
  p50_latency_ms: number;
  p95_latency_ms: number;
  avg_first_token_ms: number | null;
  top_cited: { section_id: string; count: number }[];
}

export interface AppConfig {
  retrieval_defaults: {
    dense_top_k: number;
    sparse_top_k: number;
    rrf_k: number;
    rerank_enabled: boolean;
    final_top_k: number;
    grounding_threshold: number;
    mmr_enabled?: boolean;
  };
  models: { embed: string; rerank: string; llm_provider: string; gguf: string };
  corpus?: {
    version: string;
    ecfr_issue_date?: string | null;
    sections?: number;
    chunks?: number;
    built_at?: string | null;
  };
}

export interface RetrievalConfigOverride {
  dense_top_k?: number | null;
  sparse_top_k?: number | null;
  rrf_k?: number | null;
  rerank_enabled?: boolean | null;
  final_top_k?: number | null;
  grounding_threshold?: number | null;
}

export interface HealthStatus {
  status: string;
  llm_provider: string;
  indexes: { qdrant_points?: number; bm25_ready?: boolean; bm25_docs?: number; error?: string };
}
