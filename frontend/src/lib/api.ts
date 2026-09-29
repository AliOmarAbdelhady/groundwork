import type {
  AppConfig,
  ChatEvent,
  EvalReport,
  HealthStatus,
  RetrievalConfigOverride,
  SearchResponse,
  StatsSummary,
  TraceRecord,
} from "./types";

const BASE = import.meta.env.DEV ? "" : ""; // vite proxy handles /api in dev

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) throw new Error(`${path}: ${res.status} ${await res.text()}`);
  return res.json();
}

export const api = {
  health: () => get<HealthStatus>("/health"),
  config: () => get<AppConfig>("/api/config"),
  collections: () => get<HealthStatus["indexes"]>("/api/collections"),
  stats: () => get<StatsSummary>("/api/stats"),
  traces: (limit = 50) => get<TraceRecord[]>(`/api/traces?limit=${limit}`),
  evalLatest: () => get<EvalReport>("/api/eval/latest"),
  search: async (query: string, top_k = 10, rerank_enabled = false) => {
    const res = await fetch("/api/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, top_k, rerank_enabled }),
    });
    if (!res.ok) throw new Error(`search failed: ${res.status}`);
    return (await res.json()) as SearchResponse;
  },
  runEval: async (withAnswers = false) => {
    const res = await fetch(`/api/eval/run${withAnswers ? "?with_answers=true" : ""}`, {
      method: "POST",
    });
    if (!res.ok) throw new Error(`eval failed: ${res.status} ${await res.text()}`);
    return res.json();
  },
};

/** POST /api/chat and consume its SSE stream. */
export async function streamChat(
  query: string,
  history: { role: string; content: string }[],
  config: RetrievalConfigOverride,
  onEvent: (ev: ChatEvent) => void,
  signal?: AbortSignal
): Promise<void> {
  const res = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, history, config }),
    signal,
  });
  if (!res.ok || !res.body) {
    let detail = `chat failed: ${res.status}`;
    try {
      const err = await res.json();
      if (typeof err.detail === "string") detail = err.detail;
      else if (Array.isArray(err.detail) && err.detail[0]?.msg) detail = err.detail[0].msg;
    } catch { /* keep status text */ }
    throw new Error(detail);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    const parts = buf.split("\n\n");
    buf = parts.pop() ?? "";
    for (const part of parts) {
      const line = part.split("\n").find((l) => l.startsWith("data: "));
      if (!line) continue;
      try {
        onEvent(JSON.parse(line.slice(6)));
      } catch {
        // ignore malformed keep-alive fragments
      }
    }
  }
}
