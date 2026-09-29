import { useState } from "react";
import { Search, ExternalLink, Gauge } from "lucide-react";
import { api } from "../lib/api";
import type { SearchResponse } from "../lib/types";

export default function ExplorePage() {
  const [q, setQ] = useState("");
  const [rerank, setRerank] = useState(true);
  const [res, setRes] = useState<SearchResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  async function run(e?: React.FormEvent) {
    e?.preventDefault();
    if (!q.trim()) return;
    setBusy(true);
    setErr(null);
    try {
      setRes(await api.search(q, 10, rerank));
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  }

  const maxSparse = res ? Math.max(...res.hits.map((h) => h.sparse_score ?? 0), 1) : 1;

  return (
    <div className="mx-auto max-w-4xl px-6 py-10">
      <header className="mb-6">
        <h1 className="font-display text-2xl font-bold tracking-tight text-ink-200">
          Explore the retrieval layer
        </h1>
        <p className="mt-1.5 text-[13.5px] text-ink-400">
          Run any query through hybrid retrieval and inspect every score: dense
          similarity, BM25, RRF fusion, and the cross-encoder — before any LLM
          touches the results.
        </p>
      </header>

      <form onSubmit={run} className="glass-panel flex items-center gap-2 rounded-2xl p-2.5">
        <Search size={17} className="ml-2 shrink-0 text-ink-500" />
        <input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="e.g. respirator medical evaluation"
          className="min-w-0 flex-1 bg-transparent px-2 py-2 text-[14.5px] text-ink-200 placeholder:text-ink-500 focus:outline-none"
        />
        <label className="flex cursor-pointer items-center gap-2 rounded-lg border border-ink-700 px-3 py-1.5 text-[12px] text-ink-300">
          <input type="checkbox" checked={rerank} onChange={(e) => setRerank(e.target.checked)} className="h-3.5 w-3.5 accent-amber-500" />
          rerank
        </label>
        <button
          disabled={busy}
          className="rounded-xl px-4 py-2 text-[13px] font-semibold text-ink-950 disabled:opacity-50"
          style={{ background: "linear-gradient(135deg,#f5a524,#fb7822)" }}
        >
          {busy ? "searching…" : "Search"}
        </button>
      </form>

      {err && <p className="mt-4 rounded-lg bg-danger/10 p-3 text-[13px] text-danger">{err}</p>}

      {res && (
        <>
          <div className="mt-4 flex flex-wrap items-center gap-2 text-[11px] text-ink-400">
            <span className="inline-flex items-center gap-1.5 rounded-full border border-ink-700 px-2.5 py-1 font-mono">
              <Gauge size={11} className="text-amber-glow" /> grounding {res.grounding_score.toFixed(2)}
            </span>
            {res.timings_ms.map((t) => (
              <span key={t.stage} className="rounded-full bg-ink-800/60 px-2 py-1 font-mono">
                {t.stage} {Math.round(t.ms)}ms
              </span>
            ))}
            <span className="rounded-full bg-ink-800/60 px-2 py-1 font-mono">
              {res.hits.length} hits
            </span>
          </div>

          <div className="mt-5 space-y-3">
            {res.hits.map((h) => (
              <article key={h.chunk_id} className="glass-panel animate-fade-up rounded-xl p-4">
                <div className="flex items-start justify-between gap-3">
                  <div className="min-w-0">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="font-mono text-[12px] font-semibold text-amber-glow">
                        #{h.rank} · {h.section_id}
                      </span>
                      <span className={`rounded px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide ${
                        h.source === "ecfr" ? "bg-amber-glow/10 text-amber-glow" : "bg-signal/10 text-signal"
                      }`}>
                        {h.source === "ecfr" ? "eCFR" : "OSHA PDF"}
                      </span>
                    </div>
                    <h3 className="mt-1 truncate text-[14px] font-medium text-ink-200">{h.heading}</h3>
                    <p className="mt-0.5 text-[11px] text-ink-500">{h.parent_path}</p>
                  </div>
                  {h.url && (
                    <a href={h.url} target="_blank" rel="noreferrer" className="shrink-0 rounded-lg border border-ink-700 p-2 text-ink-400 transition-colors hover:border-sky-info/40 hover:text-sky-info">
                      <ExternalLink size={13} />
                    </a>
                  )}
                </div>

                <p className="mt-3 line-clamp-3 text-[13px] leading-relaxed text-ink-300">{h.text}</p>

                <div className="mt-3 space-y-1.5">
                  <ScoreRow label="dense" v={h.dense_score} domain={[0, 1]} />
                  <ScoreRow label="bm25" v={h.sparse_score} domain={[0, maxSparse]} />
                  <ScoreRow label="rerank" v={h.rerank_score ? sigmoid(h.rerank_score) : null} domain={[0, 1]} />
                </div>
              </article>
            ))}
          </div>
        </>
      )}
    </div>
  );
}

function sigmoid(x: number) {
  return 1 / (1 + Math.exp(-x));
}

function ScoreRow({ label, v, domain }: { label: string; v: number | null; domain: [number, number] }) {
  const frac = v == null ? 0 : Math.max(0.02, Math.min(1, (v - domain[0]) / (domain[1] - domain[0])));
  const colors: Record<string, string> = {
    dense: "bg-sky-info/60",
    bm25: "bg-signal/60",
    rerank: "bg-amber-glow/70",
  };
  return (
    <div className="flex items-center gap-2.5">
      <span className="w-14 shrink-0 font-mono text-[10.5px] text-ink-500">{label}</span>
      <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-ink-800">
        {v != null && (
          <div className={`h-full rounded-full ${colors[label]}`} style={{ width: `${frac * 100}%` }} />
        )}
      </div>
      <span className="w-14 shrink-0 text-right font-mono text-[10.5px] text-ink-400">
        {v == null ? "—" : v.toFixed(3)}
      </span>
    </div>
  );
}
