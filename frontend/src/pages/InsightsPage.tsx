import { useEffect, useState } from "react";
import { Activity, ShieldAlert, Timer, Zap, Layers } from "lucide-react";
import { api } from "../lib/api";
import type { StatsSummary, TraceRecord } from "../lib/types";

export default function InsightsPage() {
  const [stats, setStats] = useState<StatsSummary | null>(null);
  const [traces, setTraces] = useState<TraceRecord[]>([]);

  useEffect(() => {
    const load = () => {
      api.stats().then(setStats).catch(() => {});
      api.traces(60).then(setTraces).catch(() => {});
    };
    load();
    const t = setInterval(load, 10_000);
    return () => clearInterval(t);
  }, []);

  return (
    <div className="mx-auto max-w-5xl px-6 py-10">
      <header className="mb-6">
        <h1 className="font-display text-2xl font-bold tracking-tight text-ink-200">
          Insights
        </h1>
        <p className="mt-1.5 text-[13.5px] text-ink-400">
          Live observability from the SQLite trace store — every query's latency,
          grounding score, refusal state, and cited sections. Refreshes every 10 s.
        </p>
      </header>

      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <StatCard icon={<Layers size={15} />} label="Total queries" value={stats ? stats.total_traces.toLocaleString() : "—"} />
        <StatCard
          icon={<Timer size={15} />}
          label="p50 / p95 latency"
          value={stats ? `${fmtMs(stats.p50_latency_ms)} / ${fmtMs(stats.p95_latency_ms)}` : "—"}
        />
        <StatCard
          icon={<Zap size={15} />}
          label="avg time to first token"
          value={stats?.avg_first_token_ms != null ? fmtMs(stats.avg_first_token_ms) : "—"}
        />
        <StatCard
          icon={<ShieldAlert size={15} />}
          label="refusal rate"
          value={stats ? `${(stats.refusal_rate * 100).toFixed(1)}%` : "—"}
          warn={(stats?.refusal_rate ?? 0) > 0.4}
        />
      </div>

      {stats && stats.top_cited.length > 0 && (
        <div className="glass-panel mt-4 rounded-2xl p-5">
          <h2 className="mb-3 flex items-center gap-2 font-display text-[14px] font-semibold text-ink-200">
            <Activity size={14} className="text-amber-glow" /> Most-cited sections
          </h2>
          <div className="space-y-2">
            {stats.top_cited.map((c) => {
              const max = stats.top_cited[0].count;
              return (
                <div key={c.section_id} className="flex items-center gap-3">
                  <span className="w-52 shrink-0 truncate font-mono text-[11.5px] text-ink-300">
                    {c.section_id}
                  </span>
                  <div className="h-2 flex-1 overflow-hidden rounded-full bg-ink-800">
                    <div
                      className="h-full rounded-full"
                      style={{
                        width: `${(c.count / max) * 100}%`,
                        background: "linear-gradient(90deg,#f5a524,#fb7822)",
                      }}
                    />
                  </div>
                  <span className="w-8 text-right font-mono text-[11px] text-ink-500">{c.count}</span>
                </div>
              );
            })}
          </div>
        </div>
      )}

      <div className="glass-panel mt-4 overflow-x-auto rounded-2xl">
        <div className="border-b border-ink-700 px-4 py-3">
          <h2 className="font-display text-[14px] font-semibold text-ink-200">Recent traces</h2>
        </div>
        {traces.length === 0 ? (
          <p className="px-4 py-6 text-[13px] text-ink-500">
            No traces yet — ask something in Chat first.
          </p>
        ) : (
          <table className="w-full text-[12px]">
            <thead>
              <tr className="border-b border-ink-700 text-left text-ink-400">
                <th className="px-4 py-2.5 font-semibold">time</th>
                <th className="px-3 py-2.5 font-semibold">query</th>
                <th className="px-3 py-2.5 font-mono font-semibold">latency</th>
                <th className="px-3 py-2.5 font-mono font-semibold">ttft</th>
                <th className="px-3 py-2.5 font-mono font-semibold">grounding</th>
                <th className="px-3 py-2.5 font-semibold">state</th>
                <th className="px-3 py-2.5 font-semibold">citations</th>
              </tr>
            </thead>
            <tbody>
              {traces.map((t) => (
                <tr key={t.query_id} className="border-b border-ink-800 text-ink-300">
                  <td className="px-4 py-2 font-mono text-[10.5px] text-ink-500">
                    {new Date(t.ts).toLocaleTimeString()}
                  </td>
                  <td className="max-w-[280px] truncate px-3 py-2">{t.query}</td>
                  <td className="px-3 py-2 font-mono">{fmtMs(t.latency_ms)}</td>
                  <td className="px-3 py-2 font-mono">{t.first_token_ms != null ? fmtMs(t.first_token_ms) : "—"}</td>
                  <td className="px-3 py-2 font-mono">{t.grounding_score.toFixed(2)}</td>
                  <td className="px-3 py-2">
                    {t.refused ? (
                      <span className="rounded bg-danger/10 px-1.5 py-0.5 text-[10.5px] text-danger">refused</span>
                    ) : (
                      <span className="rounded bg-signal/10 px-1.5 py-0.5 text-[10.5px] text-signal">answered</span>
                    )}
                  </td>
                  <td className="max-w-[200px] truncate px-3 py-2 font-mono text-[10.5px] text-ink-500">
                    {t.cited_sections.join(", ") || "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}

function fmtMs(ms: number): string {
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${Math.round(ms)}ms`;
}

function StatCard({
  icon, label, value, warn,
}: {
  icon: React.ReactNode; label: string; value: string; warn?: boolean;
}) {
  return (
    <div className="glass-panel rounded-2xl p-4">
      <div className="flex items-center gap-2 text-[11px] font-semibold uppercase tracking-wider text-ink-400">
        <span className="text-amber-glow">{icon}</span>
        {label}
      </div>
      <div className={`mt-2 font-display text-[22px] font-bold ${warn ? "text-danger" : "text-ink-200"}`}>
        {value}
      </div>
    </div>
  );
}
