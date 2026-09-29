import { useEffect, useState } from "react";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend, CartesianGrid,
} from "recharts";
import { Play, RefreshCw } from "lucide-react";
import { api } from "../lib/api";
import type { EvalReport } from "../lib/types";

const CFG_LABEL: Record<string, string> = {
  dense_only: "Dense only",
  bm25_only: "BM25 only",
  hybrid: "Hybrid (RRF)",
  hybrid_rerank: "Hybrid + rerank",
};

const RETRIEVAL_METRICS = ["hit@1", "hit@3", "hit@5", "hit@10", "mrr@10", "ndcg@10"];

export default function EvalPage() {
  const [rep, setRep] = useState<EvalReport | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const load = () => api.evalLatest().then(setRep).catch(() => setRep(null));

  useEffect(() => {
    load();
  }, []);

  async function run() {
    setBusy(true);
    setErr(null);
    try {
      await api.runEval(false);
      await load();
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  }

  const chartData =
    rep?.runs.map((r) => ({
      name: CFG_LABEL[r.config_name] ?? r.config_name,
      ...Object.fromEntries(RETRIEVAL_METRICS.map((m) => [m, r.retrieval[m] ?? 0])),
    })) ?? [];

  return (
    <div className="mx-auto max-w-5xl px-6 py-10">
      <header className="mb-6 flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="font-display text-2xl font-bold tracking-tight text-ink-200">
            Evaluation
          </h1>
          <p className="mt-1.5 max-w-xl text-[13.5px] text-ink-400">
            Golden set of {rep?.runs[0]?.n_items ?? 64} questions (56 in-domain +
            8 adversarial must-refuse) run through four retrieval configs.
            Metrics are deterministic and computed at section level.
          </p>
        </div>
        <button
          onClick={run}
          disabled={busy}
          className="flex items-center gap-2 rounded-xl px-4 py-2.5 text-[13px] font-semibold text-ink-950 disabled:opacity-50"
          style={{ background: "linear-gradient(135deg,#f5a524,#fb7822)" }}
        >
          {busy ? <RefreshCw size={14} className="animate-spin" /> : <Play size={14} />}
          {busy ? "running eval…" : "Run evaluation"}
        </button>
      </header>

      {err && <p className="mb-4 rounded-lg bg-danger/10 p-3 text-[13px] text-danger">{err}</p>}
      {!rep && !busy && (
        <p className="rounded-xl border border-ink-700 bg-ink-800/40 p-5 text-[13.5px] text-ink-400">
          No evaluation report yet. Click <span className="text-ink-200">Run evaluation</span> —
          it takes ~30–60 s (retrieval only, no LLM calls).
        </p>
      )}

      {rep && (
        <>
          <p className="mb-4 font-mono text-[11px] text-ink-500">
            report generated {new Date(rep.created_at).toLocaleString()}
          </p>

          <div className="glass-panel mb-6 rounded-2xl p-5">
            <h2 className="mb-4 font-display text-[15px] font-semibold text-ink-200">
              Retrieval quality by config
            </h2>
            <div className="h-72">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData} margin={{ top: 4, right: 12, bottom: 0, left: -18 }}>
                  <CartesianGrid stroke="#1c2534" strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="name" tick={{ fill: "#6b7c95", fontSize: 11 }} stroke="#1c2534" />
                  <YAxis domain={[0, 1]} tick={{ fill: "#6b7c95", fontSize: 11 }} stroke="#1c2534" />
                  <Tooltip
                    contentStyle={{ background: "#0f141d", border: "1px solid #2a3648", borderRadius: 10, fontSize: 12 }}
                    labelStyle={{ color: "#c3cedd" }}
                  />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  <Bar dataKey="hit@5" fill="#f5a524" radius={[3, 3, 0, 0]} />
                  <Bar dataKey="mrr@10" fill="#35d0ba" radius={[3, 3, 0, 0]} />
                  <Bar dataKey="ndcg@10" fill="#55b8ff" radius={[3, 3, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="glass-panel mb-6 overflow-x-auto rounded-2xl">
            <table className="w-full text-[12.5px]">
              <thead>
                <tr className="border-b border-ink-700 text-left text-ink-400">
                  <th className="px-4 py-3 font-semibold">Config</th>
                  {RETRIEVAL_METRICS.map((m) => (
                    <th key={m} className="px-3 py-3 font-mono font-semibold">{m}</th>
                  ))}
                  <th className="px-3 py-3 font-semibold">Answered</th>
                  <th className="px-3 py-3 font-semibold">Adv. refusal</th>
                </tr>
              </thead>
              <tbody>
                {rep.runs.map((r) => (
                  <tr key={r.config_name} className="border-b border-ink-800 text-ink-300">
                    <td className="px-4 py-2.5 font-medium text-ink-200">
                      {CFG_LABEL[r.config_name] ?? r.config_name}
                    </td>
                    {RETRIEVAL_METRICS.map((m) => {
                      const best = Math.max(...rep.runs.map((x) => x.retrieval[m] ?? 0));
                      const v = r.retrieval[m] ?? 0;
                      return (
                        <td key={m} className={`px-3 py-2.5 font-mono ${v === best ? "text-amber-glow" : ""}`}>
                          {v.toFixed(3)}
                        </td>
                      );
                    })}
                    <td className="px-3 py-2.5 font-mono">
                      {r.answered_rate_golden != null ? `${(r.answered_rate_golden * 100).toFixed(0)}%` : "—"}
                    </td>
                    <td className="px-3 py-2.5 font-mono">
                      {r.refusal_rate_adversarial != null ? (
                        <span className={r.refusal_rate_adversarial >= 0.75 ? "text-signal" : "text-danger"}>
                          {(r.refusal_rate_adversarial * 100).toFixed(0)}%
                        </span>
                      ) : "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {rep.runs.find((r) => r.answers) && (
            <div className="glass-panel mb-6 rounded-2xl p-5">
              <h2 className="mb-3 font-display text-[15px] font-semibold text-ink-200">
                Answer quality (with generation)
              </h2>
              <pre className="overflow-x-auto font-mono text-[11.5px] text-ink-300">
                {JSON.stringify(
                  Object.fromEntries(
                    rep.runs.filter((r) => r.answers).map((r) => [r.config_name, r.answers])
                  ),
                  null,
                  2
                )}
              </pre>
            </div>
          )}

          <PerItemTable rep={rep} />
        </>
      )}
    </div>
  );
}

function PerItemTable({ rep }: { rep: EvalReport }) {
  const run = rep.runs[rep.runs.length - 1];
  return (
    <div className="glass-panel overflow-x-auto rounded-2xl">
      <div className="border-b border-ink-700 px-4 py-3">
        <h2 className="font-display text-[15px] font-semibold text-ink-200">
          Per-item — <span className="font-mono text-[12px] text-ink-400">{run.config_name}</span>
        </h2>
      </div>
      <table className="w-full text-[12px]">
        <thead>
          <tr className="border-b border-ink-700 text-left text-ink-400">
            <th className="px-4 py-2.5 font-semibold">id</th>
            <th className="px-3 py-2.5 font-semibold">question / behavior</th>
            <th className="px-3 py-2.5 font-mono font-semibold">hit@5</th>
            <th className="px-3 py-2.5 font-mono font-semibold">mrr</th>
            <th className="px-3 py-2.5 font-mono font-semibold">grounding</th>
            <th className="px-3 py-2.5 font-mono font-semibold">expected → top retrieved</th>
          </tr>
        </thead>
        <tbody>
          {run.per_item.map((it) => (
            <tr key={it.id} className="border-b border-ink-800 text-ink-300">
              <td className="px-4 py-2 font-mono text-ink-400">{it.id}</td>
              <td className="max-w-[320px] truncate px-3 py-2">
                {it.adversarial ? (
                  <span className={it.refused ? "text-signal" : "text-danger"}>
                    🛡 {it.refused ? "refused (correct)" : "answered (should refuse!)"}
                  </span>
                ) : (
                  it.question ?? ""
                )}
              </td>
              <td className="px-3 py-2 font-mono">{it.adversarial ? "—" : it["hit@5"]?.toFixed(0)}</td>
              <td className="px-3 py-2 font-mono">{it.adversarial ? "—" : it["mrr@10"]?.toFixed(2)}</td>
              <td className="px-3 py-2 font-mono">{it.grounding_score?.toFixed(2)}</td>
              <td className="max-w-[260px] truncate px-3 py-2 font-mono text-[11px] text-ink-500">
                {it.adversarial
                  ? (it.top_retrieved ?? []).slice(0, 2).join(", ")
                  : `${it.expected?.[0] ?? ""} → ${it.retrieved?.[0] ?? "—"}`}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
