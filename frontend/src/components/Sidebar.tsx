import { NavLink } from "react-router-dom";
import { MessageSquareText, ScanSearch, Gauge, Activity, Database } from "lucide-react";
import { Logo } from "../App";
import type { HealthStatus } from "../lib/types";

const NAV = [
  { to: "/", label: "Chat", icon: MessageSquareText, end: true },
  { to: "/explore", label: "Explore", icon: ScanSearch },
  { to: "/evaluation", label: "Evaluation", icon: Gauge },
  { to: "/insights", label: "Insights", icon: Activity },
];

export default function Sidebar({ health }: { health: HealthStatus | null }) {
  const idx = health?.indexes;
  const ready = (idx?.qdrant_points ?? 0) > 0 && idx?.bm25_ready;

  return (
    <aside className="flex w-60 shrink-0 flex-col border-r border-ink-800 bg-ink-900/80">
      <div className="flex items-center gap-3 px-5 pb-5 pt-6">
        <Logo />
        <div>
          <div className="font-display text-[17px] font-700 tracking-tight text-ink-200">
            GroundWork
          </div>
          <div className="text-[11px] font-mono uppercase tracking-widest text-ink-500">
            OSHA · 29 CFR
          </div>
        </div>
      </div>

      <div className="hazard-stripe mx-5 h-[3px] rounded-full opacity-70" />

      <nav className="mt-6 flex flex-col gap-1 px-3">
        {NAV.map(({ to, label, icon: Icon, end }) => (
          <NavLink
            key={to}
            to={to}
            end={end}
            className={({ isActive }) =>
              `group flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-colors ${
                isActive
                  ? "bg-amber-glow/12 text-amber-glow"
                  : "text-ink-300 hover:bg-ink-800/70 hover:text-ink-200"
              }`
            }
          >
            <Icon size={17} strokeWidth={2} />
            {label}
          </NavLink>
        ))}
      </nav>

      <div className="mt-auto space-y-3 p-4">
        <div className="glass-panel rounded-xl p-3.5">
          <div className="flex items-center gap-2 text-[11px] font-semibold uppercase tracking-wider text-ink-400">
            <Database size={12} /> Corpus index
          </div>
          <div className="mt-2 space-y-1.5 font-mono text-[11px] text-ink-300">
            <Row label="qdrant" value={idx?.qdrant_points?.toLocaleString() ?? "…"} ok={ready} />
            <Row label="bm25" value={idx?.bm25_ready ? `${idx.bm25_docs?.toLocaleString()} docs` : "…"} ok={!!idx?.bm25_ready} />
            <Row label="llm" value={health?.llm_provider ?? "…"} ok={!!health} />
          </div>
        </div>
        <p className="px-1 text-[10.5px] leading-relaxed text-ink-500">
          Answers cite official eCFR sections &amp; OSHA publications. Not legal
          advice.
        </p>
      </div>
    </aside>
  );
}

function Row({ label, value, ok }: { label: string; value: string; ok?: boolean }) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-ink-500">{label}</span>
      <span className="flex items-center gap-1.5">
        {value}
        <span
          className={`inline-block h-1.5 w-1.5 rounded-full ${
            ok ? "bg-signal animate-pulse-soft" : "bg-ink-600"
          }`}
        />
      </span>
    </div>
  );
}
