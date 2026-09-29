import { useEffect, useState } from "react";
import { X, ExternalLink, ShieldCheck, ShieldAlert, FileText, Scale } from "lucide-react";
import type { SourceRef } from "../lib/types";

export interface DrawerState {
  open: boolean;
  sectionId: string | null;
}

export default function SourceDrawer({
  state,
  sources,
  citationValid,
  onClose,
}: {
  state: DrawerState;
  sources: SourceRef[];
  citationValid: (id: string) => boolean | null;
  onClose: () => void;
}) {
  const [sel, setSel] = useState<string | null>(null);

  useEffect(() => {
    if (state.sectionId) setSel(state.sectionId);
  }, [state.sectionId]);

  const active =
    sources.find((s) => root(s.section_id) === root(sel ?? "")) ??
    sources.find((s) => root(s.section_id) === root(state.sectionId ?? "")) ??
    sources[0];

  if (!state.open) return null;

  return (
    <div className="fixed inset-0 z-40 flex justify-end">
      <div className="absolute inset-0 bg-ink-950/70 backdrop-blur-[2px]" onClick={onClose} />
      <aside className="relative flex h-full w-[520px] max-w-[92vw] animate-fade-up flex-col border-l border-ink-700 bg-[#0b0f16] shadow-2xl">
        <header className="flex items-center justify-between border-b border-ink-800 px-5 py-4">
          <div className="flex items-center gap-2.5">
            {active?.source === "ecfr" ? (
              <Scale size={16} className="text-amber-glow" />
            ) : (
              <FileText size={16} className="text-signal" />
            )}
            <h3 className="font-display text-[15px] font-semibold text-ink-200">
              Sources ({sources.length})
            </h3>
          </div>
          <button
            onClick={onClose}
            className="rounded-lg p-1.5 text-ink-400 transition-colors hover:bg-ink-800 hover:text-ink-200"
            aria-label="Close sources"
          >
            <X size={17} />
          </button>
        </header>

        <div className="flex min-h-0 flex-1">
          <div className="w-48 shrink-0 overflow-y-auto border-r border-ink-800 py-2">
            {sources.map((s) => {
              const isSel = root(s.section_id) === root(active?.section_id ?? "");
              return (
                <button
                  key={s.section_id}
                  onClick={() => setSel(s.section_id)}
                  className={`block w-full px-4 py-2.5 text-left text-[11.5px] font-mono leading-snug transition-colors ${
                    isSel
                      ? "border-r-2 border-amber-glow bg-amber-glow/10 text-amber-glow"
                      : "border-r-2 border-transparent text-ink-300 hover:bg-ink-800/60"
                  }`}
                >
                  {s.section_id.split(" (")[0]}
                </button>
              );
            })}
          </div>

          <div className="min-w-0 flex-1 overflow-y-auto p-5">
            {active && (
              <>
                <div className="mb-1 flex items-center gap-2 font-mono text-[12px] text-amber-glow">
                  {citationValid(active.section_id) === true && (
                    <span className="inline-flex items-center gap-1 rounded bg-signal/10 px-1.5 py-0.5 text-[10px] text-signal">
                      <ShieldCheck size={11} /> cited in answer
                    </span>
                  )}
                  {citationValid(active.section_id) === false && (
                    <span className="inline-flex items-center gap-1 rounded bg-danger/10 px-1.5 py-0.5 text-[10px] text-danger">
                      <ShieldAlert size={11} /> not cited
                    </span>
                  )}
                </div>
                <h4 className="font-display text-[16px] font-semibold leading-snug text-ink-200">
                  {active.heading}
                </h4>
                <div className="mt-1.5 flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px] text-ink-500">
                  <span className="font-mono">{active.section_id}</span>
                  <span>·</span>
                  <span>{active.source === "ecfr" ? "eCFR" : "OSHA publication"}</span>
                  <span>·</span>
                  <span>retrieval score {active.score?.toFixed?.(3) ?? active.score}</span>
                </div>
                {active.url && (
                  <a
                    href={active.url}
                    target="_blank"
                    rel="noreferrer"
                    className="mt-3 inline-flex items-center gap-1.5 rounded-lg border border-ink-700 bg-ink-800/60 px-2.5 py-1.5 text-[12px] text-ink-300 transition-colors hover:border-sky-info/40 hover:text-sky-info"
                  >
                    Open official source <ExternalLink size={12} />
                  </a>
                )}
                <div className="hazard-stripe my-4 h-[2px] w-16 rounded-full opacity-60" />
                <pre className="whitespace-pre-wrap font-sans text-[13.5px] leading-relaxed text-ink-200">
                  {active.excerpt}
                </pre>
              </>
            )}
          </div>
        </div>
      </aside>
    </div>
  );
}

function root(id: string): string {
  const m = id.match(/^(?:29 CFR \d{4}\.\d+|OSHA [A-Z0-9-]{2,20})/i);
  return (m ? m[0] : id).toLowerCase();
}
