import { useEffect, useRef, useState } from "react";
import { SlidersHorizontal, RotateCcw } from "lucide-react";
import type { RetrievalConfigOverride } from "../lib/types";

interface Defaults {
  dense_top_k: number;
  sparse_top_k: number;
  final_top_k: number;
  rrf_k: number;
  rerank_enabled: boolean;
  grounding_threshold: number;
}

interface Props {
  value: RetrievalConfigOverride;
  defaults: Defaults | null;
  onChange: (v: RetrievalConfigOverride) => void;
}

export default function ConfigPopover({ value, defaults, onChange }: Props) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const onDoc = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, []);

  const d = defaults;
  const effective = (
    k: "dense_top_k" | "sparse_top_k" | "final_top_k" | "grounding_threshold"
  ): string | number => value[k] ?? d?.[k] ?? "";

  return (
    <div ref={ref} className="relative">
      <button
        onClick={() => setOpen((o) => !o)}
        className={`flex items-center gap-2 rounded-lg border px-3 py-1.5 text-[12px] font-medium transition-colors ${
          open
            ? "border-amber-glow/50 bg-amber-glow/10 text-amber-glow"
            : "border-ink-700 bg-ink-800/60 text-ink-300 hover:text-ink-200"
        }`}
        title="Retrieval configuration"
      >
        <SlidersHorizontal size={14} />
        Retrieval
        <span className="font-mono text-[10.5px] text-ink-500">
          {value.rerank_enabled === false || value.rerank_enabled === true
            ? value.rerank_enabled ? "rerank" : "no-rerank"
            : d?.rerank_enabled ? "rerank" : "no-rerank"}
        </span>
      </button>

      {open && (
        <div className="absolute bottom-full right-0 z-30 mb-2 w-80 animate-fade-up rounded-xl border border-ink-700 bg-ink-850 p-4 shadow-2xl">
          <div className="mb-3 flex items-center justify-between">
            <span className="text-[12px] font-semibold uppercase tracking-wider text-ink-400">
              Retrieval knobs
            </span>
            <button
              className="flex items-center gap-1 text-[11px] text-ink-400 hover:text-amber-glow"
              onClick={() => onChange({})}
            >
              <RotateCcw size={11} /> reset
            </button>
          </div>
          <Slider label="dense top-k" min={0} max={60} step={5} value={effective("dense_top_k")} def={d?.dense_top_k}
            onChange={(v) => onChange({ ...value, dense_top_k: v })} />
          <Slider label="bm25 top-k" min={0} max={60} step={5} value={effective("sparse_top_k")} def={d?.sparse_top_k}
            onChange={(v) => onChange({ ...value, sparse_top_k: v })} />
          <Slider label="final top-k" min={1} max={12} step={1} value={effective("final_top_k")} def={d?.final_top_k}
            onChange={(v) => onChange({ ...value, final_top_k: v })} />
          <Slider label="grounding threshold" min={0} max={0.8} step={0.05} value={effective("grounding_threshold")} def={d?.grounding_threshold}
            onChange={(v) => onChange({ ...value, grounding_threshold: v })} format={(v) => Number(v).toFixed(2)} />
          <label className="mt-3 flex cursor-pointer items-center justify-between rounded-lg border border-ink-700 bg-ink-800/50 px-3 py-2">
            <span className="text-[12.5px] text-ink-300">Cross-encoder rerank</span>
            <input
              type="checkbox"
              checked={value.rerank_enabled ?? d?.rerank_enabled ?? true}
              onChange={(e) => onChange({ ...value, rerank_enabled: e.target.checked })}
              className="h-4 w-4 accent-amber-500"
            />
          </label>
          <p className="mt-3 text-[11px] leading-relaxed text-ink-500">
            Changes apply to your next question. Threshold too low → the assistant
            may answer off-topic questions; too high → it refuses valid ones.
          </p>
        </div>
      )}
    </div>
  );
}

function Slider({
  label, min, max, step, value, def, onChange, format,
}: {
  label: string; min: number; max: number; step: number; value: string | number;
  def?: number; onChange: (v: number) => void; format?: (v: number) => string;
}) {
  const v = Number(value);
  return (
    <div className="mb-2.5">
      <div className="mb-1 flex justify-between text-[11.5px]">
        <span className="text-ink-400">{label}</span>
        <span className={`font-mono ${v === def ? "text-ink-500" : "text-amber-glow"}`}>
          {format ? format(v) : v}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={v}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full accent-amber-500"
      />
    </div>
  );
}
