import { useCallback, useEffect, useRef, useState } from "react";
import {
  SendHorizonal, ShieldAlert, Zap, BookOpenCheck, CornerDownLeft, OctagonAlert,
} from "lucide-react";
import Markdown from "../components/Markdown";
import SourceDrawer, { type DrawerState } from "../components/SourceDrawer";
import ConfigPopover from "../components/ConfigPopover";
import { api, streamChat } from "../lib/api";
import type { AppConfig, ChatMessage, RetrievalConfigOverride, SourceRef } from "../lib/types";

const EXAMPLES = [
  "At what height do construction workers need fall protection?",
  "Who pays for PPE — the employer or the employee?",
  "How quickly must a fatality be reported to OSHA?",
  "What are the first aid steps for heat stroke?",
];

export default function ChatPage() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [config, setConfig] = useState<RetrievalConfigOverride>({});
  const [appCfg, setAppCfg] = useState<AppConfig | null>(null);
  const [drawer, setDrawer] = useState<DrawerState>({ open: false, sectionId: null });
  const [drawerSources, setDrawerSources] = useState<SourceRef[]>([]);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    api.config().then(setAppCfg).catch(() => {});
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        e.preventDefault();
        inputRef.current?.focus();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages]);

  const lastSources = [...messages].reverse().find((m) => m.role === "assistant" && m.sources)?.sources;
  const sourcesForDrawer = drawerSources.length ? drawerSources : lastSources ?? [];

  const citationValid = useCallback(
    (id: string): boolean | null => {
      const root = (s: string) => s.match(/^(?:29 CFR \d{4}\.\d+|OSHA [A-Z0-9-]{2,20})/i)?.[0]?.toLowerCase() ?? s.toLowerCase();
      const lastAssistant = [...messages].reverse().find((m) => m.role === "assistant");
      const cites = lastAssistant?.meta?.citations;
      if (!cites) return null;
      const validRoots = cites.valid.map(root);
      if (validRoots.includes(root(id))) return true;
      if (cites.cited.map(root).includes(root(id))) return false;
      return null;
    },
    [messages]
  );

  async function send(text?: string) {
    const q = (text ?? input).trim();
    if (!q || busy || q.length < 2) return;
    setInput("");
    setBusy(true);

    const userMsg: ChatMessage = { role: "user", content: q };
    const asst: ChatMessage = { role: "assistant", content: "", streaming: true };
    setMessages((m) => [...m, userMsg, asst]);

    const history = messages.slice(-4).map((m) => ({ role: m.role, content: m.content }));
    const patch = (fn: (m: ChatMessage) => ChatMessage) =>
      setMessages((ms) => ms.map((m, i) => (i === ms.length - 1 ? fn(m) : m)));

    try {
      await streamChat(q, history, config, (ev) => {
        if (ev.type === "sources") {
          patch((m) => ({ ...m, sources: ev.sources }));
          setDrawerSources(ev.sources);
        } else if (ev.type === "token") {
          patch((m) => ({ ...m, content: m.content + ev.text }));
        } else if (ev.type === "meta") {
          patch((m) => ({
            ...m,
            meta: {
              grounding_score: ev.grounding_score,
              timings: ev.timings,
              exact_citation: ev.exact_citation,
              corpus_version: ev.corpus_version,
            },
          }));
        } else if (ev.type === "refused") {
          patch((m) => ({
            ...m,
            content: "",
            meta: {
              grounding_score: ev.grounding_score,
              timings: [],
              refused: true,
              refusal_reason: ev.reason,
            },
          }));
        } else if (ev.type === "done") {
          patch((m) => ({
            ...m,
            meta: {
              grounding_score: m.meta?.grounding_score ?? 0,
              timings: m.meta?.timings ?? [],
              exact_citation: m.meta?.exact_citation,
              corpus_version: m.meta?.corpus_version,
              citations: ev.citations,
              warning: ev.warning,
              first_token_ms: ev.first_token_ms,
              total_ms: ev.total_ms,
            },
          }));
        } else if (ev.type === "error") {
          patch((m) => ({ ...m, content: `Error: ${ev.detail}` }));
        }
      });
    } catch (e) {
      patch((m) => ({ ...m, content: `Connection failed: ${String(e)}` }));
    } finally {
      patch((m) => ({ ...m, streaming: false }));
      setBusy(false);
    }
  }

  return (
    <div className="mx-auto flex h-full min-h-0 max-w-3xl flex-col px-4 pb-4">
      <div ref={scrollRef} className="min-h-0 flex-1 overflow-y-auto pt-8">
        {messages.length === 0 ? <EmptyState onPick={(q) => send(q)} /> : null}

        <div className="space-y-7 pb-6">
          {messages.map((m, i) =>
            m.role === "user" ? (
              <div key={i} className="flex animate-fade-up justify-end">
                <div className="max-w-[85%] rounded-2xl rounded-br-md border border-ink-700 bg-ink-800/80 px-4 py-2.5 text-[14.5px] text-ink-200">
                  {m.content}
                </div>
              </div>
            ) : (
              <AssistantMessage
                key={i}
                msg={m}
                onCite={(id) => {
                  setDrawerSources(m.sources ?? []);
                  setDrawer({ open: true, sectionId: id });
                }}
              />
            )
          )}
        </div>
      </div>

      <div className="pt-3">
        <div className="glass-panel rounded-2xl p-2.5 shadow-2xl shadow-black/40">
          <div className="flex items-end gap-2">
            <textarea
              ref={inputRef}
              rows={1}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  send();
                }
              }}
              placeholder="Ask about OSHA regulations… (⌘K to focus)"
              className="max-h-36 flex-1 resize-none bg-transparent px-3 py-2.5 text-[14.5px] text-ink-200 placeholder:text-ink-500 focus:outline-none"
            />
            <ConfigPopover value={config} defaults={appCfg?.retrieval_defaults ?? null} onChange={setConfig} />
            <button
              onClick={() => send()}
              disabled={busy || !input.trim()}
              className="grid h-10 w-10 shrink-0 place-items-center rounded-xl transition-all disabled:opacity-40"
              style={{ background: busy ? "#2a3648" : "linear-gradient(135deg,#f5a524,#fb7822)" }}
              aria-label="Send"
            >
              {busy ? (
                <span className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-ink-300 border-t-transparent" />
              ) : (
                <SendHorizonal size={17} className="text-ink-950" />
              )}
            </button>
          </div>
        </div>
        <p className="mt-2 text-center text-[11px] text-ink-500">
          Grounded in 29 CFR (eCFR) + OSHA publications{appCfg?.corpus?.version && appCfg.corpus.version !== "unknown" ? ` · corpus ${appCfg.corpus.version}` : ""} · answers include verifiable citations · not legal advice
        </p>
      </div>

      <SourceDrawer
        state={drawer}
        sources={sourcesForDrawer}
        citationValid={citationValid}
        onClose={() => setDrawer({ open: false, sectionId: null })}
      />
    </div>
  );
}

function AssistantMessage({ msg, onCite }: { msg: ChatMessage; onCite: (id: string) => void }) {
  const meta = msg.meta;

  if (meta?.refused) {
    const outside = meta.refusal_reason === "out_of_domain" || meta.refusal_reason === "weak_evidence";
    return (
      <div className="animate-fade-up rounded-2xl border border-danger/25 bg-danger/[0.06] p-4">
        <div className="flex items-center gap-2 text-[13px] font-semibold text-danger">
          <OctagonAlert size={15} /> {outside ? "Outside the corpus scope" : "Refused — insufficient grounding"}
          <span className="font-mono text-[11px] font-normal text-ink-400">
            grounding {meta.grounding_score?.toFixed(2)}
          </span>
        </div>
        <p className="mt-2 text-[13.5px] leading-relaxed text-ink-300">
          I can't answer this confidently from the OSHA regulations and guidance I
          have indexed. This question appears outside the scope of 29 CFR and OSHA
          publications, or nothing sufficiently relevant was found. Try rephrasing
          with more OSHA-specific terms.
        </p>
      </div>
    );
  }

  return (
    <div className="animate-fade-up">
      <div className="mb-2 flex flex-wrap items-center gap-2 text-[11px]">
        <span className="inline-flex items-center gap-1.5 rounded-full border border-ink-700 bg-ink-800/60 px-2.5 py-1 font-mono text-ink-300">
          <Zap size={11} className="text-amber-glow" />
          grounding {meta?.grounding_score?.toFixed(2) ?? "—"}
        </span>
        {meta?.exact_citation && (
          <span className="rounded-full bg-signal/10 px-2 py-1 font-mono text-[10.5px] text-signal">
            exact citation match
          </span>
        )}
        {meta?.timings?.map((t) => (
          <span key={t.stage} className="rounded-full bg-ink-800/50 px-2 py-1 font-mono text-[10.5px] text-ink-500">
            {t.stage} {t.ms >= 1000 ? `${(t.ms / 1000).toFixed(1)}s` : `${Math.round(t.ms)}ms`}
          </span>
        ))}
        {meta?.first_token_ms != null && (
          <span className="rounded-full bg-ink-800/50 px-2 py-1 font-mono text-[10.5px] text-ink-500">
            ttft {(meta.first_token_ms / 1000).toFixed(1)}s
          </span>
        )}
        {meta?.citations && meta.citations.n_cited > 0 && (
          <span
            className={`inline-flex items-center gap-1 rounded-full px-2 py-1 font-mono text-[10.5px] ${
              meta.citations.n_hallucinated > 0
                ? "bg-danger/10 text-danger"
                : "bg-signal/10 text-signal"
            }`}
          >
            <BookOpenCheck size={10} />
            {meta.citations.n_valid}/{meta.citations.n_cited} citations verified
            {meta.citations.n_hallucinated > 0 ? ` · ${meta.citations.n_hallucinated} unverified` : ""}
          </span>
        )}
      </div>

      {msg.content ? (
        <Markdown
          content={msg.content}
          onCite={onCite}
          invalidCitations={meta?.citations?.detail
            ?.filter((d) => !d.valid)
            .map((d) => d.citation)}
        />
      ) : (
        <div className="flex gap-1.5 py-2">
          {[0, 1, 2].map((i) => (
            <span
              key={i}
              className="h-2 w-2 animate-pulse-soft rounded-full bg-amber-glow/70"
              style={{ animationDelay: `${i * 0.25}s` }}
            />
          ))}
        </div>
      )}

      {meta?.warning && !msg.streaming && (
        <div className="mt-2.5 rounded-xl border border-danger/25 bg-danger/[0.05] px-3.5 py-2.5">
          <div className="flex items-center gap-2 text-[12px] font-semibold text-danger">
            <ShieldAlert size={13} /> {meta.warning}
          </div>
          {meta.citations?.detail && meta.citations.detail.some((d) => !d.valid) && (
            <ul className="mt-1.5 space-y-0.5">
              {meta.citations.detail
                .filter((d) => !d.valid)
                .map((d) => (
                  <li key={d.citation} className="font-mono text-[11px] text-ink-400">
                    {d.citation}
                    {d.reason ? ` — ${d.reason}` : ""}
                  </li>
                ))}
            </ul>
          )}
        </div>
      )}

      {msg.sources && msg.sources.length > 0 && !msg.streaming && (
        <div className="mt-3 flex flex-wrap gap-1.5">
          {msg.sources.map((s) => (
            <button
              key={s.section_id}
              onClick={() => onCite(s.section_id)}
              className="group flex items-center gap-1.5 rounded-lg border border-ink-700 bg-ink-800/50 px-2.5 py-1.5 font-mono text-[11px] text-ink-300 transition-colors hover:border-amber-glow/40 hover:text-amber-glow"
            >
              <span className={`h-1.5 w-1.5 rounded-full ${s.source === "ecfr" ? "bg-amber-glow/70" : "bg-signal/70"}`} />
              {s.section_id.split(" (")[0]}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

function EmptyState({ onPick }: { onPick: (q: string) => void }) {
  return (
    <div className="pb-10 pt-6 text-center">
      <div className="mx-auto mb-5 h-14 w-14 rounded-2xl hazard-stripe opacity-80" />
      <h1 className="font-display text-[26px] font-bold tracking-tight text-ink-200">
        Ask OSHA. Get the regulation, cited.
      </h1>
      <p className="mx-auto mt-2 max-w-md text-[13.5px] leading-relaxed text-ink-400">
        GroundWork grounds every answer in the official eCFR text of 29 CFR and
        OSHA publications — each claim carries a clickable section citation, and
        off-topic questions are refused, not guessed.
      </p>
      <div className="mx-auto mt-6 grid max-w-xl grid-cols-1 gap-2 sm:grid-cols-2">
        {EXAMPLES.map((q) => (
          <button
            key={q}
            onClick={() => onPick(q)}
            className="rounded-xl border border-ink-700 bg-ink-800/40 px-4 py-3 text-left text-[13px] text-ink-300 transition-all hover:border-amber-glow/40 hover:bg-amber-glow/[0.06] hover:text-ink-200"
          >
            <CornerDownLeft size={13} className="mr-1.5 inline text-ink-500" />
            {q}
          </button>
        ))}
      </div>
    </div>
  );
}
