import { memo, type ReactNode } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

const CITE_RE =
  /[\[【](29\s*CFR\s*§?\s*[\d.]+(?:\([0-9a-zA-Z]+\))*|OSHA\s*[A-Z0-9-]{2,20}(?:\s*\((?:pp?\.|pages?)\s*[\d–-]+\))?)[\]】]/g;

/** Turn [29 CFR 1910.132(b)] / 【OSHA 3146 (pp. 7-9)】 into clickable chips. */
function preprocessCitations(md: string): string {
  return md.replace(CITE_RE, (_m, id: string) => `[${id}](#cite:${encodeURIComponent(id)})`);
}

function flatten(node: ReactNode): string {
  if (node == null || typeof node === "boolean") return "";
  if (typeof node === "string" || typeof node === "number") return String(node);
  if (Array.isArray(node)) return node.map(flatten).join("");
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const el = node as any;
  if (el?.props?.children) return flatten(el.props.children);
  return "";
}

const norm = (s: string) => s.toLowerCase().replace(/\s+/g, " ").trim();

export default memo(function Markdown({
  content,
  onCite,
  invalidCitations,
}: {
  content: string;
  onCite?: (id: string) => void;
  /** Raw citation strings that failed evidence validation (rendered in red). */
  invalidCitations?: string[];
}) {
  const invalid = new Set((invalidCitations ?? []).map(norm));
  return (
    <div className="prose-gw text-[14.5px] text-ink-200">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          a({ href, children }) {
            if (href?.startsWith("#cite:")) {
              const id = decodeURIComponent(href.slice(6));
              const bad = invalid.size > 0 && invalid.has(norm(id));
              return (
                <button
                  type="button"
                  onClick={() => onCite?.(id)}
                  className={`mx-0.5 inline-flex translate-y-[-1px] items-center rounded-md border px-1.5 py-px font-mono text-[11.5px] font-medium transition-colors ${
                    bad
                      ? "border-danger/40 bg-danger/10 text-danger hover:bg-danger/20"
                      : "border-amber-glow/30 bg-amber-glow/10 text-amber-glow hover:bg-amber-glow/20 hover:border-amber-glow/50"
                  }`}
                  title={bad ? "Citation could not be verified against the retrieved text" : undefined}
                >
                  {flatten(children)}
                </button>
              );
            }
            return (
              <a href={href} target="_blank" rel="noreferrer" className="text-sky-info underline underline-offset-2">
                {children}
              </a>
            );
          },
          table({ children }) {
            return (
              <div className="my-3 overflow-x-auto rounded-lg border border-ink-700">
                <table className="w-full text-[13px]">{children}</table>
              </div>
            );
          },
          th({ children }) {
            return <th className="border-b border-ink-700 bg-ink-800/60 px-3 py-2 text-left font-semibold text-ink-200">{children}</th>;
          },
          td({ children }) {
            return <td className="border-b border-ink-800 px-3 py-2 align-top text-ink-300">{children}</td>;
          },
        }}
      >
        {preprocessCitations(content)}
      </ReactMarkdown>
    </div>
  );
});
