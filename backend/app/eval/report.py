"""Markdown report rendering for eval runs (published as EVALUATION.md)."""

from __future__ import annotations

from ..schemas import EvalReport

_CFG_LABEL = {
    "dense_only": "Dense only (bge-small)",
    "bm25_only": "BM25 only",
    "hybrid": "Hybrid (RRF fusion)",
    "hybrid_rerank": "Hybrid + cross-encoder rerank (default)",
}

_METRICS = ["hit@1", "hit@3", "hit@5", "hit@10", "mrr@10", "ndcg@10"]


def render_markdown(report: EvalReport) -> str:
    lines: list[str] = []
    lines.append("# GroundWork — Evaluation Report")
    lines.append("")
    lines.append(f"*Generated: {report.created_at} (all metrics deterministic; "
                 "golden set = `backend/app/eval/golden.yaml`)*")
    lines.append("")

    lines.append("## Retrieval quality (section-level, golden set)")
    lines.append("")
    header = ("| Config | " + " | ".join(_METRICS)
              + " | Subsection evidence | Answered | Adversarial refusal |")
    lines.append(header)
    lines.append("|" + "---|" * (len(_METRICS) + 4))
    for run in report.runs:
        vals = [f"{run.retrieval.get(m, 0.0):.3f}" if m in run.retrieval else "—"
                for m in _METRICS]
        subsec = (f"{run.subsection_evidence:.0%}"
                  if run.subsection_evidence is not None else "—")
        answered = f"{run.answered_rate_golden:.0%}" if run.answered_rate_golden is not None else "—"
        refusal = (f"{run.refusal_rate_adversarial:.0%}"
                   if run.refusal_rate_adversarial is not None else "—")
        lines.append(
            f"| {_CFG_LABEL.get(run.config_name, run.config_name)} | "
            + " | ".join(vals)
            + f" | {subsec} | {answered} | {refusal} |"
        )
    lines.append("")

    answer_runs = [r for r in report.runs if r.answers]
    if answer_runs:
        lines.append("## Answer quality (generation eval)")
        lines.append("")
        am = ["citation_precision", "citation_rate", "hallucinated_citations",
              "groundedness", "relevance"]
        lines.append("| Config | " + " | ".join(am) + " |")
        lines.append("|" + "---|" * (len(am) + 1))
        for run in answer_runs:
            vals = [f"{run.answers.get(m, 0.0):.3f}" for m in am]
            lines.append(
                f"| {_CFG_LABEL.get(run.config_name, run.config_name)} | "
                + " | ".join(vals) + " |"
            )
        lines.append("")

    default = report.runs[-1]
    lines.append(f"## Per-item detail — `{default.config_name}`")
    lines.append("")
    lines.append("| id | hit@5 | MRR | grounding | refused | expected → retrieved top-1 |")
    lines.append("|---|---|---|---|---|---|")
    for it in default.per_item:
        if it.get("adversarial"):
            lines.append(f"| {it['id']} (adv) | — | — | {it.get('grounding_score', 0)} | "
                         f"{'refused (correct)' if it['refused'] else 'answered (should refuse)'} | "
                         f"{', '.join(it.get('top_retrieved', [])[:2])} |")
        else:
            top = (it.get("retrieved") or ["—"])[0]
            exp = (it.get("expected") or ["—"])[0]
            mark = "ok" if it.get("hit@5") else "MISS"
            lines.append(
                f"| {it['id']} {mark} | {it.get('hit@5', 0):.0f} | "
                f"{it.get('mrr@10', 0):.2f} | {it.get('grounding_score', 0)} | "
                f"{'refused' if it.get('refused') else 'no'} | {exp} -> {top} |"
            )
    lines.append("")
    return "\n".join(lines)
