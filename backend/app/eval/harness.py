"""Evaluation harness.

Runs the golden set through the retrieval pipeline (and optionally the full
generation path) under multiple retrieval configs, producing an EvalReport
with per-item detail. Retrieval metrics are exact/deterministic; answer
metrics use citation validation + embedding similarity — no flaky LLM judge
in the critical path.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path

import yaml

from ..config import Settings, get_settings
from ..obs.logging import get_logger
from ..rag import citations as cite
from ..rag.generate import build_messages, get_llm
from ..rag.retrieve import RetrievalPipeline
from ..schemas import EvalHit, EvalReport, GoldenItem, RetrievalConfig
from .metrics import aggregate, answer_metrics, retrieval_metrics

log = get_logger(__name__)

EVAL_TOP_K = 10  # rank depth evaluated

DEFAULT_CONFIGS: dict[str, RetrievalConfig] = {
    "dense_only": RetrievalConfig(dense_top_k=30, sparse_top_k=0, rerank_enabled=False,
                                  final_top_k=EVAL_TOP_K),
    "bm25_only": RetrievalConfig(dense_top_k=0, sparse_top_k=30, rerank_enabled=False,
                                 final_top_k=EVAL_TOP_K),
    "hybrid": RetrievalConfig(dense_top_k=30, sparse_top_k=30, rerank_enabled=False,
                              final_top_k=EVAL_TOP_K),
    "hybrid_rerank": RetrievalConfig(dense_top_k=30, sparse_top_k=30, rerank_enabled=True,
                                     final_top_k=EVAL_TOP_K),
}


def load_golden(path: Path | None = None) -> list[GoldenItem]:
    path = path or Path(__file__).parent / "golden.yaml"
    raw = yaml.safe_load(path.read_text())["items"]
    return [GoldenItem(**item) for item in raw]


def run_eval(
    configs: dict[str, RetrievalConfig] | None = None,
    with_answers: bool = False,
    settings: Settings | None = None,
    pipeline: RetrievalPipeline | None = None,
) -> EvalReport:
    settings = settings or get_settings()
    pipeline = pipeline or RetrievalPipeline(settings)
    configs = configs or DEFAULT_CONFIGS
    golden = load_golden()
    embedder = pipeline.embedder
    llm = get_llm() if with_answers else None

    runs: list[EvalHit] = []
    for name, cfg in configs.items():
        t0 = time.perf_counter()
        per_retrieval: list[dict[str, float]] = []
        per_answer: list[dict[str, float]] = []
        refusal_adv: list[bool] = []
        answered_gold: list[bool] = []
        per_item: list[dict] = []

        for item in golden:
            result = pipeline.retrieve(item.question, cfg)
            ranked = [h.section_id for h in result.hits]

            if item.adversarial:
                refusal_adv.append(result.refused)
                per_item.append({
                    "id": item.id, "adversarial": True, "refused": result.refused,
                    "grounding_score": result.grounding_score,
                    "top_retrieved": ranked[:3],
                })
                continue

            m = retrieval_metrics(ranked, item.expected_sections, k=EVAL_TOP_K)
            per_retrieval.append(m)
            row = {
                "id": item.id, "question": item.question,
                "expected": item.expected_sections, "retrieved": ranked[:EVAL_TOP_K],
                **m, "grounding_score": result.grounding_score, "refused": result.refused,
            }
            answered_gold.append(not result.refused)

            if with_answers and llm is not None and not result.refused:
                messages = build_messages(item.question, result.context, [])
                answer = "".join(
                    llm.stream(messages, max_tokens=settings.llm_max_tokens, temperature=0.0)
                )
                info = cite.validate_citations(answer, ranked)
                am = answer_metrics(
                    info,
                    embedder.embed_documents([answer])[0].tolist(),
                    embedder.embed_documents([result.context[:6000]])[0].tolist(),
                    embedder.embed_queries([item.question])[0].tolist(),
                )
                per_answer.append(am)
                row["answer"] = answer[:400]
                row.update(am)
                row["citations"] = info["cited"]
                row["hallucinated"] = info["hallucinated"]
            per_item.append(row)

        run = EvalHit(
            config_name=name,
            retrieval=aggregate(per_retrieval) if per_retrieval else {},
            answers=aggregate(per_answer) if per_answer else None,
            refusal_rate_adversarial=(
                round(sum(refusal_adv) / len(refusal_adv), 4) if refusal_adv else None
            ),
            answered_rate_golden=(
                round(sum(answered_gold) / len(answered_gold), 4) if answered_gold else None
            ),
            n_items=len(golden),
            duration_s=round(time.perf_counter() - t0, 1),
            per_item=per_item,
        )
        runs.append(run)
        log.info("eval_config_done", config=name,
                 **{k: v for k, v in run.retrieval.items()},
                 refusal_rate=run.refusal_rate_adversarial)

    report = EvalReport(created_at=datetime.now(UTC).isoformat(timespec="seconds"),
                        runs=runs)
    _persist(report, settings)
    return report


def _persist(report: EvalReport, settings: Settings) -> None:
    out = settings.artifacts_dir / "eval"
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval_report_latest.json").write_text(report.model_dump_json(indent=2))


def load_latest_report(settings: Settings | None = None) -> EvalReport | None:
    settings = settings or get_settings()
    path = settings.artifacts_dir / "eval" / "eval_report_latest.json"
    if not path.exists():
        return None
    return EvalReport.model_validate_json(path.read_text())


if __name__ == "__main__":
    r = run_eval(with_answers="--answers" in __import__("sys").argv)
    print(json.dumps([x.model_dump(exclude={"per_item"}) for x in r.runs], indent=2))
