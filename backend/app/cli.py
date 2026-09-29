"""GroundWork CLI: python -m app.cli <command>.

Commands: ingest | index | eval | serve | report | download-llm
"""

from __future__ import annotations

import argparse
import json
import sys


def main() -> None:
    parser = argparse.ArgumentParser(prog="groundwork")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("ingest", help="download + parse + chunk the corpus")
    sub.add_parser("index", help="build dense (Qdrant) + sparse (BM25) indexes")
    p_eval = sub.add_parser("eval", help="run the golden-set evaluation")
    p_eval.add_argument("--answers", action="store_true", help="also evaluate generation")
    p_eval.add_argument("--config", help="eval a single config (dense_only|bm25_only|hybrid|hybrid_rerank)")
    p_eval.add_argument("--json", action="store_true", help="dump raw JSON")
    sub.add_parser("report", help="render an evaluation report from the latest run")
    p_dl = sub.add_parser("download-llm", help="pre-download the local GGUF model")
    p_dl.add_argument("--force", action="store_true")
    p_serve = sub.add_parser("serve", help="run the API server")
    p_serve.add_argument("--host", default=None)
    p_serve.add_argument("--port", type=int, default=None)

    args = parser.parse_args()

    if args.cmd == "ingest":
        from .ingest.pipeline import run_ingest
        print(json.dumps(run_ingest(), indent=2))

    elif args.cmd == "index":
        from .index.build import run_index
        print(json.dumps(run_index(), indent=2))

    elif args.cmd == "eval":
        from .eval.harness import DEFAULT_CONFIGS, run_eval
        configs = {args.config: DEFAULT_CONFIGS[args.config]} if args.config else DEFAULT_CONFIGS
        rep = run_eval(configs, with_answers=args.answers)
        if args.json:
            print(json.dumps([r.model_dump(exclude={"per_item"}) for r in rep.runs], indent=2))
        else:
            from .eval.report import render_markdown
            print(render_markdown(rep))

    elif args.cmd == "report":
        from .eval.harness import load_latest_report
        from .eval.report import render_markdown
        rep = load_latest_report()
        if rep is None:
            sys.exit("no report found — run `eval` first")
        md = render_markdown(rep)
        from .config import get_settings
        out = get_settings().project_root / "EVALUATION.md"
        out.write_text(md)
        print(f"wrote {out}")

    elif args.cmd == "download-llm":
        from huggingface_hub import hf_hub_download

        from .config import get_settings
        s = get_settings()
        p = hf_hub_download(repo_id=s.gguf_repo_id, filename=s.gguf_filename,
                            local_dir=str(s.models_dir / "gguf"), force_download=args.force)
        print("model at", p)

    elif args.cmd == "serve":
        import uvicorn

        from .config import get_settings
        s = get_settings()
        uvicorn.run(
            "app.main:app",
            host=args.host or s.api_host,
            port=args.port or s.api_port,
        )


if __name__ == "__main__":
    main()
