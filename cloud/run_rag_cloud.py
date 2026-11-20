"""Run the multi-tenant RAG service end to end: load data, build the vector database,
run the demo queries, evaluate retrieval, and optionally answer your own question.

    python cloud/run_rag_cloud.py                       # install missing packages, demo + evaluation
    python cloud/run_rag_cloud.py --ask "Who invented the voltaic pile?" --tenant wiki
    python cloud/run_rag_cloud.py --ask "..." --generate  # adds a generated answer (Claude / free Gemini in Colab)
    python cloud/run_rag_cloud.py --self-test           # offline checks, no downloads

Works on Google Colab (`!python cloud/run_rag_cloud.py`), Kaggle, Binder or locally.
No Hugging Face, Qdrant or OpenAI account is needed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import rag_cloud as rc  # noqa: E402

PACKAGES = {"qdrant_client": "qdrant-client>=1.12", "sentence_transformers": "sentence-transformers>=3.4",
            "pandas": "pandas", "pyarrow": "pyarrow"}


def table(rows: list[dict], cols: list[str], width: int = 70) -> str:
    def cut(v):
        s = str(v).replace("\n", " ")
        return s if len(s) <= width else s[: width - 1] + "…"
    widths = {c: max(len(c), *(len(cut(r[c])) for r in rows)) for c in cols}
    line = " | ".join(c.ljust(widths[c]) for c in cols)
    out = [line, "-+-".join("-" * widths[c] for c in cols)]
    out += [" | ".join(cut(r[c]).ljust(widths[c]) for c in cols) for r in rows]
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["auto", "hf", "offline"], default="auto")
    ap.add_argument("--embed", choices=["auto", "bge", "hashing"], default="auto")
    ap.add_argument("--store", choices=["auto", "qdrant", "numpy"], default="auto")
    ap.add_argument("--qdrant-path", default=None, help="persist Qdrant on disk instead of memory")
    ap.add_argument("--limit", type=int, default=None, help="max Wikipedia passages to index")
    ap.add_argument("--eval-n", type=int, default=200)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--ask", default=None)
    ap.add_argument("--tenant", default=rc.PUBLIC_TENANT)
    ap.add_argument("--generate", action="store_true",
                    help="also generate a cited answer: Claude (ANTHROPIC_API_KEY), free Gemini in Colab, "
                         "or Gemini API (GEMINI_API_KEY)")
    ap.add_argument("--no-install", action="store_true")
    ap.add_argument("--json", default=None, help="write a JSON summary to this path")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        rc.self_test()
        return 0

    env, online = rc.detect_environment(), rc.internet_ok()
    print(f"Environment: {env} | internet: {'yes' if online else 'no'} | Python {sys.version.split()[0]}")
    if online and not args.no_install:
        print("Packages:", rc.ensure_packages(PACKAGES))
    elif not online:
        print("Offline: using whatever is installed; missing pieces fall back automatically.")

    docs, qa, data_info = rc.load_documents(args.source, limit=args.limit)
    print("Data:", json.dumps(data_info))
    service, build_info = rc.RAGCloudService.build(docs, rc.make_embedder(args.embed),
                                                   rc.make_store(args.store, args.qdrant_path), progress=print)
    print("Index:", json.dumps(build_info))

    rows = rc.run_demo(service)
    print("\nDemo queries\n" + table(rows, ["purpose", "tenant", "answer", "sources", "cached", "blocked", "pii"]))
    leaks = sum(r["cross_tenant"] for r in rows)
    print(f"Cross-tenant passages returned: {leaks}")

    result = rc.evaluate(service, qa, n=args.eval_n, top_k=args.top_k)
    print("\nRetrieval evaluation:", json.dumps(result))

    asked = None
    if args.ask:
        r = service.handle(rc.RAGRequest(args.ask, args.tenant, top_k=3), generate=args.generate)
        asked = {"question": args.ask, "tenant": args.tenant, "answer": r.answer, "sources": r.sources,
                 "generated": r.generated, "generator": r.generator}
        print(f"\nQ ({args.tenant}): {args.ask}\nA: {r.answer}\nSources: {', '.join(r.sources) or '-'}")
        if args.generate:
            print(f"Generated ({r.generator or 'none'}):", r.generated or
                  "(skipped: no generator available - set ANTHROPIC_API_KEY or GEMINI_API_KEY, or run in Colab)")

    if args.json:
        Path(args.json).write_text(json.dumps({"environment": env, "data": data_info, "index": build_info,
                                               "demo": rows, "evaluation": result, "asked": asked}, indent=1))
    return 1 if leaks else 0


if __name__ == "__main__":
    raise SystemExit(main())
