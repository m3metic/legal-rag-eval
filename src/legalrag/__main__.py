"""Command line entry point.

    python -m legalrag run configs/obliqa/bm25_passage.yaml [--limit 200]
    python -m legalrag compare results/obliqa_*
"""

from __future__ import annotations

import argparse
import json

from .experiment import compare, run


def main() -> None:
    parser = argparse.ArgumentParser(prog="legalrag")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Run one or more experiment configs")
    p_run.add_argument("configs", nargs="+")
    p_run.add_argument("--limit", type=int, help="Evaluate only the first N queries (quick checks)")
    p_run.add_argument("--out", default="results")

    p_cmp = sub.add_parser("compare", help="Print a markdown table comparing result folders")
    p_cmp.add_argument("paths", nargs="+")
    p_cmp.add_argument("--columns", nargs="*")

    args = parser.parse_args()
    if args.command == "run":
        for config in args.configs:
            result = run(config, limit=args.limit, out_root=args.out)
            print(f"\n== {result['name']} ({result['n_queries']} queries, {result['corpus']['n_chunks']} chunks)")
            print(json.dumps({k: round(v, 4) for k, v in result["metrics"].items()}, indent=2))
    elif args.command == "compare":
        print(compare(args.paths, args.columns))


if __name__ == "__main__":
    main()
