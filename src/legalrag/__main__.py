"""Command line entry point.

    python -m legalrag run configs/obliqa/bm25_passage.yaml [--limit 200]
    python -m legalrag compare results/obliqa_*
    python -m legalrag inspect results/maud_bm25_section300 --group "No-Shop" --n 5
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

    p_ins = sub.add_parser("inspect", help="Show failed queries with correct and retrieved text")
    p_ins.add_argument("run_dir", help="A results folder, e.g. results/maud_hybrid_section300")
    p_ins.add_argument("--metric", default="recall@5", help="Metric that defines a failure (default recall@5)")
    p_ins.add_argument("--below", type=float, default=1.0, help="Failure means metric < this (default 1.0)")
    p_ins.add_argument("--group", help="Only this group, e.g. a MAUD clause type or an ObliQA passage count")
    p_ins.add_argument("--n", type=int, default=5, help="How many failures to show")
    p_ins.add_argument("--top", type=int, default=3, help="How many retrieved results to show per failure")
    p_ins.add_argument("--words", type=int, default=80, help="Words of text to show per passage")
    p_ins.add_argument("--seed", type=int, default=0, help="Change to see a different random sample")

    args = parser.parse_args()
    if args.command == "run":
        for config in args.configs:
            result = run(config, limit=args.limit, out_root=args.out)
            print(f"\n== {result['name']} ({result['n_queries']} queries, {result['corpus']['n_chunks']} chunks)")
            print(json.dumps({k: round(v, 4) for k, v in result["metrics"].items()}, indent=2))
    elif args.command == "compare":
        print(compare(args.paths, args.columns))
    elif args.command == "inspect":
        from .inspect_failures import inspect

        print(inspect(args.run_dir, args.metric, args.below, args.group, args.n, args.top, args.words, args.seed))


if __name__ == "__main__":
    main()
