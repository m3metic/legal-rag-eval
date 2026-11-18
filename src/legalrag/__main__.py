"""Command line entry point.

    python -m legalrag run configs/obliqa/bm25_passage.yaml [--limit 200]
    python -m legalrag compare results/obliqa_*
    python -m legalrag inspect results/maud_bm25_section300 --group "No-Shop" --n 5
    python -m legalrag gate [configs/gate.yaml] [--update]
    python -m legalrag answer configs/answers/obliqa_bm25.yaml [--sample 20] [--no-grade]
"""

from __future__ import annotations

import argparse
import json
import sys

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

    p_gate = sub.add_parser("gate", help="Rerun the gate experiments and fail if any metric drops")
    p_gate.add_argument("gate_file", nargs="?", default="configs/gate.yaml")
    p_gate.add_argument("--out", default="results/gate")
    p_gate.add_argument("--update", action="store_true", help="Write the new scores as the baseline")

    p_ans = sub.add_parser("answer", help="Generate cited answers for a sample of questions and grade them")
    p_ans.add_argument("config", help="An answer config, e.g. configs/answers/obliqa_bm25.yaml")
    p_ans.add_argument("--sample", type=int, help="Override the number of questions in the config")
    p_ans.add_argument("--generator", help='Override the writer model, e.g. "openai:gpt-6.1-terra"')
    p_ans.add_argument("--grader", help='Override the grader model, e.g. "anthropic:claude-sonnet-5-5"')
    p_ans.add_argument("--no-grade", action="store_true", help="Skip the grader; citation scores only")
    p_ans.add_argument("--out", default="results")

    args = parser.parse_args()
    if args.command == "answer":
        import os

        from .answering import run_answers, summary_markdown

        result = run_answers(
            args.config, args.out, args.sample, args.generator, args.grader, grade=not args.no_grade
        )
        if os.environ.get("GITHUB_STEP_SUMMARY"):
            with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as f:
                f.write("## Answer quality\n\n" + summary_markdown(result) + "\n")
        print(f"\n== {result['name']} ({result['n_questions']} questions)")
        print(f"writer: {result['generator']}   grader: {result['grader']}")
        if result.get("warning"):
            print(f"warning: {result['warning']}")
        print(json.dumps(result["metrics"], indent=2))
        print(json.dumps(result["usage"], indent=2))
    elif args.command == "gate":
        from .gate import gate

        sys.exit(0 if gate(args.gate_file, args.out, args.update) else 1)
    elif args.command == "run":
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
