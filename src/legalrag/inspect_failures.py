"""Print retrieval failures side by side, for error analysis.

For each failed query: the question, the text the dataset marks as correct,
and the top results that were retrieved instead, so you can judge *why* the
retriever missed. Reads a run's per_query.jsonl and rebuilds the chunks from
the run's own config, so the text shown is exactly what the retriever saw.

    python -m legalrag inspect results/maud_hybrid_section300 \
        --group "Negative interim operating covenant" --n 5
"""

from __future__ import annotations

import json
import random
import textwrap
from pathlib import Path

from .datasets import load_dataset
from .types import EvalItem


def _clip(text: str, words: int) -> str:
    parts = " ".join(text.split()).split(" ")
    return " ".join(parts[:words]) + (" ..." if len(parts) > words else "")


def _wrap(text: str, indent: str = "    ") -> str:
    return textwrap.fill(text, width=100, initial_indent=indent, subsequent_indent=indent)


def _gold_texts(item: EvalItem, dataset, by_id: dict) -> list[str]:
    texts = []
    for unit in sorted(item.gold_units):
        chunk = by_id.get(unit)  # ObliQA passage chunks use the unit key as their ID
        texts.append(chunk.text if chunk else unit)
    for span in item.gold_spans:
        texts.append(dataset.contracts[span.doc_id][span.start : span.end])
    return texts


def inspect(
    run_dir: str | Path,
    metric: str = "recall@5",
    below: float = 1.0,
    group: str | None = None,
    n: int = 5,
    top: int = 3,
    words: int = 80,
    seed: int = 0,
) -> str:
    run_dir = Path(run_dir)
    result = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    cfg = result["config"]
    ds_cfg = dict(cfg["dataset"])
    ds_name = ds_cfg.pop("name")
    eval_cfg = ds_cfg.pop("eval", {})
    breakdown_key = cfg.get("breakdown_by")

    with open(run_dir / "per_query.jsonl", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    failures = [r for r in records if r["scores"][metric] < below]
    if group is not None:
        failures = [r for r in failures if str(r["metadata"].get(breakdown_key)) == group]
    header = f"{len(failures)} of {len(records)} queries have {metric} < {below}"
    if group is not None:
        header += f" in {breakdown_key} = {group!r}"
    if not failures:
        return header
    sample = random.Random(seed).sample(failures, min(n, len(failures)))

    dataset = load_dataset(ds_name, **ds_cfg)
    by_id = {c.chunk_id: c for c in dataset.chunks(**cfg.get("chunking", {}))}
    items = {i.query_id: i for i in dataset.eval_items(**eval_cfg)}

    out = [header, f"Showing {len(sample)} (random sample, seed {seed})", ""]
    for k, rec in enumerate(sample, 1):
        item = items[rec["query_id"]]
        out.append("=" * 100)
        out.append(f"[{k}] {rec['query_id']}   {metric} = {rec['scores'][metric]:.2f}")
        out.append(f"QUESTION: {rec['query']}")
        out.append("")
        out.append("CORRECT TEXT (from the dataset):")
        for text in _gold_texts(item, dataset, by_id):
            out.append(_wrap(_clip(text, words)))
        out.append("")
        out.append(f"RETRIEVED (top {top}):")
        for rank, hit in enumerate(rec["retrieved"][:top], 1):
            mark = "CORRECT" if hit["relevant"] else "wrong"
            chunk = by_id.get(hit["chunk_id"])
            out.append(f"  #{rank} [{mark}] {hit['chunk_id']}  score {hit['score']}")
            if chunk:
                out.append(_wrap(_clip(chunk.text, words)))
        first = next((i for i, h in enumerate(rec["retrieved"], 1) if h["relevant"]), None)
        out.append(f"  First correct result at rank: {first or 'not in results'}")
        out.append("")
    return "\n".join(out)
