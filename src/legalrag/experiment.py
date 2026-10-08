"""Run one retrieval experiment from a YAML config and save the results.

results/<name>/
    metrics.json      config, corpus stats, overall scores, breakdowns, timings
    per_query.jsonl   every query with its top results and scores (for error analysis)
"""

from __future__ import annotations

import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

import yaml

from .datasets import load_dataset
from .metrics import aggregate, gold_found_by, score_ranking
from .retrievers import build_retriever


def run(config_path: str | Path, limit: int | None = None, out_root: str | Path = "results") -> dict:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    name = cfg["name"] + (f"__limit{limit}" if limit else "")
    ds_cfg = dict(cfg["dataset"])
    ds_name = ds_cfg.pop("name")
    eval_cfg = ds_cfg.pop("eval", {})
    if limit:
        eval_cfg["limit"] = limit
    depth = cfg.get("depth", 20)
    breakdown_key = cfg.get("breakdown_by")

    t0 = time.perf_counter()
    dataset = load_dataset(ds_name, **ds_cfg)
    chunks = dataset.chunks(**cfg.get("chunking", {}))
    items = dataset.eval_items(**eval_cfg)
    t_load = time.perf_counter() - t0

    retriever = build_retriever(cfg["retriever"])
    t0 = time.perf_counter()
    retriever.index(chunks)
    t_index = time.perf_counter() - t0

    rows, records = [], []
    t0 = time.perf_counter()
    for item in items:
        hits = retriever.search(item.query, depth, item.doc_filter)
        ranked = [c for c, _ in hits]
        scores = score_ranking(item, ranked)
        rows.append(scores)
        records.append(
            {
                "query_id": item.query_id,
                "query": item.query,
                "n_gold": item.n_gold,
                "metadata": item.metadata,
                "scores": scores,
                "retrieved": [
                    {"chunk_id": c.chunk_id, "score": round(float(s), 4), "relevant": bool(gold_found_by(item, c))}
                    for c, s in hits
                ],
            }
        )
    t_search = time.perf_counter() - t0

    breakdown = {}
    if breakdown_key:
        groups = defaultdict(list)
        for item, row in zip(items, rows):
            groups[str(item.metadata.get(breakdown_key))].append(row)
        breakdown = {g: {"n": len(r), **aggregate(r)} for g, r in sorted(groups.items())}

    words = [c.n_words for c in chunks]
    result = {
        "name": name,
        "config": cfg,
        "corpus": {
            "n_chunks": len(chunks),
            "chunk_words_mean": round(statistics.mean(words), 1),
            "chunk_words_median": statistics.median(words),
        },
        "n_queries": len(items),
        "dataset_notes": getattr(dataset, "locate_stats", None),
        "metrics": aggregate(rows),
        "breakdown_by": breakdown_key,
        "breakdown": breakdown,
        "timing_s": {
            "load_and_chunk": round(t_load, 2),
            "index": round(t_index, 2),
            "search_total": round(t_search, 2),
            "search_ms_per_query": round(1000 * t_search / max(1, len(items)), 2),
        },
    }

    out = Path(out_root) / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with open(out / "per_query.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in records)
    return result


DEFAULT_COLUMNS = ["recall@1", "recall@5", "recall@10", "mrr", "ndcg@10", "recall@500w", "recall@1000w"]


def compare(paths: list[str | Path], columns: list[str] | None = None) -> str:
    """Markdown table of results, ready to paste into the README."""
    columns = columns or DEFAULT_COLUMNS
    results = []
    for p in paths:
        p = Path(p)
        f = p / "metrics.json" if p.is_dir() else p
        if f.exists():
            results.append(json.loads(f.read_text(encoding="utf-8")))
    header = ["experiment", "chunks", "words/chunk", *columns, "ms/query"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in sorted(results, key=lambda r: r["name"]):
        m = r["metrics"]
        cells = [
            r["name"],
            str(r["corpus"]["n_chunks"]),
            str(r["corpus"]["chunk_words_mean"]),
            *(f"{m.get(c, float('nan')):.3f}" for c in columns),
            str(r["timing_s"]["search_ms_per_query"]),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
