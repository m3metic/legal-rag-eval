"""Retrieval metrics.

Two families:

* Rank metrics at a fixed k (recall@k, hit@k, MRR, nDCG). Easy to compare
  with published numbers, but biased toward big chunks: ten 500-word chunks
  cover more ground than ten 50-word chunks.
* Budget metrics (recall@Nw). Fill a fixed word budget, which is roughly what
  an LLM context window costs you, with chunks in rank order. This is the
  fair way to compare chunking strategies against each other.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from .types import Chunk, EvalItem, Span

# A gold span counts as found when a chunk overlaps it by at least this many
# characters, or by half the span if the span is shorter than that.
MIN_SPAN_OVERLAP_CHARS = 200


def _span_found(span: Span, chunk: Chunk) -> bool:
    need = min(MIN_SPAN_OVERLAP_CHARS, max(1, (span.end - span.start) // 2))
    return span.overlap(chunk) >= need


def gold_found_by(item: EvalItem, chunk: Chunk) -> set[str]:
    """Keys of the gold units/spans that this chunk provides evidence for."""
    found = {u for u in item.gold_units if u in chunk.unit_ids}
    for i, span in enumerate(item.gold_spans):
        if _span_found(span, chunk):
            found.add(f"span{i}")
    return found


def score_ranking(
    item: EvalItem,
    ranked: Sequence[Chunk],
    ks: Sequence[int] = (1, 3, 5, 10),
    word_budgets: Sequence[int] = (500, 1000, 2000),
) -> dict[str, float]:
    n_gold = item.n_gold
    if n_gold == 0:
        raise ValueError(f"{item.query_id} has no gold evidence")

    per_rank = [gold_found_by(item, c) for c in ranked]
    relevant = [bool(f) for f in per_rank]
    # For nDCG a chunk only earns credit if it adds gold evidence not already
    # found higher up. Overlapping chunks (sliding windows) can contain the
    # same passage several times; counting each copy would push nDCG above 1.
    novel, seen = [], set()
    for found in per_rank:
        novel.append(bool(found - seen))
        seen |= found
    out: dict[str, float] = {}

    for k in ks:
        covered: set[str] = set().union(*per_rank[:k]) if k else set()
        out[f"recall@{k}"] = len(covered) / n_gold
        out[f"hit@{k}"] = float(any(relevant[:k]))
        dcg = sum(1 / math.log2(r + 2) for r, rel in enumerate(novel[:k]) if rel)
        ideal = sum(1 / math.log2(r + 2) for r in range(min(k, n_gold)))
        out[f"ndcg@{k}"] = dcg / ideal if ideal else 0.0

    first = next((r for r, rel in enumerate(relevant) if rel), None)
    out["mrr"] = 0.0 if first is None else 1 / (first + 1)

    for budget in word_budgets:
        used, covered = 0, set()
        for chunk, found in zip(ranked, per_rank):
            if used + chunk.n_words > budget:
                break
            used += chunk.n_words
            covered |= found
        out[f"recall@{budget}w"] = len(covered) / n_gold

    return out


def aggregate(rows: Sequence[dict[str, float]]) -> dict[str, float]:
    """Macro-average per-query scores."""
    if not rows:
        return {}
    keys = [k for k in rows[0] if isinstance(rows[0][k], (int, float))]
    return {k: sum(r[k] for r in rows) / len(rows) for k in keys}
