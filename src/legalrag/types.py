"""Core data types shared by every dataset, chunker and retriever.

The key idea: retrieval quality is measured against *gold units*, not chunks.
A gold unit is the smallest piece of evidence the dataset labels as relevant
(an ObliQA passage ID, or a character span of a MAUD merger agreement).
Each chunk records which units it covers, so we can change the chunking
strategy freely and still score it against the same ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    # Gold-unit keys this chunk contains (ObliQA: "doc::passage_id").
    unit_ids: set[str] = field(default_factory=set)
    # Character span in the source document (MAUD), used for span overlap.
    start: int | None = None
    end: int | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def n_words(self) -> int:
        return len(self.text.split())


@dataclass
class Span:
    doc_id: str
    start: int
    end: int

    def overlap(self, chunk: Chunk) -> int:
        if chunk.doc_id != self.doc_id or chunk.start is None or chunk.end is None:
            return 0
        return max(0, min(self.end, chunk.end) - max(self.start, chunk.start))


@dataclass
class EvalItem:
    query_id: str
    query: str
    # Either unit IDs (passage-labelled datasets) or spans (span-labelled datasets).
    gold_units: set[str] = field(default_factory=set)
    gold_spans: list[Span] = field(default_factory=list)
    # Restrict search to these documents (e.g. "the agreement under review").
    doc_filter: set[str] | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def n_gold(self) -> int:
        return len(self.gold_units) + len(self.gold_spans)
