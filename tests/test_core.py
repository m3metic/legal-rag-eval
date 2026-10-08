"""Tests for the parts where a silent bug would corrupt every result: metrics,
chunk offsets, and locating MAUD excerpts in the contracts."""

import math

import pytest

from legalrag.chunking import sections, word_windows
from legalrag.datasets.maud import locate, normalise
from legalrag.datasets.obliqa import _ancestors
from legalrag.metrics import score_ranking
from legalrag.retrievers import BM25Retriever, HybridRetriever
from legalrag.types import Chunk, EvalItem, Span


def chunk(cid, units=(), doc="d", start=None, end=None, text="x"):
    return Chunk(cid, doc, text, set(units), start, end)


# -- metrics ---------------------------------------------------------------
def test_recall_and_mrr_with_units():
    item = EvalItem("q", "query", gold_units={"a", "b"})
    ranked = [chunk("1"), chunk("2", {"a"}), chunk("3", {"b"})]
    s = score_ranking(item, ranked, ks=(1, 2, 3), word_budgets=())
    assert s["recall@1"] == 0
    assert s["recall@2"] == 0.5
    assert s["recall@3"] == 1
    assert s["mrr"] == 0.5
    assert s["hit@2"] == 1


def test_ndcg_perfect_ranking_is_one():
    item = EvalItem("q", "query", gold_units={"a", "b"})
    s = score_ranking(item, [chunk("1", {"a"}), chunk("2", {"b"})], ks=(2,), word_budgets=())
    assert math.isclose(s["ndcg@2"], 1.0)


def test_ndcg_never_exceeds_one_with_duplicate_evidence():
    # Overlapping windows can contain the same gold passage more than once.
    item = EvalItem("q", "query", gold_units={"a"})
    ranked = [chunk("1", {"a"}), chunk("2", {"a"}), chunk("3", {"a"})]
    s = score_ranking(item, ranked, ks=(3,), word_budgets=())
    assert math.isclose(s["ndcg@3"], 1.0)


def test_budget_excludes_chunks_that_do_not_fit():
    item = EvalItem("q", "query", gold_units={"a"})
    big = chunk("1", {"a"}, text="w " * 600)
    s = score_ranking(item, [big], ks=(1,), word_budgets=(500, 1000))
    assert s["recall@500w"] == 0  # a 600-word chunk does not fit in 500 words
    assert s["recall@1000w"] == 1


def test_span_gold_needs_real_overlap():
    item = EvalItem("q", "query", gold_spans=[Span("d", 1000, 2000)])
    s = score_ranking(item, [chunk("touch", start=1990, end=2500)], ks=(1,), word_budgets=())
    assert s["recall@1"] == 0  # 10 characters of overlap is not finding the clause
    s = score_ranking(item, [chunk("inside", start=1200, end=1500)], ks=(1,), word_budgets=())
    assert s["recall@1"] == 1


def test_item_without_gold_is_an_error():
    with pytest.raises(ValueError):
        score_ranking(EvalItem("q", "query"), [])


# -- chunking --------------------------------------------------------------
def test_word_windows_offsets_point_at_source_text():
    text = " ".join(f"w{i}" for i in range(25))
    chunks = word_windows("d", text, size=10, overlap=3)
    for c in chunks:
        assert text[c.start : c.end] == c.text
    assert chunks[0].text.split()[-3:] == chunks[1].text.split()[:3]
    assert chunks[-1].text.endswith("w24")


def test_sections_split_on_headings_and_add_heading_prefix():
    text = (
        "Preamble text here.   Section 1.1   Definitions. Alpha beta gamma.   "
        "Section 7.1   Termination. The agreement may be terminated."
    )
    chunks = sections("d", text, max_words=50, overlap=5)
    assert len(chunks) == 3
    assert chunks[2].text.startswith("[Section 7.1 Termination.]")
    assert "terminated" in text[chunks[2].start : chunks[2].end]


# -- dataset helpers -------------------------------------------------------
def test_obliqa_ancestors():
    assert _ancestors("14.2.3.Guidance.10.") == ["14", "14.2", "14.2.3", "14.2.3.Guidance"]
    assert _ancestors("1.") == []


def test_maud_locate_maps_back_to_original_offsets():
    original = "ARTICLE I\n\n“Company Material Adverse Effect”   means any event, change or effect that..."
    piece = "“Company Material Adverse Effect” means any event, change or effect"
    norm_doc, idx = normalise(original)
    s, e = locate(norm_doc, normalise(piece)[0].strip())
    found = original[idx[s] : idx[e - 1] + 1]
    assert found.startswith("Company Material") and found.endswith("effect")


def test_maud_locate_tolerates_trimmed_middle():
    doc = normalise("one two three four five six seven eight nine ten MIDDLE WORDS eleven twelve")[0]
    piece = normalise("one two three four five six seven eight nine ten eleven twelve")[0]
    assert locate(doc, piece, anchor_words=3) is not None


# -- retrievers ------------------------------------------------------------
CORPUS = [
    Chunk("a1", "A", "the company shall pay a termination fee of five million dollars"),
    Chunk("a2", "A", "material adverse effect means any change in the business"),
    Chunk("b1", "B", "termination fee payable by parent upon a financing failure"),
]


def test_bm25_ranks_keyword_match_first_and_respects_doc_filter():
    r = BM25Retriever()
    r.index(CORPUS)
    assert r.search("termination fee", k=1)[0][0].chunk_id in {"a1", "b1"}
    hits = r.search("termination fee", k=3, doc_filter={"B"})
    assert [c.chunk_id for c, _ in hits] == ["b1"]


def test_hybrid_fuses_without_duplicates():
    r = HybridRetriever([BM25Retriever(), BM25Retriever(k1=0.9)])
    r.index(CORPUS)
    ids = [c.chunk_id for c, _ in r.search("material adverse effect", k=3)]
    assert ids[0] == "a2" and len(ids) == len(set(ids))
