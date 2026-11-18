"""Tests for answer generation and grading, using fake models (no network, no API keys)."""

import json

import pytest

from legalrag.answering import (
    ABSTAIN,
    aggregate_answers,
    citation_scores,
    parse_citations,
    parse_grade,
    run_answers,
    sample_items,
)
from legalrag.llm import LLM
from legalrag.types import Chunk, EvalItem


def chunk(cid):
    return Chunk(chunk_id=cid, doc_id=cid.split("::")[0], text=f"text of {cid}", unit_ids={cid})


CONTEXT = [chunk("1::a"), chunk("1::b"), chunk("2::c")]
ITEM = EvalItem(query_id="q", query="?", gold_units={"1::a", "2::c", "9::z"})


def test_parse_citations_handles_both_styles_and_dedupes():
    assert parse_citations("Must notify [1][3]. Also [3, 2]; see [1].") == [1, 3, 2]
    assert parse_citations("No citations here.") == []


def test_citation_scores():
    s = citation_scores(ITEM, CONTEXT, [1, 2], abstained=False)
    assert s["context_recall"] == pytest.approx(2 / 3)  # 9::z was never retrieved
    assert s["citation_precision"] == 0.5  # 1::a gold, 1::b not
    assert s["citation_recall"] == pytest.approx(1 / 3)
    assert s["citation_recall_in_context"] == 0.5  # cited 1 of the 2 gold it was shown
    assert not s["invalid_citation"] and not s["uncited"]


def test_out_of_range_citation_is_invalid_and_ignored():
    s = citation_scores(ITEM, CONTEXT, [1, 7], abstained=False)
    assert s["invalid_citation"]
    assert s["citation_precision"] == 1.0


def test_answer_without_citations_is_flagged_but_abstention_is_not():
    assert citation_scores(ITEM, CONTEXT, [], abstained=False)["uncited"]
    assert not citation_scores(ITEM, CONTEXT, [], abstained=True)["uncited"]


def test_parse_grade_tolerates_code_fences_and_rejects_garbage():
    g = parse_grade('```json\n{"claims": [], "contradicts_reference": false, "obligations": []}\n```')
    assert g == {"claims": [], "contradicts_reference": False, "obligations": []}
    assert parse_grade("I cannot grade this.") is None


def test_abstentions_count_as_zero_coverage_and_are_left_out_of_precision():
    rows = [
        {**citation_scores(ITEM, CONTEXT, [1], False), "obligation_coverage": 1.0, "faithfulness": 1.0,
         "contradiction": False},
        {**citation_scores(ITEM, CONTEXT, [], True), "obligation_coverage": 0.0},
    ]
    m = aggregate_answers(rows)
    assert m["abstain_rate"] == 0.5
    assert m["citation_precision"] == 1.0  # only the answered question
    assert m["obligation_coverage"] == 0.5
    assert m["faithfulness"] == 1.0


def test_sample_is_fixed_and_keeps_order():
    items = [EvalItem(query_id=str(i), query="") for i in range(50)]
    a, b = sample_items(items, 10, seed=0), sample_items(items, 10, seed=0)
    assert [i.query_id for i in a] == [i.query_id for i in b]
    assert [int(i.query_id) for i in a] == sorted(int(i.query_id) for i in a)
    assert sample_items(items, None) == items


def test_llm_cache_avoids_repeat_calls(tmp_path):
    calls = []
    llm = LLM("fake:x", cache_dir=tmp_path, responder=lambda s, p: calls.append(p) or "hi")
    assert llm.complete("sys", "p") == "hi" and llm.complete("sys", "p") == "hi"
    assert len(calls) == 1 and llm.usage.cached_calls == 1


def test_llm_rejects_unknown_provider():
    with pytest.raises(ValueError):
        LLM("acme:model")


# -- end to end on the real ObliQA data, with fake models --------------------------
def _obliqa_available():
    from pathlib import Path

    return Path("data/raw/ObliQADataset/StructuredRegulatoryDocuments").exists()


@pytest.mark.skipif(not _obliqa_available(), reason="ObliQA not downloaded")
def test_run_answers_end_to_end_with_fake_models(tmp_path):
    def writer(system, prompt):
        return "The firm must do this [1]." if "QUESTION" in prompt else ABSTAIN

    def grader(system, prompt):
        return json.dumps(
            {"claims": [{"text": "x", "supported": True}], "contradicts_reference": False,
             "obligations": [{"text": "a", "covered": True}, {"text": "b", "covered": False}]}
        )

    fakes = {
        "generator": LLM("fake:writer", cache_dir=None, responder=writer),
        "grader": LLM("fake:grader", cache_dir=None, responder=grader),
    }
    result = run_answers("configs/answers/obliqa_bm25.yaml", tmp_path, sample=5, llm_overrides=fakes)
    m = result["metrics"]
    assert result["n_questions"] == 5
    assert m["faithfulness"] == 1.0 and m["obligation_coverage"] == 0.5 and m["abstain_rate"] == 0.0
    # Both fakes share a provider, so the same-family warning must appear.
    assert result["grader_same_family_as_writer"] and "warning" in result
    lines = (tmp_path / result["name"] / "answers.jsonl").read_text().splitlines()
    assert len(lines) == 5 and json.loads(lines[0])["cited"] == [1]
