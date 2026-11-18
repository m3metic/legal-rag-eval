"""Answer generation with citations, and how good those answers are.

For each question in a fixed random sample:
  1. retrieve the top-k passages with the experiment's retriever
  2. a writer model answers using only those passages, citing them as [1], [2]
  3. citations are scored against the gold passages, with no model involved
  4. a grader model (ideally from another provider) checks every claim against
     the passages it cites, and checks which obligations in the gold passages
     the answer covers

Scores, all averaged over the sample:
  context_recall            gold passages that were in the top-k at all (the ceiling)
  citation_precision        cited passages that are gold (answered questions only)
  citation_recall           gold passages that were cited
  citation_recall_in_context  gold passages cited, out of those the writer was shown;
                            separates "retrieval missed it" from "the writer ignored it"
  invalid_citation_rate     answers citing a number that was not in the context
  uncited_rate              answers with no citation at all
  abstain_rate              answers saying the passages are not enough
  faithfulness              claims supported by the passages they cite (grader)
  obligation_coverage       obligations in the gold passages the answer states (grader)
  contradiction_rate        answers contradicting the gold passages (grader)

ObliQA has no reference answers (the RIRAG shared task it comes from also grades
without them), so the gold passages are the reference.

results/<name>/
    answer_metrics.json   config, scores, token usage
    answers.jsonl         every question, its context, answer, citations and grades
"""

from __future__ import annotations

import json
import random
import re
import statistics
import time
from pathlib import Path

import yaml

from .datasets import load_dataset
from .llm import LLM
from .retrievers import build_retriever
from .types import Chunk, EvalItem

ABSTAIN = "INSUFFICIENT EVIDENCE"

WRITER_SYSTEM = f"""You answer compliance questions about Abu Dhabi Global Market (ADGM) financial \
regulations, using only the numbered passages you are given.

- State every obligation, requirement or condition in the passages that answers the question.
- After each sentence, cite the passage or passages it relies on, like [2] or [1][3].
- Do not use outside knowledge or cite anything not in the list.
- If the passages do not answer the question, reply with exactly: {ABSTAIN}"""

GRADER_SYSTEM = """You grade answers to regulatory compliance questions strictly and return only JSON."""

GRADER_TEMPLATE = """QUESTION
{question}

ANSWER
{answer}

PASSAGES THE ANSWER CITED
{cited}

REFERENCE PASSAGES (what a complete answer should draw on)
{reference}

Do three things:
1. Split the ANSWER into its individual factual claims. For each claim, decide whether the \
PASSAGES THE ANSWER CITED fully support it. A claim with no cited passage is unsupported.
2. Decide whether the ANSWER contradicts anything in the REFERENCE PASSAGES.
3. List each distinct obligation, requirement, condition or permission in the REFERENCE \
PASSAGES that is relevant to the QUESTION, and decide whether the ANSWER states it.

Return JSON only, in this shape:
{{"claims": [{{"text": "...", "supported": true}}],
  "contradicts_reference": false,
  "obligations": [{{"text": "...", "covered": true}}]}}"""


# -- prompts and citations ----------------------------------------------------
def build_prompt(question: str, context: list[Chunk]) -> str:
    passages = "\n\n".join(f"[{i}] {c.chunk_id}\n{c.text}" for i, c in enumerate(context, 1))
    return f"QUESTION\n{question}\n\nPASSAGES\n{passages}"


_CITE = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]")


def parse_citations(answer: str) -> list[int]:
    """All citation numbers in order of first use: "[1][3]" and "[1, 3]" both give [1, 3]."""
    seen: list[int] = []
    for group in _CITE.findall(answer):
        for n in re.split(r"\s*[,;]\s*", group):
            if int(n) not in seen:
                seen.append(int(n))
    return seen


def citation_scores(item: EvalItem, context: list[Chunk], cited: list[int], abstained: bool) -> dict:
    gold = item.gold_units
    valid = [context[n - 1] for n in cited if 1 <= n <= len(context)]
    in_context = gold & set().union(*(c.unit_ids for c in context)) if context else set()
    cited_units = set().union(*(c.unit_ids for c in valid)) if valid else set()
    return {
        "context_recall": len(in_context) / len(gold) if gold else 0.0,
        "citation_precision": (sum(bool(c.unit_ids & gold) for c in valid) / len(valid)) if valid else 0.0,
        "citation_recall": len(gold & cited_units) / len(gold) if gold else 0.0,
        "citation_recall_in_context": (len(in_context & cited_units) / len(in_context)) if in_context else None,
        "invalid_citation": any(not 1 <= n <= len(context) for n in cited),
        "uncited": not abstained and not valid,
        "abstained": abstained,
    }


# -- grading --------------------------------------------------------------------
def parse_grade(text: str) -> dict | None:
    """The grader's JSON, tolerating code fences or text around it. None if unreadable."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        g = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return g if isinstance(g, dict) else None


def grade_scores(grade: dict | None) -> dict:
    if grade is None:
        return {"faithfulness": None, "obligation_coverage": None, "contradiction": None}
    claims = grade.get("claims") or []
    obligations = grade.get("obligations") or []
    return {
        "faithfulness": (sum(bool(c.get("supported")) for c in claims) / len(claims)) if claims else None,
        "obligation_coverage": (
            sum(bool(o.get("covered")) for o in obligations) / len(obligations) if obligations else None
        ),
        "contradiction": bool(grade.get("contradicts_reference")),
    }


# -- running --------------------------------------------------------------------
def _mean(values) -> float | None:
    vals = [float(v) for v in values if v is not None]
    return round(statistics.mean(vals), 4) if vals else None


def aggregate_answers(rows: list[dict]) -> dict:
    answered = [r for r in rows if not r["abstained"]]
    return {
        "context_recall": _mean(r["context_recall"] for r in rows),
        "citation_precision": _mean(r["citation_precision"] for r in answered),
        "citation_recall": _mean(r["citation_recall"] for r in rows),
        "citation_recall_in_context": _mean(r["citation_recall_in_context"] for r in rows),
        "invalid_citation_rate": _mean(r["invalid_citation"] for r in rows),
        "uncited_rate": _mean(r["uncited"] for r in rows),
        "abstain_rate": _mean(r["abstained"] for r in rows),
        # An abstention covers no obligations, so it counts as 0 rather than being skipped.
        "obligation_coverage": _mean(
            0.0 if r["abstained"] else r.get("obligation_coverage") for r in rows if "obligation_coverage" in r
        ),
        "faithfulness": _mean(r.get("faithfulness") for r in answered),
        "contradiction_rate": _mean(r.get("contradiction") for r in answered),
    }


LABELS = {
    "context_recall": "Gold passages retrieved into the context (ceiling)",
    "citation_precision": "Cited passages that are gold",
    "citation_recall": "Gold passages cited",
    "citation_recall_in_context": "Gold passages cited, of those shown to the writer",
    "invalid_citation_rate": "Answers citing a passage that was not shown",
    "uncited_rate": "Answers with no citation",
    "abstain_rate": "Answers saying the passages are not enough",
    "faithfulness": "Claims supported by their cited passages (grader)",
    "obligation_coverage": "Gold obligations the answer states (grader)",
    "contradiction_rate": "Answers contradicting the gold passages (grader)",
}


def summary_markdown(result: dict) -> str:
    """Results as a markdown table, for the GitHub Actions run page."""
    lines = [
        (
            f"**{result['name']}**: {result['n_questions']} questions · writer `{result['generator']}` · "
            f"grader `{result['grader']}`"
        ),
        "",
        "| metric | value |",
        "|---|---|",
    ]
    for key, label in LABELS.items():
        v = result["metrics"].get(key)
        lines.append(f"| {label} | {'n/a' if v is None else f'{v:.3f}'} |")
    if result.get("warning"):
        lines += ["", f"> {result['warning']}"]
    u = result["usage"]
    lines += ["", "Tokens: " + "; ".join(
        f"{role} {d['input_tokens']:,} in / {d['output_tokens']:,} out ({d['cached_calls']} cached calls)"
        for role, d in u.items()
    )]
    return "\n".join(lines)


def sample_items(items: list[EvalItem], n: int | None, seed: int = 0) -> list[EvalItem]:
    """A fixed random sample, so every run and every model answers the same questions."""
    if not n or n >= len(items):
        return items
    picked = set(random.Random(seed).sample(range(len(items)), n))
    return [it for i, it in enumerate(items) if i in picked]


def run_answers(
    config_path: str | Path,
    out_root: str | Path = "results",
    sample: int | None = None,
    generator: str | None = None,
    grader: str | None = None,
    grade: bool = True,
    llm_overrides: dict | None = None,
) -> dict:
    """Run one answer experiment. `llm_overrides` lets tests pass fake models."""
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    rcfg = yaml.safe_load(Path(cfg["retrieval"]).read_text(encoding="utf-8"))
    k = int(cfg.get("k", 5))
    n = sample if sample is not None else cfg.get("sample")
    generator = generator or cfg["generator"]
    grader = grader or cfg.get("grader")
    name = cfg["name"] + (f"__sample{n}" if sample is not None else "")

    ds_cfg = dict(rcfg["dataset"])
    ds_name = ds_cfg.pop("name")
    if ds_name != "obliqa":
        raise NotImplementedError("Answer generation is implemented for ObliQA so far")
    eval_cfg = ds_cfg.pop("eval", {})
    dataset = load_dataset(ds_name, **ds_cfg)
    chunks = dataset.chunks(**rcfg.get("chunking", {}))
    # Gold passages in full, as the grader's reference, whatever the experiment's chunking.
    unit_text = {c.chunk_id: c.text for c in dataset.chunks(strategy="passage")}
    items = sample_items(dataset.eval_items(**eval_cfg), n, int(cfg.get("seed", 0)))

    retriever = build_retriever(rcfg["retriever"])
    retriever.index(chunks)

    overrides = llm_overrides or {}
    writer = overrides.get("generator") or LLM(generator, max_tokens=int(cfg.get("max_tokens", 800)))
    judge = None
    if grade and grader:
        judge = overrides.get("grader") or LLM(grader, max_tokens=2000)

    rows, records = [], []
    t0 = time.perf_counter()
    for item in items:
        context = [c for c, _ in retriever.search(item.query, k, item.doc_filter)]
        answer = writer.complete(WRITER_SYSTEM, build_prompt(item.query, context)).strip()
        abstained = answer.upper().startswith(ABSTAIN)
        cited = [] if abstained else parse_citations(answer)
        row = citation_scores(item, context, cited, abstained)

        grade_json = None
        if judge and not abstained:
            cited_text = "\n\n".join(
                f"[{n}] {context[n - 1].text}" for n in cited if 1 <= n <= len(context)
            ) or "(none)"
            reference = "\n\n".join(unit_text.get(u, "") for u in sorted(item.gold_units))
            raw = judge.complete(
                GRADER_SYSTEM,
                GRADER_TEMPLATE.format(question=item.query, answer=answer, cited=cited_text, reference=reference),
            )
            grade_json = parse_grade(raw)
            row.update(grade_scores(grade_json))
        elif judge:
            row.update({"obligation_coverage": 0.0})

        rows.append(row)
        records.append(
            {
                "query_id": item.query_id,
                "question": item.query,
                "gold": sorted(item.gold_units),
                "context": [c.chunk_id for c in context],
                "answer": answer,
                "cited": cited,
                "scores": row,
                "grade": grade_json,
            }
        )

    same_family = bool(judge) and judge.family == writer.family
    result = {
        "name": name,
        "config": cfg,
        "generator": writer.spec,
        "grader": judge.spec if judge else None,
        "grader_same_family_as_writer": same_family,
        "n_questions": len(items),
        "metrics": aggregate_answers(rows),
        "ungradable": sum(1 for r in records if judge and r["grade"] is None and not r["scores"]["abstained"]),
        "usage": {"generator": writer.usage.as_dict(), **({"grader": judge.usage.as_dict()} if judge else {})},
        "seconds": round(time.perf_counter() - t0, 1),
    }
    if same_family:
        result["warning"] = "The grader is from the same model family as the writer and may score it kindly."

    out = Path(out_root) / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "answer_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with open(out / "answers.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in records)
    return result
