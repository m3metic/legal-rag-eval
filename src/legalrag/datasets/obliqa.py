"""ObliQA: compliance questions over 40 ADGM financial regulation documents.

Source: https://github.com/RegNLP/ObliQADataset (Gokhan et al., 2024).

The regulations come pre-split into numbered passages ("1.2.1.(1)"), and each
question lists the passages that answer it. Gold units are therefore
"<DocumentID>::<PassageID>" keys.

Chunking strategies:
  passage         one chunk per non-empty passage (the dataset's native unit)
  passage_header  same, prefixed with the titles of its parent sections
  window          `size` consecutive passages per chunk, sliding by `stride`
"""

from __future__ import annotations

import json
from pathlib import Path

from ..types import Chunk, EvalItem

DEFAULT_ROOT = Path("data/raw/ObliQADataset")


def unit_key(doc_id, passage_id) -> str:
    return f"{doc_id}::{passage_id}"


def _ancestors(passage_id: str) -> list[str]:
    """'14.2.3.Guidance.10.' -> ['14', '14.2', '14.2.3', '14.2.3.Guidance']"""
    parts = [p for p in passage_id.strip(".").split(".") if p]
    return [".".join(parts[:i]) for i in range(1, len(parts))]


class ObliQA:
    name = "obliqa"

    def __init__(self, root: str | Path = DEFAULT_ROOT):
        self.root = Path(root)
        doc_dir = self.root / "StructuredRegulatoryDocuments"
        if not doc_dir.exists():
            raise FileNotFoundError(f"{doc_dir} not found. Run scripts/download_data.sh first.")
        self.docs: dict[str, list[dict]] = {}
        for f in sorted(doc_dir.glob("*.json"), key=lambda p: int(p.stem)):
            passages = json.loads(f.read_text(encoding="utf-8"))
            self.docs[str(passages[0]["DocumentID"])] = passages

    # -- corpus -------------------------------------------------------------
    def chunks(self, strategy: str = "passage", **params) -> list[Chunk]:
        if strategy == "passage":
            return self._passages(with_headers=False)
        if strategy == "passage_header":
            return self._passages(with_headers=True)
        if strategy == "window":
            return self._windows(**params)
        raise ValueError(f"Unknown ObliQA chunking strategy: {strategy!r}")

    def _titles(self, passages: list[dict]) -> dict[str, str]:
        # Short passages are section titles ("Application", "INTRODUCTION").
        return {
            p["PassageID"].strip("."): p["Passage"].strip()
            for p in passages
            if 0 < len(p["Passage"].split()) <= 12
        }

    def _passages(self, with_headers: bool) -> list[Chunk]:
        out = []
        for doc_id, passages in self.docs.items():
            titles = self._titles(passages) if with_headers else {}
            for p in passages:
                text = p["Passage"].strip()
                if not text:
                    continue
                if with_headers:
                    path = [titles[a] for a in _ancestors(p["PassageID"]) if a in titles]
                    if path:
                        text = f"[{' > '.join(path)}] {text}"
                out.append(
                    Chunk(
                        chunk_id=unit_key(doc_id, p["PassageID"]),
                        doc_id=doc_id,
                        text=text,
                        unit_ids={unit_key(doc_id, p["PassageID"])},
                    )
                )
        return out

    def _windows(self, size: int = 3, stride: int = 2) -> list[Chunk]:
        out = []
        for doc_id, passages in self.docs.items():
            ps = [p for p in passages if p["Passage"].strip()]
            for i in range(0, max(1, len(ps) - size + stride), stride):
                group = ps[i : i + size]
                if not group:
                    break
                out.append(
                    Chunk(
                        chunk_id=f"{doc_id}::win{i}",
                        doc_id=doc_id,
                        text="\n".join(p["Passage"].strip() for p in group),
                        unit_ids={unit_key(doc_id, p["PassageID"]) for p in group},
                    )
                )
        return out

    # -- evaluation set -----------------------------------------------------
    def eval_items(self, split: str = "test", limit: int | None = None) -> list[EvalItem]:
        rows = json.loads((self.root / f"ObliQA_{split}.json").read_text(encoding="utf-8"))
        items = [
            EvalItem(
                query_id=q["QuestionID"],
                query=q["Question"],
                gold_units={unit_key(p["DocumentID"], p["PassageID"]) for p in q["Passages"]},
                metadata={"n_passages": len(q["Passages"]), "group": q.get("Group")},
            )
            for q in rows
        ]
        return items[:limit] if limit else items
