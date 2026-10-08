"""MAUD: 152 public merger agreements annotated by lawyers for deal points.

Source: https://github.com/TheAtticusProject/maud (Wang et al., EMNLP 2023),
CC BY 4.0.

MAUD is built for clause *classification*: given the excerpt, answer a
deal-point question. We repurpose it for *retrieval*: given a deal point,
find the excerpt inside the full agreement, the way a reviewer working
through a data room would. Each excerpt is stored with "<omitted>" markers
between its pieces, so we locate every piece in the full contract text and
use those character spans as ground truth.

Search is restricted to the agreement under review (`doc_filter`), so the
task is "find the MAE definition in *this* contract", not across all 152.
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

import yaml

from ..chunking import chunk_text
from ..types import Chunk, EvalItem, Span

DEFAULT_ROOT = Path("data/raw/maud")
DEFAULT_QUERIES = Path("configs/maud_queries.yaml")
MIN_PIECE_WORDS = 6


def normalise(text: str) -> tuple[str, list[int]]:
    """Lowercase, keep alphanumerics, collapse everything else to one space.

    Returns the normalised string and, for each of its characters, the index
    of the original character it came from, so matches map back to offsets.
    """
    out, idx, prev_space = [], [], True
    for i, ch in enumerate(text):
        if ch.isalnum():
            out.append(ch.lower())
            idx.append(i)
            prev_space = False
        elif not prev_space:
            out.append(" ")
            idx.append(i)
            prev_space = True
    return "".join(out), idx


def _find_all(text: str, sub: str) -> list[int]:
    out, i = [], text.find(sub)
    while i >= 0:
        out.append(i)
        i = text.find(sub, i + 1)
    return out


def candidates(norm_doc: str, piece: str, anchor_words: int = 10) -> list[tuple[int, int]]:
    """Every place a normalised excerpt piece could sit in a normalised contract.

    Exact matches first. Annotators sometimes trimmed words inside a piece, so
    fall back to matching its first and last `anchor_words` words. A piece can
    match several places: a section heading also appears in the table of
    contents, and boilerplate phrases repeat. `choose_spans` picks between them.
    """
    words = piece.split()
    if len(words) < MIN_PIECE_WORDS:
        return []
    exact = _find_all(norm_doc, piece)
    if exact:
        return [(i, i + len(piece)) for i in exact]
    head, tail = " ".join(words[:anchor_words]), " ".join(words[-anchor_words:])
    out = []
    for i in _find_all(norm_doc, head):
        j = norm_doc.find(tail, i)
        out.append((i, j + len(tail)) if 0 <= j and j - i <= 2 * len(piece) else (i, min(len(norm_doc), i + len(piece))))
    return out


def choose_spans(per_piece: list[list[tuple[int, int]]]) -> list[tuple[int, int]]:
    """Pick one location per piece, keeping an excerpt's pieces close together.

    Pieces that match exactly one place are anchors. An ambiguous piece takes
    the candidate nearest the anchors' median position. With no anchors, the
    reference is the median of each piece's last candidate, which steers away
    from the table of contents at the front of the contract.
    """
    found = [c for c in per_piece if c]
    if not found:
        return []
    anchors = sorted(c[0][0] for c in found if len(c) == 1)
    refs = anchors or sorted(c[-1][0] for c in found)
    ref = refs[len(refs) // 2]
    return [min(c, key=lambda span: abs(span[0] - ref)) for c in found]


def locate(norm_doc: str, piece: str, anchor_words: int = 10) -> tuple[int, int] | None:
    """First match of a single piece (kept for simple lookups and tests)."""
    found = candidates(norm_doc, piece, anchor_words)
    return found[0] if found else None


class MAUD:
    name = "maud"

    def __init__(
        self,
        root: str | Path = DEFAULT_ROOT,
        queries: str | Path = DEFAULT_QUERIES,
        max_contracts: int | None = None,
    ):
        self.root = Path(root)
        data = self.root / "data"
        if not (data / "contracts").exists():
            raise FileNotFoundError(f"{data}/contracts not found. Run scripts/download_data.sh first.")
        names = sorted((p.stem for p in (data / "contracts").glob("*.txt")), key=lambda s: int(s.split("_")[1]))
        if max_contracts:
            names = names[:max_contracts]
        self.contracts = {n: (data / "contracts" / f"{n}.txt").read_text(encoding="utf-8", errors="ignore") for n in names}
        self.queries: dict[str, str] = yaml.safe_load(Path(queries).read_text(encoding="utf-8"))
        self._excerpts = self._load_excerpts(data)

    def _load_excerpts(self, data: Path) -> dict[tuple[str, str], str]:
        csv.field_size_limit(sys.maxsize)
        excerpts = {}
        for split in ("train", "dev", "test"):
            with open(data / f"MAUD_{split}.csv", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    # "abridged" rows are shortened copies; "main" rows are the full excerpt.
                    if row["data_type"] == "main" and row["contract_name"] in self.contracts:
                        excerpts.setdefault((row["contract_name"], row["text_type"]), row["text"])
        return excerpts

    # -- corpus -------------------------------------------------------------
    def chunks(self, strategy: str = "window", **params) -> list[Chunk]:
        out = []
        for name, text in self.contracts.items():
            out.extend(chunk_text(name, text, strategy, **params))
        return out

    # -- evaluation set -----------------------------------------------------
    def eval_items(self, query_mode: str = "question", limit: int | None = None, **_) -> list[EvalItem]:
        """query_mode: "question" uses configs/maud_queries.yaml; "label" uses the raw deal-point name."""
        norm_cache: dict[str, tuple[str, list[int]]] = {}
        items, stats = [], Counter()
        for (contract, deal_point), excerpt in sorted(self._excerpts.items()):
            if contract not in norm_cache:
                norm_cache[contract] = normalise(self.contracts[contract])
            norm_doc, idx = norm_cache[contract]
            per_piece = [candidates(norm_doc, normalise(p)[0].strip()) for p in excerpt.split("<omitted>")]
            spans = [Span(contract, idx[s], idx[e - 1] + 1) for s, e in choose_spans(per_piece)]
            if not spans:
                stats["unlocated"] += 1
                continue
            stats["located"] += 1
            query = deal_point if query_mode == "label" else self.queries.get(deal_point, deal_point)
            items.append(
                EvalItem(
                    query_id=f"{contract}::{deal_point}",
                    query=query,
                    gold_spans=spans,
                    doc_filter={contract},
                    metadata={"deal_point": deal_point, "contract": contract},
                )
            )
        self.locate_stats = dict(stats)
        return items[:limit] if limit else items
