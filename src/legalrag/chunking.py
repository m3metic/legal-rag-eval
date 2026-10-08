"""Chunkers for raw text documents (used for the MAUD merger agreements).

Every chunk keeps the character offsets of the source text it came from, so
span-labelled ground truth can be matched to it however it was cut.
"""

from __future__ import annotations

import itertools
import re

from .types import Chunk

_WORD = re.compile(r"\S+")

# "Section 5.3   Efforts; Notice of Certain Events." / "ARTICLE VI"
SECTION_HEADING = re.compile(
    r"(?:(?<=\s)|^)("
    r"(?:Section|SECTION)\s+\d+(?:\.\d+)+\s+[A-Z][^.]{2,150}\."
    r"|(?:ARTICLE|Article)\s+(?:[IVXLC]+|\d+)\b"
    r")"
)


def word_windows(
    doc_id: str,
    text: str,
    size: int = 200,
    overlap: int = 50,
    offset: int = 0,
    prefix: str = "",
    id_prefix: str = "",
) -> list[Chunk]:
    """Fixed-size sliding windows of `size` words, `overlap` words shared."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    words = list(_WORD.finditer(text))
    chunks: list[Chunk] = []
    step = size - overlap
    for n, i in enumerate(range(0, max(1, len(words) - overlap), step)):
        window = words[i : i + size]
        if not window:
            break
        s, e = window[0].start(), window[-1].end()
        body = text[s:e]
        chunks.append(
            Chunk(
                chunk_id=f"{doc_id}::{id_prefix}w{n}",
                doc_id=doc_id,
                text=f"{prefix}{body}" if prefix else body,
                start=offset + s,
                end=offset + e,
            )
        )
    return chunks


def sections(
    doc_id: str,
    text: str,
    max_words: int = 300,
    overlap: int = 50,
    add_heading: bool = True,
) -> list[Chunk]:
    """Split on contract section headings, then window any long sections.

    With `add_heading`, each chunk is prefixed with its section heading, so a
    window from the middle of "Section 7.1 Termination" still mentions
    termination. This is a cheap form of contextual chunking.
    """
    starts = [m.start(1) for m in SECTION_HEADING.finditer(text)]
    bounds = [0, *starts, len(text)]
    chunks: list[Chunk] = []
    for n, (s, e) in enumerate(itertools.pairwise(bounds)):
        part = text[s:e]
        if not part.strip():
            continue
        m = SECTION_HEADING.match(part.lstrip())
        heading = re.sub(r"\s+", " ", m.group(1)).strip() if m else ""
        prefix = f"[{heading}] " if (add_heading and heading) else ""
        chunks.extend(
            word_windows(doc_id, part, max_words, overlap, offset=s, prefix=prefix, id_prefix=f"s{n}.")
        )
    return chunks


def chunk_text(doc_id: str, text: str, strategy: str, **params) -> list[Chunk]:
    if strategy == "window":
        return word_windows(doc_id, text, **params)
    if strategy == "section":
        return sections(doc_id, text, **params)
    raise ValueError(f"Unknown text chunking strategy: {strategy!r}")
