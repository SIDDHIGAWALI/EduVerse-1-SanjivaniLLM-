"""
Extractive answer builder — a fallback for when EduVerse-1 isn't trained
enough yet to generate fluent text. Produces clean, correct answers built
directly from retrieved context, with no generation and no hallucination
risk. See pipeline.py's --extractive flag for how this plugs in.
"""

from __future__ import annotations
import re

from src.rag.retrieve import RetrievedChunk


def _clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([.,;:])", r"\1", text)
    return text


def extractive_answer(chunks: list[RetrievedChunk], max_chunks: int = 2, score_gap: float = 0.05) -> str:
    """
    Build a readable answer directly from the top retrieved chunk(s).
    Takes the top chunk, plus any additional chunk within `score_gap` of it.
    """
    if not chunks:
        return "I could not find this information in the provided university documents."

    top = chunks[0]
    selected = [top]
    for c in chunks[1:max_chunks]:
        if top.score - c.score < score_gap:
            selected.append(c)

    body = "\n\n".join(_clean(c.text) for c in selected)
    sources = ", ".join(sorted(set(c.source_file for c in selected)))
    return f"{body}\n\n(Source: {sources})"