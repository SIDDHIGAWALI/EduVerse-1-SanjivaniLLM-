"""
Chunking module for the Sanjivani Institutional LLM RAG pipeline.

Splits cleaned source documents into overlapping chunks.

Design:
- chunk_size: ~150-300 words
- overlap: ~15-20%
- paragraph-aware splitting
- syllabus section detection (UNIT I, UNIT II, etc.)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable


# ============================================================
# DATA MODEL
# ============================================================

@dataclass
class Chunk:
    chunk_id: str
    source_file: str
    section: str | None
    text: str
    word_count: int


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text: str) -> str:
    """
    Basic text normalization.
    """
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Collapse spaces/tabs
    text = re.sub(r"[ \t]+", " ", text)

    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ============================================================
# PARAGRAPH SPLITTING
# ============================================================

def split_into_paragraphs(text: str) -> list[str]:
    """
    Split document into paragraphs.
    """
    paragraphs = [p.strip() for p in text.split("\n\n")]
    return [p for p in paragraphs if p]


# ============================================================
# SECTION DETECTION
# ============================================================

def detect_section(
    text: str,
    current_section: str | None
) -> str | None:
    """
    Detect syllabus-style sections.

    Examples:
        UNIT I
        UNIT II
        UNIT III
        UNIT IV
        UNIT V

    Also supports:
        Unit 1
        Unit 2
    """

    match = re.search(
        r"\b(UNIT\s+[IVX]+|UNIT\s+\d+)\b",
        text,
        flags=re.IGNORECASE
    )

    if match:
        return match.group(1).upper()

    return current_section


# ============================================================
# CHUNKING
# ============================================================

def chunk_text(
    text: str,
    source_file: str,
    chunk_size: int = 150,
    overlap: int = 30,
) -> list[Chunk]:
    """
    Paragraph-aware sliding-window chunker.

    Parameters
    ----------
    text:
        Input document text.

    source_file:
        Name/path of source document.

    chunk_size:
        Maximum number of words in a normal chunk.

    overlap:
        Number of words carried from the previous chunk.

    Returns
    -------
    list[Chunk]
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if overlap < 0:
        raise ValueError("overlap cannot be negative")

    if overlap >= chunk_size:
        raise ValueError(
            "overlap must be smaller than chunk_size"
        )

    text = clean_text(text)
    paragraphs = split_into_paragraphs(text)

    chunks: list[Chunk] = []

    buffer_words: list[str] = []
    current_section: str | None = None

    # --------------------------------------------------------
    # Flush helper
    # --------------------------------------------------------

    def flush(
        buf: list[str],
        section: str | None
    ) -> None:

        if not buf:
            return

        chunk_str = " ".join(buf).strip()

        if not chunk_str:
            return

        # Include source + section in hash so identical text
        # from different documents does not necessarily collide.
        hash_input = (
            f"{source_file}|{section}|{chunk_str}"
        )

        cid = hashlib.md5(
            hash_input.encode("utf-8")
        ).hexdigest()[:12]

        chunks.append(
            Chunk(
                chunk_id=cid,
                source_file=source_file,
                section=section,
                text=chunk_str,
                word_count=len(buf),
            )
        )

    # --------------------------------------------------------
    # Process paragraphs
    # --------------------------------------------------------

    for para in paragraphs:

        # Detect section before processing paragraph
        detected = detect_section(
            para,
            current_section
        )

        if detected:
            current_section = detected

        words = para.split()

        if not words:
            continue

        # ----------------------------------------------------
        # Paragraph larger than chunk_size
        # ----------------------------------------------------

        if len(words) > chunk_size:

            # Flush existing buffer first
            flush(
                buffer_words,
                current_section
            )

            buffer_words = []

            # Sliding window inside large paragraph
            step = chunk_size - overlap

            i = 0

            while i < len(words):

                window = words[
                    i:i + chunk_size
                ]

                flush(
                    window,
                    current_section
                )

                i += step

            continue

        # ----------------------------------------------------
        # Normal paragraph
        # ----------------------------------------------------

        if len(buffer_words) + len(words) > chunk_size:

            # Save current chunk
            flush(
                buffer_words,
                current_section
            )

            # Keep overlap from previous chunk
            tail = (
                buffer_words[-overlap:]
                if overlap > 0
                else []
            )

            buffer_words = tail + words

        else:

            buffer_words.extend(words)

    # --------------------------------------------------------
    # Flush final buffer
    # --------------------------------------------------------

    flush(
        buffer_words,
        current_section
    )

    return chunks


# ============================================================
# DIRECTORY CHUNKING
# ============================================================

def chunk_directory(
    input_dir: str,
    output_jsonl: str,
    chunk_size: int = 150,
    overlap: int = 30,
    extensions: Iterable[str] = (
        ".txt",
        ".md",
    ),
) -> int:
    """
    Walk through input directory and create JSONL chunks.

    Returns number of chunks written.
    """

    in_path = Path(input_dir)
    out_path = Path(output_jsonl)

    if not in_path.exists():
        raise FileNotFoundError(
            f"Input directory does not exist: {in_path}"
        )

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    extensions = {
        ext.lower()
        for ext in extensions
    }

    total = 0

    files = sorted(
        p
        for p in in_path.rglob("*")
        if p.is_file()
        and p.suffix.lower() in extensions
    )

    print(
        f"Found {len(files)} source files."
    )

    with out_path.open(
        "w",
        encoding="utf-8"
    ) as out_f:

        for fp in files:

            raw = fp.read_text(
                encoding="utf-8",
                errors="ignore"
            )

            file_chunks = chunk_text(
                raw,
                source_file=str(
                    fp.relative_to(in_path)
                ),
                chunk_size=chunk_size,
                overlap=overlap,
            )

            for chunk in file_chunks:

                out_f.write(
                    json.dumps(
                        asdict(chunk),
                        ensure_ascii=False
                    )
                    + "\n"
                )

                total += 1

    return total


# ============================================================
# CLI
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Chunk Sanjivani institutional "
            "documents into JSONL."
        )
    )

    parser.add_argument(
        "--input_dir",
        default="data/cleaned",
        help="Directory containing cleaned text files"
    )

    parser.add_argument(
        "--output_jsonl",
        default="data/chunks.jsonl",
        help="Output JSONL file"
    )

    parser.add_argument(
        "--chunk_size",
        type=int,
        default=150,
        help="Maximum words per chunk"
    )

    parser.add_argument(
        "--overlap",
        type=int,
        default=30,
        help="Overlap words between chunks"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Sanjivani RAG Chunking")
    print("=" * 60)

    print(
        f"Input directory : {args.input_dir}"
    )

    print(
        f"Output file     : {args.output_jsonl}"
    )

    print(
        f"Chunk size      : {args.chunk_size}"
    )

    print(
        f"Overlap         : {args.overlap}"
    )

    print("=" * 60)

    total = chunk_directory(
        input_dir=args.input_dir,
        output_jsonl=args.output_jsonl,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )

    print()
    print(
        f"Successfully wrote {total} chunks."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()