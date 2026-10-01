"""
Merge and clean collected source files into one training-ready corpus file.

Takes whatever your team has collected — a folder of loose .txt/.md/.csv files,
possibly messy, possibly with duplicates — and produces:

    1. One combined, cleaned .txt corpus file (ready for train_tokenizer.py / train.py)
    2. A report showing exactly what was included, skipped, or cleaned, so you
       can sanity-check the corpus before spending compute time training on it.

Usage:
    python -m src.data_pipeline.merge_corpus \
        --input_dir data/raw_docs/bucket_a_general \
        --output_file data/corpus_bucket_a.txt \
        --report_file data/corpus_bucket_a_report.txt

Run once per bucket (Bucket A general corpus, Bucket B Sanjivani corpus),
since they should stay as separate files for the curriculum stages.
"""

from __future__ import annotations
import re
import csv
import hashlib
import argparse
from pathlib import Path
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Cleaning rules
# ---------------------------------------------------------------------------

PAGE_NUMBER_PATTERNS = [
    re.compile(r"^\s*page\s+\d+\s*(of\s+\d+)?\s*$", re.IGNORECASE),
    re.compile(r"^\s*\d+\s*/\s*\d+\s*$"),          # "3/12"
    re.compile(r"^\s*-\s*\d+\s*-\s*$"),             # "- 3 -"
    re.compile(r"^\s*\d{1,4}\s*$"),                 # a lone page number on its own line
]

WEB_JUNK_PATTERNS = [
    re.compile(r"^\s*(skip to content|cookie policy|accept cookies|home\s*\|\s*about\s*\|\s*contact)\s*$", re.IGNORECASE),
    re.compile(r"^\s*©.*\d{4}.*$"),                  # copyright footer lines
    re.compile(r"^\s*all rights reserved\.?\s*$", re.IGNORECASE),
]

MIN_LINE_LENGTH = 3           # drop lines shorter than this (likely junk/artifacts)
MIN_FILE_WORDS = 5            # skip files with almost no real content


def clean_line(line: str) -> str | None:
    """Return a cleaned line, or None if the line should be dropped entirely."""
    stripped = line.strip()

    if not stripped:
        return ""  # keep blank lines — they mark paragraph breaks, don't drop them

    for pattern in PAGE_NUMBER_PATTERNS + WEB_JUNK_PATTERNS:
        if pattern.match(stripped):
            return None

    if len(stripped) < MIN_LINE_LENGTH:
        return None

    # collapse internal whitespace, normalize weird unicode spaces
    stripped = re.sub(r"[ \t\u00A0]+", " ", stripped)
    return stripped


def clean_document(raw_text: str) -> str:
    raw_text = raw_text.replace("\r\n", "\n")
    lines = raw_text.split("\n")

    cleaned_lines = []
    for line in lines:
        result = clean_line(line)
        if result is not None:
            cleaned_lines.append(result)

    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)  # collapse 3+ blank lines down to 1
    return text.strip()


# ---------------------------------------------------------------------------
# CSV -> text conversion (for Bucket data that arrives as FAQ-style CSVs)
# ---------------------------------------------------------------------------

def csv_to_text(csv_path: Path, text_cols: list[str] | None = None) -> str:
    """Turn a CSV (e.g. question/answer FAQ sheet) into paragraph-style text,
    one paragraph per row, so it merges naturally into the corpus."""
    text_cols = text_cols or ["question", "answer"]
    paragraphs = []

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        cols = [c for c in text_cols if reader.fieldnames and c in reader.fieldnames]
        if not cols:
            # fall back to a single "text" column if present
            if reader.fieldnames and "text" in reader.fieldnames:
                cols = ["text"]
            else:
                return ""  # unrecognized CSV shape, skip

        for row in reader:
            parts = [str(row[c]).strip() for c in cols if row.get(c)]
            if parts:
                paragraphs.append(" ".join(parts))

    return "\n\n".join(paragraphs)


# ---------------------------------------------------------------------------
# Merge logic
# ---------------------------------------------------------------------------

@dataclass
class MergeReport:
    included: list[str] = field(default_factory=list)
    skipped_empty: list[str] = field(default_factory=list)
    skipped_duplicate: list[str] = field(default_factory=list)
    skipped_unreadable: list[str] = field(default_factory=list)
    total_words: int = 0

    def write(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            f.write("=== Corpus Merge Report ===\n\n")
            f.write(f"Files included:      {len(self.included)}\n")
            f.write(f"Files skipped (empty/too short): {len(self.skipped_empty)}\n")
            f.write(f"Files skipped (duplicate):       {len(self.skipped_duplicate)}\n")
            f.write(f"Files skipped (unreadable):      {len(self.skipped_unreadable)}\n")
            f.write(f"Total words in final corpus:     {self.total_words:,}\n")
            f.write(f"Rough estimated tokens (~1.3x words): {int(self.total_words * 1.3):,}\n\n")

            f.write("--- Included ---\n")
            for f_name in self.included:
                f.write(f"  {f_name}\n")

            if self.skipped_empty:
                f.write("\n--- Skipped: empty or too short ---\n")
                for f_name in self.skipped_empty:
                    f.write(f"  {f_name}\n")

            if self.skipped_duplicate:
                f.write("\n--- Skipped: exact duplicate of another file ---\n")
                for f_name in self.skipped_duplicate:
                    f.write(f"  {f_name}\n")

            if self.skipped_unreadable:
                f.write("\n--- Skipped: could not read file ---\n")
                for f_name in self.skipped_unreadable:
                    f.write(f"  {f_name}\n")


def merge_corpus(
    input_dir: str,
    output_file: str,
    report_file: str | None = None,
    extensions: tuple[str, ...] = (".txt", ".md", ".csv"),
    csv_text_cols: list[str] | None = None,
) -> MergeReport:
    in_path = Path(input_dir)
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    report = MergeReport()
    seen_hashes: set[str] = set()

    files = sorted(p for p in in_path.rglob("*") if p.suffix.lower() in extensions)

    with out_path.open("w", encoding="utf-8") as out_f:
        for fp in files:
            rel_name = str(fp.relative_to(in_path))

            try:
                if fp.suffix.lower() == ".csv":
                    raw_text = csv_to_text(fp, text_cols=csv_text_cols)
                else:
                    raw_text = fp.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                report.skipped_unreadable.append(rel_name)
                continue

            cleaned = clean_document(raw_text)
            word_count = len(cleaned.split())

            if word_count < MIN_FILE_WORDS:
                report.skipped_empty.append(rel_name)
                continue

            content_hash = hashlib.md5(cleaned.encode("utf-8")).hexdigest()
            if content_hash in seen_hashes:
                report.skipped_duplicate.append(rel_name)
                continue
            seen_hashes.add(content_hash)

            # write a document separator comment + the cleaned text
            out_f.write(f"\n\n")
            out_f.write(cleaned)
            out_f.write("\n")

            report.included.append(rel_name)
            report.total_words += word_count

    if report_file:
        report.write(report_file)

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge and clean collected files into one training corpus")
    parser.add_argument("--input_dir", required=True, help="Folder containing collected .txt/.md/.csv files")
    parser.add_argument("--output_file", required=True, help="Path to write the merged corpus .txt file")
    parser.add_argument("--report_file", default=None, help="Path to write a summary report (optional but recommended)")
    parser.add_argument("--csv_text_cols", nargs="+", default=["question", "answer"],
                         help="Column names to use when a .csv file is encountered")
    args = parser.parse_args()

    report_path = args.report_file or (str(Path(args.output_file).with_suffix("")) + "_report.txt")
    report = merge_corpus(
        args.input_dir, args.output_file, report_path,
        csv_text_cols=args.csv_text_cols,
    )

    print(f"Merged {len(report.included)} files -> {args.output_file}")
    print(f"  Skipped: {len(report.skipped_empty)} empty, {len(report.skipped_duplicate)} duplicate, "
          f"{len(report.skipped_unreadable)} unreadable")
    print(f"  Total words: {report.total_words:,}  (~{int(report.total_words * 1.3):,} estimated tokens)")
    print(f"  Full report written to: {report_path}")
