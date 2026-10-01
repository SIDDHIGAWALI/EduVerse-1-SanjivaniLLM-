"""
Clean scraped website .txt files before merging into the training corpus. v2.

FIXES FROM v1 (after real-world testing on 162 files):
1. Binary/image content detection: two "webp" image files were being read as
   text and slipping through as 5000+ "words" of pure garbage bytes
   (recognizable by a high ratio of the U+FFFD replacement character, from
   decoding raw bytes as UTF-8). These are now detected and excluded outright.
2. The "menu/list junk" heuristic (avg words/line) was too aggressive and
   auto-deleted genuinely valuable content — department pages, director's
   messages, several PDFs — that happen to be formatted with short lines
   (course lists, eligibility bullets). This heuristic now routes files to a
   separate "needs_review" folder instead of silently deleting them, so you
   can quickly eyeball and decide rather than losing real content.

Three output categories:
  - KEPT: passes all checks, copied straight to output_dir
  - NEEDS REVIEW: low avg words/line, copied to output_dir/_needs_review/
    for you to skim — likely a mix of real list-heavy content and some junk
  - EXCLUDED: near-empty after cleaning, OR detected as binary garbage —
    not copied anywhere, listed in the report only

Usage:
    python -m src.data_pipeline.clean_web_scrape --input_dir data/raw_docs --output_dir data/raw_docs_clean --min_real_words 15
"""

from __future__ import annotations
import re
import argparse
from pathlib import Path


JUNK_LINES = {
    "high contrast", "gray scale", "increase font", "decrease font",
    "original font", "underline links", "text reader", "select language​▼",
    "select language", "reset", "whatsapp", "admissions", "facebook",
    "linkedin", "youtube", "instagram", "university relations", "erp portal",
    "projects & innovation", "examination", "photo gallery", "events",
    "apply now", "home", "about us", "academics", "admission",
    "students corner", "placement", "iqac", "careers", "contact",
    "quick links", "homephoto gallerysitemapscholarshipsports facilities",
    "timing", "09:40 am to 5:40 pm", "site visitors", "how to reach us",
    "shirdi international airport: 25 kms", "nashik airport: 75 kms",
    "chh. sambhajinagar airport: 105 kms", "shirdi railway station: 13 kms",
    "kopargaon railway station: 0.5 kms", "kopargaon bus stand: 1 kms",
    "samruddhi highway: 05 kms", "no content added yet",
    "no content available", "please wait",
}

FOOTER_START_PATTERN = re.compile(r"^sanjivani university,\s*kopargaon,?\s*$", re.IGNORECASE)
COPYRIGHT_PATTERN = re.compile(r"^©\s*\d{4}\s*sanjivani university", re.IGNORECASE)
TEMPLATE_ARTIFACT_PATTERN = re.compile(r"\{\{.*?\}\}")
LONE_DIGIT_PATTERN = re.compile(r"^\d{1,2}$")

# Signs of raw binary data misread as text: the UTF-8 replacement character
# (appears when invalid byte sequences get decoded), or a very high ratio of
# non-printable/control characters.
REPLACEMENT_CHAR = "\ufffd"
MOJIBAKE_MARKER = "ï¿½"  # what U+FFFD often looks like when double-mis-decoded


def is_binary_garbage(text: str, sample_chars: int = 3000) -> bool:
    sample = text[:sample_chars]
    if not sample:
        return False
    bad_char_count = sample.count(REPLACEMENT_CHAR) + sample.count(MOJIBAKE_MARKER)
    ratio = bad_char_count / max(len(sample), 1)
    return ratio > 0.01  # more than 1% garbage-marker density = binary data, not text


def clean_scraped_page(text: str) -> tuple[str, int, int]:
    original_word_count = len(text.split())

    lines = text.split("\n")
    kept_lines = []
    in_footer = False

    for line in lines:
        stripped = line.strip()
        lower = stripped.lower()

        if not stripped:
            kept_lines.append("")
            continue

        if FOOTER_START_PATTERN.match(stripped):
            in_footer = True
        if in_footer:
            continue

        if lower in JUNK_LINES:
            continue
        if COPYRIGHT_PATTERN.match(stripped):
            continue
        if LONE_DIGIT_PATTERN.match(stripped):
            continue
        if TEMPLATE_ARTIFACT_PATTERN.search(stripped):
            continue

        kept_lines.append(stripped)

    cleaned = "\n".join(kept_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    cleaned_word_count = len(cleaned.split())

    return cleaned, original_word_count, cleaned_word_count


def avg_words_per_line(text: str) -> float:
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return 0.0
    return sum(len(l.split()) for l in lines) / len(lines)


def process_directory(
    input_dir: str,
    output_dir: str,
    min_real_words: int = 15,
    review_threshold_words_per_line: float = 3.5,
) -> None:
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    review_path = out_path / "_needs_review"
    out_path.mkdir(parents=True, exist_ok=True)
    review_path.mkdir(parents=True, exist_ok=True)

    files = sorted(in_path.glob("*.txt"))
    if not files:
        print(f"No .txt files found in {input_dir}")
        return

    kept, review, excluded_empty, excluded_binary = 0, 0, 0, 0

    print(f"{'FILE':<55} {'BEFORE':>7} {'AFTER':>7}  STATUS")
    print("-" * 95)

    for fp in files:
        raw_text = fp.read_text(encoding="utf-8", errors="ignore")

        if is_binary_garbage(raw_text):
            print(f"{fp.name:<55} {'-':>7} {'-':>7}  EXCLUDED - binary/image data detected")
            excluded_binary += 1
            continue

        cleaned, before, after = clean_scraped_page(raw_text)
        avg_wpl = avg_words_per_line(cleaned)

        if after < min_real_words:
            status = "EXCLUDED - near-empty"
            excluded_empty += 1
        elif avg_wpl < review_threshold_words_per_line:
            out_file = review_path / fp.name
            out_file.write_text(cleaned, encoding="utf-8")
            status = f"NEEDS REVIEW - list-heavy (avg {avg_wpl:.1f} words/line)"
            review += 1
        else:
            out_file = out_path / fp.name
            out_file.write_text(cleaned, encoding="utf-8")
            status = "OK"
            kept += 1

        print(f"{fp.name:<55} {before:>7} {after:>7}  {status}")

    print(f"\nDone.")
    print(f"  {kept} files kept outright -> {output_dir}")
    print(f"  {review} files flagged for manual review -> {output_dir}\\_needs_review\\")
    print(f"  {excluded_empty} files excluded as near-empty")
    print(f"  {excluded_binary} files excluded as binary/image data")
    print(f"\nIMPORTANT: check the _needs_review folder before merging — it likely contains "
          f"a mix of real content (department pages, list-formatted info) and genuine junk. "
          f"Skim each file; move the good ones into {output_dir} directly, delete the rest.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean scraped website text before adding to training corpus")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--min_real_words", type=int, default=15)
    parser.add_argument("--review_threshold_words_per_line", type=float, default=3.5)
    args = parser.parse_args()

    process_directory(
        args.input_dir, args.output_dir,
        min_real_words=args.min_real_words,
        review_threshold_words_per_line=args.review_threshold_words_per_line,
    )