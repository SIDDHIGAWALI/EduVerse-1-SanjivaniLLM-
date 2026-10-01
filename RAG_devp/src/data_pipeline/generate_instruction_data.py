"""
Generate instruction-tuning examples from your existing chunks.jsonl.

WHY THIS EXISTS: Your base training corpus is pure informational text — it
never demonstrates the "Context: ... Question: ... Answer: ..." pattern
that pipeline.py's PROMPT_TEMPLATE uses at inference time. A 23M-parameter
model won't infer this format from raw text alone; it needs to see the
pattern directly, repeated, to learn "Answer:" means "produce a relevant
answer using the context," not "keep free-associating."

This script auto-generates one (Context, Question, Answer) example per
chunk using keyword-based question templates, then writes them out in
EXACTLY the same prompt format pipeline.py builds at inference time —
so continued training directly teaches the model to complete that pattern.

This is a lightweight, template-based approach — NOT hand-labeled QA data.
Review a sample of the output before training on it; if a generated
question doesn't fit its chunk well, either fix the keyword map below or
manually edit the output file before training.

Usage:
    python -m src.data_pipeline.generate_instruction_data --chunks data/chunks.jsonl --output data/instruction_corpus.txt
"""

from __future__ import annotations
import json
import re
import argparse
from pathlib import Path


# Keyword -> question template. Checked in order; first match wins.
# Add more entries here as you notice ungenerated/mismatched topics.
KEYWORD_QUESTIONS = [
    (r"\battendance\b", "What is the minimum attendance requirement for examinations?"),
    (r"\bbacklog\b", "How many backlog papers can a student appear for?"),
    (r"\brevaluation\b", "What is the process for answer sheet revaluation?"),
    (r"\blibrary\b|\btimings\b", "What are the library timings?"),
    (r"\bborrow(ing)?\b.*\bbooks?\b", "How many books can a student borrow from the library?"),
    (r"\badmission\b", "What is the admission process?"),
    (r"\bfee structure\b|\btuition fee\b", "What is the fee structure?"),
    (r"\bb\.?tech\b.*programmes?\b|\bprogrammes? (offered|include)\b", "What B.Tech programmes are offered?"),
    (r"\bcentre of excellence\b|\bfortinet\b|\bibm\b", "What industry Centres of Excellence exist at the university?"),
    (r"\bsyllabus\b|\bunit i\b|\bcourse contents\b", "What does the syllabus cover?"),
    (r"\bresearch design\b", "What is covered under research design in the syllabus?"),
    (r"\bstudent development\b|\bincubation\b|\bmaker space", "What student development facilities are available?"),
    (r"\bacademic environment\b", "What is the academic environment like at the university?"),
]

GENERIC_FALLBACK = "What information is provided about {topic}?"


def guess_question(source_file: str, text: str) -> str:
    combined = (source_file + " " + text).lower()
    for pattern, question in KEYWORD_QUESTIONS:
        if re.search(pattern, combined):
            return question

    # fallback: derive a generic topic name from the filename
    topic = Path(source_file).stem.replace("_", " ").replace("-", " ")
    return GENERIC_FALLBACK.format(topic=topic)


PROMPT_TEMPLATE = """You are a helpful university assistant.

Answer the user's question using ONLY the provided context.

Rules:
- Give exactly ONE answer.
- Do not repeat the context.
- Do not repeat the question.
- Do not generate multiple possible answers.
- Do not mention the retrieval process.
- If the context does not contain the answer, say:
  "I could not find this information in the provided university documents."
- Answer clearly and naturally.

Context:
- {context}

Question:
{question}

Answer:
{answer}"""


def generate_instruction_corpus(chunks_jsonl: str, output_path: str, sample_report: int = 5) -> int:
    chunks = []
    with open(chunks_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    if not chunks:
        raise ValueError(f"No chunks found in {chunks_jsonl}")

    examples = []
    for c in chunks:
        question = guess_question(c["source_file"], c["text"])
        example = PROMPT_TEMPLATE.format(
            context=c["text"],
            question=question,
            answer=c["text"],  # the chunk text itself IS the correct answer content
        )
        examples.append(example)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n\n<|endofexample|>\n\n".join(examples), encoding="utf-8")

    print(f"Generated {len(examples)} instruction examples -> {output_path}")
    print(f"\n--- Sample of generated (question, source) pairs — REVIEW these for quality ---")
    for c in chunks[:sample_report]:
        q = guess_question(c["source_file"], c["text"])
        print(f"  [{c['source_file']}] -> \"{q}\"")

    return len(examples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate instruction-tuning examples from chunks.jsonl")
    parser.add_argument("--chunks", default="data/chunks.jsonl")
    parser.add_argument("--output", default="data/instruction_corpus.txt")
    args = parser.parse_args()

    generate_instruction_corpus(args.chunks, args.output)
