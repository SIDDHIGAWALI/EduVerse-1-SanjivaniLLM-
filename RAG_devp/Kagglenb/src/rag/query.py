from __future__ import annotations

import argparse
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM


EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# Start with a small instruction model for testing
LLM_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


def load_metadata(meta_path: str) -> list[dict]:
    metadata = []

    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                metadata.append(json.loads(line))

    return metadata


def retrieve(
    question: str,
    index,
    metadata,
    embedder,
    top_k: int = 3,
):
    query_vector = embedder.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    scores, indices = index.search(query_vector, top_k)

    results = []

    for score, idx in zip(scores[0], indices[0]):

        if idx < 0 or idx >= len(metadata):
            continue

        results.append(
            {
                "score": float(score),
                "chunk": metadata[idx],
            }
        )

    return results


def build_prompt(question: str, results: list[dict]) -> str:

    context_parts = []

    for i, result in enumerate(results, start=1):

        chunk = result["chunk"]

        context_parts.append(
            f"""SOURCE {i}
File: {chunk.get("source_file", "unknown")}

{chunk["text"]}
"""
        )

    context = "\n\n".join(context_parts)

    prompt = f"""You are a helpful university assistant.

Answer the user's question using ONLY the information provided in the context.

Rules:
- Give ONE clear answer.
- Do not repeat the context.
- Do not mention retrieval, embeddings, FAISS, chunks, or the prompt.
- Do not invent information.
- If the answer is not present in the context, say:
  "I could not find this information in the available university documents."
- Use a clear and concise format.
- For questions asking for multiple items, use bullet points or numbered lists.
- For questions asking for a syllabus, organize it unit-wise.
- Do not produce multiple alternative answers.

Context:
{context}

Question:
{question}

Answer:
"""

    return prompt


def generate_answer(
    prompt: str,
    tokenizer,
    model,
):
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful university information assistant. "
                "Answer only from the supplied context."
            ),
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=500,
        do_sample=False,
        temperature=None,
        pad_token_id=tokenizer.eos_token_id,
    )

    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()

    return answer


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--question",
        required=True,
    )

    parser.add_argument(
        "--index",
        default="data/sanjivani.index",
    )

    parser.add_argument(
        "--meta",
        default="data/sanjivani_meta.jsonl",
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
    )

    args = parser.parse_args()

    print("Loading embedding model...")

    embedder = SentenceTransformer(
        EMBED_MODEL_NAME
    )

    print("Loading FAISS index...")

    index = faiss.read_index(args.index)

    metadata = load_metadata(args.meta)

    print("Loading language model...")

    tokenizer = AutoTokenizer.from_pretrained(
        LLM_MODEL_NAME
    )

    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME,
        torch_dtype="auto",
        device_map="auto",
    )

    print("\nRetrieving relevant documents...")

    results = retrieve(
        args.question,
        index,
        metadata,
        embedder,
        top_k=args.top_k,
    )

    prompt = build_prompt(
        args.question,
        results,
    )

    answer = generate_answer(
        prompt,
        tokenizer,
        model,
    )

    print("\n" + "=" * 70)
    print("ANSWER")
    print("=" * 70)

    print(answer)

    print("=" * 70)


if __name__ == "__main__":
    main()