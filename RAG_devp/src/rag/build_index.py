"""
Build the FAISS vector index for the Sanjivani knowledge base.

Reads chunks.jsonl (produced by src/data_pipeline/chunking.py), embeds each
chunk with all-MiniLM-L6-v2, and writes:
    - sanjivani.index      (FAISS index, binary)
    - sanjivani_meta.jsonl (chunk_id -> text/source mapping, same row order as index)

Matches Model_Design_&_Architecture.pdf Section 7:
    embedding model = all-MiniLM-L6-v2
    vector DB       = FAISS
"""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"


def load_chunks(chunks_jsonl: str) -> list[dict]:
    chunks = []
    with open(chunks_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def build_index(
    chunks_jsonl: str,
    index_out: str,
    meta_out: str,
    model_name: str = EMBED_MODEL_NAME,
    batch_size: int = 64,
) -> None:
    chunks = load_chunks(chunks_jsonl)
    if not chunks:
        raise ValueError(f"No chunks found in {chunks_jsonl}. Run chunking.py first.")

    print(f"Loaded {len(chunks)} chunks. Loading embedding model '{model_name}'...")
    embedder = SentenceTransformer(model_name)

    texts = [c["text"] for c in chunks]
    print("Encoding chunks...")
    vectors = embedder.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # so L2 distance behaves like cosine similarity
    ).astype("float32")

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product on normalized vecs = cosine sim
    index.add(vectors)

    Path(index_out).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, index_out)

    with open(meta_out, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"Wrote index ({index.ntotal} vectors, dim={dim}) -> {index_out}")
    print(f"Wrote metadata -> {meta_out}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build Sanjivani RAG vector index")
    parser.add_argument("--chunks", default="data/chunks.jsonl")
    parser.add_argument("--index_out", default="data/sanjivani.index")
    parser.add_argument("--meta_out", default="data/sanjivani_meta.jsonl")
    parser.add_argument("--model", default=EMBED_MODEL_NAME)
    args = parser.parse_args()

    build_index(args.chunks, args.index_out, args.meta_out, model_name=args.model)
