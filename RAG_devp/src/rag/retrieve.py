"""
Query-time retrieval for the Sanjivani RAG pipeline.

Loads the FAISS index + metadata built by build_index.py, embeds an incoming
user question with the SAME embedding model, and returns the top-k most
relevant chunks.

k defaults to 3 per Model_Design_&_Architecture.pdf Section 7: with only
512 tokens of context, 2-3 short high-relevance chunks beat 5+ chunks that
get truncated.
"""

from __future__ import annotations
import json
from dataclasses import dataclass

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from .build_index import EMBED_MODEL_NAME


@dataclass
class RetrievedChunk:
    chunk_id: str
    source_file: str
    text: str
    score: float


class Retriever:
    def __init__(self, index_path: str, meta_path: str, model_name: str = EMBED_MODEL_NAME):
        self.index = faiss.read_index(index_path)
        self.embedder = SentenceTransformer(model_name)
        self.meta: list[dict] = []
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.meta.append(json.loads(line))

        if self.index.ntotal != len(self.meta):
            raise ValueError(
                f"Index size ({self.index.ntotal}) != metadata rows ({len(self.meta)}). "
                "Rebuild the index and metadata together."
            )

    def retrieve(self, query: str, k: int = 3, min_score: float = 0.0) -> list[RetrievedChunk]:
        q_vec = self.embedder.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")

        scores, idxs = self.index.search(q_vec, k)
        results: list[RetrievedChunk] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1 or score < min_score:
                continue
            row = self.meta[idx]
            results.append(
                RetrievedChunk(
                    chunk_id=row["chunk_id"],
                    source_file=row["source_file"],
                    text=row["text"],
                    score=float(score),
                )
            )
        return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Query the Sanjivani knowledge base")
    parser.add_argument("--index", default="data/sanjivani.index")
    parser.add_argument("--meta", default="data/sanjivani_meta.jsonl")
    parser.add_argument("--query", required=True)
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()

    retriever = Retriever(args.index, args.meta)
    for r in retriever.retrieve(args.query, k=args.k):
        print(f"[{r.score:.3f}] ({r.source_file}) {r.text[:120]}...")
