"""
Full end-to-end demo: retrieval (FAISS + MiniLM) -> prompt -> EduVerse-1 generation.

Run this AFTER you have:
  1. Built the vector index:      python src/rag/build_index.py
  2. Trained a tokenizer:         python src/model/train_tokenizer.py
  3. Trained at least one stage:  python src/model/train.py ...

Usage:
    python run_demo.py --query "When is the library open?"
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent / "src" / "model"))

from rag.retrieve import Retriever
from rag.pipeline import RAGPipeline
from generate import EduVerseGenerator


def main():
    parser = argparse.ArgumentParser(description="Sanjivani Institutional LLM — full RAG demo")
    parser.add_argument("--index", default="data/sanjivani.index")
    parser.add_argument("--meta", default="data/sanjivani_meta.jsonl")
    parser.add_argument("--checkpoint", default="checkpoints/stage3_sanjivani/checkpoint.pt")
    parser.add_argument("--tokenizer", default="data/eduverse_tokenizer.json")
    parser.add_argument("--query", required=True)
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()

    print("Loading retriever...")
    retriever = Retriever(args.index, args.meta)

    print("Loading EduVerse-1...")
    generator = EduVerseGenerator(args.checkpoint, args.tokenizer)

    pipeline = RAGPipeline(retriever, generate_fn=generator, k=args.k)
    result = pipeline.answer(args.query)

    print("\n=== Retrieved chunks ===")
    for c in result.retrieved_chunks:
        print(f"  [{c.score:.3f}] {c.source_file}: {c.text[:100]}...")

    print("\n=== Answer ===")
    print(result.answer)


if __name__ == "__main__":
    main()
