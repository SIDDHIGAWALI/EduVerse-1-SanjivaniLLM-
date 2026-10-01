"""
End-to-end RAG pipeline: user question -> retrieval -> prompt -> EduVerse-1 generation.
"""

from __future__ import annotations

from dataclasses import dataclass

from .retrieve import Retriever, RetrievedChunk


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
{context}

Question:
{question}

Answer:"""


def build_prompt(
    chunks: list[RetrievedChunk],
    question: str
) -> str:
    context = "\n".join(
        f"- {c.text}"
        for c in chunks
    )

    return PROMPT_TEMPLATE.format(
        context=context,
        question=question
    )


@dataclass
class RAGResult:
    question: str
    prompt: str
    retrieved_chunks: list[RetrievedChunk]
    answer: str


class RAGPipeline:
    """
    Wraps a Retriever + a generator function.

    Normal questions use k=3.
    Detailed/comprehensive questions retrieve more chunks.
    """

    def __init__(
        self,
        retriever: Retriever,
        generate_fn,
        k: int = 3,
        max_context_words: int = 350
    ):
        self.retriever = retriever
        self.generate_fn = generate_fn
        self.k = k
        self.max_context_words = max_context_words

    def _trim_to_budget(
        self,
        chunks: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:

        kept = []
        total_words = 0

        for c in chunks:
            wc = len(c.text.split())

            # Keep complete chunks only.
            # Never cut a chunk in the middle.
            if total_words + wc > self.max_context_words:
                if kept:
                    break

            kept.append(c)
            total_words += wc

        return kept

    def answer(self, question: str) -> RAGResult:

        detailed_terms = [
            "in detail",
            "detailed",
            "complete",
            "entire",
            "full",
            "all",
            "all chapters",
            "all units",
            "all topics",
            "explain everything",
            "syllabus",
        ]

        question_lower = question.lower()

        # Normal questions → k=3
        # Comprehensive questions → k=8
        if any(term in question_lower for term in detailed_terms):
            retrieval_k = 8
        else:
            retrieval_k = self.k

        raw_chunks = self.retriever.retrieve(
            question,
            k=retrieval_k
        )

        chunks = self._trim_to_budget(raw_chunks)

        prompt = build_prompt(
            chunks,
            question
        )

        # generate_fn is called EXACTLY ONCE. One call in, one answer out.
        answer = self.generate_fn(prompt)

        return RAGResult(
            question=question,
            prompt=prompt,
            retrieved_chunks=chunks,
            answer=answer
        )


def echo_stub_generator(prompt: str) -> str:
    """
    Placeholder generator used before EduVerse-1 is connected.
    NOTE: this deliberately returns the whole prompt verbatim, which is why
    it LOOKS like duplicated context/question/answer when printed — it isn't
    actually duplicating anything, it's just showing you the input prompt
    because no real model has been wired in yet. Pass --checkpoint and
    --tokenizer to use the real EduVerse-1 model instead.
    """
    return (
        "[EduVerse-1 not yet loaded — this is a stub response "
        "echoing the prompt. Pass --checkpoint/--tokenizer to fix this]\n"
        + prompt
    )


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Run the full Sanjivani RAG pipeline"
    )

    parser.add_argument(
        "--index",
        default="data/sanjivani.index"
    )

    parser.add_argument(
        "--meta",
        default="data/sanjivani_meta.jsonl"
    )

    parser.add_argument(
        "--query",
        required=True
    )

    parser.add_argument(
        "--k",
        type=int,
        default=3
    )

    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to a trained EduVerse-1 checkpoint.pt. "
             "If omitted, falls back to the echo stub (no real generation)."
    )

    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Path to the trained tokenizer .json. Required if --checkpoint is set."
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=80
    )

    args = parser.parse_args()

    retriever = Retriever(
        args.index,
        args.meta
    )

    if args.checkpoint:
        if not args.tokenizer:
            parser.error("--tokenizer is required when --checkpoint is set")

        # Import here (not top-level) so this file still works standalone
        # for retrieval-only testing without requiring torch/the model code.
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / "model"))
        from generate import EduVerseGenerator

        print(f"Loading EduVerse-1 from {args.checkpoint} ...")
        generator = EduVerseGenerator(args.checkpoint, args.tokenizer)

        def generate_fn(prompt: str) -> str:
            return generator(prompt, max_new_tokens=args.max_new_tokens)
    else:
        print("WARNING: no --checkpoint given, using the echo stub. "
              "You will NOT get a real answer. Pass --checkpoint and --tokenizer "
              "to use the actual trained model.")
        generate_fn = echo_stub_generator

    pipeline = RAGPipeline(
        retriever,
        generate_fn=generate_fn,
        k=args.k
    )

    result = pipeline.answer(args.query)

    print("\n" + "=" * 70)
    print("RETRIEVED CONTEXT")
    print("=" * 70)

    for c in result.retrieved_chunks:
        print(
            f"  [{c.score:.3f}] "
            f"{c.source_file}: "
            f"{c.text[:150]}..."
        )

    print("\n" + "=" * 70)
    print("ANSWER")
    print("=" * 70)
    print(result.answer)