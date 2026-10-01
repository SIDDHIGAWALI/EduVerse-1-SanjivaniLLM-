"""
Load a trained EduVerse-1 checkpoint and wrap it as a `generate_fn(prompt) -> str`
callable, ready to plug into src/rag/pipeline.py's RAGPipeline.

Includes repetition_penalty and no_repeat_ngram_size to fix stuttering/looping
output (e.g. "Tech CSE Interdisciplinary Studies.Tech CSE Interdisciplinary
Studies..."). This is a decoding-time fix, independent of corpus size or
training — worth using regardless of how much more training data you add later.

Also strips excessive blank-line output, which tends to surface once the
repetition fix blocks a model's favorite phrase-loop — the underlying
content is still limited by training data volume, but this keeps output
readable rather than mostly whitespace.
"""

from __future__ import annotations
import re
import torch

from eduverse_model import EduVerse1, EduVerseConfig
from train_tokenizer import load_tokenizer


def _clean_output(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


class EduVerseGenerator:
    def __init__(self, checkpoint_path: str, tokenizer_path: str, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = load_tokenizer(tokenizer_path)

        state = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        cfg = state["config"]
        if not isinstance(cfg, EduVerseConfig):
            cfg = EduVerseConfig(**cfg)

        self.model = EduVerse1(cfg).to(self.device)
        self.model.load_state_dict(state["model"])
        self.model.eval()

    def __call__(
        self,
        prompt: str,
        max_new_tokens: int = 80,
        temperature: float = 0.7,
        top_k: int = 40,
        repetition_penalty: float = 1.3,
        no_repeat_ngram_size: int = 3,
    ) -> str:
        ids = self.tokenizer.encode(prompt).ids
        budget = self.model.cfg.context_length - max_new_tokens
        if len(ids) > budget:
            ids = ids[-budget:]

        idx = torch.tensor([ids], dtype=torch.long, device=self.device)
        out_ids = self.model.generate(
            idx,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
        new_tokens = out_ids[0, len(ids):].tolist()
        raw_text = self.tokenizer.decode(new_tokens)
        return _clean_output(raw_text)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate text from a trained EduVerse-1 checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=60)
    parser.add_argument("--repetition_penalty", type=float, default=1.3,
                         help="1.0 = no penalty. 1.2-1.5 is a reasonable range to reduce repetition.")
    parser.add_argument("--no_repeat_ngram_size", type=int, default=3,
                         help="0 disables. 3 blocks any 3-token sequence from repeating.")
    args = parser.parse_args()

    gen = EduVerseGenerator(args.checkpoint, args.tokenizer)
    print(gen(
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
    ))