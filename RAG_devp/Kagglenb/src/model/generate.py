"""
Load a trained EduVerse-1 checkpoint and wrap it as a `generate_fn(prompt) -> str`
callable, ready to plug into src/rag/pipeline.py's RAGPipeline.

This is the final piece of the Section 7 diagram: "EduVerse-1 (Generator)" ->
"Generated Response".
"""

from __future__ import annotations
import torch

from eduverse_model import EduVerse1, EduVerseConfig
from train_tokenizer import load_tokenizer


class EduVerseGenerator:
    def __init__(self, checkpoint_path: str, tokenizer_path: str, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = load_tokenizer(tokenizer_path)

        state = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        cfg = state["config"]
        if not isinstance(cfg, EduVerseConfig):
            cfg = EduVerseConfig(**cfg)  # config is saved as a plain dict via vars(cfg)

        self.model = EduVerse1(cfg).to(self.device)
        self.model.load_state_dict(state["model"])
        self.model.eval()

    def __call__(self, prompt: str, max_new_tokens: int = 80,
                 temperature: float = 0.7, top_k: int = 40) -> str:
        ids = self.tokenizer.encode(prompt).ids
        # keep the tail if the prompt is longer than context - 1 - max_new_tokens
        budget = self.model.cfg.context_length - max_new_tokens
        if len(ids) > budget:
            ids = ids[-budget:]

        idx = torch.tensor([ids], dtype=torch.long, device=self.device)
        out_ids = self.model.generate(
            idx, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k
        )
        new_tokens = out_ids[0, len(ids):].tolist()
        return self.tokenizer.decode(new_tokens)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate text from a trained EduVerse-1 checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=60)
    args = parser.parse_args()

    gen = EduVerseGenerator(args.checkpoint, args.tokenizer)
    print(gen(args.prompt, max_new_tokens=args.max_new_tokens))
