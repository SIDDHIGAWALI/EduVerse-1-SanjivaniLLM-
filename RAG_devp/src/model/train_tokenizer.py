"""
Train a custom BPE tokenizer from scratch on the educational corpus.

Uses HuggingFace `tokenizers` (the low-level Rust library — NOT a pretrained
tokenizer, NOT `transformers.AutoTokenizer`). We train the merges ourselves
on our own text, matching the "Custom BPE Tokenizer (16K Vocabulary)" box
in the architecture diagram.
"""

from __future__ import annotations
from pathlib import Path

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

VOCAB_SIZE = 16_000
SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]


def train_bpe_tokenizer(
    corpus_files: list[str],
    output_path: str,
    vocab_size: int = VOCAB_SIZE,
    min_frequency: int = 2,
) -> Tokenizer:
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )

    tokenizer.train(files=corpus_files, trainer=trainer)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(output_path)
    print(f"Trained tokenizer: vocab_size={tokenizer.get_vocab_size()} -> saved to {output_path}")
    return tokenizer


def load_tokenizer(path: str) -> Tokenizer:
    return Tokenizer.from_file(path)


if __name__ == "__main__":
    import argparse
    import glob

    parser = argparse.ArgumentParser(description="Train EduVerse-1's custom BPE tokenizer")
    parser.add_argument("--corpus_glob", default="data/raw_docs/*.txt",
                         help="Glob pattern for training text files")
    parser.add_argument("--output", default="data/eduverse_tokenizer.json")
    parser.add_argument("--vocab_size", type=int, default=VOCAB_SIZE)
    args = parser.parse_args()

    files = glob.glob(args.corpus_glob)
    if not files:
        raise ValueError(f"No files matched {args.corpus_glob}")

    train_bpe_tokenizer(files, args.output, vocab_size=args.vocab_size)
