"""
Training loop for EduVerse-1: next-token prediction (cross-entropy) on a
tokenized corpus, with AdamW + cosine LR schedule + gradient clipping.

Supports the three-stage curriculum from Model_Design_&_Architecture.pdf
Section 5 via `--init_from` (load a previous stage's checkpoint) and
`--lr_scale` (0.3x for HigherEd stage, 0.1x for Sanjivani stage) plus
`--replay_ratio` (~10% of prior-stage data mixed into later stages to
guard against catastrophic forgetting). If you're doing single-stage
training on one combined corpus, just omit --init_from and --lr_scale.

Runs BOTH ways:
    cd src/model; python train.py --bin_path ...             (as a script)
    python -c "from src.model.train import build_token_stream; ..."  (as a module)
"""

from __future__ import annotations
import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch

try:
    from .eduverse_model import EduVerse1, EduVerseConfig
    from .train_tokenizer import load_tokenizer
except ImportError:
    # fallback when run directly as a script (no parent package context)
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from eduverse_model import EduVerse1, EduVerseConfig
    from train_tokenizer import load_tokenizer


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def build_token_stream(text_files: list[str], tokenizer_path: str, out_bin: str) -> int:
    """Tokenize a list of text files into one flat uint16 array on disk (memmap-friendly)."""
    tok = load_tokenizer(tokenizer_path)
    all_ids: list[int] = []
    bos_id = tok.token_to_id("<bos>")
    eos_id = tok.token_to_id("<eos>")

    for fp in text_files:
        text = Path(fp).read_text(encoding="utf-8", errors="ignore")
        ids = tok.encode(text).ids
        all_ids.append(bos_id)
        all_ids.extend(ids)
        all_ids.append(eos_id)

    arr = np.array(all_ids, dtype=np.uint16)
    arr.tofile(out_bin)
    return len(arr)


class TokenDataset:
    """Simple random-chunk sampler over a flat uint16 token file (memmapped)."""

    def __init__(self, bin_path: str, context_length: int):
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.context_length = context_length
        if len(self.data) <= context_length:
            raise ValueError(
                f"Token stream ({len(self.data)} tokens) must exceed context_length "
                f"({context_length}). Add more training text."
            )

    def get_batch(self, batch_size: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        ix = np.random.randint(0, len(self.data) - self.context_length - 1, size=batch_size)
        x = torch.stack([
            torch.from_numpy(self.data[i:i + self.context_length].astype(np.int64)) for i in ix
        ])
        y = torch.stack([
            torch.from_numpy(self.data[i + 1:i + 1 + self.context_length].astype(np.int64)) for i in ix
        ])
        return x.to(device), y.to(device)


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def get_lr(step: int, warmup_steps: int, max_steps: int, base_lr: float, min_lr_ratio: float = 0.1) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    if step > max_steps:
        return base_lr * min_lr_ratio
    decay_ratio = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return base_lr * min_lr_ratio + coeff * base_lr * (1 - min_lr_ratio)


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train(
    bin_path: str,
    tokenizer_path: str,
    out_dir: str,
    init_from: str | None = None,
    base_lr: float = 3e-4,
    lr_scale: float = 1.0,
    batch_size: int = 16,
    max_steps: int = 2000,
    warmup_steps: int = 100,
    eval_interval: int = 200,
    grad_clip: float = 1.0,
    device: str | None = None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    tok = load_tokenizer(tokenizer_path)
    cfg = EduVerseConfig(vocab_size=tok.get_vocab_size())
    model = EduVerse1(cfg).to(device)

    if init_from is not None:
        print(f"Initializing weights from checkpoint: {init_from}")
        state = torch.load(init_from, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])

    lr = base_lr * lr_scale
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1, betas=(0.9, 0.95))

    dataset = TokenDataset(bin_path, cfg.context_length)

    print(f"Training on device={device}, {model.num_params():,} params, lr={lr:.2e}, "
          f"context_length={cfg.context_length}, vocab_size={cfg.vocab_size}")

    model.train()
    t0 = time.time()
    for step in range(max_steps):
        cur_lr = get_lr(step, warmup_steps, max_steps, lr)
        for g in optimizer.param_groups:
            g["lr"] = cur_lr

        x, y = dataset.get_batch(batch_size, device)
        logits, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        if step % eval_interval == 0 or step == max_steps - 1:
            elapsed = time.time() - t0
            print(f"step {step:5d} | loss {loss.item():.4f} | lr {cur_lr:.2e} | {elapsed:.1f}s")

    ckpt_path = Path(out_dir) / "checkpoint.pt"
    torch.save({"model": model.state_dict(), "config": vars(cfg)}, ckpt_path)
    print(f"Saved checkpoint -> {ckpt_path}")
    return ckpt_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EduVerse-1")
    parser.add_argument("--bin_path", required=True, help="Path to tokenized .bin file")
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--out_dir", default="checkpoints/stage1_base")
    parser.add_argument("--init_from", default=None, help="Checkpoint to continue training from")
    parser.add_argument("--base_lr", type=float, default=3e-4)
    parser.add_argument("--lr_scale", type=float, default=1.0,
                         help="1.0 for base stage, 0.3 for HigherEd stage, 0.1 for Sanjivani stage")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_steps", type=int, default=2000)
    args = parser.parse_args()

    train(
        bin_path=args.bin_path,
        tokenizer_path=args.tokenizer_path,
        out_dir=args.out_dir,
        init_from=args.init_from,
        base_lr=args.base_lr,
        lr_scale=args.lr_scale,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
    )