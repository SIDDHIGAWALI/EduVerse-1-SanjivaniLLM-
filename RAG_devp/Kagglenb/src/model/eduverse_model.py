"""
EduVerse-1 — decoder-only Transformer, built from scratch in PyTorch.

Matches the "Model Configuration" table in Model_Design_&_Architecture.pdf:

    Architecture          Decoder-only Transformer
    Layers                6
    Attention Heads       8
    Hidden Dimension      512
    FFN Inner Dimension   2048 (4x hidden)
    Context Length        512 tokens
    Vocabulary Size       16,000
    Positional Encoding   Learned (RoPE optional, see §6)
    Normalization         Pre-LayerNorm
    Activation            GELU
    Parameters            25-35M
    Training Objective    Next Token Prediction (cross-entropy)

No pretrained weights are loaded anywhere in this file. Every layer is
initialized from scratch, per the project's "No pretrained LLMs" constraint.
"""

from __future__ import annotations
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class EduVerseConfig:
    vocab_size: int = 16_000
    context_length: int = 512
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 512
    ffn_hidden: int = 2048       # 4x n_embd
    dropout: float = 0.1
    bias: bool = True


class CausalSelfAttention(nn.Module):
    """Masked multi-head self-attention (8 heads, causal mask) — matches the
    'Masked Multi-Head Self-Attention' box in the decoder block diagram."""

    def __init__(self, cfg: EduVerseConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head

        self.qkv_proj = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.out_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

        # causal mask: position i can only attend to positions <= i
        mask = torch.tril(torch.ones(cfg.context_length, cfg.context_length))
        self.register_buffer("causal_mask", mask.view(1, 1, cfg.context_length, cfg.context_length))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        qkv = self.qkv_proj(x)  # (B, T, 3C)
        q, k, v = qkv.split(C, dim=2)

        # reshape to (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, nh, T, T)
        att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        out = att @ v                                    # (B, nh, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_dropout(self.out_proj(out))
        return out


class FeedForward(nn.Module):
    """Linear -> GELU -> Linear, 512 -> 2048 -> 512, per the architecture diagram."""

    def __init__(self, cfg: EduVerseConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.n_embd, cfg.ffn_hidden, bias=cfg.bias),
            nn.GELU(),
            nn.Linear(cfg.ffn_hidden, cfg.n_embd, bias=cfg.bias),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DecoderBlock(nn.Module):
    """Pre-Norm decoder block:
         x -> LN -> MHSA -> +residual -> LN -> FFN -> +residual -> next block
       matches the 'Transformer Decoder Block (Pre-Norm)' diagram exactly.
    """

    def __init__(self, cfg: EduVerseConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.ffn = FeedForward(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))   # pre-norm + residual around attention
        x = x + self.ffn(self.ln2(x))    # pre-norm + residual around FFN
        return x


class EduVerse1(nn.Module):
    """Full decoder-only LM: token embedding + positional encoding -> 6x DecoderBlock
    -> final LayerNorm -> linear projection to vocab -> softmax (next-token prediction)."""

    def __init__(self, cfg: EduVerseConfig):
        super().__init__()
        self.cfg = cfg

        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.context_length, cfg.n_embd)  # learned positional encoding
        self.drop = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList([DecoderBlock(cfg) for _ in range(cfg.n_layer)])
        self.final_ln = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        # weight tying: input embedding and output projection share weights
        # (standard trick, cuts params meaningfully at this vocab size)
        self.lm_head.weight = self.token_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """
        idx:     (B, T) token ids
        targets: (B, T) token ids shifted by 1, or None for inference
        returns: logits (B, T, vocab_size), loss (scalar or None)
        """
        B, T = idx.shape
        assert T <= self.cfg.context_length, (
            f"Sequence length {T} exceeds context_length {self.cfg.context_length}"
        )

        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)  # (1, T)
        x = self.token_emb(idx) + self.pos_emb(pos)               # added, not concatenated
        x = self.drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.final_ln(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: int | None = 40,
    ) -> torch.Tensor:
        """Autoregressive greedy/sampled generation, one token at a time."""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.cfg.context_length else idx[:, -self.cfg.context_length:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.pos_emb.weight.numel()
        return n


if __name__ == "__main__":
    cfg = EduVerseConfig()
    model = EduVerse1(cfg)

    n_params = model.num_params()
    print(f"EduVerse-1 initialized: {n_params:,} parameters ({n_params / 1e6:.1f}M)")
    print(f"Target from spec: 25-35M params")

    # sanity check forward pass with dummy batch
    B, T = 4, 64
    dummy_idx = torch.randint(0, cfg.vocab_size, (B, T))
    dummy_targets = torch.randint(0, cfg.vocab_size, (B, T))

    logits, loss = model(dummy_idx, dummy_targets)
    print(f"logits shape: {tuple(logits.shape)}  (expected ({B}, {T}, {cfg.vocab_size}))")
    print(f"loss: {loss.item():.4f}  (expect ~{math.log(cfg.vocab_size):.2f} at init, i.e. ln(vocab_size))")

    # sanity check generation
    prompt = torch.randint(0, cfg.vocab_size, (1, 10))
    out = model.generate(prompt, max_new_tokens=15)
    print(f"generated shape: {tuple(out.shape)}  (expected (1, 25))")
