"""
EduVerse-1 — decoder-only Transformer, built from scratch in PyTorch.

generate() method includes repetition_penalty and no_repeat_ngram_size to
fix stuttering/looping output. This is a decoding-time fix — doesn't
require retraining, works regardless of corpus size.
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
    ffn_hidden: int = 2048
    dropout: float = 0.1
    bias: bool = True


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: EduVerseConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head

        self.qkv_proj = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.out_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

        mask = torch.tril(torch.ones(cfg.context_length, cfg.context_length))
        self.register_buffer("causal_mask", mask.view(1, 1, cfg.context_length, cfg.context_length))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_dropout(self.out_proj(out))
        return out


class FeedForward(nn.Module):
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
    def __init__(self, cfg: EduVerseConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.ffn = FeedForward(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class EduVerse1(nn.Module):
    def __init__(self, cfg: EduVerseConfig):
        super().__init__()
        self.cfg = cfg

        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.context_length, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList([DecoderBlock(cfg) for _ in range(cfg.n_layer)])
        self.final_ln = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
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
        B, T = idx.shape
        assert T <= self.cfg.context_length

        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.token_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.final_ln(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100
            )
        return logits, loss

    @staticmethod
    def _apply_repetition_penalty(logits: torch.Tensor, generated_ids: list[int], penalty: float) -> torch.Tensor:
        if penalty == 1.0 or not generated_ids:
            return logits
        unique_ids = set(generated_ids)
        for tid in unique_ids:
            score = logits[0, tid]
            if score > 0:
                logits[0, tid] = score / penalty
            else:
                logits[0, tid] = score * penalty
        return logits

    @staticmethod
    def _apply_no_repeat_ngram(logits: torch.Tensor, generated_ids: list[int], n: int) -> torch.Tensor:
        if n <= 0 or len(generated_ids) < n - 1:
            return logits

        seen_ngrams: dict[tuple, list[int]] = {}
        for i in range(len(generated_ids) - n + 1):
            ngram = tuple(generated_ids[i:i + n])
            seen_ngrams.setdefault(ngram[:-1], []).append(ngram[-1])

        prefix = tuple(generated_ids[-(n - 1):])
        for banned_token in seen_ngrams.get(prefix, []):
            logits[0, banned_token] = float("-inf")
        return logits

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: int | None = 40,
        repetition_penalty: float = 1.3,
        no_repeat_ngram_size: int = 3,
    ) -> torch.Tensor:
        self.eval()
        generated_ids: list[int] = idx[0].tolist()

        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.cfg.context_length else idx[:, -self.cfg.context_length:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :].clone() / max(temperature, 1e-6)

            logits = self._apply_repetition_penalty(logits, generated_ids, repetition_penalty)
            logits = self._apply_no_repeat_ngram(logits, generated_ids, no_repeat_ngram_size)

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
            generated_ids.append(next_id.item())

        return idx

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.pos_emb.weight.numel()
        return n