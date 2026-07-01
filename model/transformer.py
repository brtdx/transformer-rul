#!/usr/bin/env python3
"""Transformer encoder model for cycle-level RUL prediction (multi-horizon regression).

Architecture reference: Vaswani et al., "Attention Is All You Need", NeurIPS 2017.
arXiv:1706.03762.

This is a PyTorch reimplementation of the original Transformer ENCODER (decoder is
omitted since the task is regression, not seq2seq generation). No code is ported from
the original paper — the paper does not include source code — this implementation
is written from scratch following the architectural description.

Adaptations for battery RUL regression:
- Encoder-only (no decoder)
- Token = cycle feature vector (15-dim) projected to d_model
- Sinusoidal positional encoding (exact original formula)
- Multi-horizon regression heads (h+5, h+10, h+20)
- Pre-LN + GELU (modern stable variant)
- Weighted HuberLoss for multi-horizon training
"""
import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal position embedding using the cycle index (1..N) as position."""

    def __init__(self, d_model, max_len=1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(pos * div)
        else:
            pe[:, 1::2] = torch.cos(pos * div[:-1])
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x, offset=1):
        # x: (B, L, d_model); offset shifts position index (so cycle=1..N not 0..N-1)
        L = x.size(1)
        return x + self.pe[:, offset:offset + L, :]


class RULTransformer(nn.Module):
    def __init__(self,
                 n_features=15,
                 d_model=32,
                 n_heads=2,
                 n_layers=2,
                 ffn_dim=64,
                 dropout=0.1,
                 horizons=(5, 10, 20)):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=2048)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        # Separate regression heads for each horizon
        self.heads = nn.ModuleList([nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        ) for _ in horizons])

    def forward(self, x, src_key_padding_mask=None):
        # x: (B, L, F)
        h = self.input_proj(x)
        h = self.pos_enc(h, offset=1)
        h = self.encoder(h, src_key_padding_mask=src_key_padding_mask)
        h = self.norm(h)
        # Use the last token (most recent cycle) as summary
        pooled = h[:, -1, :]
        pooled = self.dropout(pooled)
        outs = [head(pooled).squeeze(-1) for head in self.heads]
        return torch.stack(outs, dim=-1)  # (B, H)


def weighted_huber_loss(pred, target, weights=(1.0, 0.8, 0.6), delta=0.01):
    """Multi-horizon weighted Huber loss (targets normalized to [0,1]).

    pred, target: (B, H) in [0,1] range
    delta=0.01 means errors below 1% (in normalized scale) are treated as squared,
    larger errors are linear — robust to outliers while stable for small targets.
    """
    weights = torch.tensor(weights, dtype=pred.dtype, device=pred.device)
    huber = nn.functional.huber_loss(pred, target, reduction='none', delta=delta)
    return (huber * weights.unsqueeze(0)).sum(dim=-1).mean()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    m = RULTransformer()
    n = count_parameters(m)
    x = torch.randn(8, 20, 15)
    y = m(x)
    print(f"output shape: {y.shape}  (B, H=3)")
    print(f"params: {n:,}  (~{n/1e6:.2f}M)")
    loss = weighted_huber_loss(y, torch.randn(8, 3))
    print(f"loss: {loss.item():.4f}")