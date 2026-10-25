"""Transformer velocity network shared by every learned method in the paper.

JointVelocityNetV2: pre-norm transformer encoder over the trajectory nodes with
sinusoidal flow-time embedding and an MLP embedding of the maneuver condition.
The paper configuration is hidden=384, 6 layers, 8 heads (11.1M parameters with
the FM-Riem head). FM-Eucl, DDPM-Eucl and Regression wrap the same backbone with
a 10-dim output head, so all learned baselines have identical capacity.
"""
import math
import torch
import torch.nn as nn


class SinusoidalT(nn.Module):
    def __init__(self, dim):
        super().__init__(); self.dim = dim
    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0)
                          * torch.arange(half, device=t.device).float() / max(half - 1, 1))
        ang = t.unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([ang.sin(), ang.cos()], dim=-1)


class JointVelocityNetV2(nn.Module):
    """Same I/O contract as JointVelocityNet, larger and tuned.

    forward(state, t, cond) -> (B, T, 9)
      state: (B, T, 10) raw [q, omega, tau]
      t:     (B,)
      cond:  (B, cond_dim) raw
    """
    def __init__(self, seq_len, cond_dim, hidden=384, n_layers=6, n_heads=8,
                 dropout=0.0):
        super().__init__()
        self.seq_len = seq_len
        self.in_proj = nn.Linear(10, hidden)
        self.pos_emb = nn.Parameter(torch.randn(1, seq_len, hidden) * 0.02)
        self.t_embed = nn.Sequential(SinusoidalT(hidden),
                                      nn.Linear(hidden, hidden), nn.SiLU(),
                                      nn.Linear(hidden, hidden))
        self.cond_embed = nn.Sequential(nn.Linear(cond_dim, hidden), nn.SiLU(),
                                         nn.Linear(hidden, hidden))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=n_heads, dim_feedforward=hidden * 4,
            dropout=dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, n_layers)
        self.out_norm = nn.LayerNorm(hidden)
        self.out_proj = nn.Linear(hidden, 9)

    def forward(self, state, t, cond):
        h = self.in_proj(state) + self.pos_emb[:, :state.size(1), :]
        emb = self.t_embed(t) + self.cond_embed(cond)
        h = h + emb[:, None, :]
        h = self.transformer(h)
        h = self.out_norm(h)
        return self.out_proj(h)
