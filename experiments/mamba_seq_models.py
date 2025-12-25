# experiments/mamba_seq_models.py

"""
Sequence models used in the Mamba sleep experiment.

This file:
- Defines a stacked Mamba block (MambaStack) for sequence modeling
- Implements a bidirectional Mamba-based sequence head (SeqHead)
- Provides a masked cross-entropy loss (masked_ce) over valid time steps only
"""


from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MambaStack(nn.Module):
    """
    Stack of Mamba layers with LayerNorm, residual connections, and dropout.

    Input:
        x    : (B, T, d_model) sequence of embeddings
        mask : (B, T) boolean mask (True = valid, False = padded), optional

    Behavior:
        - Applies `depth` Mamba blocks in series
        - Each block: LayerNorm → Mamba → Dropout → Residual
        - Padded positions are zeroed out using `mask`
    """
    def __init__(self, d_model: int, depth: int = 2, p_drop: float = 0.1):
        super().__init__()
        from mamba_ssm import Mamba

        self.layers = nn.ModuleList([Mamba(d_model) for _ in range(depth)])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(depth)])
        self.drop = nn.Dropout(p_drop)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # mask: (B,T)
        if mask is not None:
            x = x.masked_fill(~mask.unsqueeze(-1), 0.0)

        for ln, layer in zip(self.norms, self.layers):
            h = layer(ln(x))
            h = self.drop(h)
            x = x + h
            if mask is not None:
                x = x.masked_fill(~mask.unsqueeze(-1), 0.0)

        return x


class SeqHead(nn.Module):
    """
    Bidirectional Mamba-based sequence classifier for sleep staging.

    - Takes per-epoch embeddings from MOMENT (shape: [B, T, in_dim])
    - Projects them to d_model, encodes the sequence with MambaStack
    - Optionally runs both forward and backward Mamba and concatenates
    - Outputs per-epoch logits over sleep stages (shape: [B, T, n_classes])
    """
    def __init__(
        self,
        in_dim: int = 1024,
        n_classes: int = 5,
        d_model: int = 256,
        depth: int = 2,
        bidi: bool = True,
        p_drop: float = 0.1,
    ):
        super().__init__()
        self.bidi = bidi

        self.inp = nn.Linear(in_dim, d_model)
        self.fwd = MambaStack(d_model, depth, p_drop)
        self.bwd = MambaStack(d_model, depth, p_drop) if bidi else None

        out_dim = d_model * (2 if bidi else 1)
        self.head = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, n_classes),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        """
        x: (B, T, in_dim)
        mask: (B, T) bool
        """
        x = self.inp(x)  # (B, T, d_model)

        # forward direction
        xf = self.fwd(x, mask)

        if self.bidi:
            # backward direction
            xr = torch.flip(x, [1])
            mr = torch.flip(mask, [1])
            xb = self.bwd(xr, mr)
            xb = torch.flip(xb, [1])
            x = torch.cat([xf, xb], dim=-1)
        else:
            x = xf

        logits = self.head(x)
        logits = logits.masked_fill(~mask.unsqueeze(-1), -1e30)
        return logits


def masked_ce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    class_weights: Optional[torch.Tensor] = None,
    label_smooth: float = 0.05,
):
    """
    Masked cross-entropy for variable-length sequences.

    - Computes CE only on valid timesteps (mask == True)
    - Supports optional label smoothing
    - Used to avoid computing loss on padded positions in night-level batches
    """
    B, T, C = logits.shape
    logits = logits.view(B * T, C)
    targets = targets.view(B * T)
    mask = mask.view(B * T)

    if label_smooth and label_smooth > 0:
        with torch.no_grad():
            true = torch.zeros_like(logits)
            true.fill_(label_smooth / (C - 1))
            true.scatter_(1, torch.clamp(targets, min=0).unsqueeze(1), 1 - label_smooth)
        loss = -(true * F.log_softmax(logits, dim=-1)).sum(dim=1)
    else:
        loss = F.cross_entropy(logits, targets, reduction="none", weight=class_weights)

    return loss[mask].mean()
