from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPBaseline(nn.Module):
    """
    Simple baseline: plain MLP on raw binary seeds.

    Input:  x ∈ {0,1}^{B×n}
    Output: logits ∈ R^{B×1}
    """

    def __init__(
        self,
        n: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n = n

        layers = []
        in_dim = n
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        # Final binary classifier head
        layers.append(nn.Linear(in_dim, 1))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, n) seeds in {0,1} (float or int)

        Returns:
            logits: (B, 1)
        """
        if x.dim() != 2 or x.size(1) != self.n:
            raise ValueError(f"Expected input of shape (B, {self.n}), got {tuple(x.shape)}")

        if not x.is_floating_point():
            x = x.to(torch.float32)

        logits = self.mlp(x)
        return logits