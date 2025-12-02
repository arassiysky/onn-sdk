from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from onn_sdk.onn.config import ONNConfig
from onn_sdk.onn.block import ONNBlock


class ONNClassifier(nn.Module):
    """
    Thin wrapper around ONNBlock to expose logits and probabilities.
    """

    def __init__(self, cfg: ONNConfig):
        super().__init__()
        self.block = ONNBlock(cfg)

    def forward(self, x: torch.Tensor, return_probs: bool = False):
        """
        x: (B, n) binary seeds

        Returns:
            logits: (B, 1)
            (optionally) probs: (B, 1) after sigmoid
        """
        logits = self.block(x)
        if return_probs:
            probs = torch.sigmoid(logits)
            return logits, probs
        return logits


class ONNMultiTaskModel(nn.Module):
    """
    Multi-task ONN model with a shared operator backbone and two heads:
      - head_bal   : balancedness (binary)
      - head_orbit : orbit size class (4-way for sizes {1,2,3,6})
    """

    def __init__(self, cfg: ONNConfig):
        super().__init__()
        self.cfg = cfg

        # Shared backbone producing latent features z
        self.backbone = ONNBlock(cfg)

        # We assume cfg.num_layers > 0, so z has dimension cfg.hidden_dim.
        # If you ever use num_layers == 0, you may want to adapt these dims.
        hidden_dim = cfg.hidden_dim if cfg.num_layers > 0 else triangle_feature_dim(cfg.n)

        self.head_bal = nn.Linear(hidden_dim, 1)
        self.head_orbit = nn.Linear(hidden_dim, cfg.num_orbit_classes)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, n) seeds in {0,1}.

        Returns:
            logits_bal:   (B, 1)   for balancedness (before sigmoid)
            logits_orbit: (B, C)   for orbit size classes (before softmax)
        """
        z = self.backbone(x, return_features=True)  # (B, hidden_dim or T(n))
        logits_bal = self.head_bal(z)               # (B, 1)
        logits_orbit = self.head_orbit(z)           # (B, num_orbit_classes)
        return logits_bal, logits_orbit
