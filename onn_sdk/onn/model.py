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
