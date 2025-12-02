from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from onn_sdk.onn.config import ONNConfig
from onn_sdk.operators.l1_triangle import xnor_triangle_torch_flat_batch
from onn_sdk.operators.l3_dihedral import dihedral_sides_torch_batch
from onn_sdk.operators.l4_masks import stack_mask_family_torch


def triangle_feature_dim(n: int) -> int:
    """T(n) = n(n+1)/2"""
    return n * (n + 1) // 2


class ONNBlock(nn.Module):
    """
    ONN block for XNOR-based classification with internal L3/L4 logic.

    Batched pipeline for x ∈ {0,1}^{B×n}:
        1) Optionally expand to dihedral orbit seeds (L3) → (B, K3, n).
        2) Optionally apply L4 masks to each orbit seed → (B, K, n).
        3) For each transformed seed, build XNOR triangle and flatten (L1) → (B, K, T(n)).
        4) Aggregate features across transforms (mean pooling) → (B, T(n)).
        5) Map {0,1} → {-1,+1} and pass through MLP trunk.

    This module can be used either as:
        - a standalone binary classifier (balancedness) when return_features=False, or
        - a backbone / feature extractor when return_features=True.
    """

    def __init__(self, cfg: ONNConfig):
        super().__init__()
        self.cfg = cfg
        self.n = cfg.n
        self.feature_dim = triangle_feature_dim(self.n)

        self.use_orbits = cfg.use_orbits
        self.use_masks = cfg.use_masks
        self.extra_random_masks = cfg.extra_random_masks

        # Masks cached as a single (M, n) tensor on a specific device
        self._masks: Optional[torch.Tensor] = None
        self._masks_device: Optional[torch.device] = None

        # MLP trunk split into body + head_bal for single-task mode
        body_layers = []
        in_dim = self.feature_dim
        for _ in range(cfg.num_layers):
            body_layers.append(nn.Linear(in_dim, cfg.hidden_dim))
            body_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                body_layers.append(nn.Dropout(cfg.dropout))
            in_dim = cfg.hidden_dim

        self.mlp_body = nn.Sequential(*body_layers)
        # Head used in single-task mode (balancedness)
        self.head_bal = nn.Linear(in_dim, 1)

    # ------------------------------------------------------------------
    # L4 mask cache (batched)
    # ------------------------------------------------------------------

    def _get_masks(self, device: torch.device) -> Optional[torch.Tensor]:
        """
        Return cached L4 masks as a tensor of shape (M, n) on the given device,
        generating them on first use.
        """
        if not self.use_masks:
            return None

        if self._masks is None or self._masks_device != device:
            self._masks = stack_mask_family_torch(
                n=self.n,
                device=device,
                use_sierpinski=True,
                use_thue_morse=True,
                extra_random=self.extra_random_masks,
                p_random=0.5,
            )  # (M, n)
            self._masks_device = device
        return self._masks

    # ------------------------------------------------------------------
    # Seeds → triangle features (batched L1+L3+L4)
    # ------------------------------------------------------------------

    def _seeds_to_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, n) in {0,1} (float or int)
        returns: (B, T(n)) float32
        """
        if x.dim() != 2 or x.size(1) != self.n:
            raise ValueError(f"Expected input of shape (B, {self.n}), got {tuple(x.shape)}")

        # Ensure binary 0/1 uint8 for operator layers
        if x.is_floating_point():
            x_bin = torch.round(x).to(torch.uint8)
        else:
            x_bin = x.to(torch.uint8)

        if not torch.all((x_bin == 0) | (x_bin == 1)):
            raise ValueError("Input x must contain only 0/1 values")

        B, n = x_bin.shape
        device = x_bin.device

        # -------- L3: dihedral orbit seeds (or identity) --------
        if self.use_orbits:
            base_seeds = dihedral_sides_torch_batch(
                x_bin,
                include_reversals=True,
            )  # (B, Kb, n), uint8
        else:
            base_seeds = x_bin.unsqueeze(1)  # (B, 1, n)

        B, Kb, n = base_seeds.shape

        # -------- L4: masks (optional) --------
        masks = self._get_masks(device)  # (M, n) or None

        if masks is not None and masks.numel() > 0:
            M = masks.size(0)

            seeds_orig = base_seeds  # (B, Kb, n)

            masked = torch.bitwise_xor(
                base_seeds.unsqueeze(2),        # (B, Kb, 1, n)
                masks.view(1, 1, M, n),         # (1, 1, M, n)
            )                                     # (B, Kb, M, n)

            masked = masked.view(B, Kb * M, n)    # (B, Kb*M, n)

            all_seeds = torch.cat([seeds_orig, masked], dim=1)  # (B, K, n)
        else:
            all_seeds = base_seeds  # (B, Kb, n)

        B, K, n = all_seeds.shape

        # -------- L1: XNOR triangles for all transformed seeds --------
        seeds_flat = all_seeds.view(B * K, n)                         # (B*K, n)
        tri_flat = xnor_triangle_torch_flat_batch(seeds_flat)         # (B*K, T(n))
        T = tri_flat.size(1)

        tri_features = tri_flat.view(B, K, T).float()                 # (B, K, T)
        feats_agg = tri_features.mean(dim=1)                          # (B, T)

        # Map {0,1} → {-1,+1} for better conditioning
        feats = 2.0 * feats_agg - 1.0                                 # (B, T)
        return feats

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        """
        Args:
            x: (B, n) seeds in {0,1}, type float or int.
            return_features: if True, return backbone features z instead of logits.

        Returns:
            If return_features=False:
                logits: (B, 1) tensor of raw logits for balancedness.
            If return_features=True:
                z: (B, D) tensor of latent features (D = hidden_dim if num_layers > 0, else T(n)).
        """
        feats = self._seeds_to_features(x)  # (B, T(n))

        # Pass through MLP body if any layers are defined.
        if self.cfg.num_layers > 0:
            z = self.mlp_body(feats)       # (B, hidden_dim)
        else:
            z = feats                      # (B, T(n))

        if return_features:
            return z

        logits = self.head_bal(z)          # (B, 1)
        return logits

    