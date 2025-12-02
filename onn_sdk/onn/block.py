from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from onn_sdk.onn.config import ONNConfig
from onn_sdk.operators.l1_triangle import (
    xnor_triangle_torch_flat_batch,   # batched L1: (N, n) -> (N, T(n))
)
from onn_sdk.operators.l3_dihedral import (
    dihedral_sides_torch_batch,       # batched L3: (B, n) -> (B, K3, n)
)
from onn_sdk.operators.l4_masks import (
    stack_mask_family_torch,          # (M, n) masks
)


def triangle_feature_dim(n: int) -> int:
    """T(n) = n(n+1)//2"""
    return n * (n + 1) // 2


class ONNBlock(nn.Module):
    """
    ONN block for XNOR-based classification with internal L3/L4 logic.

    Batched pipeline for x ∈ {0,1}^{B×n}:
        1) Optionally expand to dihedral orbit seeds (L3) → (B, K3, n).
        2) Optionally apply L4 masks to each orbit seed → (B, K, n).
        3) Build XNOR triangles and flatten for all transformed seeds (L1) in batch.
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
    # L4 mask cache
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

        # ---- L3: dihedral orbits (batched) ----
        if self.use_orbits:
            # Expect dihedral_sides_torch_batch(x_bin, include_reversals=True)
            # to return something like (B, K3, n), where K3 ∈ {1,2,3,6}.
            seeds = dihedral_sides_torch_batch(
                x_bin,
                include_reversals=True,
            )  # (B, K3, n)
        else:
            seeds = x_bin.unsqueeze(1)  # (B, 1, n)

        # ---- L4: masks (apply via XOR, batched) ----
        if self.use_masks:
            masks = self._get_masks(device)  # (M, n) or None
            if masks is not None and masks.numel() > 0:
                B, K3, n = seeds.shape
                M = masks.shape[0]

                # base seeds kept
                seeds_base = seeds  # (B, K3, n)

                # expand seeds and masks for pairwise XOR
                seeds_exp = seeds_base.unsqueeze(2)          # (B, K3, 1, n)
                masks_exp = masks.view(1, 1, M, n)          # (1, 1, M, n)
                seeds_exp = seeds_exp.expand(B, K3, M, n)   # (B, K3, M, n)
                masks_exp = masks_exp.expand(B, K3, M, n)   # (B, K3, M, n)

                masked = torch.bitwise_xor(
                    seeds_exp,
                    masks_exp.to(device=device, dtype=torch.uint8),
                )                                           # (B, K3, M, n)

                # Flatten mask dimension into orbit dimension
                masked = masked.reshape(B, K3 * M, n)       # (B, K3*M, n)

                # Concatenate unmasked + masked along orbit dimension
                seeds_all = torch.cat([seeds_base, masked], dim=1)  # (B, K, n)
            else:
                seeds_all = seeds  # no masks
        else:
            seeds_all = seeds      # no masks

        B, K, n = seeds_all.shape

        # ---- L1: XNOR triangles for all transformed seeds (batched) ----
        # Flatten (B, K, n) -> (B*K, n)
        seeds_flat = seeds_all.reshape(B * K, n)            # (B*K, n)

        # Compute flattened XNOR triangles for all seeds in one call
        feats_flat = xnor_triangle_torch_flat_batch(seeds_flat)  # (B*K, T(n))

        # Reshape back to (B, K, T(n))
        feats = feats_flat.reshape(B, K, self.feature_dim)  # (B, K, T(n))

        # Aggregate across transforms (mean pooling over K)
        feats_agg = feats.mean(dim=1)                       # (B, T(n))

        # Map {0,1} → {-1,+1} for better conditioning
        feat_tensor = feats_agg.to(torch.float32)
        feat_tensor = 2.0 * feat_tensor - 1.0               # (B, T(n))
        return feat_tensor

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
