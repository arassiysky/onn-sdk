from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from onn_sdk.onn.config import ONNConfig
from onn_sdk.operators.l1_triangle import xnor_triangle_torch_flat
from onn_sdk.operators.l3_dihedral import dihedral_orbit_torch
from onn_sdk.operators.l4_masks import (
    generate_mask_family_torch,
    apply_mask_xor_torch,
)


def triangle_feature_dim(n: int) -> int:
    """T(n) = n(n+1)/2"""
    return n * (n + 1) // 2


class ONNBlock(nn.Module):
    """
    Minimal ONN block for XNOR-based classification with internal L3/L4 logic.

    Pipeline per seed x ∈ {0,1}^n:
        1) Optionally expand to dihedral orbit seeds (L3).
        2) Optionally apply L4 masks to each orbit seed.
        3) For each transformed seed, build XNOR triangle and flatten.
        4) Aggregate features across all transforms (mean pooling).
        5) Map {0,1} → {-1,+1} and pass through MLP trunk.
    """

    def __init__(self, cfg: ONNConfig):
        super().__init__()
        self.cfg = cfg
        self.n = cfg.n
        self.feature_dim = triangle_feature_dim(self.n)

        self.use_orbits = cfg.use_orbits
        self.use_masks = cfg.use_masks
        self.extra_random_masks = cfg.extra_random_masks

        # We cache masks per device
        self._masks: Optional[List[torch.Tensor]] = None
        self._masks_device: Optional[torch.device] = None

        layers = []
        in_dim = self.feature_dim
        for _ in range(cfg.num_layers):
            layers.append(nn.Linear(in_dim, cfg.hidden_dim))
            layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                layers.append(nn.Dropout(cfg.dropout))
            in_dim = cfg.hidden_dim

        # Final binary classifier head
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

    # ------------------------------------------------------------------
    # L4 mask cache
    # ------------------------------------------------------------------

    def _get_masks(self, device: torch.device) -> List[torch.Tensor]:
        """
        Return cached L4 masks on the given device, generating them on first use.
        """
        if not self.use_masks:
            return []

        if self._masks is None or self._masks_device != device:
            self._masks = generate_mask_family_torch(
                n=self.n,
                device=device,
                use_sierpinski=True,
                use_thue_morse=True,
                extra_random=self.extra_random_masks,
            )
            self._masks_device = device
        return self._masks

    # ------------------------------------------------------------------
    # L3+L4 augmentation per seed
    # ------------------------------------------------------------------

    def _augment_seed(self, seed: torch.Tensor) -> List[torch.Tensor]:
        """
        Apply L3 dihedral orbit and L4 masks to a single seed.

        Args:
            seed: 1D tensor of shape (n,) in {0,1}, dtype uint8.

        Returns:
            List of transformed seeds (each shape (n,), uint8).
        """
        device = seed.device

        # Start with the original seed
        seeds: List[torch.Tensor] = [seed]

        # L3 orbits
        if self.use_orbits:
            seeds = dihedral_orbit_torch(seed, include_reversals=True)

        # L4 masks
        if self.use_masks:
            masks = self._get_masks(device)
            if masks:
                expanded: List[torch.Tensor] = []
                for s in seeds:
                    expanded.append(s)
                    for m in masks:
                        expanded.append(apply_mask_xor_torch(s, m))
                seeds = expanded

        return seeds

    # ------------------------------------------------------------------
    # Seeds → triangle features
    # ------------------------------------------------------------------

    def _seeds_to_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, n) in {0,1} (float or int)
        returns: (B, T(n)) float32
        """
        if x.dim() != 2 or x.size(1) != self.n:
            raise ValueError(f"Expected input of shape (B, {self.n}), got {tuple(x.shape)}")

        # Ensure binary 0/1 uint8 for L1 operator
        if x.is_floating_point():
            x_bin = torch.round(x).to(torch.uint8)
        else:
            x_bin = x.to(torch.uint8)

        batch_size = x_bin.size(0)
        device = x_bin.device

        feats_batch: List[torch.Tensor] = []

        for i in range(batch_size):
            seed = x_bin[i]  # (n,)
            transformed_seeds = self._augment_seed(seed)

            # Build triangle features for each transformed seed
            feats_list: List[torch.Tensor] = []
            for s in transformed_seeds:
                flat = xnor_triangle_torch_flat(s)  # (T(n),), uint8
                feats_list.append(flat.to(torch.float32))

            # Aggregate across transforms (mean pooling)
            feats_stack = torch.stack(feats_list, dim=0)  # (K, T(n))
            feats_agg = feats_stack.mean(dim=0)           # (T(n),)

            feats_batch.append(feats_agg)

        feat_tensor = torch.stack(feats_batch, dim=0)     # (B, T(n))

        # Map {0,1} → {-1,+1} for better conditioning
        feat_tensor = 2.0 * feat_tensor - 1.0
        return feat_tensor

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, n) seeds in {0,1}, type float or int.

        Returns:
            logits: (B, 1) tensor of raw logits (before sigmoid).
        """
        feats = self._seeds_to_features(x)
        logits = self.mlp(feats)
        return logits