"""
Dataset builder utilities for XNOR / ONN experiments.

For now we implement:
  - Random seed generation
  - Balancedness label via L1 XNOR triangle
  - Optional L3 orbit and L4 mask augmentation

The main entry point is:
  build_xnor_balanced_dataset(...)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np

from onn_sdk.operators.l1_triangle import is_balanced_numpy
from onn_sdk.operators.l3_dihedral import dihedral_orbit_numpy
from onn_sdk.operators.l4_masks import (
    generate_mask_family_numpy,
    apply_mask_family_numpy,
)


# ---------------------------------------------------------------------------
# Basic seed utilities
# ---------------------------------------------------------------------------

def random_seed_numpy(n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Draw a random binary seed of length n.
    """
    if rng is None:
        rng = np.random.default_rng()
    return rng.integers(0, 2, size=n, dtype=np.uint8)


def sample_seeds_with_filter(
    n: int,
    target_count: int,
    predicate: Callable[[np.ndarray], bool],
    max_attempts: int = 1_000_000,
    rng: Optional[np.random.Generator] = None,
) -> List[np.ndarray]:
    """
    Generic sampler: collect seeds of length n that satisfy a predicate.

    Args:
        n: seed length.
        target_count: how many seeds to collect.
        predicate: function s -> bool.
        max_attempts: global cap on random draws.
        rng: optional RNG.

    Returns:
        List of seeds (each np.ndarray of shape (n,)).

    Raises:
        RuntimeError if unable to find enough seeds within max_attempts.
    """
    if rng is None:
        rng = np.random.default_rng()

    seeds: List[np.ndarray] = []
    attempts = 0

    while len(seeds) < target_count and attempts < max_attempts:
        s = random_seed_numpy(n, rng=rng)
        attempts += 1
        if predicate(s):
            seeds.append(s)

    if len(seeds) < target_count:
        raise RuntimeError(
            f"Failed to find {target_count} seeds after {attempts} attempts "
            f"for n={n}. Collected {len(seeds)}."
        )
    return seeds


# ---------------------------------------------------------------------------
# Config and main builder
# ---------------------------------------------------------------------------

@dataclass
class XNORBalancedDatasetConfig:
    n: int                      # seed length
    n_pos: int                  # number of balanced seeds
    n_neg: int                  # number of non-balanced seeds
    use_orbits: bool = False    # L3 dihedral orbits
    use_masks: bool = False     # L4 masks
    extra_random_masks: int = 0 # additional random masks if use_masks=True
    shuffle: bool = True        # shuffle dataset at the end
    seed: Optional[int] = None  # RNG seed for reproducibility


def build_xnor_balanced_dataset(
    cfg: XNORBalancedDatasetConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a simple binary classification dataset:
        X: binary seeds of shape (N, n)
        y: labels in {0,1}, where 1 = balanced triangle, 0 = non-balanced.

    Steps:
        1. Sample n_pos balanced seeds.
        2. Sample n_neg non-balanced seeds.
        3. Optionally augment each seed via:
           - L3 dihedral orbits
           - L4 mask families
        4. Return all resulting seeds and labels.
    """
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n

    # --- Step 1: sample positive (balanced) seeds ---
    balanced_seeds = sample_seeds_with_filter(
        n=n,
        target_count=cfg.n_pos,
        predicate=is_balanced_numpy,
        rng=rng,
    )

    # --- Step 2: sample negative (non-balanced) seeds ---
    def not_balanced(s: np.ndarray) -> bool:
        return not is_balanced_numpy(s)

    neg_seeds = sample_seeds_with_filter(
        n=n,
        target_count=cfg.n_neg,
        predicate=not_balanced,
        rng=rng,
    )

    all_seeds: List[np.ndarray] = []
    all_labels: List[int] = []

    # Precompute masks if needed
    masks: Optional[List[np.ndarray]] = None
    if cfg.use_masks:
        masks = generate_mask_family_numpy(
            n=n,
            use_sierpinski=True,
            use_thue_morse=True,
            extra_random=cfg.extra_random_masks,
            rng=rng,
        )

    # Helper: augment a single seed with L3/L4 if requested
    def augment_seed(s: np.ndarray, label: int):
        # Start from the base seed
        seeds_here: List[np.ndarray] = [s]

        # L3 orbits
        if cfg.use_orbits:
            # dihedral_orbit_numpy already includes s in general, so we can
            # just replace seeds_here
            seeds_here = dihedral_orbit_numpy(s, include_reversals=True)

        # L4 masks
        if cfg.use_masks and masks is not None:
            expanded: List[np.ndarray] = []
            for base in seeds_here:
                expanded.append(base)
                expanded.extend(apply_mask_family_numpy(base, masks))
            seeds_here = expanded

        # Append to global lists
        for v in seeds_here:
            all_seeds.append(v.astype(np.uint8))
            all_labels.append(label)

    # --- Add positives ---
    for s in balanced_seeds:
        augment_seed(s, label=1)

    # --- Add negatives ---
    for s in neg_seeds:
        augment_seed(s, label=0)

    X = np.stack(all_seeds, axis=0)  # shape (N, n)
    y = np.asarray(all_labels, dtype=np.int64)

    # --- Shuffle ---
    if cfg.shuffle:
        perm = rng.permutation(X.shape[0])
        X = X[perm]
        y = y[perm]

    return X, y
