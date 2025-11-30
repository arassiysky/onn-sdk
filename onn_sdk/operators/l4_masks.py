"""
L4: Mask Operators (Sierpiński, Thue–Morse, etc.)

These are global affine-style operators acting on seeds in {0,1}^n via XOR with
fixed binary masks. This corresponds to a subset of L4 in the ONN hierarchy.

We provide:
- Sierpinski mask   (Pascal triangle mod 2)
- Thue–Morse mask   (bit-parity of index)
- Simple combinators and XOR-application helpers

Backends:
- NumPy
- PyTorch
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch


# =============================================================================
# NumPy mask generators
# =============================================================================

def sierpinski_mask_numpy(n: int) -> np.ndarray:
    """
    Row (n-1) of Pascal's triangle mod 2, using the classic bitwise test:
        C(n-1, k) is odd  <=>  (k & (n-1)) == k.

    Args:
        n: length of the seed.

    Returns:
        mask: uint8 array of shape (n,), entries in {0,1}.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    last = n - 1
    idx = np.arange(n, dtype=np.uint64)
    mask = ((idx & last) == idx).astype(np.uint8)
    return mask


def thue_morse_mask_numpy(n: int) -> np.ndarray:
    """
    Thue–Morse sequence of length n:
        t(k) = parity of number of 1-bits in k.

    So t(k) = 0 if popcount(k) is even,
             1 if popcount(k) is odd.

    Args:
        n: length of the seed.

    Returns:
        mask: uint8 array of shape (n,), entries in {0,1}.
    """
    if n <= 0:
        raise ValueError("n must be positive")

    idx = np.arange(n, dtype=np.uint64)
    x = idx.copy()
    parity = np.zeros_like(idx, dtype=np.uint8)

    # Compute parity of bits: repeated shift-and-xor
    while np.any(x):
        parity ^= (x & 1).astype(np.uint8)
        x >>= 1

    return parity.astype(np.uint8)


def random_mask_numpy(n: int, p: float = 0.5, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Simple random binary mask with Bernoulli(p) entries.

    Args:
        n: length.
        p: probability of 1.
        rng: optional NumPy Generator.

    Returns:
        mask: uint8 array of shape (n,).
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if rng is None:
        rng = np.random.default_rng()
    return (rng.random(n) < p).astype(np.uint8)


def apply_mask_xor_numpy(seed: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Apply a binary mask via XOR: seed' = seed XOR mask.

    Args:
        seed: 1D binary array (0/1 or bool).
        mask: 1D binary array of same length.

    Returns:
        new_seed: 1D uint8 array in {0,1}.
    """
    seed = np.asarray(seed).astype(np.uint8)
    mask = np.asarray(mask).astype(np.uint8)
    if seed.shape != mask.shape:
        raise ValueError(f"Shape mismatch: seed {seed.shape}, mask {mask.shape}")
    return np.bitwise_xor(seed, mask).astype(np.uint8)


def generate_mask_family_numpy(
    n: int,
    use_sierpinski: bool = True,
    use_thue_morse: bool = True,
    extra_random: int = 0,
    rng: Optional[np.random.Generator] = None,
) -> List[np.ndarray]:
    """
    Build a small family of masks to act as L4 operators.

    Args:
        n: length.
        use_sierpinski: include Sierpinski mask.
        use_thue_morse: include Thue–Morse mask.
        extra_random: number of additional random masks to include.
        rng: optional RNG.

    Returns:
        List of masks (each uint8 array of shape (n,)).
    """
    masks: List[np.ndarray] = []
    if use_sierpinski:
        masks.append(sierpinski_mask_numpy(n))
    if use_thue_morse:
        masks.append(thue_morse_mask_numpy(n))
    if extra_random > 0:
        if rng is None:
            rng = np.random.default_rng()
        for _ in range(extra_random):
            masks.append(random_mask_numpy(n, rng=rng))
    return masks


def apply_mask_family_numpy(seed: np.ndarray, masks: List[np.ndarray]) -> List[np.ndarray]:
    """
    Apply a list of masks to a single seed, returning all resulting seeds.

    Args:
        seed: 1D binary array.
        masks: list of 1D binary masks of same length.

    Returns:
        List of seeds[i] = seed XOR masks[i].
    """
    out: List[np.ndarray] = []
    for m in masks:
        out.append(apply_mask_xor_numpy(seed, m))
    return out


# =============================================================================
# PyTorch mask generators
# =============================================================================

def sierpinski_mask_torch(n: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Torch version of sierpinski_mask_numpy.

    Args:
        n: length.
        device: optional device (cpu/cuda). If None, uses cpu.

    Returns:
        mask: uint8 tensor of shape (n,) on given device.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if device is None:
        device = torch.device("cpu")

    last = n - 1
    idx = torch.arange(n, dtype=torch.int64, device=device)
    mask = ((idx & last) == idx).to(torch.uint8)
    return mask


def thue_morse_mask_torch(n: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Torch version of thue_morse_mask_numpy.

    Args:
        n: length.
        device: optional device.

    Returns:
        mask: uint8 tensor of shape (n,).
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if device is None:
        device = torch.device("cpu")

    idx = torch.arange(n, dtype=torch.int64, device=device)
    x = idx.clone()
    parity = torch.zeros_like(idx, dtype=torch.uint8)

    # parity ^= (x & 1), then shift x >>= 1
    # loop until all bits zero
    while torch.any(x != 0):
        parity ^= (x & 1).to(torch.uint8)
        x = x >> 1

    return parity.to(torch.uint8)


def random_mask_torch(n: int, p: float = 0.5, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Random binary mask in torch.

    Args:
        n: length.
        p: probability of 1.
        device: optional device.

    Returns:
        mask: uint8 tensor of shape (n,).
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if device is None:
        device = torch.device("cpu")

    return (torch.rand(n, device=device) < p).to(torch.uint8)


def apply_mask_xor_torch(seed: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Apply a binary mask via XOR in torch.

    Args:
        seed: 1D tensor of shape (n,) with {0,1}.
        mask: 1D tensor of shape (n,) with {0,1}.

    Returns:
        new_seed: 1D uint8 tensor in {0,1}.
    """
    if seed.shape != mask.shape:
        raise ValueError(f"Shape mismatch: seed {seed.shape}, mask {mask.shape}")
    seed_u = seed.to(torch.uint8)
    mask_u = mask.to(torch.uint8)
    return torch.bitwise_xor(seed_u, mask_u)


def generate_mask_family_torch(
    n: int,
    device: Optional[torch.device] = None,
    use_sierpinski: bool = True,
    use_thue_morse: bool = True,
    extra_random: int = 0,
    p_random: float = 0.5,
) -> List[torch.Tensor]:
    """
    Build a small family of masks in torch.

    Args:
        n: length.
        device: target device.
        use_sierpinski: include Sierpinski mask.
        use_thue_morse: include Thue–Morse mask.
        extra_random: number of additional random masks.
        p_random: Bernoulli parameter for random masks.

    Returns:
        List of masks (uint8 tensors of shape (n,) on device).
    """
    if device is None:
        device = torch.device("cpu")

    masks: List[torch.Tensor] = []
    if use_sierpinski:
        masks.append(sierpinski_mask_torch(n, device=device))
    if use_thue_morse:
        masks.append(thue_morse_mask_torch(n, device=device))
    for _ in range(extra_random):
        masks.append(random_mask_torch(n, p=p_random, device=device))
    return masks


def apply_mask_family_torch(seed: torch.Tensor, masks: List[torch.Tensor]) -> List[torch.Tensor]:
    """
    Apply a list of masks to a torch seed.

    Args:
        seed: 1D tensor of shape (n,).
        masks: list of tensors of shape (n,).

    Returns:
        List of seeds[i] = seed XOR masks[i].
    """
    out: List[torch.Tensor] = []
    for m in masks:
        out.append(apply_mask_xor_torch(seed, m))
    return out

