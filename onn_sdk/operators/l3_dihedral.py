"""
L3: Dihedral / Orbit Operators

Implements the dihedral-style side maps on XNOR triangles:
- T(s): top side  (the seed itself)
- L(s): left side
- R(s): right side
- rev(s): reversal of a side
- orbit(s): all distinct seeds reachable by side maps + reversal

This corresponds to the D3 action on the triangle boundary in the XNOR papers.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch

from .l1_triangle import (
    xnor_triangle_numpy,
    xnor_triangle_torch,
    xnor_triangle_torch_batch,
)

# =============================================================================
# NumPy implementation
# =============================================================================


def triangle_sides_numpy(s: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Given a seed s (top side), build its XNOR triangle and return (T, L, R),
    where each is a binary array of length n.

    T: top row          (original seed)
    L: left boundary    (first column)
    R: right boundary   (descending diagonal)
    """
    s = np.asarray(s, dtype=np.uint8)
    if s.ndim != 1:
        raise ValueError(f"Expected 1D array, got shape {s.shape}")

    tri = xnor_triangle_numpy(s)
    n = tri.shape[0]

    top = tri[0, :n].copy()
    left = tri[:, 0].copy()
    right = np.array([tri[i, n - 1 - i] for i in range(n)], dtype=tri.dtype)

    return top, left, right


def side_T_numpy(s: np.ndarray) -> np.ndarray:
    """Top side: identity (just return seed)."""
    return np.asarray(s, dtype=np.uint8).copy()


def side_L_numpy(s: np.ndarray) -> np.ndarray:
    """Left side of the XNOR triangle induced by seed s."""
    _, left, _ = triangle_sides_numpy(s)
    return left


def side_R_numpy(s: np.ndarray) -> np.ndarray:
    """Right side of the XNOR triangle induced by seed s."""
    _, _, right = triangle_sides_numpy(s)
    return right


def rev_numpy(s: np.ndarray) -> np.ndarray:
    """Reverse a side."""
    s = np.asarray(s, dtype=np.uint8)
    return s[::-1].copy()


def dihedral_orbit_numpy(s: np.ndarray, include_reversals: bool = True) -> List[np.ndarray]:
    """
    Compute the dihedral orbit of a seed s under side maps + reversal.

    Orbit elements (before deduplication):
        T(s), L(s), R(s),
        rev(T(s)), rev(L(s)), rev(R(s))

    Args:
        s: binary 1D NumPy array
        include_reversals: if False, only {T, L, R} are used.

    Returns:
        List of distinct seeds (np.ndarray) in orbit, in a stable order.
    """
    s = np.asarray(s, dtype=np.uint8)
    T = side_T_numpy(s)
    L = side_L_numpy(s)
    R = side_R_numpy(s)

    seeds = [T, L, R]
    if include_reversals:
        seeds += [rev_numpy(T), rev_numpy(L), rev_numpy(R)]

    # Deduplicate while preserving order
    unique: List[np.ndarray] = []
    seen = set()
    for v in seeds:
        key = tuple(int(x) for x in v.tolist())
        if key not in seen:
            seen.add(key)
            unique.append(v)
    return unique


# =============================================================================
# PyTorch implementation (single-seed)
# =============================================================================


def triangle_sides_torch(s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Torch version of triangle_sides_numpy for a single seed.

    Args:
        s: 1D binary tensor of length n, dtype uint8 or bool or int/float 0/1.

    Returns:
        (top, left, right): each is a 1D tensor of shape (n,) on s.device.
    """
    if s.dim() != 1:
        raise ValueError(f"Expected 1D tensor, got shape {tuple(s.shape)}")

    # xnor_triangle_torch does its own binary checks/casting;
    # we just pass s through.
    tri = xnor_triangle_torch(s)
    n = tri.shape[0]
    device = tri.device
    dtype = tri.dtype

    top = tri[0, :n].clone()
    left = tri[:, 0].clone()
    right = torch.empty(n, dtype=dtype, device=device)
    for i in range(n):
        right[i] = tri[i, n - 1 - i]
    return top, left, right


def side_T_torch(s: torch.Tensor) -> torch.Tensor:
    """Top side: identity (clone)."""
    return s.clone()


def side_L_torch(s: torch.Tensor) -> torch.Tensor:
    """Left side of the XNOR triangle induced by seed s (torch)."""
    _, left, _ = triangle_sides_torch(s)
    return left


def side_R_torch(s: torch.Tensor) -> torch.Tensor:
    """Right side of the XNOR triangle induced by seed s (torch)."""
    _, _, right = triangle_sides_torch(s)
    return right


def rev_torch(s: torch.Tensor) -> torch.Tensor:
    """Reverse a side (torch)."""
    return torch.flip(s, dims=[0])


def dihedral_orbit_torch(
    s: torch.Tensor,
    include_reversals: bool = True,
) -> List[torch.Tensor]:
    """
    Torch version of dihedral_orbit_numpy for a single seed.

    Returns:
        List of distinct side seeds on the same device as s.
    """
    T, L, R = side_T_torch(s), side_L_torch(s), side_R_torch(s)
    seeds = [T, L, R]
    if include_reversals:
        seeds += [rev_torch(T), rev_torch(L), rev_torch(R)]

    unique: List[torch.Tensor] = []
    seen = set()
    for v in seeds:
        key = tuple(int(x) for x in v.detach().cpu().to(torch.uint8).tolist())
        if key not in seen:
            seen.add(key)
            unique.append(v)
    return unique


# =============================================================================
# PyTorch implementation (batched)
# =============================================================================


def triangle_sides_torch_batch(
    tri: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Batched triangle sides from a batched triangle.

    Args:
        tri: (B, n, n) uint8 tensor from xnor_triangle_torch_batch.

    Returns:
        top:  (B, n)
        left: (B, n)
        right:(B, n)
    """
    if tri.dim() != 3:
        raise ValueError(f"Expected (B, n, n) tensor, got shape {tuple(tri.shape)}")

    B, n, m = tri.shape
    if n != m:
        raise ValueError(f"Triangle must be square (n x n), got {n}x{m}")

    top = tri[:, 0, :n]   # (B, n)
    left = tri[:, :, 0]   # (B, n)

    idx = torch.arange(n, device=tri.device)
    right = tri[:, idx, n - 1 - idx]  # (B, n)

    return top, left, right


def dihedral_sides_torch_batch(
    s: torch.Tensor,
    include_reversals: bool = True,
) -> torch.Tensor:
    """
    Batched dihedral sides.

    Args:
        s: (B, n) binary {0,1} (uint8 / bool / ints or floats ~0/1).
        include_reversals: if True, include reversals.

    Returns:
        sides: (B, K, n) uint8 tensor where:
            if include_reversals:
                K = 6  -> [T, L, R, rev(T), rev(L), rev(R)]
            else:
                K = 3  -> [T, L, R]
    """
    if s.dim() != 2:
        raise ValueError(f"Expected input (B, n), got shape {tuple(s.shape)}")

    # Let xnor_triangle_torch_batch handle casting & binary checks
    tri = xnor_triangle_torch_batch(s)  # (B, n, n)
    top, left, right = triangle_sides_torch_batch(tri)  # each (B, n)

    sides = [top, left, right]  # list of (B, n)

    if include_reversals:
        sides += [
            torch.flip(top,  dims=[1]),
            torch.flip(left, dims=[1]),
            torch.flip(right, dims=[1]),
        ]

    sides_tensor = torch.stack(sides, dim=1)   # (B, K, n), uint8
    return sides_tensor


# =============================================================================
# Orbit size helpers (NumPy) for dataset building
# =============================================================================

# Mapping from raw orbit size to a compact class label.
# 1 → 0, 2 → 1, 3 → 2, 6 → 3
_ORBIT_SIZE_TO_CLASS = {
    1: 0,
    2: 1,
    3: 2,
    6: 3,
}


def orbit_size_numpy(s: np.ndarray, include_reversals: bool = True) -> int:
    """
    Compute the dihedral orbit size of a seed s using NumPy.

    Args:
        s: 1D binary array of shape (n,).
        include_reversals: passed to dihedral_orbit_numpy.

    Returns:
        size: orbit size, guaranteed to be one of {1, 2, 3, 6}.

    Raises:
        ValueError if an unexpected orbit size is encountered.
    """
    s = np.asarray(s, dtype=np.uint8)
    orbit = dihedral_orbit_numpy(s, include_reversals=include_reversals)
    size = len(orbit)
    if size not in _ORBIT_SIZE_TO_CLASS:
        raise ValueError(
            f"Unexpected orbit size {size}; "
            f"expected one of {list(_ORBIT_SIZE_TO_CLASS.keys())}."
        )
    return size


def orbit_size_class_numpy(s: np.ndarray, include_reversals: bool = True) -> int:
    """
    Orbit size as a class label in {0,1,2,3} corresponding to sizes {1,2,3,6}.

    Args:
        s: 1D binary array.
        include_reversals: passed to orbit_size_numpy.

    Returns:
        class_id: int in {0,1,2,3}.
    """
    size = orbit_size_numpy(s, include_reversals=include_reversals)
    return _ORBIT_SIZE_TO_CLASS[size]
