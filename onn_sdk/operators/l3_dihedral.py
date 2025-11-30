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
# PyTorch implementation
# =============================================================================

def triangle_sides_torch(s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Torch version of triangle_sides_numpy.

    Args:
        s: 1D binary tensor of length n.

    Returns:
        (top, left, right): each is a 1D tensor of shape (n,) on s.device.
    """
    tri = xnor_triangle_torch(s)
    n = tri.shape[0]
    device = s.device
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
    Torch version of dihedral_orbit_numpy.

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

