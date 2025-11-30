"""
L1: XNOR Triangle Operator

This module implements the core XNOR reduction triangle on binary seeds s ∈ {0,1}^n.

Two main backends:
    - NumPy (CPU, simple, good for analysis)
    - PyTorch (CPU/GPU, integrates with ONN models)

API (NumPy):
    xnor_triangle_numpy(s)         -> 2D triangle, shape (n, n)
    xnor_triangle_numpy_flat(s)    -> flattened triangle, length T(n) = n(n+1)/2
    triangle_one_count_numpy(s)    -> number of ones in the triangle
    is_balanced_numpy(s)           -> balancedness predicate

API (PyTorch):
    xnor_triangle_torch(s)         -> 2D triangle, shape (n, n), device = s.device
    xnor_triangle_torch_flat(s)    -> flattened triangle
    triangle_one_count_torch(s)    -> number of ones in the triangle
    is_balanced_torch(s)           -> balancedness predicate

All inputs are 1D binary arrays/tensors (0/1 or bool).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Helpers: sanity checks
# ---------------------------------------------------------------------------

def _check_binary_numpy(s: np.ndarray) -> np.ndarray:
    """
    Ensure s is a 1D binary NumPy array with values in {0,1}.
    Returns a uint8 array.
    """
    s = np.asarray(s)
    if s.ndim != 1:
        raise ValueError(f"Expected 1D array, got shape {s.shape}")
    # Allow bool or numeric 0/1
    s_uint = s.astype(np.uint8)
    if not np.array_equal(s_uint, (s_uint & 1)):
        raise ValueError("Input must be binary (0/1 or bool)")
    return s_uint


def _check_binary_torch(s: torch.Tensor) -> torch.Tensor:
    """
    Ensure s is a 1D binary PyTorch tensor with values in {0,1}.
    Returns a uint8 tensor on the same device.
    """
    if s.dim() != 1:
        raise ValueError(f"Expected 1D tensor, got shape {tuple(s.shape)}")

    if s.dtype == torch.bool:
        return s.to(torch.uint8)

    if s.is_floating_point():
        # Allow floats very close to 0/1
        s_rounded = torch.round(s).to(torch.uint8)
        if not torch.all((s_rounded == 0) | (s_rounded == 1)):
            raise ValueError("Floating tensor must contain only ~0.0/~1.0 values")
        return s_rounded

    if s.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        s_uint = s.to(torch.uint8)
        if not torch.all((s_uint == 0) | (s_uint == 1)):
            raise ValueError("Integer tensor must contain only 0/1 values")
        return s_uint

    raise TypeError(f"Unsupported torch dtype for binary seed: {s.dtype}")


# ---------------------------------------------------------------------------
# NumPy implementation
# ---------------------------------------------------------------------------

def xnor_triangle_numpy(s: np.ndarray) -> np.ndarray:
    """
    Build the XNOR reduction triangle for a 1D binary NumPy array s ∈ {0,1}^n.

    Triangle definition:
        row 0 = s
        row i+1[j] = s_i,j XNOR s_i,j+1

    We store it in an (n, n) matrix with the upper triangle filled and
    the rest zeroed.

    Args:
        s: 1D binary array of length n (0/1 or bool).

    Returns:
        tri: uint8 array of shape (n, n).
             Row 0 has n valid entries, row 1 has n-1, ..., row n-1 has 1.
    """
    s = _check_binary_numpy(s)
    n = s.shape[0]

    tri = np.zeros((n, n), dtype=np.uint8)
    tri[0, :n] = s

    # XNOR for bits: a XNOR b = 1 if a == b, else 0
    # For 0/1 bits, this can be computed as 1 - (a XOR b).
    for row in range(1, n):
        prev = tri[row - 1, : n - (row - 1)]
        xored = np.bitwise_xor(prev[:-1], prev[1:])
        tri[row, : n - row] = 1 - xored

    return tri


def xnor_triangle_numpy_flat(s: np.ndarray) -> np.ndarray:
    """
    Flatten the XNOR triangle row-by-row into a 1D vector of length T(n) = n(n+1)/2.

    Args:
        s: 1D binary array.

    Returns:
        flat: 1D uint8 array with all triangle entries.
    """
    tri = xnor_triangle_numpy(s)
    n = tri.shape[0]
    rows = [tri[i, : n - i] for i in range(n)]
    return np.concatenate(rows, axis=0)


def triangle_one_count_numpy(s: np.ndarray) -> int:
    """
    Count the number of ones in the XNOR triangle of s.
    """
    flat = xnor_triangle_numpy_flat(s)
    return int(flat.sum())


def is_balanced_numpy(s: np.ndarray) -> bool:
    """
    Check XNOR triangle balancedness of a seed s.

    Balancedness predicate:
        triangle has exactly T(n)/2 ones,
        where T(n) = n(n+1)/2 must be even.

    If T(n) is odd (e.g., n ≡ 1 or 2 mod 4), no balanced seeds exist,
    and this function returns False.
    """
    s = _check_binary_numpy(s)
    n = s.shape[0]
    T = n * (n + 1) // 2
    if T % 2 != 0:
        # Non-admissible n: balanced seeds do not exist
        return False
    return triangle_one_count_numpy(s) * 2 == T


# ---------------------------------------------------------------------------
# PyTorch implementation
# ---------------------------------------------------------------------------

def xnor_triangle_torch(s: torch.Tensor) -> torch.Tensor:
    """
    XNOR triangle in PyTorch.

    Args:
        s: 1D tensor of length n, binary {0,1}, any device.

    Returns:
        tri: uint8 tensor of shape (n, n) on same device as s.
    """
    s = _check_binary_torch(s)
    n = s.shape[0]
    device = s.device

    tri = torch.zeros((n, n), dtype=torch.uint8, device=device)
    tri[0, :n] = s

    for row in range(1, n):
        prev = tri[row - 1, : n - (row - 1)]
        xored = torch.bitwise_xor(prev[:-1], prev[1:])
        tri[row, : n - row] = 1 - xored

    return tri


def xnor_triangle_torch_flat(s: torch.Tensor) -> torch.Tensor:
    """
    Flatten the XNOR triangle into a 1D tensor of length T(n) = n(n+1)/2.
    """
    tri = xnor_triangle_torch(s)
    n = tri.shape[0]
    rows = [tri[i, : n - i] for i in range(n)]
    return torch.cat(rows, dim=0)


def triangle_one_count_torch(s: torch.Tensor) -> int:
    """
    Count the number of ones in the XNOR triangle of s (PyTorch).
    """
    flat = xnor_triangle_torch_flat(s)
    return int(flat.sum().item())


def is_balanced_torch(s: torch.Tensor) -> bool:
    """
    Balancedness check in PyTorch.

    See is_balanced_numpy for definition.
    """
    s = _check_binary_torch(s)
    n = s.shape[0]
    T = n * (n + 1) // 2
    if T % 2 != 0:
        return False
    return triangle_one_count_torch(s) * 2 == T

def _check_binary_torch_batch(s: torch.Tensor) -> torch.Tensor:
    """
    Ensure s is a 2D binary tensor with values in {0,1}.
    Shape: (B, n).
    Returns a uint8 tensor on the same device.
    """
    if s.dim() != 2:
        raise ValueError(f"Expected 2D tensor (batch, n), got shape {tuple(s.shape)}")

    if s.dtype == torch.bool:
        return s.to(torch.uint8)

    if s.is_floating_point():
        s_rounded = torch.round(s).to(torch.uint8)
        if not torch.all((s_rounded == 0) | (s_rounded == 1)):
            raise ValueError("Floating tensor must contain only ~0.0/~1.0 values")
        return s_rounded

    if s.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        s_uint = s.to(torch.uint8)
        if not torch.all((s_uint == 0) | (s_uint == 1)):
            raise ValueError("Integer tensor must contain only 0/1 values")
        return s_uint

    raise TypeError(f"Unsupported torch dtype for binary seed batch: {s.dtype}")


def xnor_triangle_torch_batch(s: torch.Tensor) -> torch.Tensor:
    """
    Batched XNOR triangle in PyTorch.

    Args:
        s: (B, n) tensor, binary {0,1}, any device.

    Returns:
        tri: uint8 tensor of shape (B, n, n) on same device as s.
             For each batch b, tri[b, 0, :n] is the seed,
             tri[b, row, :n-row] is the row-th triangle row.
    """
    s = _check_binary_torch_batch(s)
    B, n = s.shape
    device = s.device

    tri = torch.zeros((B, n, n), dtype=torch.uint8, device=device)
    tri[:, 0, :n] = s

    for row in range(1, n):
        prev = tri[:, row - 1, : n - (row - 1)]          # (B, n - row + 1)
        xored = torch.bitwise_xor(prev[:, :-1], prev[:, 1:])  # (B, n - row)
        tri[:, row, : n - row] = 1 - xored

    return tri


def xnor_triangle_torch_flat_batch(s: torch.Tensor) -> torch.Tensor:
    """
    Batched flattened XNOR triangle.

    Args:
        s: (B, n) binary {0,1}.

    Returns:
        flat: (B, T(n)) float32 tensor, where T(n) = n(n+1)/2.
    """
    tri = xnor_triangle_torch_batch(s)      # (B, n, n)
    B, n, _ = tri.shape

    rows = [tri[:, i, : n - i] for i in range(n)]  # list of (B, n-i)
    flat_uint8 = torch.cat(rows, dim=1)            # (B, T(n))
    return flat_uint8.float()
