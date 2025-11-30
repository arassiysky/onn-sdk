import numpy as np
import torch

from onn_sdk.operators.l1_triangle import (
    xnor_triangle_numpy,
    triangle_one_count_numpy,
    is_balanced_numpy,
    xnor_triangle_torch,
    triangle_one_count_torch,
    is_balanced_torch,
)


def main():
    # Simple seed of length 7 (just an example)
    s_np = np.array([0, 1, 0, 1, 1, 0, 1], dtype=np.uint8)
    print("Seed (NumPy):", s_np)

    tri_np = xnor_triangle_numpy(s_np)
    print("XNOR triangle (NumPy):")
    print(tri_np)

    ones_np = triangle_one_count_numpy(s_np)
    print("NumPy triangle ones:", ones_np)
    print("NumPy balanced?:", is_balanced_numpy(s_np))

    # Torch version
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device:", device)

    s_t = torch.from_numpy(s_np).to(device)
    tri_t = xnor_triangle_torch(s_t)
    print("XNOR triangle (Torch, on device):")
    print(tri_t.cpu().numpy())

    ones_t = triangle_one_count_torch(s_t)
    print("Torch triangle ones:", ones_t)
    print("Torch balanced?:", is_balanced_torch(s_t))


if __name__ == "__main__":
    main()

