import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from onn_sdk.core.dataset_builder import (
    XNORBalancedDatasetConfig,
    build_xnor_balanced_dataset,
)
from onn_sdk.onn.config import ONNConfig
from onn_sdk.onn.model import ONNClassifier


class SeedDataset(Dataset):
    """
    Simple torch Dataset wrapping (X, y) numpy arrays.

    X: (N, n) in {0,1}
    y: (N,) in {0,1}
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        assert X.ndim == 2
        assert y.ndim == 1
        assert X.shape[0] == y.shape[0]
        self.X = X.astype(np.uint8)
        self.y = y.astype(np.int64)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]          # shape (n,)
        y = self.y[idx]          # scalar
        x_t = torch.from_numpy(x).to(torch.float32)  # (n,)
        y_t = torch.tensor(y, dtype=torch.float32)   # scalar
        return x_t, y_t


def train_epoch(model, loader, optimizer, device):
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for x, y in loader:
        x = x.to(device)              # (B, n)
        y = y.to(device).unsqueeze(1) # (B, 1)

        optimizer.zero_grad()
        logits = model(x)             # (B, 1)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)

        with torch.no_grad():
            preds = (torch.sigmoid(logits) > 0.5).float()
            total_correct += (preds == y).sum().item()
            total_examples += x.size(0)

    avg_loss = total_loss / total_examples
    acc = total_correct / total_examples
    return avg_loss, acc


def eval_epoch(model, loader, device):
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)

            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item() * x.size(0)

            preds = (torch.sigmoid(logits) > 0.5).float()
            total_correct += (preds == y).sum().item()
            total_examples += x.size(0)

    avg_loss = total_loss / total_examples
    acc = total_correct / total_examples
    return avg_loss, acc


def main():
    # ----------------- 1. Build dataset -----------------
    n = 100   # XNOR length; T(n) even
    cfg_ds = XNORBalancedDatasetConfig(
        n=n,
        n_pos=1000,
        n_neg=1000,
        use_orbits=False,
        use_masks=True,
        extra_random_masks=0,
        shuffle=True,
        seed=123,
    )

    X, y = build_xnor_balanced_dataset(cfg_ds)
    print("Dataset:", X.shape, y.shape)

    # Simple train/val split
    N = X.shape[0]
    split = int(0.8 * N)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    train_ds = SeedDataset(X_train, y_train)
    val_ds = SeedDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    # ----------------- 2. Build model -----------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    cfg_onn = ONNConfig(
        n=n,
        hidden_dim=128,
        num_layers=2,
        dropout=0.1,
        use_orbits=True,          # L3 inside the model
        use_masks=False,          # L4 inside the model
        extra_random_masks=0,     # same as dataset builder for now
    )

    model = ONNClassifier(cfg_onn).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # ----------------- 3. Training loop -----------------
    epochs = 10
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = eval_epoch(model, val_loader, device)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.3f} | "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.3f}"
        )


if __name__ == "__main__":
    main()
