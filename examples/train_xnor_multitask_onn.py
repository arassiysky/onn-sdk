import torch
from torch.utils.data import TensorDataset, DataLoader

from onn_sdk.core.dataset_builder import (
    XNORBalancedDatasetConfig,
    build_xnor_balanced_multitask_dataset,
)
from onn_sdk.onn.config import ONNConfig
from onn_sdk.onn.model import ONNMultiTaskModel


def train_xnor_multitask_onn(
    model,
    X_train,
    y_train_bal,
    y_train_orbit,
    X_val,
    y_val_bal,
    y_val_orbit,
    *,
    batch_size: int = 128,
    num_epochs: int = 20,
    lr: float = 1e-3,
    device: str = "cuda",
    lambda_bal: float = 1.0,
    lambda_orbit: float = 0.1,
):
    """
    Multi-task training loop for ONNMultiTaskModel on XNOR + orbit-size.

    Args:
        model: ONNMultiTaskModel instance.
        X_*: numpy arrays or torch tensors, shape (N, n)
        y_*_bal: labels in {0,1}, shape (N,)
        y_*_orbit: orbit-size classes in {0,1,2,3}, shape (N,)
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Convert to tensors if needed
    def to_tensor(x):
        return x if isinstance(x, torch.Tensor) else torch.from_numpy(x)

    X_train = to_tensor(X_train).float()
    y_train_bal = to_tensor(y_train_bal).long()
    y_train_orbit = to_tensor(y_train_orbit).long()

    X_val = to_tensor(X_val).float()
    y_val_bal = to_tensor(y_val_bal).long()
    y_val_orbit = to_tensor(y_val_orbit).long()

    # Datasets & loaders
    train_ds = TensorDataset(X_train, y_train_bal, y_train_orbit)
    val_ds = TensorDataset(X_val, y_val_bal, y_val_orbit)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    bce = torch.nn.BCEWithLogitsLoss()
    ce = torch.nn.CrossEntropyLoss()

    print(f"Dataset (train): {X_train.shape} {y_train_bal.shape}")
    print(f"Dataset (val):   {X_val.shape} {y_val_bal.shape}")
    print(f"Using device: {device.type}")

    for epoch in range(1, num_epochs + 1):
        # ===== Train =====
        model.train()
        total_loss = 0.0
        total_bal_loss = 0.0
        total_orbit_loss = 0.0
        total = 0

        correct_bal = 0
        correct_orbit = 0

        for xb, yb_bal, yb_orbit in train_loader:
            xb = xb.to(device)
            yb_bal = yb_bal.to(device)
            yb_orbit = yb_orbit.to(device)

            opt.zero_grad()
            logits_bal, logits_orbit = model(xb)   # (B,1), (B, C)

            # Losses
            bal_loss = bce(logits_bal.squeeze(-1), yb_bal.float())
            orbit_loss = ce(logits_orbit, yb_orbit)
            loss = lambda_bal * bal_loss + lambda_orbit * orbit_loss

            loss.backward()
            opt.step()

            batch_size_actual = xb.size(0)
            total += batch_size_actual

            total_loss += loss.item() * batch_size_actual
            total_bal_loss += bal_loss.item() * batch_size_actual
            total_orbit_loss += orbit_loss.item() * batch_size_actual

            with torch.no_grad():
                # Balancedness accuracy
                preds_bal = (torch.sigmoid(logits_bal.squeeze(-1)) > 0.5).long()
                correct_bal += (preds_bal == yb_bal).sum().item()

                # Orbit-size accuracy
                preds_orbit = torch.argmax(logits_orbit, dim=-1)
                correct_orbit += (preds_orbit == yb_orbit).sum().item()

        train_loss = total_loss / total
        train_bal_loss = total_bal_loss / total
        train_orbit_loss = total_orbit_loss / total
        train_bal_acc = correct_bal / total
        train_orbit_acc = correct_orbit / total

        # ===== Validation =====
        model.eval()
        val_total_loss = 0.0
        val_total_bal_loss = 0.0
        val_total_orbit_loss = 0.0
        val_total = 0

        val_correct_bal = 0
        val_correct_orbit = 0

        with torch.no_grad():
            for xb, yb_bal, yb_orbit in val_loader:
                xb = xb.to(device)
                yb_bal = yb_bal.to(device)
                yb_orbit = yb_orbit.to(device)

                logits_bal, logits_orbit = model(xb)

                bal_loss = bce(logits_bal.squeeze(-1), yb_bal.float())
                orbit_loss = ce(logits_orbit, yb_orbit)
                loss = lambda_bal * bal_loss + lambda_orbit * orbit_loss

                batch_size_actual = xb.size(0)
                val_total += batch_size_actual

                val_total_loss += loss.item() * batch_size_actual
                val_total_bal_loss += bal_loss.item() * batch_size_actual
                val_total_orbit_loss += orbit_loss.item() * batch_size_actual

                # Accuracies
                preds_bal = (torch.sigmoid(logits_bal.squeeze(-1)) > 0.5).long()
                val_correct_bal += (preds_bal == yb_bal).sum().item()

                preds_orbit = torch.argmax(logits_orbit, dim=-1)
                val_correct_orbit += (preds_orbit == yb_orbit).sum().item()

        val_loss = val_total_loss / val_total
        val_bal_loss = val_total_bal_loss / val_total
        val_orbit_loss = val_total_orbit_loss / val_total
        val_bal_acc = val_correct_bal / val_total
        val_orbit_acc = val_correct_orbit / val_total

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f}, "
            f"train_bal_loss={train_bal_loss:.4f}, train_bal_acc={train_bal_acc:.3f}, "
            f"train_orbit_loss={train_orbit_loss:.4f}, train_orbit_acc={train_orbit_acc:.3f} | "
            f"val_loss={val_loss:.4f}, "
            f"val_bal_loss={val_bal_loss:.4f}, val_bal_acc={val_bal_acc:.3f}, "
            f"val_orbit_loss={val_orbit_loss:.4f}, val_orbit_acc={val_orbit_acc:.3f}"
        )
    # end for epoch

if __name__ == "__main__":
    # 1) Build dataset (multi-task)
    cfg_data = XNORBalancedDatasetConfig(
        n=15,
        n_pos=6000,
        n_neg=6000,
        use_orbits=False,
        use_masks=True,
        extra_random_masks=0,
        shuffle=True,
        seed=123,
    )

    X, y_bal, y_orbit = build_xnor_balanced_multitask_dataset(cfg_data)

    # Simple train/val split (e.g. 80/20)
    N = X.shape[0]
    split = int(0.8 * N)
    X_train, X_val = X[:split], X[split:]
    y_train_bal, y_val_bal = y_bal[:split], y_bal[split:]
    y_train_orbit, y_val_orbit = y_orbit[:split], y_orbit[split:]

    # 2) Build model config
    cfg_model = ONNConfig(
        n=15,
        hidden_dim=64,
        num_layers=1,
        dropout=0.0,
        use_orbits=False,
        use_masks=True,
        extra_random_masks=0,
        multi_task=True,
        num_orbit_classes=4,
        lambda_bal=1.0,
        lambda_orbit=0.1,
    )

    model = ONNMultiTaskModel(cfg_model)

    # 3) Train with nice prints
    train_xnor_multitask_onn(
        model,
        X_train,
        y_train_bal,
        y_train_orbit,
        X_val,
        y_val_bal,
        y_val_orbit,
        batch_size=128,
        num_epochs=20,
        lr=1e-3,
        device="cuda",
        lambda_bal=cfg_model.lambda_bal,
        lambda_orbit=cfg_model.lambda_orbit,
    )
