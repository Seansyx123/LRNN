"""
train.py

Training utilities for LRNN experiments.

This module provides:
- reproducible training loop
- optional rule injection
- simple evaluation (accuracy, precision, recall, f1)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 0.0
    seed: int = 42
    threshold: float = 0.5
    device: str = "cpu"


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _to_tensor(x: np.ndarray, y: np.ndarray, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    x_t = torch.tensor(x, dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.float32, device=device).view(-1, 1)
    return x_t, y_t


def train(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    rules: Optional[List] = None,
    config: Optional[TrainConfig] = None,
) -> nn.Module:
    """
    Train LRNN model.

    Args:
        model: LRNNModel instance (torch.nn.Module).
        X_train: training features (N, D) numpy array.
        y_train: training labels (N,) numpy array in {0,1}.
        rules: optional list of rule functions.
        config: training configuration.

    Returns:
        Trained model.
    """
    if config is None:
        config = TrainConfig()

    set_seed(config.seed)
    device = config.device
    model.to(device)
    model.train()

    X_t, y_t = _to_tensor(X_train, y_train, device)
    loader = DataLoader(
        TensorDataset(X_t, y_t),
        batch_size=config.batch_size,
        shuffle=True,
    )

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    for epoch in range(1, config.epochs + 1):
        epoch_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()

            # LRNNModel supports rules as an optional argument
            preds = model(xb, rules=rules) if rules is not None else model(xb, rules=None)

            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * xb.size(0)

        epoch_loss /= len(loader.dataset)

        # Keep logging minimal and clean (TR-friendly)
        if epoch == 1 or epoch % 10 == 0 or epoch == config.epochs:
            print(f"[Epoch {epoch:03d}/{config.epochs}] loss={epoch_loss:.6f}")

    return model


@torch.no_grad()
def predict_proba(
    model: nn.Module,
    X: np.ndarray,
    rules: Optional[List] = None,
    device: str = "cpu",
) -> np.ndarray:
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    probs = model(X_t, rules=rules) if rules is not None else model(X_t, rules=None)
    return probs.detach().cpu().numpy().reshape(-1)


def evaluate_binary(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute simple binary classification metrics.

    Returns:
        dict with accuracy, precision, recall, f1
    """
    y_pred = (y_proba >= threshold).astype(int)
    y_true = y_true.astype(int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    acc = (tp + tn) / max(1, (tp + tn + fp + fn))
    prec = tp / max(1, (tp + fp))
    rec = tp / max(1, (tp + fn))
    f1 = 2 * prec * rec / max(1e-12, (prec + rec))

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
    }
