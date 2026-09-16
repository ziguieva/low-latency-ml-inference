from copy import deepcopy
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.low_latency_ml.model import FraudMLP
from src.low_latency_ml.preprocessing import build_preprocessor


DATA_DIR = Path("data/processed")
MODEL_PATH = Path("models/fraud_mlp.pt")

TARGET = "is_fraud"

SEED = 42
BATCH_SIZE = 4096
LEARNING_RATE = 0.001
MAX_EPOCHS = 20
PATIENCE = 3


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_split(name: str):
    df = pd.read_parquet(DATA_DIR / f"{name}.parquet")

    X = df.drop(columns=[TARGET])
    y = df[TARGET].to_numpy(dtype=np.float32)

    return X, y


def main() -> None:
    set_seed(SEED)

    print("=== LOADING DATA ===")

    X_train_raw, y_train = load_split("train")
    X_val_raw, y_val = load_split("validation")

    print(f"Train rows:      {len(X_train_raw):,}")
    print(f"Validation rows: {len(X_val_raw):,}")

    print("\n=== PREPROCESSING ===")

    preprocessor = build_preprocessor()

    X_train = preprocessor.fit_transform(X_train_raw)
    X_val = preprocessor.transform(X_val_raw)

    X_train = X_train.astype(np.float32)
    X_val = X_val.astype(np.float32)

    print(f"Input dimension: {X_train.shape[1]}")

    train_dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train).unsqueeze(1),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    model = FraudMLP(input_dim=X_train.shape[1])

    fraud_count = y_train.sum()
    legitimate_count = len(y_train) - fraud_count

    pos_weight_value = legitimate_count / fraud_count

    pos_weight = torch.tensor(
        [pos_weight_value],
        dtype=torch.float32,
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    print(f"Positive class weight: {pos_weight_value:.2f}")

    X_val_tensor = torch.from_numpy(X_val)

    best_pr_auc = -1.0
    best_state = None
    epochs_without_improvement = 0

    print("\n=== TRAINING ===")

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()

        total_loss = 0.0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()

            logits = model(batch_X)

            loss = criterion(
                logits,
                batch_y,
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch_X)

        train_loss = total_loss / len(train_dataset)

        model.eval()

        with torch.no_grad():
            val_logits = model(X_val_tensor)

            val_probabilities = torch.sigmoid(
                val_logits
            ).squeeze(1).numpy()

        val_pr_auc = average_precision_score(
            y_val,
            val_probabilities,
        )

        print(
            f"Epoch {epoch:02d} | "
            f"Loss={train_loss:.6f} | "
            f"Val PR-AUC={val_pr_auc:.6f}"
        )

        if val_pr_auc > best_pr_auc:
            best_pr_auc = val_pr_auc
            best_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print("\nEarly stopping triggered.")
            break

    if best_state is None:
        raise RuntimeError("No valid model state was produced.")

    model.load_state_dict(best_state)

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        model.state_dict(),
        MODEL_PATH,
    )

    print("\n=== TRAINING COMPLETE ===")
    print(f"Best validation PR-AUC: {best_pr_auc:.6f}")
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()