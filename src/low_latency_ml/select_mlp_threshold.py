from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
)

from src.low_latency_ml.model import FraudMLP
from src.low_latency_ml.preprocessing import build_preprocessor


DATA_DIR = Path("data/processed")
MODEL_PATH = Path("models/fraud_mlp.pt")

TARGET = "is_fraud"


def load_split(name: str):
    df = pd.read_parquet(DATA_DIR / f"{name}.parquet")

    X = df.drop(columns=[TARGET])
    y = df[TARGET].to_numpy(dtype=np.float32)

    return X, y


def main() -> None:
    X_train, _ = load_split("train")
    X_val, y_val = load_split("validation")

    # Fit preprocessing only on the training set
    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)

    X_val_processed = preprocessor.transform(X_val).astype(np.float32)

    # Load trained MLP
    model = FraudMLP(
        input_dim=X_val_processed.shape[1]
    )

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location="cpu",
            weights_only=True,
        )
    )

    model.eval()

    X_val_tensor = torch.from_numpy(X_val_processed)

    with torch.no_grad():
        logits = model(X_val_tensor)

        probabilities = torch.sigmoid(
            logits
        ).squeeze(1).numpy()

    thresholds = np.linspace(
        0.001,
        0.999,
        999,
    )

    best_threshold = None
    best_f1 = -1.0

    for threshold in thresholds:
        predictions = (
            probabilities >= threshold
        ).astype(int)

        score = f1_score(
            y_val,
            predictions,
        )

        if score > best_f1:
            best_f1 = score
            best_threshold = threshold

    predictions = (
        probabilities >= best_threshold
    ).astype(int)

    precision = precision_score(
        y_val,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_val,
        predictions,
        zero_division=0,
    )

    print("=== MLP VALIDATION THRESHOLD ===")
    print(f"Threshold: {best_threshold:.3f}")
    print(f"F1:        {best_f1:.6f}")
    print(f"Precision: {precision:.6f}")
    print(f"Recall:    {recall:.6f}")


if __name__ == "__main__":
    main()