from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.low_latency_ml.model import FraudMLP
from src.low_latency_ml.preprocessing import build_preprocessor


DATA_DIR = Path("data/processed")
MODEL_PATH = Path("models/fraud_mlp.pt")

TARGET = "is_fraud"
THRESHOLD = 0.986


def load_split(name: str):
    df = pd.read_parquet(DATA_DIR / f"{name}.parquet")

    X = df.drop(columns=[TARGET])
    y = df[TARGET].to_numpy(dtype=np.float32)

    return X, y


def evaluate_split(
    model: FraudMLP,
    preprocessor,
    split_name: str,
) -> None:
    X, y = load_split(split_name)

    X_processed = preprocessor.transform(X).astype(np.float32)
    X_tensor = torch.from_numpy(X_processed)

    model.eval()

    with torch.no_grad():
        logits = model(X_tensor)

        probabilities = torch.sigmoid(
            logits
        ).squeeze(1).numpy()

    predictions = (
        probabilities >= THRESHOLD
    ).astype(int)

    pr_auc = average_precision_score(
        y,
        probabilities,
    )

    roc_auc = roc_auc_score(
        y,
        probabilities,
    )

    precision = precision_score(
        y,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y,
        predictions,
        zero_division=0,
    )

    tn, fp, fn, tp = confusion_matrix(
        y,
        predictions,
    ).ravel()

    print(f"\n=== {split_name.upper()} ===")
    print(f"PR-AUC:    {pr_auc:.6f}")
    print(f"ROC-AUC:   {roc_auc:.6f}")
    print(f"Precision: {precision:.6f}")
    print(f"Recall:    {recall:.6f}")
    print(f"F1:        {f1:.6f}")

    print("\nConfusion matrix:")
    print(f"TN:        {tn:,}")
    print(f"FP:        {fp:,}")
    print(f"FN:        {fn:,}")
    print(f"TP:        {tp:,}")


def main() -> None:
    print("=== LOADING TRAINING PREPROCESSOR ===")

    X_train, _ = load_split("train")

    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)

    input_dim = len(
        preprocessor.get_feature_names_out()
    )

    print(f"Input dimension: {input_dim}")

    print("\n=== LOADING MLP ===")

    model = FraudMLP(
        input_dim=input_dim
    )

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location="cpu",
            weights_only=True,
        )
    )

    print(f"Decision threshold: {THRESHOLD:.3f}")

    evaluate_split(
        model,
        preprocessor,
        "test",
    )

    evaluate_split(
        model,
        preprocessor,
        "drift_test",
    )


if __name__ == "__main__":
    main()