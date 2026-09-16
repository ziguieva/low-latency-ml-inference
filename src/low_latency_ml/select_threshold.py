import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline

from src.low_latency_ml.preprocessing import build_preprocessor


TARGET = "is_fraud"


def main() -> None:
    train = pd.read_parquet("data/processed/train.parquet")
    validation = pd.read_parquet("data/processed/validation.parquet")

    X_train = train.drop(columns=[TARGET])
    y_train = train[TARGET]

    X_val = validation.drop(columns=[TARGET])
    y_val = validation[TARGET]

    model = Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    solver="lbfgs",
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_val)[:, 1]

    thresholds = np.linspace(0.001, 0.999, 999)

    best_threshold = None
    best_f1 = -1.0

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        score = f1_score(y_val, predictions)

        if score > best_f1:
            best_f1 = score
            best_threshold = threshold

    predictions = (probabilities >= best_threshold).astype(int)

    precision = precision_score(y_val, predictions)
    recall = recall_score(y_val, predictions)

    print("=== VALIDATION THRESHOLD ===")
    print(f"Threshold: {best_threshold:.3f}")
    print(f"F1:        {best_f1:.6f}")
    print(f"Precision: {precision:.6f}")
    print(f"Recall:    {recall:.6f}")


if __name__ == "__main__":
    main()