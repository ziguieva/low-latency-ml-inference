from pathlib import Path
from time import perf_counter

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline

from src.low_latency_ml.preprocessing import build_preprocessor


DATA_DIR = Path("data/processed")

TARGET = "is_fraud"


def load_split(name: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_parquet(DATA_DIR / f"{name}.parquet")

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    return X, y


def main() -> None:
    X_train, y_train = load_split("train")
    X_validation, y_validation = load_split("validation")

    pipeline = Pipeline(
        steps=[
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

    print("=== BASELINE TRAINING ===")
    print(f"Train rows: {len(X_train):,}")
    print(f"Validation rows: {len(X_validation):,}")

    start = perf_counter()

    pipeline.fit(X_train, y_train)

    training_time = perf_counter() - start

    validation_scores = pipeline.predict_proba(X_validation)[:, 1]

    pr_auc = average_precision_score(
        y_validation,
        validation_scores,
    )

    roc_auc = roc_auc_score(
        y_validation,
        validation_scores,
    )

    print("\n=== VALIDATION RESULTS ===")
    print(f"PR-AUC: {pr_auc:.6f}")
    print(f"ROC-AUC: {roc_auc:.6f}")
    print(f"Training time: {training_time:.2f} s")


if __name__ == "__main__":
    main()