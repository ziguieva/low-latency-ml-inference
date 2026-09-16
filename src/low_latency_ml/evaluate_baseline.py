import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src.low_latency_ml.preprocessing import build_preprocessor


TARGET = "is_fraud"
THRESHOLD = 0.121


def load_split(name: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_parquet(f"data/processed/{name}.parquet")

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    return X, y


def evaluate_split(
    model: Pipeline,
    split_name: str,
) -> None:
    X, y = load_split(split_name)

    scores = model.predict_proba(X)[:, 1]
    predictions = (scores >= THRESHOLD).astype(int)

    pr_auc = average_precision_score(y, scores)
    roc_auc = roc_auc_score(y, scores)
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
    X_train, y_train = load_split("train")

    model = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    solver="lbfgs",
                ),
            ),
        ]
    )

    print("=== TRAINING BASELINE MODEL ===")
    print(f"Train rows: {len(X_train):,}")
    print(f"Decision threshold: {THRESHOLD:.3f}")

    model.fit(
        X_train,
        y_train,
    )

    evaluate_split(
        model,
        "test",
    )

    evaluate_split(
        model,
        "drift_test",
    )


if __name__ == "__main__":
    main()