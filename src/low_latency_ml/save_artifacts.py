import json
from pathlib import Path

import joblib
import pandas as pd

from src.low_latency_ml.preprocessing import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    SCALED_NUMERIC_FEATURES,
    build_preprocessor,
)


DATA_PATH = Path("data/processed/train.parquet")
MODEL_DIR = Path("models")

PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.joblib"
CONTRACT_PATH = MODEL_DIR / "inference_contract.json"
MODEL_PATH = MODEL_DIR / "fraud_mlp.pt"

TARGET = "is_fraud"
THRESHOLD = 0.986


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    train = pd.read_parquet(DATA_PATH)

    X_train = train.drop(columns=[TARGET])

    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        preprocessor,
        PREPROCESSOR_PATH,
    )

    categorical_pipeline = (
        preprocessor
        .named_transformers_["categorical"]
    )

    encoder = categorical_pipeline.named_steps["encoder"]

    numeric_pipeline = (
        preprocessor
        .named_transformers_["numeric"]
    )

    scaler = numeric_pipeline.named_steps["scaler"]

    categories = {
        feature: encoder.categories_[index].tolist()
        for index, feature in enumerate(
            CATEGORICAL_FEATURES
        )
    }

    scaler_mean = {
        feature: float(scaler.mean_[index])
        for index, feature in enumerate(
            SCALED_NUMERIC_FEATURES
        )
    }

    scaler_scale = {
        feature: float(scaler.scale_[index])
        for index, feature in enumerate(
            SCALED_NUMERIC_FEATURES
        )
    }

    processed_features = (
        preprocessor
        .get_feature_names_out()
        .tolist()
    )

    contract = {
        "model": {
            "type": "FraudMLP",
            "input_dimension": len(processed_features),
            "threshold": THRESHOLD,
        },
        "raw_features": {
            "categorical": CATEGORICAL_FEATURES,
            "scaled_numeric": SCALED_NUMERIC_FEATURES,
            "binary": BINARY_FEATURES,
        },
        "preprocessing": {
            "missing_value": {
                "time_since_last_txn": -1.0,
            },
            "categories": categories,
            "scaler_mean": scaler_mean,
            "scaler_scale": scaler_scale,
        },
        "processed_features": processed_features,
    }

    with open(
        CONTRACT_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            contract,
            file,
            indent=2,
        )

    print("=== INFERENCE ARTIFACTS ===")
    print(f"Raw features:       {X_train.shape[1]}")
    print(f"Processed features: {len(processed_features)}")
    print(f"Decision threshold: {THRESHOLD:.3f}")
    print(f"Preprocessor:       {PREPROCESSOR_PATH}")
    print(f"Contract:           {CONTRACT_PATH}")
    print(f"Model:              {MODEL_PATH}")


if __name__ == "__main__":
    main()