from pathlib import Path

import pandas as pd


RAW_DATA_PATH = Path("data/raw/synsep_full_dataset.csv")
OUTPUT_DIR = Path("data/processed")

FEATURES = [
    "persona",
    "beneficiary_country",
    "country_type",
    "amount",
    "remittance_category",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "time_since_last_txn",
    "is_new_beneficiary",
]

TARGET = "is_fraud"


def prepare_data() -> None:
    columns = ["timestamp", *FEATURES, TARGET]

    df = pd.read_csv(
        RAW_DATA_PATH,
        usecols=columns,
        parse_dates=["timestamp"],
    )

    splits = {
        "train": df[
            (df["timestamp"] >= "2024-01-01")
            & (df["timestamp"] < "2024-07-01")
        ],
        "validation": df[
            (df["timestamp"] >= "2024-07-01")
            & (df["timestamp"] < "2024-09-01")
        ],
        "test": df[
            (df["timestamp"] >= "2024-09-01")
            & (df["timestamp"] < "2024-12-01")
        ],
        "drift_test": df[
            (df["timestamp"] >= "2024-12-01")
            & (df["timestamp"] < "2025-01-01")
        ],
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for name, split in splits.items():
        output = split[FEATURES + [TARGET]].copy()

        output.to_parquet(
            OUTPUT_DIR / f"{name}.parquet",
            index=False,
        )

        print(
            f"{name:12s} | "
            f"rows={len(output):,} | "
            f"frauds={output[TARGET].sum():,} | "
            f"fraud_rate={output[TARGET].mean() * 100:.4f}%"
        )


if __name__ == "__main__":
    prepare_data()