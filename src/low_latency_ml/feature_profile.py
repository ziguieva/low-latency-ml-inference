from pathlib import Path

import pandas as pd


DATA_DIR = Path("data/processed")

SPLITS = [
    "train",
    "validation",
    "test",
    "drift_test",
]

CATEGORICAL_FEATURES = [
    "persona",
    "beneficiary_country",
    "country_type",
    "remittance_category",
]

NUMERIC_FEATURES = [
    "amount",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "time_since_last_txn",
    "is_new_beneficiary",
]


def main() -> None:
    datasets = {
        name: pd.read_parquet(DATA_DIR / f"{name}.parquet")
        for name in SPLITS
    }

    train = datasets["train"]

    print("=== CATEGORICAL FEATURES ===")

    for column in CATEGORICAL_FEATURES:
        train_values = set(train[column].dropna().unique())

        print(f"\n{column}")
        print(f"Train categories: {len(train_values)}")

        for split_name in SPLITS[1:]:
            split_values = set(
                datasets[split_name][column].dropna().unique()
            )

            unseen = split_values - train_values

            print(
                f"{split_name:12s} | "
                f"categories={len(split_values):2d} | "
                f"unseen={sorted(unseen)}"
            )

    print("\n=== MISSING VALUES ===")

    for split_name, df in datasets.items():
        missing = df[CATEGORICAL_FEATURES + NUMERIC_FEATURES].isna().sum()
        missing = missing[missing > 0]

        print(f"\n{split_name}:")
        print(missing if not missing.empty else "None")


if __name__ == "__main__":
    main()