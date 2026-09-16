from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/raw/synsep_full_dataset.csv")
CHUNK_SIZE = 200_000


def audit_dataset(path: Path) -> None:
    total_rows = 0
    fraud_count = 0
    missing_values = None
    min_timestamp = None
    max_timestamp = None

    for chunk in pd.read_csv(path, chunksize=CHUNK_SIZE):
        total_rows += len(chunk)
        fraud_count += int(chunk["is_fraud"].sum())

        chunk_missing = chunk.isna().sum()

        if missing_values is None:
            missing_values = chunk_missing
        else:
            missing_values = missing_values.add(chunk_missing, fill_value=0)

        timestamps = pd.to_datetime(chunk["timestamp"])

        chunk_min = timestamps.min()
        chunk_max = timestamps.max()

        min_timestamp = chunk_min if min_timestamp is None else min(min_timestamp, chunk_min)
        max_timestamp = chunk_max if max_timestamp is None else max(max_timestamp, chunk_max)

    fraud_rate = fraud_count / total_rows * 100

    sample = pd.read_csv(path, nrows=10_000)

    print("\n=== DATASET AUDIT ===")
    print(f"Rows: {total_rows:,}")
    print(f"Columns: {len(sample.columns)}")
    print(f"Period: {min_timestamp} -> {max_timestamp}")

    print("\n=== TARGET ===")
    print(f"Fraud cases: {fraud_count:,}")
    print(f"Fraud rate: {fraud_rate:.4f}%")

    print("\n=== DATA TYPES ===")
    print(sample.dtypes)

    print("\n=== MISSING VALUES ===")
    print(missing_values[missing_values > 0].sort_values(ascending=False))

    print("\n=== TARGET VALUES ===")
    print(sample["is_fraud"].unique())


if __name__ == "__main__":
    audit_dataset(DATA_PATH)