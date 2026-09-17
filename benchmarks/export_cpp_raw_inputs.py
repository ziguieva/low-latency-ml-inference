from pathlib import Path
import struct

import numpy as np
import pandas as pd


INPUT_PATH = Path("data/processed/test.parquet")
OUTPUT_PATH = Path("benchmarks/cpp_raw_inputs.bin")

N_ROWS = 50_000

MAGIC = b"LLMLRAW1"

# 16 bytes persona
# 16 bytes remittance_category
# 6 float32:
# amount, hour_of_day, day_of_week,
# time_since_last_txn, is_weekend, is_new_beneficiary
RECORD_STRUCT = struct.Struct("<16s16s6f")


def encode_string(value: str) -> bytes:
    encoded = value.encode("utf-8")

    if len(encoded) >= 16:
        raise ValueError(
            f"String too long for fixed-width encoding: {value}"
        )

    return encoded


def main() -> None:
    df = pd.read_parquet(INPUT_PATH).head(N_ROWS)

    required_columns = [
        "persona",
        "remittance_category",
        "amount",
        "hour_of_day",
        "day_of_week",
        "time_since_last_txn",
        "is_weekend",
        "is_new_beneficiary",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open("wb") as file:
        file.write(MAGIC)
        file.write(
            struct.pack(
                "<QQ",
                len(df),
                RECORD_STRUCT.size,
            )
        )

        for row in df.itertuples(index=False):
            time_since_last_txn = (
                np.nan
                if pd.isna(row.time_since_last_txn)
                else float(row.time_since_last_txn)
            )

            file.write(
                RECORD_STRUCT.pack(
                    encode_string(row.persona),
                    encode_string(
                        row.remittance_category
                    ),
                    float(row.amount),
                    float(row.hour_of_day),
                    float(row.day_of_week),
                    time_since_last_txn,
                    float(row.is_weekend),
                    float(row.is_new_beneficiary),
                )
            )

    print(f"Rows: {len(df):,}")
    print(
        f"Record size: "
        f"{RECORD_STRUCT.size} bytes"
    )
    print(f"Output: {OUTPUT_PATH}")
    print(
        f"File size: "
        f"{OUTPUT_PATH.stat().st_size / 1024 / 1024:.2f} MiB"
    )


if __name__ == "__main__":
    main()