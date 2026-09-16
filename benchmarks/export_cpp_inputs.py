from pathlib import Path
import struct

import joblib
import numpy as np
import pandas as pd


DATA_PATH = Path("data/processed/test.parquet")
PREPROCESSOR_PATH = Path("models/preprocessor.joblib")
OUTPUT_PATH = Path("benchmarks/cpp_inputs.bin")

TARGET = "is_fraud"
SAMPLE_SIZE = 50_000


def main() -> None:
    df = pd.read_parquet(DATA_PATH).head(SAMPLE_SIZE)

    X_raw = df.drop(columns=[TARGET])

    preprocessor = joblib.load(PREPROCESSOR_PATH)

    X = preprocessor.transform(X_raw).astype(np.float32)

    rows, cols = X.shape

    with OUTPUT_PATH.open("wb") as file:
        file.write(b"LLMLBEN1")
        file.write(struct.pack("<QQ", rows, cols))
        file.write(X.tobytes(order="C"))

    print("=== C++ BENCHMARK INPUTS ===")
    print(f"Rows:       {rows:,}")
    print(f"Features:   {cols}")
    print(f"Data type:  {X.dtype}")
    print(f"File:       {OUTPUT_PATH}")
    print(f"Size:       {OUTPUT_PATH.stat().st_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()