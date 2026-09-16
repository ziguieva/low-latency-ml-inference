from pathlib import Path
from time import perf_counter_ns

import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd
import torch

from src.low_latency_ml.model import FraudMLP


MODEL_DIR = Path("models")
DATA_PATH = Path("data/processed/test.parquet")

PYTORCH_MODEL_PATH = MODEL_DIR / "fraud_mlp.pt"
ONNX_MODEL_PATH = MODEL_DIR / "fraud_mlp.onnx"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.joblib"

TARGET = "is_fraud"

INPUT_DIM = 19
WARMUP_RUNS = 2_000
BENCHMARK_RUNS = 50_000


def summarize(name: str, latencies_ns: np.ndarray) -> None:
    latencies_us = latencies_ns / 1_000.0

    mean = np.mean(latencies_us)
    p50 = np.percentile(latencies_us, 50)
    p95 = np.percentile(latencies_us, 95)
    p99 = np.percentile(latencies_us, 99)

    total_seconds = latencies_ns.sum() / 1_000_000_000
    throughput = len(latencies_ns) / total_seconds

    print(f"\n=== {name} ===")
    print(f"Mean:       {mean:.3f} us")
    print(f"P50:        {p50:.3f} us")
    print(f"P95:        {p95:.3f} us")
    print(f"P99:        {p99:.3f} us")
    print(f"Throughput: {throughput:,.0f} inference/s")


def main() -> None:
    # Single-thread configuration
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    # Load and preprocess inputs before benchmarking
    df = pd.read_parquet(DATA_PATH).head(BENCHMARK_RUNS)
    X_raw = df.drop(columns=[TARGET])

    preprocessor = joblib.load(PREPROCESSOR_PATH)

    X = preprocessor.transform(X_raw).astype(np.float32)

    print("=== BENCHMARK CONFIGURATION ===")
    print(f"Samples:     {len(X):,}")
    print(f"Input dim:   {X.shape[1]}")
    print(f"Warm-up:     {WARMUP_RUNS:,}")
    print(f"Threads:     1")
    print(f"Batch size:  1")

    # -------------------------
    # PyTorch
    # -------------------------

    pytorch_model = FraudMLP(input_dim=INPUT_DIM)

    pytorch_model.load_state_dict(
        torch.load(
            PYTORCH_MODEL_PATH,
            map_location="cpu",
            weights_only=True,
        )
    )

    pytorch_model.eval()

    torch_inputs = [
        torch.from_numpy(row.reshape(1, -1))
        for row in X
    ]

    with torch.inference_mode():
        for i in range(WARMUP_RUNS):
            pytorch_model(torch_inputs[i % len(torch_inputs)])

        pytorch_latencies = np.empty(
            BENCHMARK_RUNS,
            dtype=np.int64,
        )

        for i in range(BENCHMARK_RUNS):
            x = torch_inputs[i]

            start = perf_counter_ns()

            pytorch_model(x)

            end = perf_counter_ns()

            pytorch_latencies[i] = end - start

    summarize(
        "PYTORCH CPU",
        pytorch_latencies,
    )

    # -------------------------
    # ONNX Runtime
    # -------------------------

    options = ort.SessionOptions()

    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    session = ort.InferenceSession(
        str(ONNX_MODEL_PATH),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )

    onnx_inputs = [
        row.reshape(1, -1)
        for row in X
    ]

    for i in range(WARMUP_RUNS):
        session.run(
            None,
            {"features": onnx_inputs[i % len(onnx_inputs)]},
        )

    onnx_latencies = np.empty(
        BENCHMARK_RUNS,
        dtype=np.int64,
    )

    for i in range(BENCHMARK_RUNS):
        x = onnx_inputs[i]

        start = perf_counter_ns()

        session.run(
            None,
            {"features": x},
        )

        end = perf_counter_ns()

        onnx_latencies[i] = end - start

    summarize(
        "ONNX RUNTIME CPU",
        onnx_latencies,
    )


if __name__ == "__main__":
    main()