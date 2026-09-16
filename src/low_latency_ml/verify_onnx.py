from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd
import torch

from src.low_latency_ml.model import FraudMLP


MODEL_DIR = Path("models")

PYTORCH_MODEL_PATH = MODEL_DIR / "fraud_mlp.pt"
ONNX_MODEL_PATH = MODEL_DIR / "fraud_mlp.onnx"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.joblib"

TEST_DATA_PATH = Path("data/processed/test.parquet")

TARGET = "is_fraud"
INPUT_DIM = 19
SAMPLE_SIZE = 1000
TOLERANCE = 1e-5


def main() -> None:
    # Load test sample
    df = pd.read_parquet(TEST_DATA_PATH).head(SAMPLE_SIZE)

    X_raw = df.drop(columns=[TARGET])

    # Load fitted preprocessor
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    X = preprocessor.transform(X_raw).astype(np.float32)

    # PyTorch model
    pytorch_model = FraudMLP(input_dim=INPUT_DIM)

    pytorch_model.load_state_dict(
        torch.load(
            PYTORCH_MODEL_PATH,
            map_location="cpu",
            weights_only=True,
        )
    )

    pytorch_model.eval()

    with torch.no_grad():
        logits = pytorch_model(
            torch.from_numpy(X)
        )

        pytorch_probabilities = torch.sigmoid(
            logits
        ).squeeze(1).numpy()

    # ONNX Runtime
    session = ort.InferenceSession(
        str(ONNX_MODEL_PATH),
        providers=["CPUExecutionProvider"],
    )

    onnx_outputs = []

    for row in X:
        input_data = row.reshape(1, -1)

        output = session.run(
            ["fraud_probability"],
            {"features": input_data},
        )[0]

        onnx_outputs.append(
            float(output[0][0])
        )

    onnx_probabilities = np.array(
        onnx_outputs,
        dtype=np.float32,
    )

    # Numerical comparison
    absolute_error = np.abs(
        pytorch_probabilities
        - onnx_probabilities
    )

    max_error = absolute_error.max()
    mean_error = absolute_error.mean()

    print("=== PYTORCH vs ONNX PARITY ===")
    print(f"Samples:          {SAMPLE_SIZE}")
    print(f"Maximum error:    {max_error:.10f}")
    print(f"Mean error:       {mean_error:.10f}")
    print(f"Tolerance:        {TOLERANCE:.10f}")
    print(
        "Parity check:    "
        + ("PASS" if max_error < TOLERANCE else "FAIL")
    )


if __name__ == "__main__":
    main()