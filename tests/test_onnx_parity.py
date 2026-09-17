import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn

from low_latency_ml.model import FraudMLP


INPUT_DIM = 19
N_SAMPLES = 100
TOLERANCE = 1e-5


class FraudInferenceModel(nn.Module):
    def __init__(self, model: FraudMLP) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(x))


class TestOnnxParity(unittest.TestCase):
    def test_pytorch_onnx_numerical_parity(self) -> None:
        torch.manual_seed(42)
        np.random.seed(42)

        model = FraudMLP(input_dim=INPUT_DIM)
        model.eval()

        inference_model = FraudInferenceModel(model)
        inference_model.eval()

        samples = np.random.normal(
            size=(N_SAMPLES, INPUT_DIM)
        ).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir) / "fraud_mlp_test.onnx"

            dummy_input = torch.zeros(
                1,
                INPUT_DIM,
                dtype=torch.float32,
            )

            torch.onnx.export(
                inference_model,
                dummy_input,
                model_path,
                input_names=["features"],
                output_names=["fraud_probability"],
                opset_version=18,
                dynamo=False,
            )

            exported_model = onnx.load(model_path)
            onnx.checker.check_model(exported_model)

            session_options = ort.SessionOptions()
            session_options.intra_op_num_threads = 1
            session_options.inter_op_num_threads = 1

            session = ort.InferenceSession(
                str(model_path),
                sess_options=session_options,
                providers=["CPUExecutionProvider"],
            )

            max_error = 0.0

            with torch.no_grad():
                for sample in samples:
                    x = sample.reshape(1, INPUT_DIM)

                    torch_probability = (
                        inference_model(
                            torch.from_numpy(x)
                        )
                        .numpy()
                        .item()
                    )

                    onnx_probability = session.run(
                        ["fraud_probability"],
                        {"features": x},
                    )[0].item()

                    error = abs(
                        torch_probability
                        - onnx_probability
                    )

                    max_error = max(
                        max_error,
                        error,
                    )

            self.assertLessEqual(
                max_error,
                TOLERANCE,
                msg=(
                    "PyTorch/ONNX parity failure: "
                    f"max error={max_error:.10f}, "
                    f"tolerance={TOLERANCE}"
                ),
            )


if __name__ == "__main__":
    unittest.main()