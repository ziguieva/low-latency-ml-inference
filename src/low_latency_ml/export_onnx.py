from pathlib import Path

import onnx
import torch
from torch import nn

from src.low_latency_ml.model import FraudMLP


MODEL_PATH = Path("models/fraud_mlp.pt")
ONNX_PATH = Path("models/fraud_mlp.onnx")

INPUT_DIM = 19
OPSET_VERSION = 18


class FraudInferenceModel(nn.Module):
    """Inference wrapper returning a fraud probability."""

    def __init__(self, model: FraudMLP) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(x)
        return torch.sigmoid(logits)


def main() -> None:
    model = FraudMLP(input_dim=INPUT_DIM)

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location="cpu",
            weights_only=True,
        )
    )

    model.eval()

    inference_model = FraudInferenceModel(model)
    inference_model.eval()

    example_input = torch.zeros(
        1,
        INPUT_DIM,
        dtype=torch.float32,
    )

    torch.onnx.export(
        inference_model,
        example_input,
        ONNX_PATH,
        input_names=["features"],
        output_names=["fraud_probability"],
        opset_version=OPSET_VERSION,
        dynamo=False,
    )

    onnx_model = onnx.load(ONNX_PATH)
    onnx.checker.check_model(onnx_model)

    size_kb = ONNX_PATH.stat().st_size / 1024

    print("=== ONNX EXPORT ===")
    print(f"Input shape:  [1, {INPUT_DIM}]")
    print("Output shape: [1, 1]")
    print(f"Opset:        {OPSET_VERSION}")
    print(f"Model size:   {size_kb:.2f} KB")
    print(f"Saved to:     {ONNX_PATH}")
    print("ONNX check:   OK")


if __name__ == "__main__":
    main()