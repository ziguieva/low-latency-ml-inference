import torch

from src.low_latency_ml.model import FraudMLP


MODEL_PATH = "models/fraud_mlp.pt"
INPUT_DIM = 19
WARMUP_RUNS = 2_000


def main() -> None:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    model = FraudMLP(input_dim=INPUT_DIM)

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location="cpu",
            weights_only=True,
        )
    )

    model.eval()

    x = torch.zeros(
        (1, INPUT_DIM),
        dtype=torch.float32,
    )

    with torch.inference_mode():
        for _ in range(WARMUP_RUNS):
            model(x)

    print("PyTorch memory benchmark complete.")


if __name__ == "__main__":
    main()