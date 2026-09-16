import numpy as np
import onnxruntime as ort


MODEL_PATH = "models/fraud_mlp.onnx"
INPUT_DIM = 19
WARMUP_RUNS = 2_000


def main() -> None:
    options = ort.SessionOptions()

    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    session = ort.InferenceSession(
        MODEL_PATH,
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )

    x = np.zeros(
        (1, INPUT_DIM),
        dtype=np.float32,
    )

    for _ in range(WARMUP_RUNS):
        session.run(
            None,
            {"features": x},
        )

    print("ONNX Runtime Python memory benchmark complete.")


if __name__ == "__main__":
    main()