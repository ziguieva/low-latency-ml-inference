# Main Model Architecture

## Model

Compact Multi-Layer Perceptron (MLP).

## Architecture

Input: 19 features

- Linear: 19 -> 32
- ReLU
- Linear: 32 -> 16
- ReLU
- Linear: 16 -> 1

## Output

The model outputs a fraud score.

A sigmoid function converts the output into a fraud probability.

## Objective

The model is intentionally compact to support:

- low-latency CPU inference
- ONNX export
- Python inference
- C++ inference
- latency benchmarking
- throughput benchmarking