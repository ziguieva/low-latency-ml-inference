# Low-Latency ML Inference

[![CI](https://github.com/ziguieva/low-latency-ml-inference/actions/workflows/ci.yml/badge.svg)](https://github.com/ziguieva/low-latency-ml-inference/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![C++](https://img.shields.io/badge/C%2B%2B-17-blue)
![CMake](https://img.shields.io/badge/CMake-3.20%2B-blue)

End-to-end machine-learning inference benchmark comparing Python and native C++ for a realistic fraud-detection workload.

---

## Overview

The objective is to study the performance impact of moving a machine-learning inference workload from Python to C++ while keeping the same trained model and equivalent preprocessing.

The use case is binary fraud detection on synthetic European SEPA banking transactions.

The project compares:

- PyTorch inference in Python
- ONNX Runtime inference in Python
- ONNX Runtime inference in C++
- native C++ end-to-end inference including preprocessing

Metrics include:

- mean latency
- p50 latency
- p95 latency
- p99 latency
- throughput
- peak resident memory

---

## Architecture

```text
Raw banking transaction
        |
        v
+---------------------+
| Feature preprocessing|
+---------------------+
        |
        | 19 float32 features
        v
+---------------------+
| Compact MLP         |
| 19 -> 32 -> 16 -> 1|
+---------------------+
        |
        v
   Fraud probability
        |
        v
 Threshold decision
````

The production-style C++ path is:

```text
8 raw features
    |
    v
Native C++ preprocessing
    |
    v
19 model features
    |
    v
ONNX Runtime C++
    |
    v
Fraud probability
```

---

## Dataset

The project uses the **SynSEPA** synthetic banking transaction dataset.

Main transaction file:

```text
synsep_full_dataset.csv
```

Dataset characteristics:

* 1,839,560 transactions
* 7,112 fraudulent transactions
* fraud rate: approximately 0.39%
* transaction period: January to December 2024

The temporal split is intentionally designed to preserve chronology:

| Split         | Period               |
| ------------- | -------------------- |
| Train         | January - June       |
| Validation    | July - August        |
| Standard Test | September - November |
| Drift Test    | December             |

December is isolated as a distribution-drift period because its fraud regime differs strongly from the preceding months.

Raw datasets are not committed to Git.

---

## Features

Eight raw model features are used:

### Categorical

* `persona`
* `remittance_category`

### Numerical

* `amount`
* `hour_of_day`
* `day_of_week`
* `time_since_last_txn`

### Binary

* `is_weekend`
* `is_new_beneficiary`

After one-hot encoding and numerical scaling, the model receives:

```text
19 float32 features
```

Several dataset columns are intentionally excluded, including identifiers, free text, target leakage fields, and synthetic geographic shortcuts.

See:

```text
docs/features.md
docs/preprocessing.md
```

---

## Model

The final model is a compact multilayer perceptron:

```text
19 -> 32 -> 16 -> 1
```

with ReLU activations between hidden layers.

Total number of trainable parameters:

```text
1,185
```

Training uses:

* PyTorch
* `BCEWithLogitsLoss`
* class weighting
* Adam optimizer
* validation PR-AUC for model selection
* early stopping

The trained model is exported to ONNX for cross-runtime inference.

---

## Model Quality

### Logistic Regression Baseline

Standard test:

| Metric  |  Value |
| ------- | -----: |
| PR-AUC  | 0.1298 |
| ROC-AUC | 0.9898 |
| F1      | 0.1707 |

### MLP

Standard test:

| Metric  |  Value |
| ------- | -----: |
| PR-AUC  | 0.3429 |
| ROC-AUC | 0.9926 |
| F1      | 0.3024 |

The December drift set produces significantly weaker MLP performance, demonstrating that low inference latency does not imply robustness to distribution shift.

---

## Numerical Parity

PyTorch and ONNX Runtime outputs were compared on 1,000 samples.

```text
Maximum absolute error: 9.96e-8
Mean absolute error:    1.44e-8
Tolerance:              1e-5
Result:                 PASS
```

Native C++ preprocessing was also validated against the Python preprocessing pipeline with differences on the order of `1e-8`.

---

## Performance Results

Hardware:

```text
Apple MacBook Air M3
ARM64
CPU inference
1 thread
Batch size = 1
```

Benchmark protocol:

```text
2,000 warm-up runs
50,000 measured inference calls
```

### Latency and Throughput

| Runtime             |      Mean |       P50 |       P95 |       P99 |    Throughput |
| ------------------- | --------: | --------: | --------: | --------: | ------------: |
| PyTorch Python      | 10.869 µs | 10.792 µs | 11.250 µs | 12.500 µs |  92,009 inf/s |
| ONNX Runtime Python |  3.917 µs |  3.916 µs |  4.125 µs |  4.542 µs | 255,276 inf/s |
| ONNX Runtime C++    |  2.093 µs |  2.083 µs |  2.167 µs |  2.458 µs | 477,816 inf/s |
| C++ End-to-End      |  2.239 µs |  2.250 µs |  2.334 µs |  2.500 µs | 446,667 inf/s |

Compared with PyTorch Python, ONNX Runtime C++ reduces mean inference latency by approximately:

```text
5.2x
```

---

## Memory

Peak resident set size:

| Runtime             | Peak RSS |
| ------------------- | -------: |
| PyTorch Python      | 199.6 MB |
| ONNX Runtime Python |  44.8 MB |
| ONNX Runtime C++    |  22.9 MB |

The C++ ONNX Runtime process therefore uses approximately:

```text
8.7x less peak resident memory
```

than the PyTorch Python process in this benchmark.

---

## End-to-End C++ Performance

The native C++ inference path includes:

```text
Raw transaction
    -> preprocessing
    -> tensor creation
    -> ONNX Runtime
    -> probability
```

Measured mean latency:

```text
2.239 µs
```

Native preprocessing alone is approximately:

```text
0.012 µs
```

and represents only a very small fraction of total end-to-end latency.

Because this preprocessing operation is extremely short, the value should be interpreted as an order-of-magnitude microbenchmark rather than an exact nanosecond-level measurement.

---

## Project Structure

```text
.
├── benchmarks/
│   ├── benchmark_python.py
│   ├── export_cpp_inputs.py
│   └── export_cpp_raw_inputs.py
│
├── cpp/
│   ├── include/
│   │   └── preprocessor.hpp
│   │
│   ├── src/
│   │   ├── benchmark.cpp
│   │   ├── benchmark_end_to_end.cpp
│   │   ├── benchmark_preprocessor.cpp
│   │   ├── inference.cpp
│   │   ├── inference_end_to_end.cpp
│   │   ├── memory_benchmark.cpp
│   │   ├── onnx_smoke_test.cpp
│   │   └── preprocessor.cpp
│   │
│   └── CMakeLists.txt
│
├── data/
│   ├── raw/
│   └── processed/
│
├── docs/
│   ├── benchmark.md
│   ├── benchmark_results.md
│   ├── evaluation.md
│   ├── features.md
│   ├── memory_benchmark.md
│   ├── model.md
│   ├── preprocessing.md
│   ├── requirements.md
│   └── training.md
│
├── models/
│
├── src/
│   └── low_latency_ml/
│
├── tests/
├── pyproject.toml
└── README.md
```

---

## Requirements

### Python

```text
Python 3.11+
```

Create the virtual environment and install the project dependencies according to `pyproject.toml`.

### C++

Required:

```text
C++17
CMake >= 3.20
ONNX Runtime
nlohmann-json
```

On macOS with Homebrew:

```bash
brew install onnxruntime nlohmann-json
```

CMake automatically detects these Homebrew installations.

Custom dependency locations can also be supplied through:

```text
ONNXRUNTIME_ROOT
NLOHMANN_JSON_ROOT
```

---

## Build the C++ Components

From the repository root:

```bash
cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

Test the ONNX Runtime integration:

```bash
./build/onnx_smoke_test
```

Expected output:

```text
ONNX Runtime C++: OK
```

---

## Run End-to-End C++ Inference

```bash
./build/inference_end_to_end
```

Example:

```text
Fraud probability: 0.967934
Prediction: LEGITIMATE
```

---

## Run the Benchmarks

Python benchmark:

```bash
python -m benchmarks.benchmark_python
```

Export the raw transactions used by the C++ end-to-end benchmark:

```bash
python -m benchmarks.export_cpp_raw_inputs
```

C++ inference-only benchmark:

```bash
./build/benchmark_cpp
```

C++ end-to-end benchmark:

```bash
./build/benchmark_end_to_end
```

Native preprocessing microbenchmark:

```bash
./build/benchmark_preprocessor
```

Detailed methodology and results are available in:

```text
docs/benchmark.md
docs/benchmark_results.md
docs/memory_benchmark.md
```

---

## Key Takeaways

This project demonstrates that deploying the same compact neural network through ONNX Runtime and C++ can substantially reduce both inference latency and process memory compared with a conventional PyTorch Python runtime.

It also highlights two separate engineering concerns:

1. **runtime performance** — where native C++ and ONNX Runtime provide significant gains;
2. **model robustness** — where distribution drift can still degrade predictive performance regardless of inference speed.

The result is therefore not only a latency benchmark, but an end-to-end study of how a Python-trained ML model can be validated, exported, and executed efficiently in a native inference environment.

````
