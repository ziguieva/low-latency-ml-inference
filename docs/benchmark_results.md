# Benchmark Results

## Environment

- Machine: Apple MacBook Air M3
- Architecture: ARM64
- CPU inference only
- Threads: 1
- Batch size: 1
- Input dimension: 19
- Warm-up runs: 2,000
- Measured inference runs: 50,000

## Latency and Throughput

| Runtime | Mean latency | P50 | P95 | P99 | Throughput |
|---|---:|---:|---:|---:|---:|
| PyTorch Python | 10.869 µs | 10.792 µs | 11.250 µs | 12.500 µs | 92,009 inf/s |
| ONNX Runtime Python | 3.917 µs | 3.916 µs | 4.125 µs | 4.542 µs | 255,276 inf/s |
| ONNX Runtime C++ | 2.093 µs | 2.083 µs | 2.167 µs | 2.458 µs | 477,816 inf/s |

## Peak Memory

| Runtime | Peak RSS |
|---|---:|
| PyTorch Python | 199.6 MB |
| ONNX Runtime Python | 44.8 MB |
| ONNX Runtime C++ | 22.9 MB |

## Relative Performance

Compared with PyTorch Python:

- ONNX Runtime Python is approximately 2.8x faster.
- ONNX Runtime C++ is approximately 5.2x faster.
- ONNX Runtime C++ uses approximately 8.7x less peak resident memory.

Compared with ONNX Runtime Python:

- ONNX Runtime C++ is approximately 1.87x faster.
- ONNX Runtime C++ uses approximately 2x less peak resident memory.

## Numerical Parity

PyTorch and ONNX Runtime were validated on 1,000 samples.

- Maximum absolute error: 9.96e-8
- Mean absolute error: 1.44e-8
- Tolerance: 1e-5
- Result: PASS