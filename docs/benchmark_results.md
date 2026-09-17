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
## End-to-End C++ Inference

The end-to-end benchmark measures the complete native C++ inference path:

`Raw transaction -> C++ preprocessing -> ONNX tensor -> ONNX Runtime -> probability`

Dataset loading, contract loading, and model initialization are excluded from the timed section.

| Metric | ONNX Runtime C++ only | C++ End-to-End |
|---|---:|---:|
| Mean latency | 2.093 µs | 2.239 µs |
| P50 | 2.083 µs | 2.250 µs |
| P95 | 2.167 µs | 2.334 µs |
| P99 | 2.458 µs | 2.500 µs |
| Throughput | 477,816 inf/s | 446,667 inf/s |

The native preprocessing and tensor preparation add approximately 0.146 µs to the mean latency, corresponding to about 7% overhead compared with ONNX Runtime inference alone.

## Native C++ Preprocessing Microbenchmark

The native preprocessing stage was benchmarked independently on the same 50,000 raw transactions used for the end-to-end benchmark.

Because a single transformation is extremely short, each transaction was transformed 100 times inside the timed block and the measured duration was divided by 100.

Three consecutive runs produced:

| Run | Mean | P50 | P95 | P99 |
|---|---:|---:|---:|---:|
| 1 | 0.013 µs | 0.012 µs | 0.016 µs | 0.018 µs |
| 2 | 0.012 µs | 0.011 µs | 0.014 µs | 0.015 µs |
| 3 | 0.011 µs | 0.011 µs | 0.013 µs | 0.015 µs |

A representative preprocessing cost of approximately **0.012 µs (12 ns)** is therefore used as an order-of-magnitude result.

Compared directly with the 2.239 µs end-to-end mean latency, native preprocessing represents roughly **0.5%** of the complete C++ inference path.

### Latency Breakdown

| Stage | Mean latency |
|---|---:|
| ONNX Runtime C++ inference only | 2.093 µs |
| Native C++ preprocessing only | ~0.012 µs |
| Complete C++ end-to-end path | 2.239 µs |

The individual benchmarks were measured separately and should not be treated as perfectly additive. The remaining difference between inference-only and end-to-end latency includes tensor construction, runtime call-boundary overhead, result handling, and normal benchmark variability.
