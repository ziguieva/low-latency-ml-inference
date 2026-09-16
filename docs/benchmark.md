# Low-Latency Benchmark Protocol

## Hardware

- Apple MacBook Air M3
- CPU inference only

## Input

- Batch size: 1
- Input shape: [1, 19]
- Data type: float32
- Inputs sampled from the standard test dataset

## Runtimes

- PyTorch CPU
- ONNX Runtime CPU

## Threading

Both runtimes are benchmarked with a single CPU thread.

## Warm-up

2,000 inference calls are executed before measurement.

Warm-up executions are excluded from benchmark statistics.

## Measurement

50,000 inference calls are measured.

Model loading and preprocessing are excluded.

Only model inference latency is measured.

## Timer

Python:

`time.perf_counter_ns()`

## Metrics

- Mean latency
- P50 latency
- P95 latency
- P99 latency
- Throughput

## Units

Latency is reported in microseconds.

Throughput is reported in inferences per second.

## Fairness Constraints

Both runtimes must use:

- the same machine
- the same model weights
- the same input vectors
- the same batch size
- CPU execution
- one thread