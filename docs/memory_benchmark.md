# Memory Benchmark Protocol

## Goal

Measure peak resident memory usage for each inference runtime.

## Compared Runtimes

- PyTorch Python
- ONNX Runtime Python
- ONNX Runtime C++

## Conditions

All runtimes must use:

- CPU inference
- 1 thread
- batch size = 1
- input shape = [1, 19]
- float32 input
- 2,000 warm-up inferences
- model already loaded before measurement
- the same model weights

## Exclusions

The benchmark excludes:

- dataset loading
- preprocessing
- training
- benchmark input generation

Only the inference runtime, model and one reusable input tensor are considered.

## Metric

Peak RSS (Resident Set Size).

## Measurement Tool

macOS:

`/usr/bin/time -l`

Peak memory is reported in MB.