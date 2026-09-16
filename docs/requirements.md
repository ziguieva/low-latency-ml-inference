# Project Requirements

## Use Case

Real-time fraud detection for European banking transactions.

## Dataset

SynSEPA dataset:

- `synsep_full_dataset.csv`: main transaction dataset
- `accounts.csv`: complementary account-level information

### Dataset period

The SynSEPA transaction data covers:

- Start: 2024-01-01
- End: 2024-12-31
- Transactions: 1,839,560
- Fraud cases: 7,112
- Fraud rate: 0.3866%

Although the initial preference was for 2025+ data, the 2024 SynSEPA dataset is accepted because of its realistic European SEPA structure and sufficient volume for low-latency inference benchmarking.

## Input

A single banking transaction represented by numerical and categorical features.

## Output

Fraud probability:

- 0: legitimate transaction
- 1: fraudulent transaction

## Main Objective

Train a machine-learning model in Python and deploy its inference pipeline in C++.

## Performance Requirements

The implementations will be compared using:

- Mean latency
- P50 latency
- P95 latency
- P99 latency
- Throughput
- Memory usage

## Constraints

- Python 3.11
- C++17
- CPU inference
- Apple Silicon M3
- Batch size = 1 for low-latency benchmarking
- Same model and same input data for all implementations