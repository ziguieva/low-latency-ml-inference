# Model Evaluation Strategy

## Primary Metric

PR-AUC (Average Precision)

This is the primary model-selection metric because fraud cases are highly imbalanced.

## Secondary Metrics

- ROC-AUC
- Precision
- Recall
- F1-score

Accuracy must not be used as the main selection metric.

## Threshold Selection

The classification threshold must be selected using the validation set only.

The test and drift-test datasets must never be used for threshold tuning.

## Evaluation Sets

- Train: model fitting
- Validation: model selection and threshold tuning
- Test: final standard evaluation
- Drift Test: evaluation under distribution shift

## Baseline

A logistic regression classifier will be used as the first baseline before developing the main low-latency model.