# Training Strategy

## Model

Compact MLP:

- Input: 19 features
- Hidden layer 1: 32 neurons
- Hidden layer 2: 16 neurons
- Output: 1 logit

## Training Configuration

- Loss: BCEWithLogitsLoss
- Optimizer: Adam
- Learning rate: 0.001
- Batch size: 4096
- Maximum epochs: 20
- Random seed: 42
- Device: CPU

## Class Imbalance

Fraud cases are strongly under-represented.

The positive-class weight is computed only from the training set:

pos_weight = number_of_legitimate_transactions / number_of_fraud_transactions

## Model Selection

The best model is selected using validation PR-AUC.

## Early Stopping

Training stops if validation PR-AUC does not improve for 3 consecutive epochs.

## Data Leakage Prevention

- Train data is used for model fitting.
- Validation data is used for model selection.
- Test data must not be used during training.
- Drift-test data must not be used during training.