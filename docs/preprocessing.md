# Preprocessing Specification

## Categorical Features

The following features are encoded using one-hot encoding:

- persona
- remittance_category

Unknown categories must be ignored during inference.

## Numerical Features

The following features are standardized using statistics computed only on the training set:

- amount
- hour_of_day
- day_of_week
- time_since_last_txn

Standardization:

x_scaled = (x - mean_train) / std_train

## Binary Features

The following features remain unchanged:

- is_weekend
- is_new_beneficiary

## Missing Values

For `time_since_last_txn`:

- missing value means no previous transaction
- missing values are replaced by `-1`

## Leakage Prevention

All preprocessing parameters must be fitted exclusively on the training set.

Validation, test, drift test and future inference must reuse the exact same preprocessing parameters.