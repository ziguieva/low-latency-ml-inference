# Feature Specification

## Target

`is_fraud`

## Model Features

- persona
- amount
- remittance_category
- hour_of_day
- day_of_week
- is_weekend
- time_since_last_txn
- is_new_beneficiary

## Excluded Columns

- transaction_id: technical identifier
- account_id: account identifier
- sender_iban: sensitive identifier
- beneficiary_iban: high-cardinality identifier
- timestamp: temporal information already represented by derived features
- remittance_text: unstructured text, excluded from the initial scope
- fraud_type: direct target leakage
- is_fraud: prediction target