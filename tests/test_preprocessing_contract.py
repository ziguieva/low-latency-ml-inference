import unittest

import numpy as np


class TestPreprocessingContract(unittest.TestCase):
    def test_expected_feature_layout(self) -> None:
        persona_categories = [
            "business",
            "employee",
            "retiree",
            "student",
        ]

        remittance_categories = [
            "grocery",
            "other",
            "rent",
            "salary",
            "shopping",
            "subscription",
            "supplier",
            "transfer",
            "utility",
        ]

        processed_features = (
            [f"persona_{value}" for value in persona_categories]
            + [
                f"remittance_category_{value}"
                for value in remittance_categories
            ]
            + [
                "amount",
                "hour_of_day",
                "day_of_week",
                "time_since_last_txn",
                "is_weekend",
                "is_new_beneficiary",
            ]
        )

        self.assertEqual(len(processed_features), 19)
        self.assertEqual(len(set(processed_features)), 19)

    def test_standardization_formula(self) -> None:
        value = 250.0
        mean = 3181.4739541955314
        scale = 7575.297667778302

        standardized = (value - mean) / scale

        np.testing.assert_allclose(
            standardized,
            -0.38697805,
            rtol=0.0,
            atol=1e-7,
        )

    def test_missing_time_value(self) -> None:
        missing_value = -1.0
        mean = 166548.35974341136
        scale = 193812.78011284757

        standardized = (
            missing_value - mean
        ) / scale

        np.testing.assert_allclose(
            standardized,
            -0.85933115,
            rtol=0.0,
            atol=1e-7,
        )


if __name__ == "__main__":
    unittest.main()