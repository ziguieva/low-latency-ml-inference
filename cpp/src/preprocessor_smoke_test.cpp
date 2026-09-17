#include "preprocessor.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>


namespace {

constexpr float TOLERANCE = 1e-6F;


void assert_close(
    float actual,
    float expected,
    const std::string& name
) {
    const float error =
        std::fabs(actual - expected);

    if (error > TOLERANCE) {
        throw std::runtime_error(
            name
            + " mismatch: actual="
            + std::to_string(actual)
            + ", expected="
            + std::to_string(expected)
        );
    }
}


void test_complete_transaction(
    const Preprocessor& preprocessor
) {
    const RawTransaction transaction{
        "employee",
        "transfer",
        250.0F,
        14.0F,
        2.0F,
        3600.0F,
        0.0F,
        1.0F
    };

    const auto features =
        preprocessor.transform(transaction);

    const std::array<float, PROCESSED_INPUT_DIM>
        expected{
            0.0F,
            1.0F,
            0.0F,
            0.0F,
            0.0F,
            0.0F,
            0.0F,
            0.0F,
            0.0F,
            0.0F,
            0.0F,
            1.0F,
            0.0F,
            -0.38697805F,
            0.05007932F,
            -0.36615145F,
            -0.84075137F,
            0.0F,
            1.0F
        };

    for (
        std::size_t i = 0;
        i < features.size();
        ++i
    ) {
        assert_close(
            features[i],
            expected[i],
            "feature[" + std::to_string(i) + "]"
        );
    }
}


void test_missing_time(
    const Preprocessor& preprocessor
) {
    const RawTransaction transaction{
        "employee",
        "transfer",
        250.0F,
        14.0F,
        2.0F,
        std::nullopt,
        0.0F,
        1.0F
    };

    const auto features =
        preprocessor.transform(transaction);

    assert_close(
        features[16],
        -0.85933115F,
        "missing time_since_last_txn"
    );
}

}


int main(int argc, char* argv[]) {
    try {
        if (argc != 2) {
            std::cerr
                << "Usage: "
                << argv[0]
                << " <inference-contract.json>\n";

            return 1;
        }

        const Preprocessor preprocessor(
            argv[1]
        );

        test_complete_transaction(
            preprocessor
        );

        test_missing_time(
            preprocessor
        );

        std::cout
            << "C++ preprocessing parity: OK\n";

        return 0;
    }
    catch (const std::exception& error) {
        std::cerr
            << "C++ preprocessing parity: FAILED\n"
            << error.what()
            << '\n';

        return 1;
    }
}


