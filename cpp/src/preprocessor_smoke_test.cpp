#include "preprocessor.hpp"

#include <iomanip>
#include <iostream>


int main() {
    Preprocessor preprocessor(
        "models/inference_contract.json"
    );

    RawTransaction transaction{
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

    std::cout << std::fixed
              << std::setprecision(8);

    for (std::size_t i = 0; i < features.size(); ++i) {
        std::cout
            << i
            << ": "
            << features[i]
            << '\n';
    }

    return 0;
}