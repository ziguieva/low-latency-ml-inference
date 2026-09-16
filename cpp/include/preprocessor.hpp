#pragma once

#include <array>
#include <optional>
#include <string>


constexpr std::size_t PROCESSED_INPUT_DIM = 19;


struct RawTransaction {
    std::string persona;
    std::string remittance_category;

    float amount;
    float hour_of_day;
    float day_of_week;

    std::optional<float> time_since_last_txn;

    float is_weekend;
    float is_new_beneficiary;
};


class Preprocessor {
public:
    explicit Preprocessor(
        const std::string& contract_path
    );

    std::array<float, PROCESSED_INPUT_DIM> transform(
        const RawTransaction& transaction
    ) const;

private:
    std::array<std::string, 4> persona_categories_;
    std::array<std::string, 9> remittance_categories_;

    float amount_mean_;
    float amount_scale_;

    float hour_mean_;
    float hour_scale_;

    float day_mean_;
    float day_scale_;

    float time_mean_;
    float time_scale_;

    float missing_time_value_;
};