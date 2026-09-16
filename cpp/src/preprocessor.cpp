#include "preprocessor.hpp"

#include <fstream>
#include <stdexcept>
#include <vector>
#include <algorithm>

#include <nlohmann/json.hpp>


using json = nlohmann::json;


Preprocessor::Preprocessor(
    const std::string& contract_path
) {
    std::ifstream file(contract_path);

    if (!file) {
        throw std::runtime_error(
            "Cannot open inference contract: " + contract_path
        );
    }

    json contract;
    file >> contract;

    const auto& preprocessing =
        contract.at("preprocessing");

    const auto& categories =
        preprocessing.at("categories");

    const auto persona_values =
        categories.at("persona")
            .get<std::vector<std::string>>();

    const auto remittance_values =
        categories.at("remittance_category")
            .get<std::vector<std::string>>();

    if (persona_values.size() != persona_categories_.size()) {
        throw std::runtime_error(
            "Unexpected persona category count."
        );
    }

    if (
        remittance_values.size()
        != remittance_categories_.size()
    ) {
        throw std::runtime_error(
            "Unexpected remittance category count."
        );
    }

    std::copy(
        persona_values.begin(),
        persona_values.end(),
        persona_categories_.begin()
    );

    std::copy(
        remittance_values.begin(),
        remittance_values.end(),
        remittance_categories_.begin()
    );

    const auto& means =
        preprocessing.at("scaler_mean");

    const auto& scales =
        preprocessing.at("scaler_scale");

    amount_mean_ =
        means.at("amount").get<float>();

    amount_scale_ =
        scales.at("amount").get<float>();

    hour_mean_ =
        means.at("hour_of_day").get<float>();

    hour_scale_ =
        scales.at("hour_of_day").get<float>();

    day_mean_ =
        means.at("day_of_week").get<float>();

    day_scale_ =
        scales.at("day_of_week").get<float>();

    time_mean_ =
        means.at("time_since_last_txn").get<float>();

    time_scale_ =
        scales.at("time_since_last_txn").get<float>();

    missing_time_value_ =
        preprocessing
            .at("missing_value")
            .at("time_since_last_txn")
            .get<float>();
}


std::array<float, PROCESSED_INPUT_DIM>
Preprocessor::transform(
    const RawTransaction& transaction
) const {
    std::array<float, PROCESSED_INPUT_DIM> output{};

    std::size_t index = 0;

    // Persona one-hot
    for (const auto& category : persona_categories_) {
        output[index++] =
            transaction.persona == category
            ? 1.0F
            : 0.0F;
    }

    // Remittance category one-hot
    for (
        const auto& category
        : remittance_categories_
    ) {
        output[index++] =
            transaction.remittance_category == category
            ? 1.0F
            : 0.0F;
    }

    // Standardized numerical features
    output[index++] =
        (transaction.amount - amount_mean_)
        / amount_scale_;

    output[index++] =
        (transaction.hour_of_day - hour_mean_)
        / hour_scale_;

    output[index++] =
        (transaction.day_of_week - day_mean_)
        / day_scale_;

    const float time_value =
        transaction.time_since_last_txn.value_or(
            missing_time_value_
        );

    output[index++] =
        (time_value - time_mean_)
        / time_scale_;

    // Binary features
    output[index++] =
        transaction.is_weekend;

    output[index++] =
        transaction.is_new_beneficiary;

    if (index != PROCESSED_INPUT_DIM) {
        throw std::runtime_error(
            "Invalid processed feature dimension."
        );
    }

    return output;
}