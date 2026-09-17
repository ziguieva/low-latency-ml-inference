#include "preprocessor.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>


namespace {

constexpr std::size_t WARMUP_RUNS = 2'000;
constexpr std::size_t INNER_REPEATS = 100;


struct RawRecord {
    char persona[16];
    char remittance_category[16];

    float amount;
    float hour_of_day;
    float day_of_week;
    float time_since_last_txn;
    float is_weekend;
    float is_new_beneficiary;
};


static_assert(
    sizeof(RawRecord) == 56,
    "Unexpected RawRecord size."
);


std::string fixed_string(
    const char* data,
    std::size_t size
) {
    const auto end = std::find(
        data,
        data + size,
        '\0'
    );

    return std::string(data, end);
}


std::vector<RawTransaction> load_transactions(
    const std::string& path
) {
    std::ifstream file(
        path,
        std::ios::binary
    );

    if (!file) {
        throw std::runtime_error(
            "Cannot open raw input file: " + path
        );
    }

    std::array<char, 8> magic{};

    file.read(
        magic.data(),
        magic.size()
    );

    const std::array<char, 8> expected_magic{
        'L', 'L', 'M', 'L',
        'R', 'A', 'W', '1'
    };

    if (magic != expected_magic) {
        throw std::runtime_error(
            "Invalid raw input file magic."
        );
    }

    std::uint64_t rows = 0;
    std::uint64_t record_size = 0;

    file.read(
        reinterpret_cast<char*>(&rows),
        sizeof(rows)
    );

    file.read(
        reinterpret_cast<char*>(&record_size),
        sizeof(record_size)
    );

    if (record_size != sizeof(RawRecord)) {
        throw std::runtime_error(
            "Unexpected raw record size."
        );
    }

    std::vector<RawTransaction> transactions;
    transactions.reserve(rows);

    for (std::uint64_t i = 0; i < rows; ++i) {
        RawRecord record{};

        file.read(
            reinterpret_cast<char*>(&record),
            sizeof(record)
        );

        if (!file) {
            throw std::runtime_error(
                "Unexpected end of raw input file."
            );
        }

        std::optional<float> time_value;

        if (!std::isnan(record.time_since_last_txn)) {
            time_value = record.time_since_last_txn;
        }

        transactions.push_back(
            RawTransaction{
                fixed_string(
                    record.persona,
                    sizeof(record.persona)
                ),
                fixed_string(
                    record.remittance_category,
                    sizeof(record.remittance_category)
                ),
                record.amount,
                record.hour_of_day,
                record.day_of_week,
                time_value,
                record.is_weekend,
                record.is_new_beneficiary
            }
        );
    }

    return transactions;
}


double percentile(
    std::vector<double> values,
    double p
) {
    std::sort(
        values.begin(),
        values.end()
    );

    const double position =
        p * static_cast<double>(
            values.size() - 1
        );

    const auto lower =
        static_cast<std::size_t>(
            std::floor(position)
        );

    const auto upper =
        static_cast<std::size_t>(
            std::ceil(position)
        );

    if (lower == upper) {
        return values[lower];
    }

    const double weight =
        position - static_cast<double>(lower);

    return
        values[lower] * (1.0 - weight)
        + values[upper] * weight;
}

}


int main() {
    const auto transactions =
        load_transactions(
            "benchmarks/cpp_raw_inputs.bin"
        );

    std::cout
        << "Transactions: "
        << transactions.size()
        << '\n';

    Preprocessor preprocessor(
        "models/inference_contract.json"
    );

    volatile float sink = 0.0F;

    // Warm-up
    for (
        std::size_t i = 0;
        i < WARMUP_RUNS;
        ++i
    ) {
        const auto features =
            preprocessor.transform(
                transactions[
                    i % transactions.size()
                ]
            );

        sink = features[0];
    }

    std::vector<double> latencies_us;
    latencies_us.reserve(
        transactions.size()
    );


    for (
        const auto& transaction
        : transactions
    ) {
        float local_sink = 0.0F;

        const auto start =
            std::chrono::steady_clock::now();

        for (
            std::size_t repeat = 0;
            repeat < INNER_REPEATS;
            ++repeat
        ) {
            const auto features =
                preprocessor.transform(
                    transaction
                );

            local_sink +=
                features[
                    repeat % features.size()
                ];
        }

        const auto end =
            std::chrono::steady_clock::now();

        sink = sink + local_sink;

        const double block_us =
            std::chrono::duration<
                double,
                std::micro
            >(
                end - start
            ).count();

        const double latency_us =
            block_us
            / static_cast<double>(
                INNER_REPEATS
            );

        latencies_us.push_back(
            latency_us
        );
    }


    const double total_us =
        std::accumulate(
            latencies_us.begin(),
            latencies_us.end(),
            0.0
        );

    const double mean_us =
        total_us
        / static_cast<double>(
            latencies_us.size()
        );

    const double throughput =
        1'000'000.0 / mean_us;

    std::cout
        << std::fixed
        << std::setprecision(3);

    std::cout
        << "\nC++ Preprocessing Only\n";

    std::cout
        << "Mean: "
        << mean_us
        << " us\n";

    std::cout
        << "P50: "
        << percentile(
            latencies_us,
            0.50
        )
        << " us\n";

    std::cout
        << "P95: "
        << percentile(
            latencies_us,
            0.95
        )
        << " us\n";

    std::cout
        << "P99: "
        << percentile(
            latencies_us,
            0.99
        )
        << " us\n";

    std::cout
        << "Throughput: "
        << std::setprecision(0)
        << throughput
        << " transforms/s\n";

    return 0;
}