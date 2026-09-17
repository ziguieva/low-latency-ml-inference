#include "preprocessor.hpp"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>


namespace {

constexpr std::size_t WARMUP_RUNS = 2'000;

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

        if (!std::isnan(
                record.time_since_last_txn
            )) {
            time_value =
                record.time_since_last_txn;
        }

        transactions.push_back(
            RawTransaction{
                fixed_string(
                    record.persona,
                    sizeof(record.persona)
                ),
                fixed_string(
                    record.remittance_category,
                    sizeof(
                        record.remittance_category
                    )
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
    if (values.empty()) {
        throw std::runtime_error(
            "Cannot compute percentile of empty data."
        );
    }

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

    Ort::Env env(
        ORT_LOGGING_LEVEL_WARNING,
        "benchmark-end-to-end"
    );

    Ort::SessionOptions session_options;

    session_options.SetIntraOpNumThreads(1);
    session_options.SetInterOpNumThreads(1);

    session_options.SetExecutionMode(
        ExecutionMode::ORT_SEQUENTIAL
    );

    session_options.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL
    );

    Ort::Session session(
        env,
        "models/fraud_mlp.onnx",
        session_options
    );

    auto memory_info =
        Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator,
            OrtMemTypeDefault
        );

    const std::array<int64_t, 2> input_shape{
        1,
        PROCESSED_INPUT_DIM
    };

    const char* input_names[] = {
        "features"
    };

    const char* output_names[] = {
        "fraud_probability"
    };

    volatile float sink = 0.0F;

    // Warm-up
    for (
        std::size_t i = 0;
        i < WARMUP_RUNS;
        ++i
    ) {
        const auto& transaction =
            transactions[
                i % transactions.size()
            ];

        auto processed =
            preprocessor.transform(
                transaction
            );

        auto input_tensor =
            Ort::Value::CreateTensor<float>(
                memory_info,
                processed.data(),
                processed.size(),
                input_shape.data(),
                input_shape.size()
            );

        auto outputs =
            session.Run(
                Ort::RunOptions{nullptr},
                input_names,
                &input_tensor,
                1,
                output_names,
                1
            );

        sink =
            outputs[0]
                .GetTensorMutableData<float>()[0];
    }

    std::vector<double> latencies_us;
    latencies_us.reserve(
        transactions.size()
    );

    for (
        const auto& transaction
        : transactions
    ) {
        const auto start =
            std::chrono::steady_clock::now();

        auto processed =
            preprocessor.transform(
                transaction
            );

        auto input_tensor =
            Ort::Value::CreateTensor<float>(
                memory_info,
                processed.data(),
                processed.size(),
                input_shape.data(),
                input_shape.size()
            );

        auto outputs =
            session.Run(
                Ort::RunOptions{nullptr},
                input_names,
                &input_tensor,
                1,
                output_names,
                1
            );

        sink =
            outputs[0]
                .GetTensorMutableData<float>()[0];

        const auto end =
            std::chrono::steady_clock::now();

        const double latency_us =
            std::chrono::duration<double, std::micro>(
                end - start
            ).count();

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
        << "\nONNX Runtime C++ End-to-End\n";

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
        << " inf/s\n";

    return 0;
}