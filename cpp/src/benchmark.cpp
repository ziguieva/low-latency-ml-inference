#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>


constexpr std::size_t INPUT_DIM = 19;
constexpr std::size_t WARMUP_RUNS = 2'000;
constexpr std::size_t BENCHMARK_RUNS = 50'000;

const std::string MODEL_PATH = "models/fraud_mlp.onnx";
const std::string INPUTS_PATH = "benchmarks/cpp_inputs.bin";


struct BenchmarkData {
    std::uint64_t rows;
    std::uint64_t cols;
    std::vector<float> values;
};


BenchmarkData load_inputs(const std::string& path) {
    std::ifstream file(path, std::ios::binary);

    if (!file) {
        throw std::runtime_error(
            "Cannot open benchmark input file: " + path
        );
    }

    std::array<char, 8> magic{};
    file.read(magic.data(), magic.size());

    const std::string expected_magic = "LLMLBEN1";

    if (std::string(magic.data(), magic.size()) != expected_magic) {
        throw std::runtime_error(
            "Invalid benchmark input file format."
        );
    }

    std::uint64_t rows = 0;
    std::uint64_t cols = 0;

    file.read(
        reinterpret_cast<char*>(&rows),
        sizeof(rows)
    );

    file.read(
        reinterpret_cast<char*>(&cols),
        sizeof(cols)
    );

    if (cols != INPUT_DIM) {
        throw std::runtime_error(
            "Unexpected input dimension."
        );
    }

    const std::size_t total_values =
        static_cast<std::size_t>(rows * cols);

    std::vector<float> values(total_values);

    file.read(
        reinterpret_cast<char*>(values.data()),
        static_cast<std::streamsize>(
            total_values * sizeof(float)
        )
    );

    if (!file) {
        throw std::runtime_error(
            "Failed to read benchmark input data."
        );
    }

    return {
        rows,
        cols,
        std::move(values)
    };
}


double percentile(
    std::vector<std::int64_t> values,
    double p
) {
    std::sort(
        values.begin(),
        values.end()
    );

    const double position =
        (values.size() - 1) * p;

    const std::size_t lower =
        static_cast<std::size_t>(position);

    const std::size_t upper =
        std::min(
            lower + 1,
            values.size() - 1
        );

    const double fraction =
        position - lower;

    return (
        values[lower]
        + fraction
        * (values[upper] - values[lower])
    );
}


void print_results(
    const std::vector<std::int64_t>& latencies_ns
) {
    const double total_ns =
        std::accumulate(
            latencies_ns.begin(),
            latencies_ns.end(),
            0.0
        );

    const double mean_ns =
        total_ns / latencies_ns.size();

    const double p50_ns =
        percentile(latencies_ns, 0.50);

    const double p95_ns =
        percentile(latencies_ns, 0.95);

    const double p99_ns =
        percentile(latencies_ns, 0.99);

    const double total_seconds =
        total_ns / 1'000'000'000.0;

    const double throughput =
        latencies_ns.size() / total_seconds;

    std::cout << std::fixed
              << std::setprecision(3);

    std::cout << "\n=== ONNX RUNTIME C++ ===\n";

    std::cout
        << "Mean:       "
        << mean_ns / 1'000.0
        << " us\n";

    std::cout
        << "P50:        "
        << p50_ns / 1'000.0
        << " us\n";

    std::cout
        << "P95:        "
        << p95_ns / 1'000.0
        << " us\n";

    std::cout
        << "P99:        "
        << p99_ns / 1'000.0
        << " us\n";

    std::cout
        << "Throughput: "
        << std::setprecision(0)
        << throughput
        << " inference/s\n";
}


int main() {
    try {
        const BenchmarkData data =
            load_inputs(INPUTS_PATH);

        if (data.rows < BENCHMARK_RUNS) {
            throw std::runtime_error(
                "Not enough benchmark input rows."
            );
        }

        std::cout << "=== BENCHMARK CONFIGURATION ===\n";
        std::cout << "Samples:     "
                  << BENCHMARK_RUNS << '\n';

        std::cout << "Input dim:   "
                  << data.cols << '\n';

        std::cout << "Warm-up:     "
                  << WARMUP_RUNS << '\n';

        std::cout << "Threads:     1\n";
        std::cout << "Batch size:  1\n";

        Ort::Env env(
            ORT_LOGGING_LEVEL_WARNING,
            "cpp-benchmark"
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
            MODEL_PATH.c_str(),
            session_options
        );

        Ort::MemoryInfo memory_info =
            Ort::MemoryInfo::CreateCpu(
                OrtArenaAllocator,
                OrtMemTypeDefault
            );

        const std::array<int64_t, 2> input_shape{
            1,
            static_cast<int64_t>(INPUT_DIM)
        };

        const char* input_names[] = {
            "features"
        };

        const char* output_names[] = {
            "fraud_probability"
        };

        // Warm-up
        for (std::size_t i = 0; i < WARMUP_RUNS; ++i) {
            float* row =
                const_cast<float*>(
                    data.values.data()
                    + (i % data.rows) * INPUT_DIM
                );

            Ort::Value input_tensor =
                Ort::Value::CreateTensor<float>(
                    memory_info,
                    row,
                    INPUT_DIM,
                    input_shape.data(),
                    input_shape.size()
                );

            session.Run(
                Ort::RunOptions{nullptr},
                input_names,
                &input_tensor,
                1,
                output_names,
                1
            );
        }

        std::vector<std::int64_t> latencies_ns;
        latencies_ns.reserve(BENCHMARK_RUNS);

        volatile float probability_sink = 0.0F;

        // Benchmark
        for (
            std::size_t i = 0;
            i < BENCHMARK_RUNS;
            ++i
        ) {
            float* row =
                const_cast<float*>(
                    data.values.data()
                    + i * INPUT_DIM
                );

            Ort::Value input_tensor =
                Ort::Value::CreateTensor<float>(
                    memory_info,
                    row,
                    INPUT_DIM,
                    input_shape.data(),
                    input_shape.size()
                );

            const auto start =
                std::chrono::steady_clock::now();

            auto outputs = session.Run(
                Ort::RunOptions{nullptr},
                input_names,
                &input_tensor,
                1,
                output_names,
                1
            );

            const auto end =
                std::chrono::steady_clock::now();

            probability_sink +=
                outputs[0]
                    .GetTensorMutableData<float>()[0];

            const auto latency =
                std::chrono::duration_cast<
                    std::chrono::nanoseconds
                >(end - start)
                .count();

            latencies_ns.push_back(latency);
        }

        print_results(latencies_ns);

        // Prevent optimizer from discarding inference output.
        if (probability_sink < 0.0F) {
            std::cerr << probability_sink << '\n';
        }

        return 0;
    }
    catch (const Ort::Exception& error) {
        std::cerr
            << "ONNX Runtime error: "
            << error.what()
            << '\n';

        return 1;
    }
    catch (const std::exception& error) {
        std::cerr
            << "Error: "
            << error.what()
            << '\n';

        return 1;
    }
}