#include <array>
#include <cstddef>
#include <iostream>
#include <string>

#include <onnxruntime_cxx_api.h>


constexpr std::size_t INPUT_DIM = 19;
constexpr std::size_t WARMUP_RUNS = 2'000;

const std::string MODEL_PATH = "models/fraud_mlp.onnx";


int main() {
    try {
        Ort::Env env(
            ORT_LOGGING_LEVEL_WARNING,
            "memory-benchmark"
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

        std::array<float, INPUT_DIM> input_data{};

        const std::array<int64_t, 2> input_shape{
            1,
            static_cast<int64_t>(INPUT_DIM)
        };

        Ort::MemoryInfo memory_info =
            Ort::MemoryInfo::CreateCpu(
                OrtArenaAllocator,
                OrtMemTypeDefault
            );

        Ort::Value input_tensor =
            Ort::Value::CreateTensor<float>(
                memory_info,
                input_data.data(),
                input_data.size(),
                input_shape.data(),
                input_shape.size()
            );

        const char* input_names[] = {
            "features"
        };

        const char* output_names[] = {
            "fraud_probability"
        };

        volatile float probability_sink = 0.0F;

        for (
            std::size_t i = 0;
            i < WARMUP_RUNS;
            ++i
        ) {
            auto outputs = session.Run(
                Ort::RunOptions{nullptr},
                input_names,
                &input_tensor,
                1,
                output_names,
                1
            );

            probability_sink +=
                outputs[0]
                    .GetTensorMutableData<float>()[0];
        }

        std::cout
            << "ONNX Runtime C++ memory benchmark complete."
            << '\n';

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