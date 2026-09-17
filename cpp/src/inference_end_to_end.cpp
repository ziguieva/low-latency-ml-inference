#include "preprocessor.hpp"

#include <array>
#include <iostream>
#include <string>

#include <onnxruntime_cxx_api.h>


int main() {
    // 1. Transaction brute
    RawTransaction transaction{
        "employee",
        "transfer",
        250.0F,
        14.0F,
        2.0F,
        3600.0F,
        0.0F,
        1.0F
    };

    // 2. Prétraitement natif C++
    Preprocessor preprocessor(
        "models/inference_contract.json"
    );

    const auto processed =
        preprocessor.transform(transaction);

    // 3. Initialisation ONNX Runtime
    Ort::Env env(
        ORT_LOGGING_LEVEL_WARNING,
        "fraud-inference"
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

    // 4. Tensor d'entrée [1, 19]
    std::array<int64_t, 2> input_shape{
        1,
        PROCESSED_INPUT_DIM
    };

    auto memory_info =
        Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator,
            OrtMemTypeDefault
        );

    auto input_tensor =
        Ort::Value::CreateTensor<float>(
            memory_info,
            const_cast<float*>(processed.data()),
            processed.size(),
            input_shape.data(),
            input_shape.size()
        );

    // 5. Inference
    const char* input_names[] = {
        "features"
    };

    const char* output_names[] = {
        "fraud_probability"
    };

    auto outputs =
        session.Run(
            Ort::RunOptions{nullptr},
            input_names,
            &input_tensor,
            1,
            output_names,
            1
        );

    const float probability =
        outputs[0]
            .GetTensorMutableData<float>()[0];

    constexpr float threshold = 0.986F;

    // 6. Résultat
    std::cout
        << "Fraud probability: "
        << probability
        << '\n';

    std::cout
        << "Prediction: "
        << (
            probability >= threshold
            ? "FRAUD"
            : "LEGITIMATE"
        )
        << '\n';

    return 0;
}