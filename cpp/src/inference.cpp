#include <array>
#include <iostream>
#include <string>

#include <onnxruntime_cxx_api.h>


constexpr std::size_t INPUT_DIM = 19;
constexpr float FRAUD_THRESHOLD = 0.986F;


int main() {
    try {
        const std::string model_path = "models/fraud_mlp.onnx";

        // ONNX Runtime environment
        Ort::Env env(
            ORT_LOGGING_LEVEL_WARNING,
            "fraud-inference"
        );

        // Session configuration
        Ort::SessionOptions session_options;

        session_options.SetIntraOpNumThreads(1);
        session_options.SetInterOpNumThreads(1);

        session_options.SetGraphOptimizationLevel(
            GraphOptimizationLevel::ORT_ENABLE_ALL
        );

        // Load ONNX model
        Ort::Session session(
            env,
            model_path.c_str(),
            session_options
        );

        // First preprocessed input vector.
        // Neutral vector for validating the C++ inference path.
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

        auto outputs = session.Run(
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

        const bool is_fraud =
            probability >= FRAUD_THRESHOLD;

        std::cout << "=== C++ ONNX INFERENCE ===\n";
        std::cout << "Input dimension: "
                  << INPUT_DIM << '\n';

        std::cout << "Fraud probability: "
                  << probability << '\n';

        std::cout << "Threshold: "
                  << FRAUD_THRESHOLD << '\n';

        std::cout << "Prediction: "
                  << (is_fraud ? "FRAUD" : "LEGITIMATE")
                  << '\n';

        return 0;
    }
    catch (const Ort::Exception& error) {
        std::cerr
            << "ONNX Runtime error: "
            << error.what()
            << '\n';

        return 1;
    }
}