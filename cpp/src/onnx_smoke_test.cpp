#include <iostream>
#include <onnxruntime_cxx_api.h>

int main() {
    Ort::Env env(
        ORT_LOGGING_LEVEL_WARNING,
        "smoke-test"
    );

    std::cout << "ONNX Runtime C++: OK" << std::endl;

    return 0;
}