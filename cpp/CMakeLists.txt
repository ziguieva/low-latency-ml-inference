cmake_minimum_required(VERSION 3.20)

project(low_latency_ml_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(
    ONNXRUNTIME_ROOT
    "/opt/homebrew/opt/onnxruntime"
    CACHE PATH
    "ONNX Runtime installation directory"
)

find_path(
    ONNXRUNTIME_INCLUDE_DIR
    onnxruntime_cxx_api.h
    PATHS "${ONNXRUNTIME_ROOT}/include/onnxruntime"
    REQUIRED
)

find_library(
    ONNXRUNTIME_LIBRARY
    NAMES onnxruntime
    PATHS "${ONNXRUNTIME_ROOT}/lib"
    REQUIRED
)

# Step 26 - ONNX Runtime smoke test
add_executable(
    onnx_smoke_test
    src/onnx_smoke_test.cpp
)

target_include_directories(
    onnx_smoke_test
    PRIVATE
    "${ONNXRUNTIME_INCLUDE_DIR}"
)

target_link_libraries(
    onnx_smoke_test
    PRIVATE
    "${ONNXRUNTIME_LIBRARY}"
)

# Step 27 - ONNX inference
add_executable(
    inference
    src/inference.cpp
)

target_include_directories(
    inference
    PRIVATE
    "${ONNXRUNTIME_INCLUDE_DIR}"
)

target_link_libraries(
    inference
    PRIVATE
    "${ONNXRUNTIME_LIBRARY}"
)

add_executable(
    benchmark_cpp
    src/benchmark.cpp
)

target_include_directories(
    benchmark_cpp
    PRIVATE
    "${ONNXRUNTIME_INCLUDE_DIR}"
)

target_link_libraries(
    benchmark_cpp
    PRIVATE
    "${ONNXRUNTIME_LIBRARY}"
)