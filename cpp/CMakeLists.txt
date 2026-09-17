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

add_executable(
    memory_benchmark_cpp
    src/memory_benchmark.cpp
)

target_include_directories(
    memory_benchmark_cpp
    PRIVATE
    "${ONNXRUNTIME_INCLUDE_DIR}"
)

target_link_libraries(
    memory_benchmark_cpp
    PRIVATE
    "${ONNXRUNTIME_LIBRARY}"
)

set(
    NLOHMANN_JSON_ROOT
    "/opt/homebrew/opt/nlohmann-json"
    CACHE PATH
    "nlohmann-json installation directory"
)

find_path(
    NLOHMANN_JSON_INCLUDE_DIR
    nlohmann/json.hpp
    PATHS "${NLOHMANN_JSON_ROOT}/include"
    REQUIRED
)

add_library(
    preprocessor
    STATIC
    src/preprocessor.cpp
)

target_include_directories(
    preprocessor
    PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${NLOHMANN_JSON_INCLUDE_DIR}"
)

add_executable(
    preprocessor_smoke_test
    src/preprocessor_smoke_test.cpp
)

target_link_libraries(
    preprocessor_smoke_test
    PRIVATE
    preprocessor
)

add_executable(
    inference_end_to_end
    src/inference_end_to_end.cpp
)

target_include_directories(
    inference_end_to_end
    PRIVATE
    "${ONNXRUNTIME_INCLUDE_DIR}"
)

target_link_libraries(
    inference_end_to_end
    PRIVATE
    preprocessor
    "${ONNXRUNTIME_LIBRARY}"
)

add_executable(
    benchmark_end_to_end
    src/benchmark_end_to_end.cpp
)

target_include_directories(
    benchmark_end_to_end
    PRIVATE
    "${ONNXRUNTIME_INCLUDE_DIR}"
)

target_link_libraries(
    benchmark_end_to_end
    PRIVATE
    preprocessor
    "${ONNXRUNTIME_LIBRARY}"
)

add_executable(
    benchmark_preprocessor
    src/benchmark_preprocessor.cpp
)

target_link_libraries(
    benchmark_preprocessor
    PRIVATE
    preprocessor
)