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