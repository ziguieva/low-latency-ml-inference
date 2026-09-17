cmake_minimum_required(VERSION 3.20)

project(low_latency_ml_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


# ---------------------------------------------------------
# Optional installation hints
# ---------------------------------------------------------

set(
    ONNXRUNTIME_ROOT
    ""
    CACHE PATH
    "Optional ONNX Runtime installation directory"
)

set(
    NLOHMANN_JSON_ROOT
    ""
    CACHE PATH
    "Optional nlohmann-json installation directory"
)


# ---------------------------------------------------------
# Homebrew fallback
# ---------------------------------------------------------

find_program(
    HOMEBREW_EXECUTABLE
    brew
)

if(HOMEBREW_EXECUTABLE)

    if(NOT ONNXRUNTIME_ROOT)
        execute_process(
            COMMAND
                "${HOMEBREW_EXECUTABLE}"
                --prefix
                onnxruntime
            RESULT_VARIABLE
                ONNXRUNTIME_BREW_RESULT
            OUTPUT_VARIABLE
                ONNXRUNTIME_BREW_PREFIX
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )

        if(ONNXRUNTIME_BREW_RESULT EQUAL 0)
            set(
                ONNXRUNTIME_ROOT
                "${ONNXRUNTIME_BREW_PREFIX}"
            )
        endif()
    endif()


    if(NOT NLOHMANN_JSON_ROOT)
        execute_process(
            COMMAND
                "${HOMEBREW_EXECUTABLE}"
                --prefix
                nlohmann-json
            RESULT_VARIABLE
                NLOHMANN_BREW_RESULT
            OUTPUT_VARIABLE
                NLOHMANN_BREW_PREFIX
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )

        if(NLOHMANN_BREW_RESULT EQUAL 0)
            set(
                NLOHMANN_JSON_ROOT
                "${NLOHMANN_BREW_PREFIX}"
            )
        endif()
    endif()

endif()


# ---------------------------------------------------------
# ONNX Runtime
# ---------------------------------------------------------

find_path(
    ONNXRUNTIME_INCLUDE_DIR
    NAMES
        onnxruntime_cxx_api.h
    HINTS
        "${ONNXRUNTIME_ROOT}"
        "$ENV{ONNXRUNTIME_ROOT}"
    PATH_SUFFIXES
        include/onnxruntime
        include
    REQUIRED
)

find_library(
    ONNXRUNTIME_LIBRARY
    NAMES
        onnxruntime
    HINTS
        "${ONNXRUNTIME_ROOT}"
        "$ENV{ONNXRUNTIME_ROOT}"
    PATH_SUFFIXES
        lib
    REQUIRED
)


add_library(
    onnxruntime_external
    INTERFACE
)

target_include_directories(
    onnxruntime_external
    INTERFACE
    "${ONNXRUNTIME_INCLUDE_DIR}"
)

target_link_libraries(
    onnxruntime_external
    INTERFACE
    "${ONNXRUNTIME_LIBRARY}"
)


# ---------------------------------------------------------
# nlohmann-json
# ---------------------------------------------------------

find_path(
    NLOHMANN_JSON_INCLUDE_DIR
    NAMES
        nlohmann/json.hpp
    HINTS
        "${NLOHMANN_JSON_ROOT}"
        "$ENV{NLOHMANN_JSON_ROOT}"
    PATH_SUFFIXES
        include
    REQUIRED
)


add_library(
    nlohmann_json_external
    INTERFACE
)

target_include_directories(
    nlohmann_json_external
    INTERFACE
    "${NLOHMANN_JSON_INCLUDE_DIR}"
)


# ---------------------------------------------------------
# Preprocessing library
# ---------------------------------------------------------

add_library(
    preprocessor
    STATIC
    src/preprocessor.cpp
)

target_include_directories(
    preprocessor
    PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

target_link_libraries(
    preprocessor
    PUBLIC
    nlohmann_json_external
)


# ---------------------------------------------------------
# Executables
# ---------------------------------------------------------

add_executable(
    onnx_smoke_test
    src/onnx_smoke_test.cpp
)

target_link_libraries(
    onnx_smoke_test
    PRIVATE
    onnxruntime_external
)


add_executable(
    inference
    src/inference.cpp
)

target_link_libraries(
    inference
    PRIVATE
    onnxruntime_external
)


add_executable(
    benchmark_cpp
    src/benchmark.cpp
)

target_link_libraries(
    benchmark_cpp
    PRIVATE
    onnxruntime_external
)


add_executable(
    memory_benchmark_cpp
    src/memory_benchmark.cpp
)

target_link_libraries(
    memory_benchmark_cpp
    PRIVATE
    onnxruntime_external
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

target_link_libraries(
    inference_end_to_end
    PRIVATE
    preprocessor
    onnxruntime_external
)


add_executable(
    benchmark_end_to_end
    src/benchmark_end_to_end.cpp
)

target_link_libraries(
    benchmark_end_to_end
    PRIVATE
    preprocessor
    onnxruntime_external
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


# ---------------------------------------------------------
# Configuration summary
# ---------------------------------------------------------

message(
    STATUS
    "ONNX Runtime include: ${ONNXRUNTIME_INCLUDE_DIR}"
)

message(
    STATUS
    "ONNX Runtime library: ${ONNXRUNTIME_LIBRARY}"
)

message(
    STATUS
    "nlohmann-json include: ${NLOHMANN_JSON_INCLUDE_DIR}"
)