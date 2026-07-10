cmake_minimum_required(VERSION 4.0)

# Let CMake ask the installed NVIDIA driver/toolkit for the local GPU target.
# On this machine, nvidia-smi reports compute capability 12.0 for the RTX 5070.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures to compile for")
endif()

project(
    cuda_deep_learning_framework
    VERSION 0.1.0
    LANGUAGES CXX CUDA
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

if(MSVC)
    add_compile_options(
        "$<$<COMPILE_LANGUAGE:CXX>:/W4>"
        "$<$<COMPILE_LANGUAGE:CXX>:/permissive->"
        "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4>"
    )
else()
    add_compile_options(
        "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
        "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wpedantic>"
    )
endif()

function(add_cuda_example target_name source_file)
    add_executable(${target_name} ${source_file})

    target_include_directories(
        ${target_name}
        PRIVATE
            ${PROJECT_SOURCE_DIR}/include
            ${PROJECT_SOURCE_DIR}/examples/phase1
    )

    set_target_properties(
        ${target_name}
        PROPERTIES
            CUDA_SEPARABLE_COMPILATION OFF
    )

    if(MSVC)
        target_compile_options(
            ${target_name}
            PRIVATE
                "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd4211>"
        )
    endif()
endfunction()

function(add_cpp_example target_name source_file)
    add_executable(${target_name} ${source_file})

    target_include_directories(
        ${target_name}
        PRIVATE
            ${PROJECT_SOURCE_DIR}/include
    )
endfunction()

add_cuda_example(env_verify examples/phase1/env_verify.cu)
add_cuda_example(hello_cuda examples/phase1/hello_cuda.cu)
add_cuda_example(vector_add examples/phase1/vector_add.cu)
add_cuda_example(matrix_add examples/phase1/matrix_add.cu)
add_cuda_example(matmul_naive examples/phase1/matmul_naive.cu)
add_cuda_example(matmul_tiled examples/phase1/matmul_tiled.cu)
add_cuda_example(cuda_error_handling examples/phase1/cuda_error_handling.cu)
add_cuda_example(benchmarking examples/phase1/benchmarking.cu)

add_cpp_example(tensor_metadata examples/phase2/tensor_metadata.cpp)
add_cuda_example(gpu_memory_manager examples/phase2/gpu_memory_manager.cu)
add_cuda_example(tensor_allocation examples/phase2/tensor_allocation.cu)
add_cpp_example(broadcasting examples/phase2/broadcasting.cpp)
add_cuda_example(tensor_add examples/phase2/tensor_add.cu)
add_cpp_example(cuda_utility_layer examples/phase2/cuda_utility_layer.cpp)
add_cuda_example(tensor_binary_ops examples/phase2/tensor_binary_ops.cu)
add_cuda_example(tensor_matmul examples/phase2/tensor_matmul.cu)

add_cuda_example(activation_ops examples/phase3/activation_ops.cu)
add_cuda_example(linear_ops examples/phase3/linear_ops.cu)
add_cuda_example(conv2d_ops examples/phase3/conv2d_ops.cu)
add_cuda_example(pooling_ops examples/phase3/pooling_ops.cu)
add_cuda_example(loss_ops examples/phase3/loss_ops.cu)





