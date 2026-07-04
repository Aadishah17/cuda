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

add_executable(hello_cuda examples/phase1/hello_cuda.cu)

set_target_properties(
    hello_cuda
    PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
)

if(MSVC)
    target_compile_options(
        hello_cuda
        PRIVATE
            "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd4211>"
    )
endif()
