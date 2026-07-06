#pragma once

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace cuda_dl::core {

class CudaException final : public std::runtime_error {
public:
    CudaException(
        const cudaError_t error,
        const char* const operation,
        const char* const file,
        const int line)
        : std::runtime_error(format_message(error, operation, file, line))
        , error_(error)
        , operation_(operation)
        , file_(file)
        , line_(line)
    {
    }

    cudaError_t error() const noexcept
    {
        return error_;
    }

    const std::string& operation() const noexcept
    {
        return operation_;
    }

    const std::string& file() const noexcept
    {
        return file_;
    }

    int line() const noexcept
    {
        return line_;
    }

private:
    static std::string format_message(
        const cudaError_t error,
        const char* const operation,
        const char* const file,
        const int line)
    {
        return std::string("CUDA error at ") + file + ":" + std::to_string(line)
            + " while running `" + operation + "`: " + cudaGetErrorString(error);
    }

    cudaError_t error_;
    std::string operation_;
    std::string file_;
    int line_;
};

inline void check_cuda(
    const cudaError_t result,
    const char* const operation,
    const char* const file,
    const int line)
{
    if (result == cudaSuccess) {
        return;
    }

    throw CudaException(result, operation, file, line);
}

} // namespace cuda_dl::core

#define CUDADL_CUDA_CHECK(call) \
    ::cuda_dl::core::check_cuda((call), #call, __FILE__, __LINE__)

#define CUDADL_CUDA_CHECK_LAST_KERNEL(kernel_name) \
    ::cuda_dl::core::check_cuda((cudaGetLastError()), (kernel_name), __FILE__, __LINE__)

#define CUDADL_CUDA_SYNCHRONIZE(operation) \
    ::cuda_dl::core::check_cuda((cudaDeviceSynchronize()), (operation), __FILE__, __LINE__)
