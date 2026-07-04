#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace cuda_dl::examples {

inline void check_cuda(const cudaError_t result, const char* const operation)
{
    if (result == cudaSuccess) {
        return;
    }

    throw std::runtime_error(
        std::string(operation) + " failed: " + cudaGetErrorString(result));
}

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(const std::size_t element_count)
        : element_count_(element_count)
        , size_bytes_(checked_size_bytes(element_count))
    {
        if (size_bytes_ == 0) {
            return;
        }

        void* allocation = nullptr;
        check_cuda(cudaMalloc(&allocation, size_bytes_), "cudaMalloc");
        data_ = static_cast<T*>(allocation);
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : data_(std::exchange(other.data_, nullptr))
        , element_count_(std::exchange(other.element_count_, 0))
        , size_bytes_(std::exchange(other.size_bytes_, 0))
    {
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept
    {
        if (this != &other) {
            release();
            data_ = std::exchange(other.data_, nullptr);
            element_count_ = std::exchange(other.element_count_, 0);
            size_bytes_ = std::exchange(other.size_bytes_, 0);
        }

        return *this;
    }

    ~DeviceBuffer() noexcept
    {
        release();
    }

    T* get() noexcept
    {
        return data_;
    }

    const T* get() const noexcept
    {
        return data_;
    }

    std::size_t size() const noexcept
    {
        return element_count_;
    }

    std::size_t bytes() const noexcept
    {
        return size_bytes_;
    }

private:
    static std::size_t checked_size_bytes(const std::size_t element_count)
    {
        if (element_count > (std::numeric_limits<std::size_t>::max() / sizeof(T))) {
            throw std::length_error("device allocation size overflow");
        }

        return element_count * sizeof(T);
    }

    void release() noexcept
    {
        if (data_ != nullptr) {
            static_cast<void>(cudaFree(data_));
            data_ = nullptr;
        }
    }

    T* data_{nullptr};
    std::size_t element_count_{0};
    std::size_t size_bytes_{0};
};

} // namespace cuda_dl::examples
