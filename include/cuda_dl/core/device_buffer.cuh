#pragma once

#include <cuda_dl/core/cuda_error.cuh>

#include <cuda_runtime.h>

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace cuda_dl::core {

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;

    explicit DeviceBuffer(const std::size_t element_count)
        : element_count_(element_count)
        , size_bytes_(checked_size_bytes(element_count))
    {
        if (size_bytes_ == 0) {
            return;
        }

        void* allocation = nullptr;
        CUDADL_CUDA_CHECK(cudaMalloc(&allocation, size_bytes_));
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

    bool empty() const noexcept
    {
        return element_count_ == 0;
    }

    void copy_from_host(const T* source, const std::size_t element_count)
    {
        validate_copy(source, element_count, "copy_from_host");
        if (element_count == 0) {
            return;
        }

        CUDADL_CUDA_CHECK(cudaMemcpy(data_, source, checked_size_bytes(element_count), cudaMemcpyHostToDevice));
    }

    void copy_to_host(T* destination, const std::size_t element_count) const
    {
        validate_copy(destination, element_count, "copy_to_host");
        if (element_count == 0) {
            return;
        }

        CUDADL_CUDA_CHECK(cudaMemcpy(destination, data_, checked_size_bytes(element_count), cudaMemcpyDeviceToHost));
    }

    void zero()
    {
        if (size_bytes_ == 0) {
            return;
        }

        // cudaMemset writes bytes in device global memory. It is correct here
        // for bitwise zero initialization, not for arbitrary typed values.
        CUDADL_CUDA_CHECK(cudaMemset(data_, 0, size_bytes_));
    }

private:
    static std::size_t checked_size_bytes(const std::size_t element_count)
    {
        if (element_count > (std::numeric_limits<std::size_t>::max() / sizeof(T))) {
            throw std::length_error("device allocation size overflow");
        }

        return element_count * sizeof(T);
    }

    template <typename Pointer>
    void validate_copy(Pointer pointer, const std::size_t element_count, const char* operation) const
    {
        if (element_count > element_count_) {
            throw std::length_error(std::string(operation) + " exceeds device buffer size");
        }

        if (element_count > 0 && pointer == nullptr) {
            throw std::invalid_argument(std::string(operation) + " received a null host pointer");
        }
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

} // namespace cuda_dl::core
