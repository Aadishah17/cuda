#pragma once

#include <cuda_dl/core/device_buffer.cuh>
#include <cuda_dl/core/tensor.hpp>

#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <utility>

namespace cuda_dl::core {

class DeviceTensor {
public:
    explicit DeviceTensor(Tensor metadata)
        : metadata_(std::move(metadata))
        , storage_(metadata_.element_count())
    {
        if (metadata_.dtype() != DType::Float32) {
            throw std::invalid_argument("DeviceTensor currently supports float32 tensors only");
        }
    }

    DeviceTensor(TensorShape shape, const DType dtype)
        : DeviceTensor(Tensor(std::move(shape), dtype))
    {
    }

    DeviceTensor(std::initializer_list<std::size_t> dimensions, const DType dtype)
        : DeviceTensor(TensorShape(dimensions), dtype)
    {
    }

    DeviceTensor(const DeviceTensor&) = delete;
    DeviceTensor& operator=(const DeviceTensor&) = delete;

    DeviceTensor(DeviceTensor&&) noexcept = default;
    DeviceTensor& operator=(DeviceTensor&&) noexcept = default;

    const Tensor& metadata() const noexcept
    {
        return metadata_;
    }

    const TensorShape& shape() const noexcept
    {
        return metadata_.shape();
    }

    DType dtype() const noexcept
    {
        return metadata_.dtype();
    }

    std::size_t rank() const noexcept
    {
        return metadata_.rank();
    }

    std::size_t element_count() const noexcept
    {
        return metadata_.element_count();
    }

    std::size_t bytes() const noexcept
    {
        return metadata_.bytes();
    }

    float* data() noexcept
    {
        return storage_.get();
    }

    const float* data() const noexcept
    {
        return storage_.get();
    }

    void copy_from_host(const float* source, const std::size_t element_count)
    {
        storage_.copy_from_host(source, element_count);
    }

    void copy_to_host(float* destination, const std::size_t element_count) const
    {
        storage_.copy_to_host(destination, element_count);
    }

    void zero()
    {
        storage_.zero();
    }

private:
    Tensor metadata_;
    DeviceBuffer<float> storage_;
};

} // namespace cuda_dl::core
