#pragma once

#include <cuda_dl/core/device_buffer.cuh>

namespace cuda_dl::examples {

template <typename T>
using DeviceBuffer = cuda_dl::core::DeviceBuffer<T>;

} // namespace cuda_dl::examples
