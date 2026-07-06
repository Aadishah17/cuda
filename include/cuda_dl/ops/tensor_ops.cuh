#pragma once

#include <cuda_dl/core/cuda_error.cuh>
#include <cuda_dl/core/device_tensor.cuh>
#include <cuda_dl/core/launch_config.cuh>

#include <cstddef>
#include <stdexcept>

namespace cuda_dl::ops {
namespace detail {

static __global__ void add_same_shape_kernel(
    const float* lhs,
    const float* rhs,
    float* output,
    const std::size_t element_count)
{
    const std::size_t index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= element_count) {
        return;
    }

    output[index] = lhs[index] + rhs[index];
}

} // namespace detail

inline cuda_dl::core::DeviceTensor add(
    const cuda_dl::core::DeviceTensor& lhs,
    const cuda_dl::core::DeviceTensor& rhs)
{
    if (lhs.dtype() != rhs.dtype()) {
        throw std::invalid_argument("add requires tensors with matching dtypes");
    }

    if (lhs.shape().dimensions() != rhs.shape().dimensions()) {
        throw std::invalid_argument("add currently requires tensors with matching shapes");
    }

    cuda_dl::core::DeviceTensor output(cuda_dl::core::Tensor(lhs.shape(), lhs.dtype()));

    if (output.element_count() == 0) {
        return output;
    }

    const cuda_dl::core::LaunchConfig1D launch = cuda_dl::core::make_1d_launch_config(output.element_count());

    detail::add_same_shape_kernel<<<launch.blocks_per_grid, launch.threads_per_block>>>(
        lhs.data(),
        rhs.data(),
        output.data(),
        output.element_count());

    CUDADL_CUDA_CHECK_LAST_KERNEL("add_same_shape_kernel");

    // Early framework milestones synchronize inside ops so failures are caught
    // at the call site. Later stream support should make this policy explicit.
    CUDADL_CUDA_SYNCHRONIZE("add_same_shape_kernel completion");

    return output;
}

} // namespace cuda_dl::ops
