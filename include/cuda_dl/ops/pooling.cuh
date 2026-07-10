#pragma once

#include <cuda_dl/core/cuda_error.cuh>
#include <cuda_dl/core/device_buffer.cuh>
#include <cuda_dl/core/device_tensor.cuh>
#include <cuda_dl/core/launch_config.cuh>

#include <cstddef>
#include <stdexcept>
#include <utility>

namespace cuda_dl::ops {
namespace detail {

// MaxPool2D Forward GPU Kernel
static __global__ void maxpool2d_forward_kernel(
    const float* const input,
    float* const output,
    int* const argmax,
    const std::size_t B, const std::size_t C, const std::size_t H, const std::size_t W,
    const std::size_t PH, const std::size_t PW,
    const std::size_t P, const std::size_t S,
    const std::size_t OH, const std::size_t OW,
    const std::size_t total_elements)
{
    const std::size_t index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= total_elements) {
        return;
    }

    // Unflatten index to (b, c, oh, ow)
    std::size_t temp = index;
    const std::size_t ow_coord = temp % OW;
    temp /= OW;
    const std::size_t oh_coord = temp % OH;
    temp /= OH;
    const std::size_t c_coord = temp % C;
    const std::size_t b_coord = temp / C;

    float max_val = -3.402823466e+38F; // -FLT_MAX
    int max_idx = -1;

    for (std::size_t ky = 0; ky < PH; ++ky) {
        const int in_y = static_cast<int>(oh_coord * S) + static_cast<int>(ky) - static_cast<int>(P);
        if (in_y < 0 || in_y >= static_cast<int>(H)) {
            continue;
        }

        for (std::size_t kx = 0; kx < PW; ++kx) {
            const int in_x = static_cast<int>(ow_coord * S) + static_cast<int>(kx) - static_cast<int>(P);
            if (in_x < 0 || in_x >= static_cast<int>(W)) {
                continue;
            }

            const std::size_t in_offset = (((b_coord * C + c_coord) * H + static_cast<std::size_t>(in_y)) * W) + static_cast<std::size_t>(in_x);
            const float val = input[in_offset];
            if (val > max_val) {
                max_val = val;
                max_idx = static_cast<int>(in_offset);
            }
        }
    }

    output[index] = max_val;
    argmax[index] = max_idx;
}

// MaxPool2D Backward GPU Kernel
static __global__ void maxpool2d_backward_kernel(
    const float* const upstream_grad,
    const int* const argmax,
    float* const downstream_grad,
    const std::size_t total_output_elements)
{
    const std::size_t index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= total_output_elements) {
        return;
    }

    const int target_idx = argmax[index];
    if (target_idx >= 0) {
        atomicAdd(&downstream_grad[target_idx], upstream_grad[index]);
    }
}

} // namespace detail

struct MaxPool2DForwardResult {
    cuda_dl::core::DeviceTensor output;
    cuda_dl::core::DeviceBuffer<int> argmax;
};

// High-level Forward Pass Wrapper
inline MaxPool2DForwardResult maxpool2d_forward(
    const cuda_dl::core::DeviceTensor& input, // [B, C, H, W]
    const std::size_t pool_h,
    const std::size_t pool_w,
    const std::size_t padding = 0,
    const std::size_t stride = 2)
{
    if (input.rank() != 4) {
        throw std::invalid_argument("maxpool2d_forward: input must be a rank-4 tensor");
    }
    if (pool_h == 0 || pool_w == 0 || stride == 0) {
        throw std::invalid_argument("maxpool2d_forward: invalid pool or stride size");
    }

    const std::size_t B = input.shape().dimension(0);
    const std::size_t C = input.shape().dimension(1);
    const std::size_t H = input.shape().dimension(2);
    const std::size_t W = input.shape().dimension(3);

    const std::size_t OH = ((H + 2 * padding - pool_h) / stride) + 1;
    const std::size_t OW = ((W + 2 * padding - pool_w) / stride) + 1;

    cuda_dl::core::DeviceTensor output({B, C, OH, OW}, input.dtype());
    const std::size_t total_elements = output.element_count();

    cuda_dl::core::DeviceBuffer<int> argmax(total_elements);

    if (total_elements > 0) {
        const cuda_dl::core::LaunchConfig1D launch = cuda_dl::core::make_1d_launch_config(total_elements);
        detail::maxpool2d_forward_kernel<<<launch.blocks_per_grid, launch.threads_per_block>>>(
            input.data(),
            output.data(),
            argmax.get(),
            B, C, H, W,
            pool_h, pool_w,
            padding, stride,
            OH, OW,
            total_elements);

        CUDADL_CUDA_CHECK_LAST_KERNEL("maxpool2d_forward_kernel");
        CUDADL_CUDA_SYNCHRONIZE("maxpool2d_forward_kernel completion");
    }

    return MaxPool2DForwardResult{std::move(output), std::move(argmax)};
}

// High-level Backward Pass Wrapper
inline cuda_dl::core::DeviceTensor maxpool2d_backward(
    const cuda_dl::core::DeviceTensor& input, // [B, C, H, W]
    const cuda_dl::core::DeviceTensor& upstream_grad, // [B, C, OH, OW]
    const cuda_dl::core::DeviceBuffer<int>& argmax)
{
    if (input.rank() != 4 || upstream_grad.rank() != 4) {
        throw std::invalid_argument("maxpool2d_backward: operand ranks must be 4");
    }
    if (upstream_grad.element_count() != argmax.size()) {
        throw std::invalid_argument("maxpool2d_backward: upstream gradient and argmax size mismatch");
    }

    cuda_dl::core::DeviceTensor downstream_grad(input.shape(), input.dtype());
    downstream_grad.zero();

    const std::size_t total_output_elements = upstream_grad.element_count();
    if (total_output_elements > 0) {
        const cuda_dl::core::LaunchConfig1D launch = cuda_dl::core::make_1d_launch_config(total_output_elements);
        detail::maxpool2d_backward_kernel<<<launch.blocks_per_grid, launch.threads_per_block>>>(
            upstream_grad.data(),
            argmax.get(),
            downstream_grad.data(),
            total_output_elements);

        CUDADL_CUDA_CHECK_LAST_KERNEL("maxpool2d_backward_kernel");
        CUDADL_CUDA_SYNCHRONIZE("maxpool2d_backward_kernel completion");
    }

    return downstream_grad;
}

} // namespace cuda_dl::ops
