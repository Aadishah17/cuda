#pragma once

#include <cuda_dl/core/cuda_error.cuh>
#include <cuda_dl/core/device_tensor.cuh>
#include <cuda_dl/core/launch_config.cuh>

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>

namespace cuda_dl::ops {
namespace detail {

inline std::pair<std::size_t, std::size_t> conv2d_output_spatial_shape(
    const std::size_t input_height,
    const std::size_t input_width,
    const std::size_t kernel_height,
    const std::size_t kernel_width,
    const std::size_t padding,
    const std::size_t stride)
{
    if (stride == 0) {
        throw std::invalid_argument("conv2d stride cannot be zero");
    }
    if (kernel_height == 0 || kernel_width == 0) {
        throw std::invalid_argument("conv2d kernel dimensions must be positive");
    }

    if (padding > (std::numeric_limits<std::size_t>::max() - input_height) / 2
        || padding > (std::numeric_limits<std::size_t>::max() - input_width) / 2) {
        throw std::overflow_error("conv2d padded spatial size overflow");
    }

    const std::size_t padded_height = input_height + (2 * padding);
    const std::size_t padded_width = input_width + (2 * padding);
    if (kernel_height > padded_height || kernel_width > padded_width) {
        throw std::invalid_argument("conv2d kernel exceeds padded input size");
    }

    return {
        1 + ((padded_height - kernel_height) / stride),
        1 + ((padded_width - kernel_width) / stride)};
}

// Helper to compute flat NCHW coordinate offset
__device__ inline std::size_t get_nchw_offset(
    const std::size_t b, const std::size_t c, const std::size_t h, const std::size_t w,
    const std::size_t C, const std::size_t H, const std::size_t W)
{
    return (((b * C + c) * H + h) * W + w);
}

// 2D Convolution Forward Pass GPU Kernel
static __global__ void conv2d_forward_kernel(
    const float* const input,
    const float* const weight,
    const float* const bias,
    float* const output,
    const std::size_t B, const std::size_t C, const std::size_t H, const std::size_t W,
    const std::size_t F, const std::size_t KH, const std::size_t KW,
    const std::size_t P, const std::size_t S,
    const std::size_t OH, const std::size_t OW,
    const std::size_t total_elements)
{
    const std::size_t index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= total_elements) {
        return;
    }

    // Unflatten index to (b, f, oh, ow)
    std::size_t temp = index;
    const std::size_t ow_coord = temp % OW;
    temp /= OW;
    const std::size_t oh_coord = temp % OH;
    temp /= OH;
    const std::size_t f_coord = temp % F;
    const std::size_t b_coord = temp / F;

    float sum = bias[f_coord];

    for (std::size_t c = 0; c < C; ++c) {
        for (std::size_t ky = 0; ky < KH; ++ky) {
            const int in_y = static_cast<int>(oh_coord * S) + static_cast<int>(ky) - static_cast<int>(P);
            if (in_y < 0 || in_y >= static_cast<int>(H)) {
                continue;
            }

            for (std::size_t kx = 0; kx < KW; ++kx) {
                const int in_x = static_cast<int>(ow_coord * S) + static_cast<int>(kx) - static_cast<int>(P);
                if (in_x < 0 || in_x >= static_cast<int>(W)) {
                    continue;
                }

                const std::size_t in_offset = get_nchw_offset(b_coord, c, static_cast<std::size_t>(in_y), static_cast<std::size_t>(in_x), C, H, W);
                // Weight shape: [F, C, KH, KW]
                const std::size_t weight_offset = (((f_coord * C + c) * KH + ky) * KW) + kx;

                sum += input[in_offset] * weight[weight_offset];
            }
        }
    }

    output[index] = sum;
}

// 2D Convolution Backward Input Gradient GPU Kernel (computes dX)
static __global__ void conv2d_backward_input_kernel(
    const float* const upstream_grad,
    const float* const weight,
    float* const downstream_grad,
    const std::size_t B, const std::size_t C, const std::size_t H, const std::size_t W,
    const std::size_t F, const std::size_t KH, const std::size_t KW,
    const std::size_t P, const std::size_t S,
    const std::size_t OH, const std::size_t OW,
    const std::size_t total_elements)
{
    const std::size_t index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= total_elements) {
        return;
    }

    // Unflatten index to (b, c, h, w) for dX
    std::size_t temp = index;
    const std::size_t w_coord = temp % W;
    temp /= W;
    const std::size_t h_coord = temp % H;
    temp /= H;
    const std::size_t c_coord = temp % C;
    const std::size_t b_coord = temp / C;

    float sum = 0.0F;

    for (std::size_t f = 0; f < F; ++f) {
        for (std::size_t ky = 0; ky < KH; ++ky) {
            const int padded_y = static_cast<int>(h_coord) + static_cast<int>(P);
            const int offset_y = padded_y - static_cast<int>(ky);
            if (offset_y < 0 || (offset_y % static_cast<int>(S)) != 0) {
                continue;
            }
            const std::size_t oh_coord = static_cast<std::size_t>(offset_y / static_cast<int>(S));
            if (oh_coord >= OH) {
                continue;
            }

            for (std::size_t kx = 0; kx < KW; ++kx) {
                const int padded_x = static_cast<int>(w_coord) + static_cast<int>(P);
                const int offset_x = padded_x - static_cast<int>(kx);
                if (offset_x < 0 || (offset_x % static_cast<int>(S)) != 0) {
                    continue;
                }
                const std::size_t ow_coord = static_cast<std::size_t>(offset_x / static_cast<int>(S));
                if (ow_coord >= OW) {
                    continue;
                }

                const std::size_t grad_offset = get_nchw_offset(b_coord, f, oh_coord, ow_coord, F, OH, OW);
                const std::size_t weight_offset = (((f * C + c_coord) * KH + ky) * KW) + kx;

                sum += upstream_grad[grad_offset] * weight[weight_offset];
            }
        }
    }

    downstream_grad[index] = sum;
}

// 2D Convolution Backward Weight Gradient GPU Kernel (computes dW)
static __global__ void conv2d_backward_weight_kernel(
    const float* const input,
    const float* const upstream_grad,
    float* const weight_grad,
    const std::size_t B, const std::size_t C, const std::size_t H, const std::size_t W,
    const std::size_t F, const std::size_t KH, const std::size_t KW,
    const std::size_t P, const std::size_t S,
    const std::size_t OH, const std::size_t OW,
    const std::size_t total_elements)
{
    const std::size_t index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= total_elements) {
        return;
    }

    // Unflatten index to (f, c, ky, kx) for dW
    std::size_t temp = index;
    const std::size_t kx_coord = temp % KW;
    temp /= KW;
    const std::size_t ky_coord = temp % KH;
    temp /= KH;
    const std::size_t c_coord = temp % C;
    const std::size_t f_coord = temp / C;

    float sum = 0.0F;

    for (std::size_t b = 0; b < B; ++b) {
        for (std::size_t oh = 0; oh < OH; ++oh) {
            const int in_y = static_cast<int>(oh * S) + static_cast<int>(ky_coord) - static_cast<int>(P);
            if (in_y < 0 || in_y >= static_cast<int>(H)) {
                continue;
            }

            for (std::size_t ow = 0; ow < OW; ++ow) {
                const int in_x = static_cast<int>(ow * S) + static_cast<int>(kx_coord) - static_cast<int>(P);
                if (in_x < 0 || in_x >= static_cast<int>(W)) {
                    continue;
                }

                const std::size_t in_offset = get_nchw_offset(b, c_coord, static_cast<std::size_t>(in_y), static_cast<std::size_t>(in_x), C, H, W);
                const std::size_t grad_offset = get_nchw_offset(b, f_coord, oh, ow, F, OH, OW);

                sum += upstream_grad[grad_offset] * input[in_offset];
            }
        }
    }

    weight_grad[index] = sum;
}

// 2D Convolution Backward Bias Gradient GPU Kernel (computes db)
static __global__ void conv2d_backward_bias_kernel(
    const float* const upstream_grad,
    float* const bias_grad,
    const std::size_t B, const std::size_t F,
    const std::size_t OH, const std::size_t OW)
{
    const std::size_t f = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (f >= F) {
        return;
    }

    float sum = 0.0F;
    for (std::size_t b = 0; b < B; ++b) {
        for (std::size_t oh = 0; oh < OH; ++oh) {
            for (std::size_t ow = 0; ow < OW; ++ow) {
                sum += upstream_grad[get_nchw_offset(b, f, oh, ow, F, OH, OW)];
            }
        }
    }

    bias_grad[f] = sum;
}

} // namespace detail

// High-level Forward Pass Wrapper
inline cuda_dl::core::DeviceTensor conv2d_forward(
    const cuda_dl::core::DeviceTensor& input,   // [B, C, H, W]
    const cuda_dl::core::DeviceTensor& weight,  // [F, C, KH, KW]
    const cuda_dl::core::DeviceTensor& bias,    // [F]
    const std::size_t padding = 0,
    const std::size_t stride = 1,
    cudaStream_t stream = nullptr)
{
    if (input.rank() != 4 || weight.rank() != 4 || bias.rank() != 1) {
        throw std::invalid_argument("conv2d_forward: invalid operand ranks (expecting 4, 4, 1)");
    }

    const std::size_t B = input.shape().dimension(0);
    const std::size_t C = input.shape().dimension(1);
    const std::size_t H = input.shape().dimension(2);
    const std::size_t W = input.shape().dimension(3);

    const std::size_t F = weight.shape().dimension(0);
    const std::size_t weight_c = weight.shape().dimension(1);
    const std::size_t KH = weight.shape().dimension(2);
    const std::size_t KW = weight.shape().dimension(3);

    if (C != weight_c || F != bias.shape().dimension(0)) {
        throw std::invalid_argument("conv2d_forward: channel or filter size mismatch");
    }
    const auto [OH, OW] = detail::conv2d_output_spatial_shape(H, W, KH, KW, padding, stride);

    cuda_dl::core::DeviceTensor output({B, F, OH, OW}, input.dtype());
    const std::size_t total_elements = output.element_count();
    if (total_elements == 0) {
        return output;
    }

    const cuda_dl::core::LaunchConfig1D launch = cuda_dl::core::make_1d_launch_config(total_elements);
    detail::conv2d_forward_kernel<<<launch.blocks_per_grid, launch.threads_per_block, 0, stream>>>(
        input.data(),
        weight.data(),
        bias.data(),
        output.data(),
        B, C, H, W,
        F, KH, KW,
        padding, stride,
        OH, OW,
        total_elements);

    CUDADL_CUDA_CHECK_LAST_KERNEL("conv2d_forward_kernel");
    return output;
}

// Struct to hold backward gradients
struct Conv2DBackwardResult {
    cuda_dl::core::DeviceTensor input_grad;
    cuda_dl::core::DeviceTensor weight_grad;
    cuda_dl::core::DeviceTensor bias_grad;
};

// High-level Backward Pass Wrapper
inline Conv2DBackwardResult conv2d_backward(
    const cuda_dl::core::DeviceTensor& input,          // [B, C, H, W]
    const cuda_dl::core::DeviceTensor& weight,         // [F, C, KH, KW]
    const cuda_dl::core::DeviceTensor& upstream_grad,  // [B, F, OH, OW]
    const std::size_t padding = 0,
    const std::size_t stride = 1,
    cudaStream_t stream = nullptr)
{
    if (input.rank() != 4 || weight.rank() != 4 || upstream_grad.rank() != 4) {
        throw std::invalid_argument("conv2d_backward: invalid operand ranks (expecting 4, 4, 4)");
    }

    const std::size_t B = input.shape().dimension(0);
    const std::size_t C = input.shape().dimension(1);
    const std::size_t H = input.shape().dimension(2);
    const std::size_t W = input.shape().dimension(3);

    const std::size_t F = weight.shape().dimension(0);
    const std::size_t weight_c = weight.shape().dimension(1);
    const std::size_t KH = weight.shape().dimension(2);
    const std::size_t KW = weight.shape().dimension(3);

    const auto [OH, OW] = detail::conv2d_output_spatial_shape(H, W, KH, KW, padding, stride);

    if (C != weight_c) {
        throw std::invalid_argument("conv2d_backward: channel dimension mismatch");
    }
    if (upstream_grad.shape().dimensions() != std::vector<std::size_t>{B, F, OH, OW}) {
        throw std::invalid_argument("conv2d_backward: upstream gradient shape mismatch");
    }

    // 1. Compute input gradient (dX)
    cuda_dl::core::DeviceTensor input_grad(input.shape(), input.dtype());
    const std::size_t dx_elements = input_grad.element_count();
    if (dx_elements > 0) {
        const cuda_dl::core::LaunchConfig1D launch_dx = cuda_dl::core::make_1d_launch_config(dx_elements);
        detail::conv2d_backward_input_kernel<<<launch_dx.blocks_per_grid, launch_dx.threads_per_block, 0, stream>>>(
            upstream_grad.data(),
            weight.data(),
            input_grad.data(),
            B, C, H, W,
            F, KH, KW,
            padding, stride,
            OH, OW,
            dx_elements);
        CUDADL_CUDA_CHECK_LAST_KERNEL("conv2d_backward_input_kernel");
    }

    // 2. Compute weight gradient (dW)
    cuda_dl::core::DeviceTensor weight_grad(weight.shape(), weight.dtype());
    const std::size_t dw_elements = weight_grad.element_count();
    if (dw_elements > 0) {
        const cuda_dl::core::LaunchConfig1D launch_dw = cuda_dl::core::make_1d_launch_config(dw_elements);
        detail::conv2d_backward_weight_kernel<<<launch_dw.blocks_per_grid, launch_dw.threads_per_block, 0, stream>>>(
            input.data(),
            upstream_grad.data(),
            weight_grad.data(),
            B, C, H, W,
            F, KH, KW,
            padding, stride,
            OH, OW,
            dw_elements);
        CUDADL_CUDA_CHECK_LAST_KERNEL("conv2d_backward_weight_kernel");
    }

    // 3. Compute bias gradient (db)
    cuda_dl::core::DeviceTensor bias_grad({F}, input.dtype());
    if (F > 0) {
        const cuda_dl::core::LaunchConfig1D launch_db = cuda_dl::core::make_1d_launch_config(F);
        detail::conv2d_backward_bias_kernel<<<launch_db.blocks_per_grid, launch_db.threads_per_block, 0, stream>>>(
            upstream_grad.data(),
            bias_grad.data(),
            B, F,
            OH, OW);
        CUDADL_CUDA_CHECK_LAST_KERNEL("conv2d_backward_bias_kernel");
    }

    return Conv2DBackwardResult{std::move(input_grad), std::move(weight_grad), std::move(bias_grad)};
}

} // namespace cuda_dl::ops
