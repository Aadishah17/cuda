#pragma once

#include <cuda_dl/core/cuda_error.cuh>
#include <cuda_dl/core/device_tensor.cuh>
#include <cuda_dl/core/launch_config.cuh>

#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace cuda_dl::ops {
namespace detail {

// --- ReLU Kernels ---

static __global__ void relu_forward_kernel(const float* const input, float* const output, const std::size_t element_count)
{
    const std::size_t index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= element_count) {
        return;
    }

    const float val = input[index];
    output[index] = val > 0.0F ? val : 0.0F;
}

static __global__ void relu_backward_kernel(
    const float* const input,
    const float* const upstream_grad,
    float* const downstream_grad,
    const std::size_t element_count)
{
    const std::size_t index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= element_count) {
        return;
    }

    downstream_grad[index] = input[index] > 0.0F ? upstream_grad[index] : 0.0F;
}

// --- Sigmoid Kernels ---

static __global__ void sigmoid_forward_kernel(const float* const input, float* const output, const std::size_t element_count)
{
    const std::size_t index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= element_count) {
        return;
    }

    // Sigmoid: 1 / (1 + exp(-x))
    output[index] = 1.0F / (1.0F + ::expf(-input[index]));
}

static __global__ void sigmoid_backward_kernel(
    const float* const output,
    const float* const upstream_grad,
    float* const downstream_grad,
    const std::size_t element_count)
{
    const std::size_t index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= element_count) {
        return;
    }

    // Sigmoid derivative using output y: dy * y * (1 - y)
    const float y = output[index];
    downstream_grad[index] = upstream_grad[index] * y * (1.0F - y);
}

// --- Tanh Kernels ---

static __global__ void tanh_forward_kernel(const float* const input, float* const output, const std::size_t element_count)
{
    const std::size_t index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= element_count) {
        return;
    }

    output[index] = ::tanhf(input[index]);
}

static __global__ void tanh_backward_kernel(
    const float* const output,
    const float* const upstream_grad,
    float* const downstream_grad,
    const std::size_t element_count)
{
    const std::size_t index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= element_count) {
        return;
    }

    // Tanh derivative using output y: dy * (1 - y^2)
    const float y = output[index];
    downstream_grad[index] = upstream_grad[index] * (1.0F - (y * y));
}

} // namespace detail

// --- High-level Wrapper Functions ---

// ReLU
inline cuda_dl::core::DeviceTensor relu_forward(const cuda_dl::core::DeviceTensor& input)
{
    cuda_dl::core::DeviceTensor output(input.shape(), input.dtype());
    const std::size_t count = input.element_count();
    if (count == 0) {
        return output;
    }

    const cuda_dl::core::LaunchConfig1D launch = cuda_dl::core::make_1d_launch_config(count);
    detail::relu_forward_kernel<<<launch.blocks_per_grid, launch.threads_per_block>>>(
        input.data(),
        output.data(),
        count);

    CUDADL_CUDA_CHECK_LAST_KERNEL("relu_forward_kernel");
    CUDADL_CUDA_SYNCHRONIZE("relu_forward_kernel completion");

    return output;
}

inline cuda_dl::core::DeviceTensor relu_backward(
    const cuda_dl::core::DeviceTensor& input,
    const cuda_dl::core::DeviceTensor& upstream_grad)
{
    if (input.shape().dimensions() != upstream_grad.shape().dimensions()) {
        throw std::invalid_argument("Input and upstream gradient shapes must match in relu_backward");
    }

    cuda_dl::core::DeviceTensor downstream_grad(input.shape(), input.dtype());
    const std::size_t count = input.element_count();
    if (count == 0) {
        return downstream_grad;
    }

    const cuda_dl::core::LaunchConfig1D launch = cuda_dl::core::make_1d_launch_config(count);
    detail::relu_backward_kernel<<<launch.blocks_per_grid, launch.threads_per_block>>>(
        input.data(),
        upstream_grad.data(),
        downstream_grad.data(),
        count);

    CUDADL_CUDA_CHECK_LAST_KERNEL("relu_backward_kernel");
    CUDADL_CUDA_SYNCHRONIZE("relu_backward_kernel completion");

    return downstream_grad;
}

// Sigmoid
inline cuda_dl::core::DeviceTensor sigmoid_forward(const cuda_dl::core::DeviceTensor& input)
{
    cuda_dl::core::DeviceTensor output(input.shape(), input.dtype());
    const std::size_t count = input.element_count();
    if (count == 0) {
        return output;
    }

    const cuda_dl::core::LaunchConfig1D launch = cuda_dl::core::make_1d_launch_config(count);
    detail::sigmoid_forward_kernel<<<launch.blocks_per_grid, launch.threads_per_block>>>(
        input.data(),
        output.data(),
        count);

    CUDADL_CUDA_CHECK_LAST_KERNEL("sigmoid_forward_kernel");
    CUDADL_CUDA_SYNCHRONIZE("sigmoid_forward_kernel completion");

    return output;
}

inline cuda_dl::core::DeviceTensor sigmoid_backward(
    const cuda_dl::core::DeviceTensor& output,
    const cuda_dl::core::DeviceTensor& upstream_grad)
{
    if (output.shape().dimensions() != upstream_grad.shape().dimensions()) {
        throw std::invalid_argument("Output and upstream gradient shapes must match in sigmoid_backward");
    }

    cuda_dl::core::DeviceTensor downstream_grad(output.shape(), output.dtype());
    const std::size_t count = output.element_count();
    if (count == 0) {
        return downstream_grad;
    }

    const cuda_dl::core::LaunchConfig1D launch = cuda_dl::core::make_1d_launch_config(count);
    detail::sigmoid_backward_kernel<<<launch.blocks_per_grid, launch.threads_per_block>>>(
        output.data(),
        upstream_grad.data(),
        downstream_grad.data(),
        count);

    CUDADL_CUDA_CHECK_LAST_KERNEL("sigmoid_backward_kernel");
    CUDADL_CUDA_SYNCHRONIZE("sigmoid_backward_kernel completion");

    return downstream_grad;
}

// Tanh
inline cuda_dl::core::DeviceTensor tanh_forward(const cuda_dl::core::DeviceTensor& input)
{
    cuda_dl::core::DeviceTensor output(input.shape(), input.dtype());
    const std::size_t count = input.element_count();
    if (count == 0) {
        return output;
    }

    const cuda_dl::core::LaunchConfig1D launch = cuda_dl::core::make_1d_launch_config(count);
    detail::tanh_forward_kernel<<<launch.blocks_per_grid, launch.threads_per_block>>>(
        input.data(),
        output.data(),
        count);

    CUDADL_CUDA_CHECK_LAST_KERNEL("tanh_forward_kernel");
    CUDADL_CUDA_SYNCHRONIZE("tanh_forward_kernel completion");

    return output;
}

inline cuda_dl::core::DeviceTensor tanh_backward(
    const cuda_dl::core::DeviceTensor& output,
    const cuda_dl::core::DeviceTensor& upstream_grad)
{
    if (output.shape().dimensions() != upstream_grad.shape().dimensions()) {
        throw std::invalid_argument("Output and upstream gradient shapes must match in tanh_backward");
    }

    cuda_dl::core::DeviceTensor downstream_grad(output.shape(), output.dtype());
    const std::size_t count = output.element_count();
    if (count == 0) {
        return downstream_grad;
    }

    const cuda_dl::core::LaunchConfig1D launch = cuda_dl::core::make_1d_launch_config(count);
    detail::tanh_backward_kernel<<<launch.blocks_per_grid, launch.threads_per_block>>>(
        output.data(),
        upstream_grad.data(),
        downstream_grad.data(),
        count);

    CUDADL_CUDA_CHECK_LAST_KERNEL("tanh_backward_kernel");
    CUDADL_CUDA_SYNCHRONIZE("tanh_backward_kernel completion");

    return downstream_grad;
}

} // namespace cuda_dl::ops
