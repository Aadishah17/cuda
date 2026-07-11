#pragma once

#include <cuda_dl/core/cuda_error.cuh>
#include <cuda_dl/core/device_tensor.cuh>
#include <cuda_dl/core/launch_config.cuh>

#include <cstddef>
#include <stdexcept>

namespace cuda_dl::ops {
namespace detail {

// SGD update kernel
static __global__ void sgd_step_kernel(
    float* const data,
    const float* const grad,
    const float lr,
    const float weight_decay,
    const std::size_t size)
{
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        float g = grad[index];
        if (weight_decay != 0.0F) {
            g += weight_decay * data[index];
        }
        data[index] -= lr * g;
    }
}

// SGD with Momentum update kernel
static __global__ void sgd_momentum_step_kernel(
    float* const data,
    const float* const grad,
    float* const velocity,
    const float lr,
    const float momentum,
    const float weight_decay,
    const std::size_t size)
{
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        float g = grad[index];
        if (weight_decay != 0.0F) {
            g += weight_decay * data[index];
        }
        // Update velocity: v = beta * v + g
        const float v_new = momentum * velocity[index] + g;
        velocity[index] = v_new;
        // Update parameters: theta = theta - lr * v
        data[index] -= lr * v_new;
    }
}

// RMSProp update kernel
static __global__ void rmsprop_step_kernel(
    float* const data,
    const float* const grad,
    float* const square_avg,
    const float lr,
    const float alpha,
    const float eps,
    const float weight_decay,
    const std::size_t size)
{
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        float g = grad[index];
        if (weight_decay != 0.0F) {
            g += weight_decay * data[index];
        }
        // Update square average: v = alpha * v + (1 - alpha) * g^2
        const float v_new = alpha * square_avg[index] + (1.0F - alpha) * g * g;
        square_avg[index] = v_new;
        // Update parameters: theta = theta - lr * g / (sqrt(v) + eps)
        data[index] -= lr * g / (::sqrtf(v_new) + eps);
    }
}

// Adam/AdamW update kernel
static __global__ void adam_step_kernel(
    float* const data,
    const float* const grad,
    float* const m,
    float* const v,
    const float lr,
    const float beta1,
    const float beta2,
    const float eps,
    const float bias_correction1,
    const float bias_correction2,
    const float weight_decay,
    const bool decoupled_decay,
    const std::size_t size)
{
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        float g = grad[index];
        float param = data[index];

        if (weight_decay != 0.0F) {
            if (decoupled_decay) {
                // AdamW: weight decay applied directly to parameter
                param -= lr * weight_decay * param;
            } else {
                // Adam: weight decay applied to gradient as L2 regularization
                g += weight_decay * param;
            }
        }

        // Update biased first moment estimate: m = beta1 * m + (1 - beta1) * g
        const float m_new = beta1 * m[index] + (1.0F - beta1) * g;
        m[index] = m_new;

        // Update biased second raw moment estimate: v = beta2 * v + (1 - beta2) * g^2
        const float v_new = beta2 * v[index] + (1.0F - beta2) * g * g;
        v[index] = v_new;

        // Compute bias-corrected first and second moment estimates
        const float m_hat = m_new / bias_correction1;
        const float v_hat = v_new / bias_correction2;

        // Update parameters
        data[index] = param - lr * m_hat / (::sqrtf(v_hat) + eps);
    }
}

} // namespace detail

// C++ wrappers for launching optimizer kernels

inline void sgd_step(
    cuda_dl::core::DeviceTensor& data,
    const cuda_dl::core::DeviceTensor& grad,
    const float lr,
    const float weight_decay)
{
    if (data.shape().dimensions() != grad.shape().dimensions()) {
        throw std::invalid_argument("sgd_step: shape mismatch between data and gradient");
    }

    const std::size_t size = data.element_count();
    if (size == 0) return;

    const cuda_dl::core::LaunchConfig1D config = cuda_dl::core::make_1d_launch_config(size);
    detail::sgd_step_kernel<<<config.blocks_per_grid, config.threads_per_block>>>(
        data.data(),
        grad.data(),
        lr,
        weight_decay,
        size
    );
    CUDADL_CUDA_CHECK_LAST_KERNEL("sgd_step_kernel");
}

inline void sgd_momentum_step(
    cuda_dl::core::DeviceTensor& data,
    const cuda_dl::core::DeviceTensor& grad,
    cuda_dl::core::DeviceTensor& velocity,
    const float lr,
    const float momentum,
    const float weight_decay)
{
    if (data.shape().dimensions() != grad.shape().dimensions() ||
        data.shape().dimensions() != velocity.shape().dimensions()) {
        throw std::invalid_argument("sgd_momentum_step: shape mismatch between parameters");
    }

    const std::size_t size = data.element_count();
    if (size == 0) return;

    const cuda_dl::core::LaunchConfig1D config = cuda_dl::core::make_1d_launch_config(size);
    detail::sgd_momentum_step_kernel<<<config.blocks_per_grid, config.threads_per_block>>>(
        data.data(),
        grad.data(),
        velocity.data(),
        lr,
        momentum,
        weight_decay,
        size
    );
    CUDADL_CUDA_CHECK_LAST_KERNEL("sgd_momentum_step_kernel");
}

inline void rmsprop_step(
    cuda_dl::core::DeviceTensor& data,
    const cuda_dl::core::DeviceTensor& grad,
    cuda_dl::core::DeviceTensor& square_avg,
    const float lr,
    const float alpha,
    const float eps,
    const float weight_decay)
{
    if (data.shape().dimensions() != grad.shape().dimensions() ||
        data.shape().dimensions() != square_avg.shape().dimensions()) {
        throw std::invalid_argument("rmsprop_step: shape mismatch between parameters");
    }

    const std::size_t size = data.element_count();
    if (size == 0) return;

    const cuda_dl::core::LaunchConfig1D config = cuda_dl::core::make_1d_launch_config(size);
    detail::rmsprop_step_kernel<<<config.blocks_per_grid, config.threads_per_block>>>(
        data.data(),
        grad.data(),
        square_avg.data(),
        lr,
        alpha,
        eps,
        weight_decay,
        size
    );
    CUDADL_CUDA_CHECK_LAST_KERNEL("rmsprop_step_kernel");
}

inline void adam_step(
    cuda_dl::core::DeviceTensor& data,
    const cuda_dl::core::DeviceTensor& grad,
    cuda_dl::core::DeviceTensor& m,
    cuda_dl::core::DeviceTensor& v,
    const float lr,
    const float beta1,
    const float beta2,
    const float eps,
    const float bias_correction1,
    const float bias_correction2,
    const float weight_decay,
    const bool decoupled_decay)
{
    if (data.shape().dimensions() != grad.shape().dimensions() ||
        data.shape().dimensions() != m.shape().dimensions() ||
        data.shape().dimensions() != v.shape().dimensions()) {
        throw std::invalid_argument("adam_step: shape mismatch between parameters");
    }

    const std::size_t size = data.element_count();
    if (size == 0) return;

    const cuda_dl::core::LaunchConfig1D config = cuda_dl::core::make_1d_launch_config(size);
    detail::adam_step_kernel<<<config.blocks_per_grid, config.threads_per_block>>>(
        data.data(),
        grad.data(),
        m.data(),
        v.data(),
        lr,
        beta1,
        beta2,
        eps,
        bias_correction1,
        bias_correction2,
        weight_decay,
        decoupled_decay,
        size
    );
    CUDADL_CUDA_CHECK_LAST_KERNEL("adam_step_kernel");
}

} // namespace cuda_dl::ops
