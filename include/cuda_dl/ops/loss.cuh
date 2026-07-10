#pragma once

#include <cuda_dl/core/cuda_error.cuh>
#include <cuda_dl/core/device_buffer.cuh>
#include <cuda_dl/core/device_tensor.cuh>
#include <cuda_dl/core/launch_config.cuh>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace cuda_dl::ops {
namespace detail {

// Stable Softmax Forward GPU Kernel (one thread per batch row)
static __global__ void softmax_forward_kernel(
    const float* const logits,
    float* const probs,
    const std::size_t batch_size,
    const std::size_t classes)
{
    const std::size_t b = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (b >= batch_size) {
        return;
    }

    const std::size_t row_offset = b * classes;

    // 1. Find max logit for numerical stability
    float max_logit = logits[row_offset];
    for (std::size_t c = 1; c < classes; ++c) {
        const float val = logits[row_offset + c];
        if (val > max_logit) {
            max_logit = val;
        }
    }

    // 2. Compute sum of exponents
    float sum_exp = 0.0F;
    for (std::size_t c = 0; c < classes; ++c) {
        const float exp_val = ::expf(logits[row_offset + c] - max_logit);
        probs[row_offset + c] = exp_val; // temporarily store exp values in probs
        sum_exp += exp_val;
    }

    // 3. Divide by sum to get final probabilities
    for (std::size_t c = 0; c < classes; ++c) {
        probs[row_offset + c] /= sum_exp;
    }
}

// Cross-Entropy Loss GPU Kernel (one thread per batch row)
static __global__ void cross_entropy_forward_kernel(
    const float* const probs,
    const int* const targets,
    float* const losses,
    const std::size_t batch_size,
    const std::size_t classes)
{
    const std::size_t b = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (b >= batch_size) {
        return;
    }

    const int target_class = targets[b];
    if (target_class < 0 || target_class >= static_cast<int>(classes)) {
        losses[b] = 0.0F;
        return;
    }

    const float prob = probs[b * classes + static_cast<std::size_t>(target_class)];
    const float eps = 1e-15F; // epsilon to prevent log(0)
    const float safe_prob = prob < eps ? eps : prob;
    losses[b] = -::logf(safe_prob);
}

// Softmax Cross-Entropy Backward GPU Kernel (computes gradient w.r.t. input logits)
static __global__ void softmax_cross_entropy_backward_kernel(
    const float* const probs,
    const int* const targets,
    float* const logits_grad,
    const std::size_t batch_size,
    const std::size_t classes,
    const std::size_t total_elements)
{
    const std::size_t index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= total_elements) {
        return;
    }

    const std::size_t b = index / classes;
    const std::size_t c = index % classes;

    const int target_class = targets[b];
    const float indicator = (static_cast<int>(c) == target_class) ? 1.0F : 0.0F;

    // dL/dx = (probs - indicator) / B
    logits_grad[index] = (probs[index] - indicator) / static_cast<float>(batch_size);
}

} // namespace detail

struct SoftmaxCrossEntropyForwardResult {
    float average_loss;
    cuda_dl::core::DeviceTensor probabilities;
};

// High-level Forward Pass Wrapper
inline SoftmaxCrossEntropyForwardResult softmax_cross_entropy_forward(
    const cuda_dl::core::DeviceTensor& logits,     // [B, C]
    const cuda_dl::core::DeviceBuffer<int>& targets) // [B]
{
    if (logits.rank() != 2) {
        throw std::invalid_argument("softmax_cross_entropy_forward: logits must be a rank-2 tensor");
    }

    const std::size_t B = logits.shape().dimension(0);
    const std::size_t C = logits.shape().dimension(1);

    if (targets.size() != B) {
        throw std::invalid_argument("softmax_cross_entropy_forward: target labels size mismatch");
    }

    cuda_dl::core::DeviceTensor probs(logits.shape(), logits.dtype());
    if (B == 0) {
        return SoftmaxCrossEntropyForwardResult{0.0F, std::move(probs)};
    }

    // 1. Calculate Softmax Probabilities
    const cuda_dl::core::LaunchConfig1D launch_sf = cuda_dl::core::make_1d_launch_config(B);
    detail::softmax_forward_kernel<<<launch_sf.blocks_per_grid, launch_sf.threads_per_block>>>(
        logits.data(),
        probs.data(),
        B, C);
    CUDADL_CUDA_CHECK_LAST_KERNEL("softmax_forward_kernel");

    // 2. Calculate Individual CE Losses
    cuda_dl::core::DeviceBuffer<float> device_losses(B);
    detail::cross_entropy_forward_kernel<<<launch_sf.blocks_per_grid, launch_sf.threads_per_block>>>(
        probs.data(),
        targets.get(),
        device_losses.get(),
        B, C);
    CUDADL_CUDA_CHECK_LAST_KERNEL("cross_entropy_forward_kernel");

    CUDADL_CUDA_SYNCHRONIZE("loss forward kernels completion");

    // 3. Download and Average Losses on Host
    std::vector<float> host_losses(B);
    device_losses.copy_to_host(host_losses.data(), B);

    float sum_loss = 0.0F;
    for (float l : host_losses) {
        sum_loss += l;
    }
    const float average_loss = sum_loss / static_cast<float>(B);

    return SoftmaxCrossEntropyForwardResult{average_loss, std::move(probs)};
}

// High-level Backward Pass Wrapper
inline cuda_dl::core::DeviceTensor softmax_cross_entropy_backward(
    const cuda_dl::core::DeviceTensor& probabilities, // [B, C]
    const cuda_dl::core::DeviceBuffer<int>& targets)  // [B]
{
    if (probabilities.rank() != 2) {
        throw std::invalid_argument("softmax_cross_entropy_backward: probabilities must be a rank-2 tensor");
    }

    const std::size_t B = probabilities.shape().dimension(0);
    const std::size_t C = probabilities.shape().dimension(1);

    if (targets.size() != B) {
        throw std::invalid_argument("softmax_cross_entropy_backward: target labels size mismatch");
    }

    cuda_dl::core::DeviceTensor logits_grad(probabilities.shape(), probabilities.dtype());
    const std::size_t total_elements = logits_grad.element_count();
    if (total_elements == 0) {
        return logits_grad;
    }

    const cuda_dl::core::LaunchConfig1D launch = cuda_dl::core::make_1d_launch_config(total_elements);
    detail::softmax_cross_entropy_backward_kernel<<<launch.blocks_per_grid, launch.threads_per_block>>>(
        probabilities.data(),
        targets.get(),
        logits_grad.data(),
        B, C,
        total_elements);

    CUDADL_CUDA_CHECK_LAST_KERNEL("softmax_cross_entropy_backward_kernel");
    CUDADL_CUDA_SYNCHRONIZE("softmax_cross_entropy_backward_kernel completion");

    return logits_grad;
}

} // namespace cuda_dl::ops
