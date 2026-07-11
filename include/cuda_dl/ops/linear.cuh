#pragma once

#include <cuda_dl/core/cuda_error.cuh>
#include <cuda_dl/core/device_tensor.cuh>
#include <cuda_dl/core/launch_config.cuh>

#include <cstddef>
#include <stdexcept>

namespace cuda_dl::ops {
namespace detail {

// Tiled Shared-Memory Matrix Multiplication supporting on-the-fly transposing
template <bool TransLHS, bool TransRHS>
static __global__ void gemm_kernel(
    const float* const lhs,
    const float* const rhs,
    float* const output,
    const std::size_t M,
    const std::size_t K,
    const std::size_t N)
{
    constexpr int tile_size = 16;
    __shared__ float tile_a[tile_size][tile_size];
    __shared__ float tile_b[tile_size][tile_size];

    const int local_col = threadIdx.x;
    const int local_row = threadIdx.y;
    const int global_col = (blockIdx.x * tile_size) + local_col;
    const int global_row = (blockIdx.y * tile_size) + local_row;

    float sum = 0.0F;

    for (std::size_t tile_start = 0; tile_start < K; tile_start += tile_size) {
        // Load LHS (A) tile cooperatively
        const std::size_t tiled_a_col = tile_start + local_col;
        if (global_row < M && tiled_a_col < K) {
            std::size_t lhs_idx;
            if constexpr (TransLHS) {
                lhs_idx = (tiled_a_col * M) + global_row;
            } else {
                lhs_idx = (global_row * K) + tiled_a_col;
            }
            tile_a[local_row][local_col] = lhs[lhs_idx];
        } else {
            tile_a[local_row][local_col] = 0.0F;
        }

        // Load RHS (B) tile cooperatively
        const std::size_t tiled_b_row = tile_start + local_row;
        if (tiled_b_row < K && global_col < N) {
            std::size_t rhs_idx;
            if constexpr (TransRHS) {
                rhs_idx = (global_col * K) + tiled_b_row;
            } else {
                rhs_idx = (tiled_b_row * N) + global_col;
            }
            tile_b[local_row][local_col] = rhs[rhs_idx];
        } else {
            tile_b[local_row][local_col] = 0.0F;
        }

        __syncthreads();

        for (int k = 0; k < tile_size; ++k) {
            sum += tile_a[local_row][k] * tile_b[k][local_col];
        }

        __syncthreads();
    }

    if (global_row < M && global_col < N) {
        output[(global_row * N) + global_col] = sum;
    }
}

// Reduction kernel to sum upstream gradients along the batch dimension for bias gradient
static __global__ void reduce_bias_kernel(
    const float* const upstream_grad,
    float* const bias_grad,
    const std::size_t batch_size,
    const std::size_t out_features)
{
    const std::size_t col = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (col >= out_features) {
        return;
    }

    float sum = 0.0F;
    for (std::size_t row = 0; row < batch_size; ++row) {
        sum += upstream_grad[(row * out_features) + col];
    }
    bias_grad[col] = sum;
}

// Element-wise addition of a 1D bias vector to a 2D matrix
static __global__ void add_bias_kernel(
    float* const output,
    const float* const bias,
    const std::size_t batch_size,
    const std::size_t out_features)
{
    const std::size_t col = (blockIdx.x * blockDim.x) + threadIdx.x;
    const std::size_t row = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (row >= batch_size || col >= out_features) {
        return;
    }

    output[(row * out_features) + col] += bias[col];
}

} // namespace detail

// High-level GEMM operator wrapper
template <bool TransLHS = false, bool TransRHS = false>
inline cuda_dl::core::DeviceTensor gemm(
    const cuda_dl::core::DeviceTensor& lhs,
    const cuda_dl::core::DeviceTensor& rhs,
    cudaStream_t stream = nullptr)
{
    if (lhs.dtype() != rhs.dtype()) {
        throw std::invalid_argument("gemm requires tensors with matching dtypes");
    }
    if (lhs.rank() != 2 || rhs.rank() != 2) {
        throw std::invalid_argument("gemm requires rank-2 tensors");
    }

    const std::size_t lhs_r = lhs.shape().dimension(0);
    const std::size_t lhs_c = lhs.shape().dimension(1);
    const std::size_t rhs_r = rhs.shape().dimension(0);
    const std::size_t rhs_c = rhs.shape().dimension(1);

    const std::size_t M = TransLHS ? lhs_c : lhs_r;
    const std::size_t K1 = TransLHS ? lhs_r : lhs_c;
    const std::size_t K2 = TransRHS ? rhs_c : rhs_r;
    const std::size_t N = TransRHS ? rhs_r : rhs_c;

    if (K1 != K2) {
        throw std::invalid_argument("gemm inner dimensions mismatch");
    }

    cuda_dl::core::DeviceTensor output({M, N}, lhs.dtype());
    if (output.element_count() == 0) {
        return output;
    }

    // Tiled shared memory kernel requires 16x16 threads blocks
    constexpr int tile_size = 16;
    const dim3 threads(tile_size, tile_size);
    const dim3 blocks(
        static_cast<unsigned int>((N + tile_size - 1) / tile_size),
        static_cast<unsigned int>((M + tile_size - 1) / tile_size)
    );

    detail::gemm_kernel<TransLHS, TransRHS><<<blocks, threads, 0, stream>>>(
        lhs.data(),
        rhs.data(),
        output.data(),
        M,
        K1,
        N);

    CUDADL_CUDA_CHECK_LAST_KERNEL("gemm_kernel");
    return output;
}

// Fully Connected (Linear) Forward Pass: Y = X * W^T + b
inline cuda_dl::core::DeviceTensor linear_forward(
    const cuda_dl::core::DeviceTensor& input,   // [B, in_features]
    const cuda_dl::core::DeviceTensor& weight,  // [out_features, in_features]
    const cuda_dl::core::DeviceTensor& bias,    // [out_features]
    cudaStream_t stream = nullptr)
{
    if (input.rank() != 2 || weight.rank() != 2 || bias.rank() != 1) {
        throw std::invalid_argument("linear_forward dimensions are invalid");
    }

    const std::size_t batch_size = input.shape().dimension(0);
    const std::size_t in_features = input.shape().dimension(1);
    const std::size_t out_features = weight.shape().dimension(0);
    const std::size_t weight_in = weight.shape().dimension(1);

    if (in_features != weight_in || out_features != bias.shape().dimension(0)) {
        throw std::invalid_argument("linear_forward feature size mismatch");
    }

    // Y_matmul = X * W^T
    cuda_dl::core::DeviceTensor output = gemm<false, true>(input, weight, stream);

    // Add bias element-wise to each row of output
    const cuda_dl::core::LaunchConfig2D launch = cuda_dl::core::make_2d_launch_config(batch_size, out_features);
    const dim3 blocks(static_cast<unsigned int>(launch.blocks_x), static_cast<unsigned int>(launch.blocks_y));
    const dim3 threads(static_cast<unsigned int>(launch.threads_x), static_cast<unsigned int>(launch.threads_y));

    detail::add_bias_kernel<<<blocks, threads, 0, stream>>>(
        output.data(),
        bias.data(),
        batch_size,
        out_features);

    CUDADL_CUDA_CHECK_LAST_KERNEL("add_bias_kernel");
    return output;
}

struct LinearBackwardResult {
    cuda_dl::core::DeviceTensor input_grad;
    cuda_dl::core::DeviceTensor weight_grad;
    cuda_dl::core::DeviceTensor bias_grad;
};

inline LinearBackwardResult linear_backward(
    const cuda_dl::core::DeviceTensor& input,          // [B, in_features]
    const cuda_dl::core::DeviceTensor& weight,         // [out_features, in_features]
    const cuda_dl::core::DeviceTensor& upstream_grad,  // [B, out_features]
    cudaStream_t stream = nullptr)
{
    if (input.rank() != 2 || weight.rank() != 2 || upstream_grad.rank() != 2) {
        throw std::invalid_argument("linear_backward dimensions are invalid");
    }

    const std::size_t batch_size = input.shape().dimension(0);
    const std::size_t in_features = input.shape().dimension(1);
    const std::size_t out_features = weight.shape().dimension(0);

    if (weight.shape().dimension(1) != in_features) {
        throw std::invalid_argument("linear_backward weight input feature dimension mismatch");
    }

    if (upstream_grad.shape().dimension(0) != batch_size || upstream_grad.shape().dimension(1) != out_features) {
        throw std::invalid_argument("linear_backward upstream gradient shape mismatch");
    }

    // 1. Calculate input gradient (dX): dX = dY * W
    cuda_dl::core::DeviceTensor dX = gemm<false, false>(upstream_grad, weight, stream);

    // 2. Calculate weight gradient (dW): dW = dY^T * X
    cuda_dl::core::DeviceTensor dW = gemm<true, false>(upstream_grad, input, stream);

    // 3. Calculate bias gradient (db): db = sum_row(dY)
    cuda_dl::core::DeviceTensor db({out_features}, input.dtype());
    const cuda_dl::core::LaunchConfig1D launch = cuda_dl::core::make_1d_launch_config(out_features);

    detail::reduce_bias_kernel<<<launch.blocks_per_grid, launch.threads_per_block, 0, stream>>>(
        upstream_grad.data(),
        db.data(),
        batch_size,
        out_features);

    CUDADL_CUDA_CHECK_LAST_KERNEL("reduce_bias_kernel");
    return LinearBackwardResult{std::move(dX), std::move(dW), std::move(db)};
}

} // namespace cuda_dl::ops
