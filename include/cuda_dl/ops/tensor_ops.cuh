#pragma once

#include <cuda_dl/core/cuda_error.cuh>
#include <cuda_dl/core/device_tensor.cuh>
#include <cuda_dl/core/launch_config.cuh>

#include <cstddef>
#include <stdexcept>

namespace cuda_dl::ops {
namespace detail {

constexpr std::size_t kMaxBroadcastRank = 8;

struct BroadcastOperand {
    std::size_t dimensions[kMaxBroadcastRank]{};
    std::size_t strides[kMaxBroadcastRank]{};
};

struct BroadcastConfig {
    std::size_t rank{0};
    std::size_t output_dimensions[kMaxBroadcastRank]{};
    BroadcastOperand lhs{};
    BroadcastOperand rhs{};
};

enum class BinaryOp {
    Add,
    Subtract,
    Multiply,
};

inline BroadcastOperand make_broadcast_operand(
    const cuda_dl::core::TensorShape& input_shape,
    const cuda_dl::core::TensorShape& output_shape)
{
    if (output_shape.rank() > kMaxBroadcastRank) {
        throw std::invalid_argument("broadcast rank exceeds supported maximum");
    }

    BroadcastOperand operand{};
    const std::size_t output_rank = output_shape.rank();
    const std::size_t input_rank = input_shape.rank();

    for (std::size_t output_axis = 0; output_axis < output_rank; ++output_axis) {
        const std::size_t axes_from_end = output_rank - 1 - output_axis;

        if (input_rank <= axes_from_end) {
            operand.dimensions[output_axis] = 1;
            operand.strides[output_axis] = 0;
            continue;
        }

        const std::size_t input_axis = input_rank - 1 - axes_from_end;
        const std::size_t input_dimension = input_shape.dimension(input_axis);
        operand.dimensions[output_axis] = input_dimension;
        operand.strides[output_axis] = input_dimension == 1 ? 0 : input_shape.stride(input_axis);
    }

    return operand;
}

inline BroadcastConfig make_broadcast_config(
    const cuda_dl::core::TensorShape& lhs_shape,
    const cuda_dl::core::TensorShape& rhs_shape,
    const cuda_dl::core::TensorShape& output_shape)
{
    if (output_shape.rank() > kMaxBroadcastRank) {
        throw std::invalid_argument("broadcast rank exceeds supported maximum");
    }

    BroadcastConfig config{};
    config.rank = output_shape.rank();
    for (std::size_t axis = 0; axis < output_shape.rank(); ++axis) {
        config.output_dimensions[axis] = output_shape.dimension(axis);
    }

    config.lhs = make_broadcast_operand(lhs_shape, output_shape);
    config.rhs = make_broadcast_operand(rhs_shape, output_shape);

    return config;
}

__device__ std::size_t broadcast_offset(
    std::size_t output_index,
    const BroadcastOperand& operand,
    const BroadcastConfig& config)
{
    std::size_t offset = 0;

    for (std::size_t axis_from_end = 0; axis_from_end < config.rank; ++axis_from_end) {
        const std::size_t axis = config.rank - 1 - axis_from_end;
        const std::size_t coordinate = output_index % config.output_dimensions[axis];
        output_index /= config.output_dimensions[axis];

        if (operand.dimensions[axis] != 1) {
            offset += coordinate * operand.strides[axis];
        }
    }

    return offset;
}

template <BinaryOp Operation>
static __global__ void binary_broadcast_kernel(
    const float* lhs,
    const float* rhs,
    float* output,
    const std::size_t element_count,
    const BroadcastConfig config)
{
    const std::size_t index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= element_count) {
        return;
    }

    const std::size_t lhs_index = broadcast_offset(index, config.lhs, config);
    const std::size_t rhs_index = broadcast_offset(index, config.rhs, config);

    if constexpr (Operation == BinaryOp::Add) {
        output[index] = lhs[lhs_index] + rhs[rhs_index];
    } else if constexpr (Operation == BinaryOp::Subtract) {
        output[index] = lhs[lhs_index] - rhs[rhs_index];
    } else {
        output[index] = lhs[lhs_index] * rhs[rhs_index];
    }
}

} // namespace detail

template <detail::BinaryOp Operation>
inline cuda_dl::core::DeviceTensor binary_elementwise(
    const cuda_dl::core::DeviceTensor& lhs,
    const cuda_dl::core::DeviceTensor& rhs,
    const char* const kernel_name)
{
    if (lhs.dtype() != rhs.dtype()) {
        throw std::invalid_argument("binary tensor operation requires tensors with matching dtypes");
    }

    const cuda_dl::core::TensorShape output_shape = cuda_dl::core::broadcast_shapes(lhs.shape(), rhs.shape());

    cuda_dl::core::DeviceTensor output(cuda_dl::core::Tensor(output_shape, lhs.dtype()));

    if (output.element_count() == 0) {
        return output;
    }

    const detail::BroadcastConfig broadcast = detail::make_broadcast_config(
        lhs.shape(),
        rhs.shape(),
        output.shape());
    const cuda_dl::core::LaunchConfig1D launch = cuda_dl::core::make_1d_launch_config(output.element_count());

    detail::binary_broadcast_kernel<Operation><<<launch.blocks_per_grid, launch.threads_per_block>>>(
        lhs.data(),
        rhs.data(),
        output.data(),
        output.element_count(),
        broadcast);

    CUDADL_CUDA_CHECK_LAST_KERNEL(kernel_name);

    // Early framework milestones synchronize inside ops so failures are caught
    // at the call site. Later stream support should make this policy explicit.
    CUDADL_CUDA_SYNCHRONIZE(kernel_name);

    return output;
}

inline cuda_dl::core::DeviceTensor add(
    const cuda_dl::core::DeviceTensor& lhs,
    const cuda_dl::core::DeviceTensor& rhs)
{
    return binary_elementwise<detail::BinaryOp::Add>(lhs, rhs, "add_broadcast_kernel");
}

inline cuda_dl::core::DeviceTensor subtract(
    const cuda_dl::core::DeviceTensor& lhs,
    const cuda_dl::core::DeviceTensor& rhs)
{
    return binary_elementwise<detail::BinaryOp::Subtract>(lhs, rhs, "subtract_broadcast_kernel");
}

inline cuda_dl::core::DeviceTensor multiply(
    const cuda_dl::core::DeviceTensor& lhs,
    const cuda_dl::core::DeviceTensor& rhs)
{
    return binary_elementwise<detail::BinaryOp::Multiply>(lhs, rhs, "multiply_broadcast_kernel");
}

namespace detail {

static __global__ void matmul_naive_kernel(
    const float* lhs,
    const float* rhs,
    float* output,
    const std::size_t rows,
    const std::size_t shared,
    const std::size_t columns)
{
    const std::size_t column = (blockIdx.x * blockDim.x) + threadIdx.x;
    const std::size_t row = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (row >= rows || column >= columns) {
        return;
    }

    float sum = 0.0F;
    for (std::size_t k = 0; k < shared; ++k) {
        sum += lhs[(row * shared) + k] * rhs[(k * columns) + column];
    }

    output[(row * columns) + column] = sum;
}

} // namespace detail

inline cuda_dl::core::DeviceTensor matmul(
    const cuda_dl::core::DeviceTensor& lhs,
    const cuda_dl::core::DeviceTensor& rhs)
{
    if (lhs.dtype() != rhs.dtype()) {
        throw std::invalid_argument("matmul requires tensors with matching dtypes");
    }

    if (lhs.rank() != 2 || rhs.rank() != 2) {
        throw std::invalid_argument("matmul currently requires rank-2 tensors");
    }

    const std::size_t rows = lhs.shape().dimension(0);
    const std::size_t shared = lhs.shape().dimension(1);
    const std::size_t rhs_rows = rhs.shape().dimension(0);
    const std::size_t columns = rhs.shape().dimension(1);

    if (shared != rhs_rows) {
        throw std::invalid_argument("matmul inner dimensions do not match");
    }

    cuda_dl::core::DeviceTensor output({rows, columns}, lhs.dtype());

    if (output.element_count() == 0) {
        return output;
    }

    const cuda_dl::core::LaunchConfig2D launch = cuda_dl::core::make_2d_launch_config(rows, columns);
    const dim3 blocks(static_cast<unsigned int>(launch.blocks_x), static_cast<unsigned int>(launch.blocks_y));
    const dim3 threads(static_cast<unsigned int>(launch.threads_x), static_cast<unsigned int>(launch.threads_y));

    detail::matmul_naive_kernel<<<blocks, threads>>>(
        lhs.data(),
        rhs.data(),
        output.data(),
        rows,
        shared,
        columns);

    CUDADL_CUDA_CHECK_LAST_KERNEL("matmul_naive_kernel");
    CUDADL_CUDA_SYNCHRONIZE("matmul_naive_kernel");

    return output;
}

} // namespace cuda_dl::ops
