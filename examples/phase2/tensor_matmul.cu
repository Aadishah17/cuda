#include <cuda_dl/core/device_tensor.cuh>
#include <cuda_dl/ops/tensor_ops.cuh>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void expect(const bool condition, const char* const message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::vector<float> cpu_matmul(
    const std::vector<float>& lhs,
    const std::vector<float>& rhs,
    const std::size_t rows,
    const std::size_t shared,
    const std::size_t columns)
{
    std::vector<float> output(rows * columns, 0.0F);
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t column = 0; column < columns; ++column) {
            float sum = 0.0F;
            for (std::size_t k = 0; k < shared; ++k) {
                sum += lhs[(row * shared) + k] * rhs[(k * columns) + column];
            }
            output[(row * columns) + column] = sum;
        }
    }
    return output;
}

} // namespace

int main()
{
    constexpr std::size_t rows = 5;
    constexpr std::size_t shared = 7;
    constexpr std::size_t columns = 3;

    cuda_dl::core::DeviceTensor lhs({rows, shared}, cuda_dl::core::DType::Float32);
    cuda_dl::core::DeviceTensor rhs({shared, columns}, cuda_dl::core::DType::Float32);

    std::vector<float> host_lhs(lhs.element_count());
    std::vector<float> host_rhs(rhs.element_count());
    for (std::size_t index = 0; index < host_lhs.size(); ++index) {
        host_lhs[index] = static_cast<float>(static_cast<int>(index % 11) - 5) * 0.25F;
    }
    for (std::size_t index = 0; index < host_rhs.size(); ++index) {
        host_rhs[index] = static_cast<float>(static_cast<int>(index % 7) - 3) * 0.5F;
    }

    lhs.copy_from_host(host_lhs.data(), host_lhs.size());
    rhs.copy_from_host(host_rhs.data(), host_rhs.size());

    cuda_dl::core::DeviceTensor output = cuda_dl::ops::matmul(lhs, rhs);
    std::vector<float> host_output(output.element_count());
    output.copy_to_host(host_output.data(), host_output.size());

    const std::vector<float> expected = cpu_matmul(host_lhs, host_rhs, rows, shared, columns);
    float max_error = 0.0F;
    for (std::size_t index = 0; index < host_output.size(); ++index) {
        max_error = std::max(max_error, std::fabs(host_output[index] - expected[index]));
    }

    expect(max_error <= 1.0e-5F, "matmul output mismatch");
    expect(output.shape().dimensions() == std::vector<std::size_t>({rows, columns}), "matmul output shape mismatch");

    bool mismatch_rejected = false;
    try {
        cuda_dl::core::DeviceTensor incompatible({shared + 1, columns}, cuda_dl::core::DType::Float32);
        static_cast<void>(cuda_dl::ops::matmul(lhs, incompatible));
    } catch (const std::invalid_argument&) {
        mismatch_rejected = true;
    }
    expect(mismatch_rejected, "matmul accepted incompatible shapes");

    std::cout << "Tensor matmul verified: output [5, 3], max error " << max_error << std::endl;

    return 0;
}
