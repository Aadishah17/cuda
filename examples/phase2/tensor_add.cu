#include <cuda_dl/core/device_tensor.cuh>
#include <cuda_dl/ops/tensor_ops.cuh>

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

} // namespace

int main()
{
    constexpr std::size_t element_count = 24;

    cuda_dl::core::DeviceTensor lhs({2, 3, 4}, cuda_dl::core::DType::Float32);
    cuda_dl::core::DeviceTensor rhs({2, 3, 4}, cuda_dl::core::DType::Float32);

    std::vector<float> host_lhs(element_count);
    std::vector<float> host_rhs(element_count);
    for (std::size_t index = 0; index < element_count; ++index) {
        host_lhs[index] = static_cast<float>(index);
        host_rhs[index] = static_cast<float>(element_count - index) * 0.25F;
    }

    lhs.copy_from_host(host_lhs.data(), host_lhs.size());
    rhs.copy_from_host(host_rhs.data(), host_rhs.size());

    cuda_dl::core::DeviceTensor output = cuda_dl::ops::add(lhs, rhs);

    std::vector<float> host_output(output.element_count());
    output.copy_to_host(host_output.data(), host_output.size());

    float max_error = 0.0F;
    for (std::size_t index = 0; index < host_output.size(); ++index) {
        const float expected = host_lhs[index] + host_rhs[index];
        max_error = std::max(max_error, std::fabs(host_output[index] - expected));
    }

    expect(max_error <= 1.0e-6F, "tensor add result mismatch");

    bool mismatch_rejected = false;
    try {
        cuda_dl::core::DeviceTensor incompatible({2, 3, 5}, cuda_dl::core::DType::Float32);
        static_cast<void>(cuda_dl::ops::add(lhs, incompatible));
    } catch (const std::invalid_argument&) {
        mismatch_rejected = true;
    }

    expect(mismatch_rejected, "tensor add accepted incompatible shapes");

    std::cout << "Tensor add verified: " << output.metadata().description()
              << ", max error " << max_error << std::endl;

    return 0;
}
