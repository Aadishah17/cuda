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

float max_error_against(
    const std::vector<float>& actual,
    const std::vector<float>& lhs,
    const std::vector<float>& rhs,
    const char operation)
{
    float max_error = 0.0F;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        const float rhs_value = rhs[index % rhs.size()];
        float expected = 0.0F;

        if (operation == '+') {
            expected = lhs[index] + rhs_value;
        } else if (operation == '-') {
            expected = lhs[index] - rhs_value;
        } else {
            expected = lhs[index] * rhs_value;
        }

        max_error = std::max(max_error, std::fabs(actual[index] - expected));
    }

    return max_error;
}

} // namespace

int main()
{
    cuda_dl::core::DeviceTensor lhs({2, 3, 4}, cuda_dl::core::DType::Float32);
    cuda_dl::core::DeviceTensor rhs({4}, cuda_dl::core::DType::Float32);

    std::vector<float> host_lhs(lhs.element_count());
    for (std::size_t index = 0; index < host_lhs.size(); ++index) {
        host_lhs[index] = static_cast<float>(index) * 0.25F;
    }

    std::vector<float> host_rhs{1.0F, 2.0F, 3.0F, 4.0F};

    lhs.copy_from_host(host_lhs.data(), host_lhs.size());
    rhs.copy_from_host(host_rhs.data(), host_rhs.size());

    cuda_dl::core::DeviceTensor added = cuda_dl::ops::add(lhs, rhs);
    cuda_dl::core::DeviceTensor subtracted = cuda_dl::ops::subtract(lhs, rhs);
    cuda_dl::core::DeviceTensor multiplied = cuda_dl::ops::multiply(lhs, rhs);

    std::vector<float> host_added(added.element_count());
    std::vector<float> host_subtracted(subtracted.element_count());
    std::vector<float> host_multiplied(multiplied.element_count());

    added.copy_to_host(host_added.data(), host_added.size());
    subtracted.copy_to_host(host_subtracted.data(), host_subtracted.size());
    multiplied.copy_to_host(host_multiplied.data(), host_multiplied.size());

    const float add_error = max_error_against(host_added, host_lhs, host_rhs, '+');
    const float subtract_error = max_error_against(host_subtracted, host_lhs, host_rhs, '-');
    const float multiply_error = max_error_against(host_multiplied, host_lhs, host_rhs, '*');

    expect(add_error <= 1.0e-6F, "broadcast add mismatch");
    expect(subtract_error <= 1.0e-6F, "broadcast subtract mismatch");
    expect(multiply_error <= 1.0e-6F, "broadcast multiply mismatch");

    std::cout << "Tensor binary ops verified: add " << add_error
              << ", subtract " << subtract_error
              << ", multiply " << multiply_error << std::endl;

    return 0;
}
