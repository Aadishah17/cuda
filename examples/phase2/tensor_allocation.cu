#include <cuda_dl/core/cuda_error.cuh>
#include <cuda_dl/core/device_tensor.cuh>

#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

__global__ void affine_kernel(float* values, const std::size_t element_count, const float scale, const float bias)
{
    const std::size_t index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= element_count) {
        return;
    }

    values[index] = (values[index] * scale) + bias;
}

} // namespace

int main()
{
    cuda_dl::core::DeviceTensor tensor({2, 3, 4}, cuda_dl::core::DType::Float32);

    std::vector<float> host_input(tensor.element_count());
    for (std::size_t index = 0; index < host_input.size(); ++index) {
        host_input[index] = static_cast<float>(index) * 0.5F;
    }

    tensor.copy_from_host(host_input.data(), host_input.size());

    constexpr float scale = 2.0F;
    constexpr float bias = 1.0F;
    constexpr int threads_per_block = 128;
    const int blocks_per_grid = static_cast<int>((tensor.element_count() + threads_per_block - 1) / threads_per_block);

    affine_kernel<<<blocks_per_grid, threads_per_block>>>(tensor.data(), tensor.element_count(), scale, bias);
    CUDADL_CUDA_CHECK_LAST_KERNEL("affine_kernel");
    CUDADL_CUDA_SYNCHRONIZE("affine_kernel completion");

    std::vector<float> host_output(tensor.element_count());
    tensor.copy_to_host(host_output.data(), host_output.size());

    float max_error = 0.0F;
    for (std::size_t index = 0; index < host_output.size(); ++index) {
        const float expected = (host_input[index] * scale) + bias;
        max_error = std::max(max_error, std::fabs(host_output[index] - expected));
    }

    if (max_error > 1.0e-6F) {
        throw std::runtime_error("DeviceTensor allocation verification failed");
    }

    std::cout << "DeviceTensor verified: " << tensor.metadata().description() << ", elements "
              << tensor.element_count() << ", bytes " << tensor.bytes()
              << ", max error " << max_error << std::endl;

    return 0;
}
