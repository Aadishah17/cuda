#include <cuda_dl/core/cuda_error.cuh>
#include <cuda_dl/core/device_buffer.cuh>

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

__global__ void add_constant_kernel(int* values, const std::size_t element_count, const int increment)
{
    const std::size_t index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= element_count) {
        return;
    }

    values[index] += increment;
}

} // namespace

int main()
{
    constexpr std::size_t element_count = 4096;
    constexpr int increment = 7;

    std::vector<int> host_input(element_count);
    for (std::size_t index = 0; index < host_input.size(); ++index) {
        host_input[index] = static_cast<int>(index);
    }

    cuda_dl::core::DeviceBuffer<int> device_values(host_input.size());
    device_values.copy_from_host(host_input.data(), host_input.size());

    constexpr int threads_per_block = 256;
    const int blocks_per_grid = static_cast<int>((element_count + threads_per_block - 1) / threads_per_block);

    add_constant_kernel<<<blocks_per_grid, threads_per_block>>>(device_values.get(), device_values.size(), increment);
    CUDADL_CUDA_CHECK_LAST_KERNEL("add_constant_kernel");
    CUDADL_CUDA_SYNCHRONIZE("add_constant_kernel completion");

    std::vector<int> host_output(element_count);
    device_values.copy_to_host(host_output.data(), host_output.size());

    for (std::size_t index = 0; index < host_output.size(); ++index) {
        const int expected = host_input[index] + increment;
        if (host_output[index] != expected) {
            throw std::runtime_error("GPU memory manager verification failed");
        }
    }

    std::cout << "GPU memory manager verified: " << device_values.size() << " int values, "
              << device_values.bytes() << " bytes" << std::endl;

    return 0;
}
