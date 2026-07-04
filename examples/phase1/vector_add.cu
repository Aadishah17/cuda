#include "cuda_example_utils.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

__global__ void vector_add_kernel(
    const float* const a,
    const float* const b,
    float* const c,
    const int element_count)
{
    // Consecutive threads access consecutive floats. That gives the hardware a
    // coalesced global-memory access pattern, which is the baseline we want for
    // bandwidth-bound elementwise tensor operations.
    const int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    // The final block may have more threads than remaining elements. This guard
    // keeps extra threads from reading or writing out of bounds.
    if (index < element_count) {
        c[index] = a[index] + b[index];
    }
}

} // namespace

int main()
{
    try {
        namespace example = cuda_dl::examples;

        constexpr int element_count = 1 << 20;
        constexpr int threads_per_block = 256;
        constexpr float tolerance = 1.0e-5F;

        const int blocks_per_grid =
            (element_count + threads_per_block - 1) / threads_per_block;

        std::vector<float> host_a(element_count);
        std::vector<float> host_b(element_count);
        std::vector<float> host_c(element_count, 0.0F);

        for (std::size_t i = 0; i < host_a.size(); ++i) {
            const float value = static_cast<float>(i % 1024U);
            host_a[i] = value * 0.25F;
            host_b[i] = 1.0F + (value * 0.5F);
        }

        example::DeviceBuffer<float> device_a(host_a.size());
        example::DeviceBuffer<float> device_b(host_b.size());
        example::DeviceBuffer<float> device_c(host_c.size());

        example::check_cuda(
            cudaMemcpy(
                device_a.get(),
                host_a.data(),
                device_a.bytes(),
                cudaMemcpyHostToDevice),
            "copy A host to device");
        example::check_cuda(
            cudaMemcpy(
                device_b.get(),
                host_b.data(),
                device_b.bytes(),
                cudaMemcpyHostToDevice),
            "copy B host to device");

        std::cout << "Launching vector_add_kernel with " << blocks_per_grid
                  << " blocks and " << threads_per_block
                  << " threads per block for " << element_count
                  << " elements.\n";

        vector_add_kernel<<<blocks_per_grid, threads_per_block>>>(
            device_a.get(),
            device_b.get(),
            device_c.get(),
            element_count);

        example::check_cuda(cudaGetLastError(), "vector_add_kernel launch");
        example::check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

        example::check_cuda(
            cudaMemcpy(
                host_c.data(),
                device_c.get(),
                device_c.bytes(),
                cudaMemcpyDeviceToHost),
            "copy C device to host");

        float max_absolute_error = 0.0F;
        int mismatch_count = 0;
        std::size_t first_mismatch = host_c.size();

        for (std::size_t i = 0; i < host_c.size(); ++i) {
            const float expected = host_a[i] + host_b[i];
            const float absolute_error = std::fabs(host_c[i] - expected);
            max_absolute_error = std::max(max_absolute_error, absolute_error);

            if (absolute_error > tolerance) {
                if (mismatch_count == 0) {
                    first_mismatch = i;
                }
                ++mismatch_count;
            }
        }

        if (mismatch_count != 0) {
            std::cerr << "Verification failed with " << mismatch_count
                      << " mismatches. First mismatch at index "
                      << first_mismatch << ": expected "
                      << host_a[first_mismatch] + host_b[first_mismatch]
                      << ", got " << host_c[first_mismatch] << '\n';
            return EXIT_FAILURE;
        }

        std::cout << "Verified " << element_count
                  << " vector additions. Max absolute error: "
                  << max_absolute_error << '\n';
        std::cout << "Sample: c[0]=" << host_c[0]
                  << ", c[123]=" << host_c[123]
                  << ", c[last]=" << host_c.back() << '\n';

        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
