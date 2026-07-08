#include "cuda_example_utils.cuh"

#include <cuda_dl/core/cuda_event_timer.cuh>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {

__global__ void vector_add_kernel(
    const float* const a,
    const float* const b,
    float* const c,
    const int element_count)
{
    const int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < element_count) {
        c[index] = a[index] + b[index];
    }
}

} // namespace

int main()
{
    try {
        namespace core = cuda_dl::core;
        namespace example = cuda_dl::examples;

        constexpr int element_count = 1 << 22;
        constexpr int threads_per_block = 256;
        constexpr int warmup_iterations = 5;
        constexpr int measured_iterations = 100;
        constexpr float tolerance = 1.0e-5F;

        const int blocks_per_grid =
            (element_count + threads_per_block - 1) / threads_per_block;

        std::vector<float> host_a(element_count);
        std::vector<float> host_b(element_count);
        std::vector<float> host_c(element_count, 0.0F);

        for (std::size_t i = 0; i < host_a.size(); ++i) {
            const float value = static_cast<float>(i % 2048U);
            host_a[i] = value * 0.125F;
            host_b[i] = 2.0F - (value * 0.03125F);
        }

        example::DeviceBuffer<float> device_a(host_a.size());
        example::DeviceBuffer<float> device_b(host_b.size());
        example::DeviceBuffer<float> device_c(host_c.size());

        device_a.copy_from_host(host_a.data(), host_a.size());
        device_b.copy_from_host(host_b.data(), host_b.size());

        for (int iteration = 0; iteration < warmup_iterations; ++iteration) {
            vector_add_kernel<<<blocks_per_grid, threads_per_block>>>(
                device_a.get(),
                device_b.get(),
                device_c.get(),
                element_count);
        }

        CUDADL_CUDA_CHECK_LAST_KERNEL("vector_add_kernel warmup");
        CUDADL_CUDA_SYNCHRONIZE("vector_add_kernel warmup");

        core::CudaEvent start;
        core::CudaEvent stop;

        start.record();
        for (int iteration = 0; iteration < measured_iterations; ++iteration) {
            vector_add_kernel<<<blocks_per_grid, threads_per_block>>>(
                device_a.get(),
                device_b.get(),
                device_c.get(),
                element_count);
        }
        stop.record();
        stop.synchronize();

        CUDADL_CUDA_CHECK_LAST_KERNEL("vector_add_kernel benchmark");

        device_c.copy_to_host(host_c.data(), host_c.size());

        float max_absolute_error = 0.0F;
        int mismatch_count = 0;

        for (std::size_t i = 0; i < host_c.size(); ++i) {
            const float expected = host_a[i] + host_b[i];
            const float absolute_error = std::fabs(host_c[i] - expected);
            max_absolute_error = std::max(max_absolute_error, absolute_error);

            if (absolute_error > tolerance) {
                ++mismatch_count;
            }
        }

        if (mismatch_count != 0) {
            std::cerr << "Benchmark verification failed with " << mismatch_count
                      << " mismatches.\n";
            return EXIT_FAILURE;
        }

        const float total_milliseconds = core::elapsed_milliseconds(start, stop);
        const double average_milliseconds =
            static_cast<double>(total_milliseconds) / measured_iterations;
        const double bytes_per_iteration =
            static_cast<double>(element_count) * sizeof(float) * 3.0;
        const double gigabytes_per_second =
            (bytes_per_iteration / average_milliseconds / 1.0e6);

        std::cout << "Vector add benchmark verified. Max absolute error: "
                  << max_absolute_error << '\n';
        std::cout << "Elements: " << element_count
                  << ", iterations: " << measured_iterations
                  << ", block size: " << threads_per_block << '\n';
        std::cout << std::fixed << std::setprecision(4)
                  << "Average kernel time: " << average_milliseconds << " ms\n"
                  << "Approximate effective bandwidth: "
                  << gigabytes_per_second << " GB/s\n";

        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
