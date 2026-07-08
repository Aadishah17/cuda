#include "cuda_example_utils.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

__global__ void matrix_add_kernel(
    const float* const a,
    const float* const b,
    float* const c,
    const int rows,
    const int columns)
{
    // x maps to columns and y maps to rows. This keeps neighboring x threads
    // reading neighboring row-major elements, which enables coalesced loads.
    const int column = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int row = (blockIdx.y * blockDim.y) + threadIdx.y;

    if ((row < rows) && (column < columns)) {
        const int index = (row * columns) + column;
        c[index] = a[index] + b[index];
    }
}

} // namespace

int main()
{
    try {
        namespace example = cuda_dl::examples;

        constexpr int rows = 768;
        constexpr int columns = 1024;
        constexpr int element_count = rows * columns;
        constexpr float tolerance = 1.0e-5F;

        std::vector<float> host_a(element_count);
        std::vector<float> host_b(element_count);
        std::vector<float> host_c(element_count, 0.0F);

        for (std::size_t i = 0; i < host_a.size(); ++i) {
            const int row = static_cast<int>(i / static_cast<std::size_t>(columns));
            const int column = static_cast<int>(i % static_cast<std::size_t>(columns));
            host_a[i] = (static_cast<float>(row) * 0.25F)
                + (static_cast<float>(column) * 0.5F);
            host_b[i] = 1.0F + (static_cast<float>(row) * 0.125F)
                - (static_cast<float>(column) * 0.25F);
        }

        example::DeviceBuffer<float> device_a(host_a.size());
        example::DeviceBuffer<float> device_b(host_b.size());
        example::DeviceBuffer<float> device_c(host_c.size());

        device_a.copy_from_host(host_a.data(), host_a.size());
        device_b.copy_from_host(host_b.data(), host_b.size());

        constexpr dim3 threads_per_block(16, 16);
        const dim3 blocks_per_grid(
            (columns + threads_per_block.x - 1) / threads_per_block.x,
            (rows + threads_per_block.y - 1) / threads_per_block.y);

        std::cout << "Launching matrix_add_kernel for " << rows << "x"
                  << columns << " matrix with grid (" << blocks_per_grid.x
                  << ", " << blocks_per_grid.y << ") and block ("
                  << threads_per_block.x << ", " << threads_per_block.y
                  << ").\n";

        matrix_add_kernel<<<blocks_per_grid, threads_per_block>>>(
            device_a.get(),
            device_b.get(),
            device_c.get(),
            rows,
            columns);

        CUDADL_CUDA_CHECK_LAST_KERNEL("matrix_add_kernel launch");
        CUDADL_CUDA_SYNCHRONIZE("matrix_add_kernel execution");

        device_c.copy_to_host(host_c.data(), host_c.size());

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
            const int row =
                static_cast<int>(first_mismatch / static_cast<std::size_t>(columns));
            const int column =
                static_cast<int>(first_mismatch % static_cast<std::size_t>(columns));

            std::cerr << "Verification failed with " << mismatch_count
                      << " mismatches. First mismatch at (" << row << ", "
                      << column << "): expected "
                      << host_a[first_mismatch] + host_b[first_mismatch]
                      << ", got " << host_c[first_mismatch] << '\n';
            return EXIT_FAILURE;
        }

        const std::size_t sample_index = (123U * static_cast<std::size_t>(columns)) + 456U;
        std::cout << "Verified " << rows << "x" << columns
                  << " matrix addition. Max absolute error: "
                  << max_absolute_error << '\n';
        std::cout << "Sample: c[0,0]=" << host_c[0]
                  << ", c[123,456]=" << host_c[sample_index]
                  << ", c[last,last]=" << host_c.back() << '\n';

        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
