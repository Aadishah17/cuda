#include "cuda_example_utils.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

constexpr int tile_size = 16;

__global__ void matmul_tiled_kernel(
    const float* const a,
    const float* const b,
    float* const c,
    const int rows_a,
    const int columns_b,
    const int shared_dimension)
{
    __shared__ float tile_a[tile_size][tile_size];
    __shared__ float tile_b[tile_size][tile_size];

    const int local_column = threadIdx.x;
    const int local_row = threadIdx.y;
    const int global_column = (blockIdx.x * tile_size) + local_column;
    const int global_row = (blockIdx.y * tile_size) + local_row;

    float sum = 0.0F;

    for (int tile_start = 0; tile_start < shared_dimension; tile_start += tile_size) {
        const int tiled_a_column = tile_start + local_column;
        const int tiled_b_row = tile_start + local_row;

        // Threads cooperatively stage one A tile and one B tile into shared
        // memory. Out-of-range entries are zero-padded so edge tiles are safe.
        tile_a[local_row][local_column] =
            ((global_row < rows_a) && (tiled_a_column < shared_dimension))
            ? a[(global_row * shared_dimension) + tiled_a_column]
            : 0.0F;

        tile_b[local_row][local_column] =
            ((tiled_b_row < shared_dimension) && (global_column < columns_b))
            ? b[(tiled_b_row * columns_b) + global_column]
            : 0.0F;

        // Every thread must finish loading before any thread consumes the tile.
        __syncthreads();

        for (int k = 0; k < tile_size; ++k) {
            sum += tile_a[local_row][k] * tile_b[k][local_column];
        }

        // Protect the next loop iteration from overwriting shared memory before
        // all threads have finished reading the current tile.
        __syncthreads();
    }

    if ((global_row < rows_a) && (global_column < columns_b)) {
        c[(global_row * columns_b) + global_column] = sum;
    }
}

void compute_cpu_reference(
    const std::vector<float>& a,
    const std::vector<float>& b,
    std::vector<float>& c,
    const int rows_a,
    const int columns_b,
    const int shared_dimension)
{
    for (int row = 0; row < rows_a; ++row) {
        for (int column = 0; column < columns_b; ++column) {
            float sum = 0.0F;

            for (int k = 0; k < shared_dimension; ++k) {
                sum += a[(row * shared_dimension) + k] * b[(k * columns_b) + column];
            }

            c[(row * columns_b) + column] = sum;
        }
    }
}

} // namespace

int main()
{
    try {
        namespace example = cuda_dl::examples;

        constexpr int rows_a = 130;
        constexpr int shared_dimension = 70;
        constexpr int columns_b = 257;
        constexpr int elements_a = rows_a * shared_dimension;
        constexpr int elements_b = shared_dimension * columns_b;
        constexpr int elements_c = rows_a * columns_b;
        constexpr float tolerance = 1.0e-3F;

        std::vector<float> host_a(elements_a);
        std::vector<float> host_b(elements_b);
        std::vector<float> host_c(elements_c, 0.0F);
        std::vector<float> reference_c(elements_c, 0.0F);

        for (int i = 0; i < elements_a; ++i) {
            host_a[static_cast<std::size_t>(i)] =
                static_cast<float>((i % 19) - 9) * 0.0625F;
        }

        for (int i = 0; i < elements_b; ++i) {
            host_b[static_cast<std::size_t>(i)] =
                static_cast<float>((i % 11) - 5) * 0.125F;
        }

        compute_cpu_reference(
            host_a,
            host_b,
            reference_c,
            rows_a,
            columns_b,
            shared_dimension);

        example::DeviceBuffer<float> device_a(host_a.size());
        example::DeviceBuffer<float> device_b(host_b.size());
        example::DeviceBuffer<float> device_c(host_c.size());

        device_a.copy_from_host(host_a.data(), host_a.size());
        device_b.copy_from_host(host_b.data(), host_b.size());

        constexpr dim3 threads_per_block(tile_size, tile_size);
        const dim3 blocks_per_grid(
            (columns_b + tile_size - 1) / tile_size,
            (rows_a + tile_size - 1) / tile_size);

        std::cout << "Launching matmul_tiled_kernel for C[" << rows_a << ", "
                  << columns_b << "] = A[" << rows_a << ", "
                  << shared_dimension << "] * B[" << shared_dimension << ", "
                  << columns_b << "] with tile size " << tile_size << ".\n";
        std::cout << "Grid (" << blocks_per_grid.x << ", " << blocks_per_grid.y
                  << "), block (" << threads_per_block.x << ", "
                  << threads_per_block.y << ").\n";

        matmul_tiled_kernel<<<blocks_per_grid, threads_per_block>>>(
            device_a.get(),
            device_b.get(),
            device_c.get(),
            rows_a,
            columns_b,
            shared_dimension);

        CUDADL_CUDA_CHECK_LAST_KERNEL("matmul_tiled_kernel launch");
        CUDADL_CUDA_SYNCHRONIZE("matmul_tiled_kernel execution");

        device_c.copy_to_host(host_c.data(), host_c.size());

        float max_absolute_error = 0.0F;
        int mismatch_count = 0;
        std::size_t first_mismatch = host_c.size();

        for (std::size_t i = 0; i < host_c.size(); ++i) {
            const float absolute_error = std::fabs(host_c[i] - reference_c[i]);
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
                static_cast<int>(first_mismatch / static_cast<std::size_t>(columns_b));
            const int column =
                static_cast<int>(first_mismatch % static_cast<std::size_t>(columns_b));

            std::cerr << "Verification failed with " << mismatch_count
                      << " mismatches. First mismatch at (" << row << ", "
                      << column << "): expected " << reference_c[first_mismatch]
                      << ", got " << host_c[first_mismatch] << '\n';
            return EXIT_FAILURE;
        }

        const std::size_t sample_index =
            (42U * static_cast<std::size_t>(columns_b)) + 17U;
        std::cout << "Verified tiled matrix multiplication. Max absolute error: "
                  << max_absolute_error << '\n';
        std::cout << "Sample: c[0,0]=" << host_c[0]
                  << ", c[42,17]=" << host_c[sample_index]
                  << ", c[last,last]=" << host_c.back() << '\n';

        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
