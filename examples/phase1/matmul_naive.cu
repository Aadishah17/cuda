#include "cuda_example_utils.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

__global__ void matmul_naive_kernel(
    const float* const a,
    const float* const b,
    float* const c,
    const int rows_a,
    const int columns_b,
    const int shared_dimension)
{
    const int column = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int row = (blockIdx.y * blockDim.y) + threadIdx.y;

    if ((row < rows_a) && (column < columns_b)) {
        float sum = 0.0F;

        // Naive matrix multiplication streams through the full dot product
        // from global memory for every output element. This is correct but
        // intentionally leaves global-memory reuse on the table.
        for (int k = 0; k < shared_dimension; ++k) {
            const float a_value = a[(row * shared_dimension) + k];
            const float b_value = b[(k * columns_b) + column];
            sum += a_value * b_value;
        }

        c[(row * columns_b) + column] = sum;
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

        constexpr int rows_a = 128;
        constexpr int shared_dimension = 64;
        constexpr int columns_b = 256;
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
                static_cast<float>((i % 17) - 8) * 0.125F;
        }

        for (int i = 0; i < elements_b; ++i) {
            host_b[static_cast<std::size_t>(i)] =
                static_cast<float>((i % 13) - 6) * 0.0625F;
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

        constexpr dim3 threads_per_block(16, 16);
        const dim3 blocks_per_grid(
            (columns_b + threads_per_block.x - 1) / threads_per_block.x,
            (rows_a + threads_per_block.y - 1) / threads_per_block.y);

        std::cout << "Launching matmul_naive_kernel for C[" << rows_a << ", "
                  << columns_b << "] = A[" << rows_a << ", "
                  << shared_dimension << "] * B[" << shared_dimension << ", "
                  << columns_b << "] with grid (" << blocks_per_grid.x
                  << ", " << blocks_per_grid.y << ") and block ("
                  << threads_per_block.x << ", " << threads_per_block.y
                  << ").\n";

        matmul_naive_kernel<<<blocks_per_grid, threads_per_block>>>(
            device_a.get(),
            device_b.get(),
            device_c.get(),
            rows_a,
            columns_b,
            shared_dimension);

        example::check_cuda(cudaGetLastError(), "matmul_naive_kernel launch");
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
        std::cout << "Verified naive matrix multiplication. Max absolute error: "
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
