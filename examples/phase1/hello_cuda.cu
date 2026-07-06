#include "cuda_example_utils.cuh"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <iostream>

namespace {

__global__ void hello_kernel()
{
    // CUDA assigns every launched thread a block index and a thread index.
    // This linear id is the foundation for mapping threads to array elements
    // in upcoming vector, matrix, and tensor kernels.
    const int global_thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;

    printf(
        "Hello from CUDA thread %d (block %d, thread %d)\n",
        global_thread_id,
        blockIdx.x,
        threadIdx.x);
}

} // namespace

int main()
{
    try {
        int device_count = 0;
        CUDADL_CUDA_CHECK(cudaGetDeviceCount(&device_count));

        if (device_count == 0) {
            std::cerr << "No CUDA-capable GPU was found.\n";
            return EXIT_FAILURE;
        }

        constexpr int device_id = 0;
        cudaDeviceProp device_properties{};

        CUDADL_CUDA_CHECK(cudaSetDevice(device_id));
        CUDADL_CUDA_CHECK(cudaGetDeviceProperties(&device_properties, device_id));

        std::cout << "CUDA device: " << device_properties.name << '\n';
        std::cout << "Compute capability: " << device_properties.major << '.'
                  << device_properties.minor << '\n';
        std::cout << "Streaming multiprocessors: "
                  << device_properties.multiProcessorCount << '\n';

        constexpr int blocks_per_grid = 1;
        constexpr int threads_per_block = 4;

        std::cout << "Launching hello_kernel with " << blocks_per_grid
                  << " block and " << threads_per_block
                  << " threads per block.\n";
        std::cout.flush();

        hello_kernel<<<blocks_per_grid, threads_per_block>>>();

        // cudaGetLastError catches invalid launch configuration immediately.
        CUDADL_CUDA_CHECK_LAST_KERNEL("hello_kernel launch");

        // Kernel execution is asynchronous. Synchronizing makes the CPU wait
        // for the GPU and surfaces runtime failures before the program exits.
        CUDADL_CUDA_SYNCHRONIZE("hello_kernel execution");

        std::cout << "Hello CUDA completed successfully.\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
