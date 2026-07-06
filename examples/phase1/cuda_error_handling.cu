#include <cuda_dl/core/cuda_error.cuh>

#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>

namespace {

__global__ void no_op_kernel()
{
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

        CUDADL_CUDA_CHECK(cudaSetDevice(0));

        no_op_kernel<<<1, 1>>>();
        CUDADL_CUDA_CHECK_LAST_KERNEL("no_op_kernel launch");
        CUDADL_CUDA_SYNCHRONIZE("no_op_kernel execution");

        bool caught_expected_error = false;
        try {
            CUDADL_CUDA_CHECK(cudaErrorInvalidValue);
        } catch (const cuda_dl::core::CudaException& error) {
            caught_expected_error = error.error() == cudaErrorInvalidValue;
            std::cout << "Captured expected CUDA exception:\n"
                      << error.what() << '\n';
        }

        if (!caught_expected_error) {
            std::cerr << "CUDA error handling validation failed.\n";
            return EXIT_FAILURE;
        }

        std::cout << "CUDA error handling validated successfully.\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
