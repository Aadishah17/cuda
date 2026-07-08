#include <cuda_dl/core/cuda_error.cuh>
#include <cuda_runtime.h>

#include <iostream>
#include <iomanip>
#include <cstdlib>

int main() {
    try {
        int driver_version = 0;
        int runtime_version = 0;

        CUDADL_CUDA_CHECK(cudaDriverGetVersion(&driver_version));
        CUDADL_CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));

        std::cout << "========================================\n";
        std::cout << "CUDA Environment Diagnostic Utility\n";
        std::cout << "========================================\n";
        std::cout << "CUDA Driver Version: " << driver_version / 1000 << "." << (driver_version % 100) / 10 << '\n';
        std::cout << "CUDA Runtime Version: " << runtime_version / 1000 << "." << (runtime_version % 100) / 10 << '\n';

        int device_count = 0;
        CUDADL_CUDA_CHECK(cudaGetDeviceCount(&device_count));

        std::cout << "Number of CUDA-capable devices: " << device_count << "\n\n";

        if (device_count == 0) {
            std::cerr << "CRITICAL ERROR: No CUDA-capable GPU found on this system.\n";
            return EXIT_FAILURE;
        }

        for (int dev = 0; dev < device_count; ++dev) {
            cudaDeviceProp prop{};
            CUDADL_CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

            std::cout << "--- Device " << dev << ": " << prop.name << " ---\n";
            std::cout << "  Compute Capability:         " << prop.major << "." << prop.minor << '\n';
            std::cout << "  Streaming Multiprocessors:  " << prop.multiProcessorCount << '\n';
            
            double total_global_mem_gib = static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0);
            std::cout << "  Total Global Memory:        " << std::fixed << std::setprecision(2) 
                      << total_global_mem_gib << " GiB\n";
            
            std::cout << "  L2 Cache Size:              " << prop.l2CacheSize / (1024.0 * 1024.0) << " MiB\n";
            std::cout << "  Total Constant Memory:      " << prop.totalConstMem / 1024.0 << " KiB\n";
            std::cout << "  Shared Memory per Block:    " << prop.sharedMemPerBlock / 1024.0 << " KiB\n";
            std::cout << "  Shared Memory per SM:       " << prop.sharedMemPerMultiprocessor / 1024.0 << " KiB\n";
            std::cout << "  Registers per Block:        " << prop.regsPerBlock << '\n';
            std::cout << "  Registers per SM:           " << prop.regsPerMultiprocessor << '\n';
            std::cout << "  Warp Size:                  " << prop.warpSize << " threads\n";
            std::cout << "  Max Threads per Block:      " << prop.maxThreadsPerBlock << '\n';
            std::cout << "  Max Threads per SM:         " << prop.maxThreadsPerMultiProcessor << '\n';
            std::cout << "  Max Grid Size:              [" << prop.maxGridSize[0] << ", " 
                      << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << "]\n";
            std::cout << "  Max Block Dimensions:       [" << prop.maxThreadsDim[0] << ", " 
                      << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << "]\n";
            
            // Query clock rates via attributes (required for CUDA 13.x)
            int clock_rate_khz = 0;
            int mem_clock_rate_khz = 0;
            CUDADL_CUDA_CHECK(cudaDeviceGetAttribute(&clock_rate_khz, cudaDevAttrClockRate, dev));
            CUDADL_CUDA_CHECK(cudaDeviceGetAttribute(&mem_clock_rate_khz, cudaDevAttrMemoryClockRate, dev));

            double clock_rate_mhz = static_cast<double>(clock_rate_khz) / 1e3;
            double mem_clock_mhz = static_cast<double>(mem_clock_rate_khz) / 1e3;
            
            std::cout << "  GPU Clock Rate:             " << clock_rate_mhz << " MHz\n";
            std::cout << "  Memory Clock Rate:          " << mem_clock_mhz << " MHz\n";
            std::cout << "  Memory Bus Width:           " << prop.memoryBusWidth << " bits\n";
            
            // Peak Memory Bandwidth = 2 * MemoryClockRate * (BusWidth / 8)
            // Divide by 1e6 to convert from kHz to GHz-equivalent base-10 bandwidth
            double peak_bandwidth = 2.0 * mem_clock_rate_khz * 1e3 * (prop.memoryBusWidth / 8.0) / 1.0e9;
            std::cout << "  Peak Memory Bandwidth:      " << peak_bandwidth << " GB/s\n";
            
            std::cout << "  Unified Addressing (UVA):   " << (prop.unifiedAddressing ? "Yes" : "No") << '\n';
            std::cout << "  Async Engine Count:         " << prop.asyncEngineCount << '\n';
            std::cout << "  Concurrent Kernels Run:     " << (prop.concurrentKernels ? "Yes" : "No") << '\n';
            std::cout << '\n';
        }

        std::cout << "Environment verification completed successfully.\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& e) {
        std::cerr << "Exception occurred during environment verification: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
