#include <cuda_dl/core/cuda_error.cuh>
#include <cuda_dl/core/device_tensor.cuh>
#include <cuda_dl/core/cuda_event_timer.cuh>
#include <cuda_dl/ops/linear.cuh>
#include <cuda_dl/ops/tensor_ops.cuh>

#include <chrono>
#include <iostream>
#include <vector>

namespace {

void run_gemm_benchmark()
{
    std::cout << "=== GEMM Performance Benchmark ===" << std::endl;
    using cuda_dl::core::DeviceTensor;
    using cuda_dl::core::DType;
    using cuda_dl::core::CudaEvent;
    using cuda_dl::core::elapsed_milliseconds;

    // Benchmark matrix size
    constexpr std::size_t M = 1024;
    constexpr std::size_t K = 1024;
    constexpr std::size_t N = 1024;

    std::cout << "Matrix sizes: A[" << M << "x" << K << "] * B[" << K << "x" << N << "]" << std::endl;

    DeviceTensor A({M, K}, DType::Float32);
    DeviceTensor B({K, N}, DType::Float32);
    A.zero();
    B.zero();

    // Warm-up
    for (int i = 0; i < 5; ++i) {
        auto temp1 = cuda_dl::ops::matmul(A, B);
        auto temp2 = cuda_dl::ops::gemm<false, false>(A, B);
    }
    CUDADL_CUDA_SYNCHRONIZE("warmup completion");

    // 1. Naive MatMul
    constexpr int iterations = 20;
    CudaEvent start_naive, stop_naive;
    start_naive.record();
    for (int i = 0; i < iterations; ++i) {
        auto out = cuda_dl::ops::matmul(A, B);
    }
    stop_naive.record();
    stop_naive.synchronize();
    float time_naive_ms = elapsed_milliseconds(start_naive, stop_naive) / static_cast<float>(iterations);

    // 2. Tiled GEMM
    CudaEvent start_tiled, stop_tiled;
    start_tiled.record();
    for (int i = 0; i < iterations; ++i) {
        auto out = cuda_dl::ops::gemm<false, false>(A, B);
    }
    stop_tiled.record();
    stop_tiled.synchronize();
    float time_tiled_ms = elapsed_milliseconds(start_tiled, stop_tiled) / static_cast<float>(iterations);

    // Calculate GFLOPS: 2 * M * N * K operations
    double ops = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
    double gflops_naive = (ops / (time_naive_ms * 1.0e-3)) / 1.0e9;
    double gflops_tiled = (ops / (time_tiled_ms * 1.0e-3)) / 1.0e9;

    std::cout << "  Naive MatMul: " << time_naive_ms << " ms (" << gflops_naive << " GFLOPS)" << std::endl;
    std::cout << "  Tiled GEMM:   " << time_tiled_ms << " ms (" << gflops_tiled << " GFLOPS)" << std::endl;
    std::cout << "  Speedup:      " << (time_naive_ms / time_tiled_ms) << "x" << std::endl;
}

void run_stream_benchmark()
{
    std::cout << "\n=== Stream Concurrency Benchmark ===" << std::endl;
    using cuda_dl::core::DeviceTensor;
    using cuda_dl::core::DType;
    using cuda_dl::core::CudaEvent;
    using cuda_dl::core::elapsed_milliseconds;

    constexpr std::size_t M = 256;
    constexpr std::size_t K = 256;
    constexpr std::size_t N = 256;
    DeviceTensor A({M, K}, DType::Float32);
    DeviceTensor B({K, N}, DType::Float32);
    A.zero();
    B.zero();

    // Create custom streams
    cudaStream_t stream1, stream2;
    CUDADL_CUDA_CHECK(cudaStreamCreate(&stream1));
    CUDADL_CUDA_CHECK(cudaStreamCreate(&stream2));

    // Warm-up
    for (int i = 0; i < 5; ++i) {
        auto temp = cuda_dl::ops::gemm<false, false>(A, B, stream1);
    }
    CUDADL_CUDA_SYNCHRONIZE("warmup completion");

    // 1. Sequential Execution on Default Stream
    CudaEvent start_seq, stop_seq;
    start_seq.record();
    for (int i = 0; i < 20; ++i) {
        auto res1 = cuda_dl::ops::gemm<false, false>(A, B, nullptr);
        auto res2 = cuda_dl::ops::gemm<false, false>(A, B, nullptr);
    }
    stop_seq.record();
    stop_seq.synchronize();
    float time_seq_ms = elapsed_milliseconds(start_seq, stop_seq);

    // 2. Concurrent Asynchronous Execution on Separate Streams
    CudaEvent start_con, stop_con;
    start_con.record();
    for (int i = 0; i < 20; ++i) {
        auto res1 = cuda_dl::ops::gemm<false, false>(A, B, stream1);
        auto res2 = cuda_dl::ops::gemm<false, false>(A, B, stream2);
    }
    stop_con.record();
    stop_con.synchronize();
    float time_con_ms = elapsed_milliseconds(start_con, stop_con);

    std::cout << "  Sequential (default stream):   " << time_seq_ms << " ms" << std::endl;
    std::cout << "  Concurrent (separate streams): " << time_con_ms << " ms" << std::endl;
    std::cout << "  Overlap Efficiency:            " << (time_seq_ms / time_con_ms) << "x" << std::endl;

    CUDADL_CUDA_CHECK(cudaStreamDestroy(stream1));
    CUDADL_CUDA_CHECK(cudaStreamDestroy(stream2));
}

} // namespace

int main()
{
    try {
        run_gemm_benchmark();
        run_stream_benchmark();
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Benchmark failed: " << error.what() << std::endl;
        return EXIT_FAILURE;
    }
}

