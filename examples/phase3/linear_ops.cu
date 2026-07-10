#include <cuda_dl/core/device_tensor.cuh>
#include <cuda_dl/ops/linear.cuh>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void expect(const bool condition, const char* const message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

// CPU reference functions

std::vector<float> cpu_gemm(
    const std::vector<float>& lhs, bool trans_lhs,
    const std::vector<float>& rhs, bool trans_rhs,
    const std::size_t M, const std::size_t K, const std::size_t N,
    const std::size_t lhs_cols, const std::size_t rhs_cols)
{
    std::vector<float> output(M * N, 0.0F);
    for (std::size_t r = 0; r < M; ++r) {
        for (std::size_t c = 0; c < N; ++c) {
            float sum = 0.0F;
            for (std::size_t k = 0; k < K; ++k) {
                const std::size_t lhs_idx = trans_lhs ? (k * M + r) : (r * lhs_cols + k);
                const std::size_t rhs_idx = trans_rhs ? (c * K + k) : (k * rhs_cols + c);
                sum += lhs[lhs_idx] * rhs[rhs_idx];
            }
            output[r * N + c] = sum;
        }
    }
    return output;
}

} // namespace

int main()
{
    try {
        constexpr std::size_t batch_size = 3;
        constexpr std::size_t in_features = 4;
        constexpr std::size_t out_features = 2;

        cuda_dl::core::DeviceTensor input({batch_size, in_features}, cuda_dl::core::DType::Float32);
        cuda_dl::core::DeviceTensor weight({out_features, in_features}, cuda_dl::core::DType::Float32);
        cuda_dl::core::DeviceTensor bias({out_features}, cuda_dl::core::DType::Float32);
        cuda_dl::core::DeviceTensor upstream_grad({batch_size, out_features}, cuda_dl::core::DType::Float32);

        // Populate host test values
        std::vector<float> host_input = {
            1.0F, 2.0F, 3.0F, 4.0F,
            -0.5F, 1.5F, -2.0F, 0.0F,
            0.5F, -1.0F, 1.0F, -0.5F
        };
        std::vector<float> host_weight = {
            0.1F, -0.2F, 0.3F, 0.4F,
            -0.5F, 0.6F, 0.7F, -0.8F
        };
        std::vector<float> host_bias = {
            0.5F, -0.2F
        };
        std::vector<float> host_upstream_grad = {
            0.1F, -0.1F,
            0.2F, 0.3F,
            -0.2F, 0.1F
        };

        // Copy data to GPU
        input.copy_from_host(host_input.data(), host_input.size());
        weight.copy_from_host(host_weight.data(), host_weight.size());
        bias.copy_from_host(host_bias.data(), host_bias.size());
        upstream_grad.copy_from_host(host_upstream_grad.data(), host_upstream_grad.size());

        // -------------------------------------------------------------
        // Test Forward Pass
        // -------------------------------------------------------------
        std::cout << "Testing Linear Forward Pass..." << std::endl;
        cuda_dl::core::DeviceTensor output = cuda_dl::ops::linear_forward(input, weight, bias);
        std::vector<float> host_output(output.element_count());
        output.copy_to_host(host_output.data(), host_output.size());

        // Compute forward reference on CPU: Y = X * W^T + bias
        std::vector<float> ref_output = cpu_gemm(
            host_input, false,
            host_weight, true,
            batch_size, in_features, out_features,
            in_features, in_features);

        for (std::size_t r = 0; r < batch_size; ++r) {
            for (std::size_t c = 0; c < out_features; ++c) {
                ref_output[r * out_features + c] += host_bias[c];
            }
        }

        float fwd_err = 0.0F;
        for (std::size_t i = 0; i < host_output.size(); ++i) {
            fwd_err = std::max(fwd_err, std::fabs(host_output[i] - ref_output[i]));
        }
        std::cout << "  Forward Max Error: " << fwd_err << std::endl;
        expect(fwd_err <= 1.0e-6F, "Forward validation mismatch");

        // -------------------------------------------------------------
        // Test Backward Pass
        // -------------------------------------------------------------
        std::cout << "Testing Linear Backward Pass..." << std::endl;
        cuda_dl::ops::LinearBackwardResult bwd_result = cuda_dl::ops::linear_backward(input, weight, upstream_grad);

        std::vector<float> host_input_grad(bwd_result.input_grad.element_count());
        std::vector<float> host_weight_grad(bwd_result.weight_grad.element_count());
        std::vector<float> host_bias_grad(bwd_result.bias_grad.element_count());

        bwd_result.input_grad.copy_to_host(host_input_grad.data(), host_input_grad.size());
        bwd_result.weight_grad.copy_to_host(host_weight_grad.data(), host_weight_grad.size());
        bwd_result.bias_grad.copy_to_host(host_bias_grad.data(), host_bias_grad.size());

        // Compute backward reference on CPU
        // 1. dX = dY * W (M = B, K = out, N = in)
        std::vector<float> ref_input_grad = cpu_gemm(
            host_upstream_grad, false,
            host_weight, false,
            batch_size, out_features, in_features,
            out_features, in_features);

        // 2. dW = dY^T * X (M = out, K = B, N = in)
        std::vector<float> ref_weight_grad = cpu_gemm(
            host_upstream_grad, true,
            host_input, false,
            out_features, batch_size, in_features,
            out_features, in_features);

        // 3. db = sum_row(dY)
        std::vector<float> ref_bias_grad(out_features, 0.0F);
        for (std::size_t r = 0; r < batch_size; ++r) {
            for (std::size_t c = 0; c < out_features; ++c) {
                ref_bias_grad[c] += host_upstream_grad[r * out_features + c];
            }
        }

        // Compare gradients
        float dx_err = 0.0F;
        for (std::size_t i = 0; i < host_input_grad.size(); ++i) {
            dx_err = std::max(dx_err, std::fabs(host_input_grad[i] - ref_input_grad[i]));
        }
        std::cout << "  Input Grad (dX) Max Error: " << dx_err << std::endl;
        expect(dx_err <= 1.0e-6F, "dX validation mismatch");

        float dw_err = 0.0F;
        for (std::size_t i = 0; i < host_weight_grad.size(); ++i) {
            dw_err = std::max(dw_err, std::fabs(host_weight_grad[i] - ref_weight_grad[i]));
        }
        std::cout << "  Weight Grad (dW) Max Error: " << dw_err << std::endl;
        expect(dw_err <= 1.0e-6F, "dW validation mismatch");

        float db_err = 0.0F;
        for (std::size_t i = 0; i < host_bias_grad.size(); ++i) {
            db_err = std::max(db_err, std::fabs(host_bias_grad[i] - ref_bias_grad[i]));
        }
        std::cout << "  Bias Grad (db) Max Error: " << db_err << std::endl;
        expect(db_err <= 1.0e-6F, "db validation mismatch");

        std::cout << "Fully Connected / Linear Layer validation completed successfully." << std::endl;
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Verification failed: " << error.what() << std::endl;
        return EXIT_FAILURE;
    }
}
