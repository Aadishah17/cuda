#include <cuda_dl/core/device_tensor.cuh>
#include <cuda_dl/ops/activations.cuh>

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

float relu_forward_cpu(const float x)
{
    return std::max(0.0F, x);
}

float relu_backward_cpu(const float x, const float dy)
{
    return x > 0.0F ? dy : 0.0F;
}

float sigmoid_forward_cpu(const float x)
{
    return 1.0F / (1.0F + std::exp(-x));
}

float sigmoid_backward_cpu(const float y, const float dy)
{
    return dy * y * (1.0F - y);
}

float tanh_forward_cpu(const float x)
{
    return std::tanh(x);
}

float tanh_backward_cpu(const float y, const float dy)
{
    return dy * (1.0F - (y * y));
}

} // namespace

int main()
{
    try {
        const std::size_t size = 1000;
        cuda_dl::core::DeviceTensor input({size}, cuda_dl::core::DType::Float32);
        cuda_dl::core::DeviceTensor upstream_grad({size}, cuda_dl::core::DType::Float32);

        // Initialize inputs with mixed negative and positive values
        std::vector<float> host_input(size);
        std::vector<float> host_upstream_grad(size);
        for (std::size_t i = 0; i < size; ++i) {
            host_input[i] = static_cast<float>(static_cast<int>(i % 20) - 10) * 0.25F; // range [-2.5, 2.25]
            host_upstream_grad[i] = static_cast<float>(static_cast<int>(i % 5) + 1) * 0.1F; // range [0.1, 0.5]
        }

        input.copy_from_host(host_input.data(), size);
        upstream_grad.copy_from_host(host_upstream_grad.data(), size);

        // -------------------------------------------------------------
        // Test ReLU
        // -------------------------------------------------------------
        std::cout << "Testing ReLU..." << std::endl;
        cuda_dl::core::DeviceTensor relu_out = cuda_dl::ops::relu_forward(input);
        cuda_dl::core::DeviceTensor relu_grad = cuda_dl::ops::relu_backward(input, upstream_grad);

        std::vector<float> host_relu_out(size);
        std::vector<float> host_relu_grad(size);
        relu_out.copy_to_host(host_relu_out.data(), size);
        relu_grad.copy_to_host(host_relu_grad.data(), size);

        float relu_fwd_err = 0.0F;
        float relu_bwd_err = 0.0F;
        for (std::size_t i = 0; i < size; ++i) {
            relu_fwd_err = std::max(relu_fwd_err, std::fabs(host_relu_out[i] - relu_forward_cpu(host_input[i])));
            relu_bwd_err = std::max(relu_bwd_err, std::fabs(host_relu_grad[i] - relu_backward_cpu(host_input[i], host_upstream_grad[i])));
        }
        expect(relu_fwd_err <= 1e-6F, "ReLU forward validation failed");
        expect(relu_bwd_err <= 1e-6F, "ReLU backward validation failed");
        std::cout << "  ReLU Forward Max Error: " << relu_fwd_err << std::endl;
        std::cout << "  ReLU Backward Max Error: " << relu_bwd_err << std::endl;

        // -------------------------------------------------------------
        // Test Sigmoid
        // -------------------------------------------------------------
        std::cout << "Testing Sigmoid..." << std::endl;
        cuda_dl::core::DeviceTensor sigmoid_out = cuda_dl::ops::sigmoid_forward(input);
        // Note: sigmoid_backward accepts (output, upstream_grad)
        cuda_dl::core::DeviceTensor sigmoid_grad = cuda_dl::ops::sigmoid_backward(sigmoid_out, upstream_grad);

        std::vector<float> host_sigmoid_out(size);
        std::vector<float> host_sigmoid_grad(size);
        sigmoid_out.copy_to_host(host_sigmoid_out.data(), size);
        sigmoid_grad.copy_to_host(host_sigmoid_grad.data(), size);

        float sigmoid_fwd_err = 0.0F;
        float sigmoid_bwd_err = 0.0F;
        for (std::size_t i = 0; i < size; ++i) {
            sigmoid_fwd_err = std::max(sigmoid_fwd_err, std::fabs(host_sigmoid_out[i] - sigmoid_forward_cpu(host_input[i])));
            sigmoid_bwd_err = std::max(sigmoid_bwd_err, std::fabs(host_sigmoid_grad[i] - sigmoid_backward_cpu(host_sigmoid_out[i], host_upstream_grad[i])));
        }
        expect(sigmoid_fwd_err <= 1e-6F, "Sigmoid forward validation failed");
        expect(sigmoid_bwd_err <= 1e-6F, "Sigmoid backward validation failed");
        std::cout << "  Sigmoid Forward Max Error: " << sigmoid_fwd_err << std::endl;
        std::cout << "  Sigmoid Backward Max Error: " << sigmoid_bwd_err << std::endl;

        // -------------------------------------------------------------
        // Test Tanh
        // -------------------------------------------------------------
        std::cout << "Testing Tanh..." << std::endl;
        cuda_dl::core::DeviceTensor tanh_out = cuda_dl::ops::tanh_forward(input);
        // Note: tanh_backward accepts (output, upstream_grad)
        cuda_dl::core::DeviceTensor tanh_grad = cuda_dl::ops::tanh_backward(tanh_out, upstream_grad);

        std::vector<float> host_tanh_out(size);
        std::vector<float> host_tanh_grad(size);
        tanh_out.copy_to_host(host_tanh_out.data(), size);
        tanh_grad.copy_to_host(host_tanh_grad.data(), size);

        float tanh_fwd_err = 0.0F;
        float tanh_bwd_err = 0.0F;
        for (std::size_t i = 0; i < size; ++i) {
            tanh_fwd_err = std::max(tanh_fwd_err, std::fabs(host_tanh_out[i] - tanh_forward_cpu(host_input[i])));
            tanh_bwd_err = std::max(tanh_bwd_err, std::fabs(host_tanh_grad[i] - tanh_backward_cpu(host_tanh_out[i], host_upstream_grad[i])));
        }
        expect(tanh_fwd_err <= 1e-6F, "Tanh forward validation failed");
        expect(tanh_bwd_err <= 1e-6F, "Tanh backward validation failed");
        std::cout << "  Tanh Forward Max Error: " << tanh_fwd_err << std::endl;
        std::cout << "  Tanh Backward Max Error: " << tanh_bwd_err << std::endl;

        std::cout << "Activation functions validation completed successfully." << std::endl;
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Verification failed: " << error.what() << std::endl;
        return EXIT_FAILURE;
    }
}
