#include <cuda_dl/core/device_buffer.cuh>
#include <cuda_dl/core/device_tensor.cuh>
#include <cuda_dl/ops/pooling.cuh>

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

// CPU reference functions for MaxPool2D

void cpu_maxpool2d_forward(
    const std::vector<float>& input,
    std::vector<float>& output,
    std::vector<int>& argmax,
    const std::size_t B, const std::size_t C, const std::size_t H, const std::size_t W,
    const std::size_t PH, const std::size_t PW,
    const std::size_t P, const std::size_t S,
    const std::size_t OH, const std::size_t OW)
{
    for (std::size_t b = 0; b < B; ++b) {
        for (std::size_t c = 0; c < C; ++c) {
            for (std::size_t oh = 0; oh < OH; ++oh) {
                for (std::size_t ow = 0; ow < OW; ++ow) {
                    float max_val = -3.402823466e+38F;
                    int max_idx = -1;

                    for (std::size_t ky = 0; ky < PH; ++ky) {
                        const int in_y = static_cast<int>(oh * S) + static_cast<int>(ky) - static_cast<int>(P);
                        if (in_y < 0 || in_y >= static_cast<int>(H)) {
                            continue;
                        }

                        for (std::size_t kx = 0; kx < PW; ++kx) {
                            const int in_x = static_cast<int>(ow * S) + static_cast<int>(kx) - static_cast<int>(P);
                            if (in_x < 0 || in_x >= static_cast<int>(W)) {
                                continue;
                            }

                            const std::size_t in_offset = (((b * C + c) * H + static_cast<std::size_t>(in_y)) * W) + static_cast<std::size_t>(in_x);
                            const float val = input[in_offset];
                            if (val > max_val) {
                                max_val = val;
                                max_idx = static_cast<int>(in_offset);
                            }
                        }
                    }

                    const std::size_t out_offset = (((b * C + c) * OH + oh) * OW) + ow;
                    output[out_offset] = max_val;
                    argmax[out_offset] = max_idx;
                }
            }
        }
    }
}

void cpu_maxpool2d_backward(
    const std::vector<float>& upstream_grad,
    const std::vector<int>& argmax,
    std::vector<float>& downstream_grad)
{
    std::fill(downstream_grad.begin(), downstream_grad.end(), 0.0F);
    for (std::size_t i = 0; i < upstream_grad.size(); ++i) {
        const int target_idx = argmax[i];
        if (target_idx >= 0) {
            downstream_grad[static_cast<std::size_t>(target_idx)] += upstream_grad[i];
        }
    }
}

} // namespace

int main()
{
    try {
        constexpr std::size_t B = 2;
        constexpr std::size_t C = 2;
        constexpr std::size_t H = 4;
        constexpr std::size_t W = 4;

        constexpr std::size_t PH = 2;
        constexpr std::size_t PW = 2;
        constexpr std::size_t P = 0;
        constexpr std::size_t S = 2;

        constexpr std::size_t OH = ((H + 2 * P - PH) / S) + 1;
        constexpr std::size_t OW = ((W + 2 * P - PW) / S) + 1;

        cuda_dl::core::DeviceTensor input({B, C, H, W}, cuda_dl::core::DType::Float32);
        cuda_dl::core::DeviceTensor upstream_grad({B, C, OH, OW}, cuda_dl::core::DType::Float32);

        // Initialize test values on host
        std::vector<float> host_input(input.element_count());
        for (std::size_t i = 0; i < host_input.size(); ++i) {
            host_input[i] = static_cast<float>(static_cast<int>(i % 13) - 6) * 0.2F;
        }

        std::vector<float> host_upstream_grad(upstream_grad.element_count());
        for (std::size_t i = 0; i < host_upstream_grad.size(); ++i) {
            host_upstream_grad[i] = static_cast<float>(i + 1) * 0.1F;
        }

        // Copy host data to GPU
        input.copy_from_host(host_input.data(), host_input.size());
        upstream_grad.copy_from_host(host_upstream_grad.data(), host_upstream_grad.size());

        // -------------------------------------------------------------
        // Test Forward Pass
        // -------------------------------------------------------------
        std::cout << "Testing MaxPool2D Forward Pass..." << std::endl;
        cuda_dl::ops::MaxPool2DForwardResult fwd_result = cuda_dl::ops::maxpool2d_forward(input, PH, PW, P, S);
        
        std::vector<float> host_output(fwd_result.output.element_count());
        std::vector<int> host_argmax(fwd_result.argmax.size());

        fwd_result.output.copy_to_host(host_output.data(), host_output.size());
        fwd_result.argmax.copy_to_host(host_argmax.data(), host_argmax.size());

        // Compute reference on CPU
        std::vector<float> ref_output(fwd_result.output.element_count());
        std::vector<int> ref_argmax(fwd_result.argmax.size());
        cpu_maxpool2d_forward(
            host_input, ref_output, ref_argmax,
            B, C, H, W, PH, PW, P, S, OH, OW);

        // Check forward outputs and cached indices
        float fwd_err = 0.0F;
        for (std::size_t i = 0; i < host_output.size(); ++i) {
            fwd_err = std::max(fwd_err, std::fabs(host_output[i] - ref_output[i]));
            expect(host_argmax[i] == ref_argmax[i], "cached argmax indices mismatch");
        }
        std::cout << "  MaxPool2D Forward Max Error: " << fwd_err << std::endl;
        expect(fwd_err <= 1e-6F, "MaxPool2D Forward validation mismatch");

        // -------------------------------------------------------------
        // Test Backward Pass
        // -------------------------------------------------------------
        std::cout << "Testing MaxPool2D Backward Pass..." << std::endl;
        cuda_dl::core::DeviceTensor downstream_grad = cuda_dl::ops::maxpool2d_backward(input, upstream_grad, fwd_result.argmax);

        std::vector<float> host_downstream_grad(downstream_grad.element_count());
        downstream_grad.copy_to_host(host_downstream_grad.data(), host_downstream_grad.size());

        // Compute reference on CPU
        std::vector<float> ref_downstream_grad(input.element_count());
        cpu_maxpool2d_backward(host_upstream_grad, ref_argmax, ref_downstream_grad);

        float bwd_err = 0.0F;
        for (std::size_t i = 0; i < host_downstream_grad.size(); ++i) {
            bwd_err = std::max(bwd_err, std::fabs(host_downstream_grad[i] - ref_downstream_grad[i]));
        }
        std::cout << "  Input Grad (dX) Max Error: " << bwd_err << std::endl;
        expect(bwd_err <= 1e-6F, "MaxPool2D dX validation mismatch");

        bool rejected_zero_stride = false;
        try {
            static_cast<void>(cuda_dl::ops::maxpool2d_forward(input, PH, PW, P, 0));
        } catch (const std::invalid_argument&) {
            rejected_zero_stride = true;
        }
        expect(rejected_zero_stride, "MaxPool2D accepted zero stride");

        bool rejected_oversized_window = false;
        try {
            static_cast<void>(cuda_dl::ops::maxpool2d_forward(input, H + (2 * P) + 1, PW, P, S));
        } catch (const std::invalid_argument&) {
            rejected_oversized_window = true;
        }
        expect(rejected_oversized_window, "MaxPool2D accepted an oversized window");

        std::cout << "MaxPool2D Layer validation completed successfully." << std::endl;
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Verification failed: " << error.what() << std::endl;
        return EXIT_FAILURE;
    }
}
