#include <cuda_dl/core/device_tensor.cuh>
#include <cuda_dl/ops/conv2d.cuh>

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

// CPU reference functions for 2D Convolution

std::size_t get_nchw_offset_cpu(
    const std::size_t b, const std::size_t c, const std::size_t h, const std::size_t w,
    const std::size_t C, const std::size_t H, const std::size_t W)
{
    return (((b * C + c) * H + h) * W + w);
}

std::vector<float> cpu_conv2d_forward(
    const std::vector<float>& input,
    const std::vector<float>& weight,
    const std::vector<float>& bias,
    const std::size_t B, const std::size_t C, const std::size_t H, const std::size_t W,
    const std::size_t F, const std::size_t KH, const std::size_t KW,
    const std::size_t P, const std::size_t S,
    const std::size_t OH, const std::size_t OW)
{
    std::vector<float> output(B * F * OH * OW, 0.0F);

    for (std::size_t b = 0; b < B; ++b) {
        for (std::size_t f = 0; f < F; ++f) {
            for (std::size_t oh = 0; oh < OH; ++oh) {
                for (std::size_t ow = 0; ow < OW; ++ow) {
                    float sum = bias[f];
                    for (std::size_t c = 0; c < C; ++c) {
                        for (std::size_t ky = 0; ky < KH; ++ky) {
                            const int in_y = static_cast<int>(oh * S) + static_cast<int>(ky) - static_cast<int>(P);
                            if (in_y < 0 || in_y >= static_cast<int>(H)) {
                                continue;
                            }

                            for (std::size_t kx = 0; kx < KW; ++kx) {
                                const int in_x = static_cast<int>(ow * S) + static_cast<int>(kx) - static_cast<int>(P);
                                if (in_x < 0 || in_x >= static_cast<int>(W)) {
                                    continue;
                                }

                                const std::size_t in_idx = get_nchw_offset_cpu(b, c, static_cast<std::size_t>(in_y), static_cast<std::size_t>(in_x), C, H, W);
                                const std::size_t weight_idx = (((f * C + c) * KH + ky) * KW) + kx;

                                sum += input[in_idx] * weight[weight_idx];
                            }
                        }
                    }
                    output[get_nchw_offset_cpu(b, f, oh, ow, F, OH, OW)] = sum;
                }
            }
        }
    }

    return output;
}

std::vector<float> cpu_conv2d_backward_input(
    const std::vector<float>& upstream_grad,
    const std::vector<float>& weight,
    const std::size_t B, const std::size_t C, const std::size_t H, const std::size_t W,
    const std::size_t F, const std::size_t KH, const std::size_t KW,
    const std::size_t P, const std::size_t S,
    const std::size_t OH, const std::size_t OW)
{
    std::vector<float> input_grad(B * C * H * W, 0.0F);

    for (std::size_t b = 0; b < B; ++b) {
        for (std::size_t c = 0; c < C; ++c) {
            for (std::size_t h = 0; h < H; ++h) {
                for (std::size_t w = 0; w < W; ++w) {
                    float sum = 0.0F;
                    for (std::size_t f = 0; f < F; ++f) {
                        for (std::size_t ky = 0; ky < KH; ++ky) {
                            const int padded_y = static_cast<int>(h) + static_cast<int>(P);
                            const int offset_y = padded_y - static_cast<int>(ky);
                            if (offset_y < 0 || (offset_y % static_cast<int>(S)) != 0) {
                                continue;
                            }
                            const std::size_t oh = static_cast<std::size_t>(offset_y / static_cast<int>(S));
                            if (oh >= OH) {
                                continue;
                            }

                            for (std::size_t kx = 0; kx < KW; ++kx) {
                                const int padded_x = static_cast<int>(w) + static_cast<int>(P);
                                const int offset_x = padded_x - static_cast<int>(kx);
                                if (offset_x < 0 || (offset_x % static_cast<int>(S)) != 0) {
                                    continue;
                                }
                                const std::size_t ow = static_cast<std::size_t>(offset_x / static_cast<int>(S));
                                if (ow >= OW) {
                                    continue;
                                }

                                const std::size_t grad_idx = get_nchw_offset_cpu(b, f, oh, ow, F, OH, OW);
                                const std::size_t weight_idx = (((f * C + c) * KH + ky) * KW) + kx;

                                sum += upstream_grad[grad_idx] * weight[weight_idx];
                            }
                        }
                    }
                    input_grad[get_nchw_offset_cpu(b, c, h, w, C, H, W)] = sum;
                }
            }
        }
    }

    return input_grad;
}

std::vector<float> cpu_conv2d_backward_weight(
    const std::vector<float>& input,
    const std::vector<float>& upstream_grad,
    const std::size_t B, const std::size_t C, const std::size_t H, const std::size_t W,
    const std::size_t F, const std::size_t KH, const std::size_t KW,
    const std::size_t P, const std::size_t S,
    const std::size_t OH, const std::size_t OW)
{
    std::vector<float> weight_grad(F * C * KH * KW, 0.0F);

    for (std::size_t f = 0; f < F; ++f) {
        for (std::size_t c = 0; c < C; ++c) {
            for (std::size_t ky = 0; ky < KH; ++ky) {
                for (std::size_t kx = 0; kx < KW; ++kx) {
                    float sum = 0.0F;
                    for (std::size_t b = 0; b < B; ++b) {
                        for (std::size_t oh = 0; oh < OH; ++oh) {
                            const int in_y = static_cast<int>(oh * S) + static_cast<int>(ky) - static_cast<int>(P);
                            if (in_y < 0 || in_y >= static_cast<int>(H)) {
                                continue;
                            }

                            for (std::size_t ow = 0; ow < OW; ++ow) {
                                const int in_x = static_cast<int>(ow * S) + static_cast<int>(kx) - static_cast<int>(P);
                                if (in_x < 0 || in_x >= static_cast<int>(W)) {
                                    continue;
                                }

                                const std::size_t in_idx = get_nchw_offset_cpu(b, c, static_cast<std::size_t>(in_y), static_cast<std::size_t>(in_x), C, H, W);
                                const std::size_t grad_idx = get_nchw_offset_cpu(b, f, oh, ow, F, OH, OW);

                                sum += upstream_grad[grad_idx] * input[in_idx];
                            }
                        }
                    }
                    weight_grad[(((f * C + c) * KH + ky) * KW) + kx] = sum;
                }
            }
        }
    }

    return weight_grad;
}

std::vector<float> cpu_conv2d_backward_bias(
    const std::vector<float>& upstream_grad,
    const std::size_t B, const std::size_t F,
    const std::size_t OH, const std::size_t OW)
{
    std::vector<float> bias_grad(F, 0.0F);
    for (std::size_t f = 0; f < F; ++f) {
        float sum = 0.0F;
        for (std::size_t b = 0; b < B; ++b) {
            for (std::size_t oh = 0; oh < OH; ++oh) {
                for (std::size_t ow = 0; ow < OW; ++ow) {
                    sum += upstream_grad[get_nchw_offset_cpu(b, f, oh, ow, F, OH, OW)];
                }
            }
        }
        bias_grad[f] = sum;
    }
    return bias_grad;
}

} // namespace

int main()
{
    try {
        constexpr std::size_t B = 2;
        constexpr std::size_t C = 2;
        constexpr std::size_t H = 4;
        constexpr std::size_t W = 4;

        constexpr std::size_t F = 2;
        constexpr std::size_t KH = 3;
        constexpr std::size_t KW = 3;

        constexpr std::size_t P = 1;
        constexpr std::size_t S = 1;

        constexpr std::size_t OH = ((H + 2 * P - KH) / S) + 1;
        constexpr std::size_t OW = ((W + 2 * P - KW) / S) + 1;

        cuda_dl::core::DeviceTensor input({B, C, H, W}, cuda_dl::core::DType::Float32);
        cuda_dl::core::DeviceTensor weight({F, C, KH, KW}, cuda_dl::core::DType::Float32);
        cuda_dl::core::DeviceTensor bias({F}, cuda_dl::core::DType::Float32);
        cuda_dl::core::DeviceTensor upstream_grad({B, F, OH, OW}, cuda_dl::core::DType::Float32);

        // Initialize test values on host
        std::vector<float> host_input(input.element_count());
        for (std::size_t i = 0; i < host_input.size(); ++i) {
            host_input[i] = static_cast<float>(static_cast<int>(i % 17) - 8) * 0.25F;
        }

        std::vector<float> host_weight(weight.element_count());
        for (std::size_t i = 0; i < host_weight.size(); ++i) {
            host_weight[i] = static_cast<float>(static_cast<int>(i % 7) - 3) * 0.5F;
        }

        std::vector<float> host_bias = {0.25F, -0.5F};

        std::vector<float> host_upstream_grad(upstream_grad.element_count());
        for (std::size_t i = 0; i < host_upstream_grad.size(); ++i) {
            host_upstream_grad[i] = static_cast<float>(static_cast<int>(i % 11) - 5) * 0.1F;
        }

        // Copy host data to GPU
        input.copy_from_host(host_input.data(), host_input.size());
        weight.copy_from_host(host_weight.data(), host_weight.size());
        bias.copy_from_host(host_bias.data(), host_bias.size());
        upstream_grad.copy_from_host(host_upstream_grad.data(), host_upstream_grad.size());

        // -------------------------------------------------------------
        // Test Forward Pass
        // -------------------------------------------------------------
        std::cout << "Testing Conv2D Forward Pass..." << std::endl;
        cuda_dl::core::DeviceTensor output = cuda_dl::ops::conv2d_forward(input, weight, bias, P, S);
        std::vector<float> host_output(output.element_count());
        output.copy_to_host(host_output.data(), host_output.size());

        std::vector<float> ref_output = cpu_conv2d_forward(
            host_input, host_weight, host_bias,
            B, C, H, W, F, KH, KW, P, S, OH, OW);

        float fwd_err = 0.0F;
        for (std::size_t i = 0; i < host_output.size(); ++i) {
            fwd_err = std::max(fwd_err, std::fabs(host_output[i] - ref_output[i]));
        }
        std::cout << "  Conv2D Forward Max Error: " << fwd_err << std::endl;
        expect(fwd_err <= 1e-6F, "Conv2D Forward validation mismatch");

        // -------------------------------------------------------------
        // Test Backward Pass
        // -------------------------------------------------------------
        std::cout << "Testing Conv2D Backward Pass..." << std::endl;
        cuda_dl::ops::Conv2DBackwardResult bwd_result = cuda_dl::ops::conv2d_backward(input, weight, upstream_grad, P, S);

        std::vector<float> host_input_grad(bwd_result.input_grad.element_count());
        std::vector<float> host_weight_grad(bwd_result.weight_grad.element_count());
        std::vector<float> host_bias_grad(bwd_result.bias_grad.element_count());

        bwd_result.input_grad.copy_to_host(host_input_grad.data(), host_input_grad.size());
        bwd_result.weight_grad.copy_to_host(host_weight_grad.data(), host_weight_grad.size());
        bwd_result.bias_grad.copy_to_host(host_bias_grad.data(), host_bias_grad.size());

        std::vector<float> ref_input_grad = cpu_conv2d_backward_input(
            host_upstream_grad, host_weight,
            B, C, H, W, F, KH, KW, P, S, OH, OW);

        std::vector<float> ref_weight_grad = cpu_conv2d_backward_weight(
            host_input, host_upstream_grad,
            B, C, H, W, F, KH, KW, P, S, OH, OW);

        std::vector<float> ref_bias_grad = cpu_conv2d_backward_bias(
            host_upstream_grad, B, F, OH, OW);

        // Compare input gradients (dX)
        float dx_err = 0.0F;
        for (std::size_t i = 0; i < host_input_grad.size(); ++i) {
            dx_err = std::max(dx_err, std::fabs(host_input_grad[i] - ref_input_grad[i]));
        }
        std::cout << "  Input Grad (dX) Max Error: " << dx_err << std::endl;
        expect(dx_err <= 1e-6F, "Conv2D dX validation mismatch");

        // Compare weight gradients (dW)
        float dw_err = 0.0F;
        for (std::size_t i = 0; i < host_weight_grad.size(); ++i) {
            dw_err = std::max(dw_err, std::fabs(host_weight_grad[i] - ref_weight_grad[i]));
        }
        std::cout << "  Weight Grad (dW) Max Error: " << dw_err << std::endl;
        expect(dw_err <= 1e-6F, "Conv2D dW validation mismatch");

        // Compare bias gradients (db)
        float db_err = 0.0F;
        for (std::size_t i = 0; i < host_bias_grad.size(); ++i) {
            db_err = std::max(db_err, std::fabs(host_bias_grad[i] - ref_bias_grad[i]));
        }
        std::cout << "  Bias Grad (db) Max Error: " << db_err << std::endl;
        expect(db_err <= 1e-6F, "Conv2D db validation mismatch");

        std::cout << "2D Convolutional Layer validation completed successfully." << std::endl;
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Verification failed: " << error.what() << std::endl;
        return EXIT_FAILURE;
    }
}
