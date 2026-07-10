#include <cuda_dl/core/device_buffer.cuh>
#include <cuda_dl/core/device_tensor.cuh>
#include <cuda_dl/ops/loss.cuh>

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

std::vector<float> cpu_softmax(const std::vector<float>& logits, const std::size_t B, const std::size_t C)
{
    std::vector<float> probs(B * C);
    for (std::size_t b = 0; b < B; ++b) {
        const std::size_t offset = b * C;

        // Find max logit
        float max_logit = logits[offset];
        for (std::size_t c = 1; c < C; ++c) {
            max_logit = std::max(max_logit, logits[offset + c]);
        }

        // Sum exponents
        float sum_exp = 0.0F;
        for (std::size_t c = 0; c < C; ++c) {
            probs[offset + c] = std::exp(logits[offset + c] - max_logit);
            sum_exp += probs[offset + c];
        }

        // Normalize
        for (std::size_t c = 0; c < C; ++c) {
            probs[offset + c] /= sum_exp;
        }
    }
    return probs;
}

float cpu_cross_entropy_loss(const std::vector<float>& probs, const std::vector<int>& targets, const std::size_t B, const std::size_t C)
{
    float sum_loss = 0.0F;
    for (std::size_t b = 0; b < B; ++b) {
        const int target_class = targets[b];
        const float prob = probs[b * C + static_cast<std::size_t>(target_class)];
        sum_loss += -std::log(prob);
    }
    return sum_loss / static_cast<float>(B);
}

std::vector<float> cpu_softmax_cross_entropy_backward(
    const std::vector<float>& probs, const std::vector<int>& targets, const std::size_t B, const std::size_t C)
{
    std::vector<float> grad(B * C);
    for (std::size_t b = 0; b < B; ++b) {
        for (std::size_t c = 0; c < C; ++c) {
            const float indicator = (static_cast<int>(c) == targets[b]) ? 1.0F : 0.0F;
            grad[b * C + c] = (probs[b * C + c] - indicator) / static_cast<float>(B);
        }
    }
    return grad;
}

} // namespace

int main()
{
    try {
        constexpr std::size_t batch_size = 3;
        constexpr std::size_t classes = 4;

        cuda_dl::core::DeviceTensor logits({batch_size, classes}, cuda_dl::core::DType::Float32);
        cuda_dl::core::DeviceBuffer<int> targets(batch_size);

        // Populate host test values
        std::vector<float> host_logits = {
            1.5F, -0.5F, 2.0F, 0.0F,
            0.5F, 1.2F, -2.0F, -0.8F,
            -1.0F, 0.0F, 0.5F, 2.5F
        };
        std::vector<int> host_targets = {2, 0, 3}; // targets are in bounds [0, 3]

        // Copy host data to GPU
        logits.copy_from_host(host_logits.data(), host_logits.size());
        targets.copy_from_host(host_targets.data(), host_targets.size());

        // -------------------------------------------------------------
        // Test Forward Pass
        // -------------------------------------------------------------
        std::cout << "Testing Softmax Cross-Entropy Forward..." << std::endl;
        cuda_dl::ops::SoftmaxCrossEntropyForwardResult fwd_result = cuda_dl::ops::softmax_cross_entropy_forward(logits, targets);

        std::vector<float> host_probs(fwd_result.probabilities.element_count());
        fwd_result.probabilities.copy_to_host(host_probs.data(), host_probs.size());

        // Compute reference on CPU
        std::vector<float> ref_probs = cpu_softmax(host_logits, batch_size, classes);
        float ref_loss = cpu_cross_entropy_loss(ref_probs, host_targets, batch_size, classes);

        // Check forward softmax outputs and average CE loss
        float prob_err = 0.0F;
        for (std::size_t i = 0; i < host_probs.size(); ++i) {
            prob_err = std::max(prob_err, std::fabs(host_probs[i] - ref_probs[i]));
        }

        const float loss_err = std::fabs(fwd_result.average_loss - ref_loss);

        std::cout << "  Average CE Loss: " << fwd_result.average_loss << " (Ref: " << ref_loss << ")" << std::endl;
        std::cout << "  Softmax Probabilities Max Error: " << prob_err << std::endl;
        std::cout << "  Loss Scalar Error: " << loss_err << std::endl;

        expect(prob_err <= 1.0e-6F, "Softmax probabilities validation mismatch");
        expect(loss_err <= 1.0e-6F, "Cross-Entropy Loss scalar validation mismatch");

        // -------------------------------------------------------------
        // Test Backward Pass
        // -------------------------------------------------------------
        std::cout << "Testing Softmax Cross-Entropy Backward..." << std::endl;
        cuda_dl::core::DeviceTensor dX = cuda_dl::ops::softmax_cross_entropy_backward(fwd_result.probabilities, targets);

        std::vector<float> host_dX(dX.element_count());
        dX.copy_to_host(host_dX.data(), host_dX.size());

        // Compute reference on CPU
        std::vector<float> ref_dX = cpu_softmax_cross_entropy_backward(ref_probs, host_targets, batch_size, classes);

        float dX_err = 0.0F;
        for (std::size_t i = 0; i < host_dX.size(); ++i) {
            dX_err = std::max(dX_err, std::fabs(host_dX[i] - ref_dX[i]));
        }
        std::cout << "  Logits Gradient (dX) Max Error: " << dX_err << std::endl;
        expect(dX_err <= 1.0e-6F, "Logits Gradient dX validation mismatch");

        std::cout << "Softmax and Cross-Entropy Loss Layer validation completed successfully." << std::endl;
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Verification failed: " << error.what() << std::endl;
        return EXIT_FAILURE;
    }
}
