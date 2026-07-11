#include <cuda_dl/core/autograd.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
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

void expect_close(
    const std::vector<float>& actual,
    const std::vector<float>& expected,
    const char* const message)
{
    expect(actual.size() == expected.size(), message);

    float max_error = 0.0F;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        max_error = std::max(max_error, std::fabs(actual[index] - expected[index]));
    }

    if (max_error > 1.0e-6F) {
        throw std::runtime_error(message);
    }
}

std::vector<float> copy_to_host(const cuda_dl::core::DeviceTensor& tensor)
{
    std::vector<float> host(tensor.element_count());
    tensor.copy_to_host(host.data(), host.size());
    return host;
}

} // namespace

int main()
{
    try {
        using cuda_dl::core::DeviceTensor;
        using cuda_dl::core::DType;
        using cuda_dl::core::Variable;

        auto x = Variable::create(DeviceTensor({3}, DType::Float32));
        auto y = Variable::create(DeviceTensor({3}, DType::Float32));

        const std::vector<float> x_host{1.0F, 2.0F, 3.0F};
        const std::vector<float> y_host{4.0F, 5.0F, 6.0F};
        x->data().copy_from_host(x_host.data(), x_host.size());
        y->data().copy_from_host(y_host.data(), y_host.size());

        // output = x * y + x. x has two paths to the output, so its
        // derivative must accumulate as y + 1 rather than overwrite.
        auto product = cuda_dl::autograd::multiply(x, y);
        auto output = cuda_dl::autograd::add(product, x);

        expect_close(copy_to_host(output->data()), {5.0F, 12.0F, 21.0F}, "forward result mismatch");

        output->backward();
        expect_close(copy_to_host(x->grad()), {5.0F, 6.0F, 7.0F}, "first x gradient mismatch");
        expect_close(copy_to_host(y->grad()), {1.0F, 2.0F, 3.0F}, "first y gradient mismatch");

        // Leaf gradients accumulate across backward calls; intermediate
        // gradients are reset internally before each traversal.
        output->backward();
        expect_close(copy_to_host(x->grad()), {10.0F, 12.0F, 14.0F}, "x gradient did not accumulate");
        expect_close(copy_to_host(y->grad()), {2.0F, 4.0F, 6.0F}, "y gradient did not accumulate");

        x->zero_grad();
        y->zero_grad();
        expect_close(copy_to_host(x->grad()), {0.0F, 0.0F, 0.0F}, "x zero_grad mismatch");
        expect_close(copy_to_host(y->grad()), {0.0F, 0.0F, 0.0F}, "y zero_grad mismatch");

        auto frozen = Variable::create(DeviceTensor({3}, DType::Float32), false);
        frozen->data().copy_from_host(y_host.data(), y_host.size());
        auto frozen_output = cuda_dl::autograd::multiply(x, frozen);
        frozen_output->backward();
        expect_close(copy_to_host(x->grad()), y_host, "frozen-parent x gradient mismatch");
        expect_close(copy_to_host(frozen->grad()), {0.0F, 0.0F, 0.0F}, "frozen variable received a gradient");

        auto relu_input = Variable::create(DeviceTensor({4}, DType::Float32));
        const std::vector<float> relu_input_host{-2.0F, 0.0F, 1.5F, 3.0F};
        relu_input->data().copy_from_host(relu_input_host.data(), relu_input_host.size());
        auto relu_output = cuda_dl::autograd::relu(relu_input);
        expect(relu_output->op() == "relu", "relu graph operation label mismatch");
        expect_close(copy_to_host(relu_output->data()), {0.0F, 0.0F, 1.5F, 3.0F}, "relu forward mismatch");

        relu_output->backward();
        expect_close(copy_to_host(relu_input->grad()), {0.0F, 0.0F, 1.0F, 1.0F}, "relu backward mismatch");

        auto linear_input = Variable::create(DeviceTensor({2, 2}, DType::Float32));
        auto linear_weight = Variable::create(DeviceTensor({3, 2}, DType::Float32));
        auto linear_bias = Variable::create(DeviceTensor({3}, DType::Float32));
        const std::vector<float> linear_input_host{1.0F, 2.0F, 3.0F, 4.0F};
        const std::vector<float> linear_weight_host{1.0F, -1.0F, 0.5F, 2.0F, -2.0F, 1.0F};
        const std::vector<float> linear_bias_host{0.1F, -0.2F, 0.3F};
        linear_input->data().copy_from_host(linear_input_host.data(), linear_input_host.size());
        linear_weight->data().copy_from_host(linear_weight_host.data(), linear_weight_host.size());
        linear_bias->data().copy_from_host(linear_bias_host.data(), linear_bias_host.size());

        auto linear_output = cuda_dl::autograd::linear(linear_input, linear_weight, linear_bias);
        expect(linear_output->op() == "linear", "linear graph operation label mismatch");
        expect_close(
            copy_to_host(linear_output->data()),
            {-0.9F, 4.3F, 0.3F, -0.9F, 9.3F, -1.7F},
            "linear forward mismatch");

        linear_output->backward();
        expect_close(copy_to_host(linear_input->grad()), {-0.5F, 2.0F, -0.5F, 2.0F}, "linear input gradient mismatch");
        expect_close(
            copy_to_host(linear_weight->grad()),
            {4.0F, 6.0F, 4.0F, 6.0F, 4.0F, 6.0F},
            "linear weight gradient mismatch");
        expect_close(copy_to_host(linear_bias->grad()), {2.0F, 2.0F, 2.0F}, "linear bias gradient mismatch");

        auto conv_input = Variable::create(DeviceTensor({1, 1, 3, 3}, DType::Float32));
        auto conv_weight = Variable::create(DeviceTensor({1, 1, 2, 2}, DType::Float32));
        auto conv_bias = Variable::create(DeviceTensor({1}, DType::Float32));
        const std::vector<float> conv_input_host{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F};
        const std::vector<float> conv_weight_host{1.0F, 2.0F, 3.0F, 4.0F};
        const std::vector<float> conv_bias_host{0.5F};
        conv_input->data().copy_from_host(conv_input_host.data(), conv_input_host.size());
        conv_weight->data().copy_from_host(conv_weight_host.data(), conv_weight_host.size());
        conv_bias->data().copy_from_host(conv_bias_host.data(), conv_bias_host.size());

        auto conv_output = cuda_dl::autograd::conv2d(conv_input, conv_weight, conv_bias);
        expect(conv_output->op() == "conv2d", "conv2d graph operation label mismatch");
        expect_close(
            copy_to_host(conv_output->data()),
            {37.5F, 47.5F, 67.5F, 77.5F},
            "conv2d forward mismatch");

        conv_output->backward();
        expect_close(
            copy_to_host(conv_input->grad()),
            {1.0F, 3.0F, 2.0F, 4.0F, 10.0F, 6.0F, 3.0F, 7.0F, 4.0F},
            "conv2d input gradient mismatch");
        expect_close(copy_to_host(conv_weight->grad()), {12.0F, 16.0F, 24.0F, 28.0F}, "conv2d weight gradient mismatch");
        expect_close(copy_to_host(conv_bias->grad()), {4.0F}, "conv2d bias gradient mismatch");

        auto pool_input = Variable::create(DeviceTensor({1, 1, 3, 3}, DType::Float32));
        const std::vector<float> pool_input_host{1.0F, 1.0F, 1.0F, 1.0F, 9.0F, 1.0F, 1.0F, 1.0F, 1.0F};
        pool_input->data().copy_from_host(pool_input_host.data(), pool_input_host.size());
        auto pool_output = cuda_dl::autograd::maxpool2d(pool_input, 2, 2, 0, 1);
        expect(pool_output->op() == "maxpool2d", "maxpool2d graph operation label mismatch");
        expect_close(copy_to_host(pool_output->data()), {9.0F, 9.0F, 9.0F, 9.0F}, "maxpool2d forward mismatch");

        pool_output->backward();
        expect_close(
            copy_to_host(pool_input->grad()),
            {0.0F, 0.0F, 0.0F, 0.0F, 4.0F, 0.0F, 0.0F, 0.0F, 0.0F},
            "maxpool2d overlapping gradient mismatch");

        auto loss_logits = Variable::create(DeviceTensor({2, 2}, DType::Float32));
        const std::vector<float> loss_logits_host{0.0F, 0.0F, 0.0F, 0.0F};
        loss_logits->data().copy_from_host(loss_logits_host.data(), loss_logits_host.size());
        cuda_dl::core::DeviceBuffer<int> loss_targets(2);
        const std::vector<int> original_targets{0, 1};
        loss_targets.copy_from_host(original_targets.data(), original_targets.size());

        auto loss_output = cuda_dl::autograd::softmax_cross_entropy(loss_logits, loss_targets);
        expect(loss_output->op() == "softmax_cross_entropy", "loss graph operation label mismatch");
        expect_close(copy_to_host(loss_output->data()), {std::log(2.0F)}, "softmax cross-entropy forward mismatch");

        const std::vector<int> overwritten_targets{1, 0};
        loss_targets.copy_from_host(overwritten_targets.data(), overwritten_targets.size());
        loss_output->backward();
        expect_close(
            copy_to_host(loss_logits->grad()),
            {-0.25F, 0.25F, 0.25F, -0.25F},
            "softmax cross-entropy saved-label gradient mismatch");

        bool rejected_shape_mismatch = false;
        try {
            auto mismatched = Variable::create(DeviceTensor({2}, DType::Float32));
            static_cast<void>(cuda_dl::autograd::add(x, mismatched));
        } catch (const std::invalid_argument&) {
            rejected_shape_mismatch = true;
        }
        expect(rejected_shape_mismatch, "autograd add accepted mismatched shapes");

        std::cout << "Autograd elementwise forward, backward, and gradient accumulation verified." << std::endl;
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Verification failed: " << error.what() << std::endl;
        return EXIT_FAILURE;
    }
}
