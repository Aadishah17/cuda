#include <cuda_dl/core/autograd.hpp>
#include <cuda_dl/core/optimizer.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <memory>
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
    const char* const message,
    const float tolerance = 1.0e-6F)
{
    expect(actual.size() == expected.size(), message);

    float max_error = 0.0F;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        max_error = std::max(max_error, std::fabs(actual[index] - expected[index]));
    }

    if (max_error > tolerance) {
        std::cerr << "Expected: ";
        for (float v : expected) std::cerr << v << " ";
        std::cerr << "\nActual:   ";
        for (float v : actual) std::cerr << v << " ";
        std::cerr << "\nMax Error: " << max_error << std::endl;
        throw std::runtime_error(message);
    }
}

std::vector<float> copy_to_host(const cuda_dl::core::DeviceTensor& tensor)
{
    std::vector<float> host(tensor.element_count());
    tensor.copy_to_host(host.data(), host.size());
    return host;
}

void test_sgd()
{
    std::cout << "Testing SGD..." << std::endl;
    using cuda_dl::core::DeviceTensor;
    using cuda_dl::core::DType;
    using cuda_dl::core::Variable;
    using cuda_dl::core::SGD;

    auto w = Variable::create(DeviceTensor({3}, DType::Float32));
    const std::vector<float> w_init{1.0F, 2.0F, 3.0F};
    const std::vector<float> g_init{0.1F, -0.2F, 0.5F};
    w->data().copy_from_host(w_init.data(), w_init.size());
    w->grad().copy_from_host(g_init.data(), g_init.size());

    // Basic SGD update: w = w - lr * g
    const float lr = 0.1F;
    SGD opt({w}, lr, 0.0F, 0.0F);
    opt.step();

    std::vector<float> expected{
        1.0F - lr * 0.1F,
        2.0F - lr * (-0.2F),
        3.0F - lr * 0.5F
    };
    expect_close(copy_to_host(w->data()), expected, "Basic SGD update failed");

    // SGD with weight decay: g_new = g + wd * w; w = w - lr * g_new
    const float wd = 0.01F;
    SGD opt_wd({w}, lr, 0.0F, wd);
    // Reset gradient for next check
    w->grad().copy_from_host(g_init.data(), g_init.size());
    opt_wd.step();

    // w is currently expected: {0.99, 2.02, 2.95}
    std::vector<float> expected_wd{
        0.99F - lr * (0.1F + wd * 0.99F),
        2.02F - lr * (-0.2F + wd * 2.02F),
        2.95F - lr * (0.5F + wd * 2.95F)
    };
    expect_close(copy_to_host(w->data()), expected_wd, "SGD with weight decay failed");
}

void test_sgd_momentum()
{
    std::cout << "Testing SGD with Momentum..." << std::endl;
    using cuda_dl::core::DeviceTensor;
    using cuda_dl::core::DType;
    using cuda_dl::core::Variable;
    using cuda_dl::core::SGD;

    auto w = Variable::create(DeviceTensor({3}, DType::Float32));
    const std::vector<float> w_init{1.0F, 2.0F, 3.0F};
    const std::vector<float> g_init{0.1F, -0.2F, 0.5F};
    w->data().copy_from_host(w_init.data(), w_init.size());
    w->grad().copy_from_host(g_init.data(), g_init.size());

    const float lr = 0.1F;
    const float momentum = 0.9F;
    SGD opt({w}, lr, momentum, 0.0F);

    // Step 1:
    // v_0 = 0
    // v_1 = momentum * v_0 + g = g = {0.1, -0.2, 0.5}
    // w_1 = w_0 - lr * v_1 = {0.99, 2.02, 2.95}
    opt.step();

    std::vector<float> expected_step1{0.99F, 2.02F, 2.95F};
    expect_close(copy_to_host(w->data()), expected_step1, "SGD Momentum step 1 failed");

    // Step 2:
    // v_2 = momentum * v_1 + g = 0.9 * {0.1, -0.2, 0.5} + {0.1, -0.2, 0.5} = {0.19, -0.38, 0.95}
    // w_2 = w_1 - lr * v_2 = {0.99, 2.02, 2.95} - 0.1 * {0.19, -0.38, 0.95} = {0.971, 2.058, 2.855}
    w->grad().copy_from_host(g_init.data(), g_init.size());
    opt.step();

    std::vector<float> expected_step2{0.971F, 2.058F, 2.855F};
    expect_close(copy_to_host(w->data()), expected_step2, "SGD Momentum step 2 failed");
}

void test_rmsprop()
{
    std::cout << "Testing RMSProp..." << std::endl;
    using cuda_dl::core::DeviceTensor;
    using cuda_dl::core::DType;
    using cuda_dl::core::Variable;
    using cuda_dl::core::RMSProp;

    auto w = Variable::create(DeviceTensor({3}, DType::Float32));
    const std::vector<float> w_init{1.0F, 2.0F, 3.0F};
    const std::vector<float> g_init{0.1F, -0.2F, 0.5F};
    w->data().copy_from_host(w_init.data(), w_init.size());
    w->grad().copy_from_host(g_init.data(), g_init.size());

    const float lr = 0.1F;
    const float alpha = 0.9F;
    const float eps = 1e-8F;
    RMSProp opt({w}, lr, alpha, eps, 0.0F);

    // Step 1:
    // v_0 = 0
    // v_1 = alpha * v_0 + (1 - alpha) * g^2 = 0.1 * g^2 = {0.001, 0.004, 0.025}
    // w_1 = w_0 - lr * g / (sqrt(v_1) + eps)
    // w_1[0] = 1.0 - 0.1 * 0.1 / (sqrt(0.001) + 1e-8) = 1.0 - 0.01 / 0.03162277 = 1.0 - 0.3162277 = 0.6837722
    opt.step();

    std::vector<float> expected_step1(3);
    for (std::size_t i = 0; i < 3; ++i) {
        float g = g_init[i];
        float v = (1.0F - alpha) * g * g;
        expected_step1[i] = w_init[i] - lr * g / (std::sqrt(v) + eps);
    }
    expect_close(copy_to_host(w->data()), expected_step1, "RMSProp step 1 failed");
}

void test_adam()
{
    std::cout << "Testing Adam..." << std::endl;
    using cuda_dl::core::DeviceTensor;
    using cuda_dl::core::DType;
    using cuda_dl::core::Variable;
    using cuda_dl::core::Adam;

    auto w = Variable::create(DeviceTensor({3}, DType::Float32));
    const std::vector<float> w_init{1.0F, 2.0F, 3.0F};
    const std::vector<float> g_init{0.1F, -0.2F, 0.5F};
    w->data().copy_from_host(w_init.data(), w_init.size());
    w->grad().copy_from_host(g_init.data(), g_init.size());

    const float lr = 0.1F;
    const float beta1 = 0.9F;
    const float beta2 = 0.99F;
    const float eps = 1e-8F;
    Adam opt({w}, lr, beta1, beta2, eps, 0.0F);

    // Step 1:
    // m_1 = beta1 * 0 + (1 - beta1) * g = 0.1 * g = {0.01, -0.02, 0.05}
    // v_1 = beta2 * 0 + (1 - beta2) * g^2 = 0.01 * g^2 = {0.0001, 0.0004, 0.0025}
    // m_hat = m_1 / (1 - beta1^1) = m_1 / 0.1 = g = {0.1, -0.2, 0.5}
    // v_hat = v_1 / (1 - beta2^1) = v_1 / 0.01 = g^2 = {0.01, 0.04, 0.25}
    // w_1 = w_0 - lr * m_hat / (sqrt(v_hat) + eps)
    // w_1[0] = 1.0 - 0.1 * 0.1 / (sqrt(0.01) + 1e-8) = 1.0 - 0.1 * 0.1 / 0.1 = 0.9
    // w_1[1] = 2.0 - 0.1 * (-0.2) / (sqrt(0.04) + 1e-8) = 2.0 - 0.1 * (-0.2) / 0.2 = 2.1
    // w_1[2] = 3.0 - 0.1 * 0.5 / (sqrt(0.25) + 1e-8) = 3.0 - 0.1 * 0.5 / 0.5 = 2.9
    opt.step();

    std::vector<float> expected_step1{0.9F, 2.1F, 2.9F};
    expect_close(copy_to_host(w->data()), expected_step1, "Adam step 1 failed");
}

void test_adamw()
{
    std::cout << "Testing AdamW..." << std::endl;
    using cuda_dl::core::DeviceTensor;
    using cuda_dl::core::DType;
    using cuda_dl::core::Variable;
    using cuda_dl::core::AdamW;

    auto w = Variable::create(DeviceTensor({3}, DType::Float32));
    const std::vector<float> w_init{1.0F, 2.0F, 3.0F};
    const std::vector<float> g_init{0.1F, -0.2F, 0.5F};
    w->data().copy_from_host(w_init.data(), w_init.size());
    w->grad().copy_from_host(g_init.data(), g_init.size());

    const float lr = 0.1F;
    const float beta1 = 0.9F;
    const float beta2 = 0.99F;
    const float eps = 1e-8F;
    const float wd = 0.05F;
    AdamW opt({w}, lr, beta1, beta2, eps, wd);

    // Step 1:
    // Decoupled weight decay applied first: w = w - lr * wd * w
    // w_decayed = w_init - 0.1 * 0.05 * w_init = w_init * 0.995 = {0.995, 1.99, 2.985}
    // Then standard Adam update is added to w_decayed:
    // w_1 = w_decayed - lr * m_hat / (sqrt(v_hat) + eps)
    // w_1[0] = 0.995 - 0.1 = 0.895
    // w_1[1] = 1.99 - 0.1 * (-1.0) = 2.09
    // w_1[2] = 2.985 - 0.1 = 2.885
    opt.step();

    std::vector<float> expected_step1{0.895F, 2.09F, 2.885F};
    expect_close(copy_to_host(w->data()), expected_step1, "AdamW step 1 failed");
}


void test_simple_fit()
{
    std::cout << "Testing optimization loop convergence (SGD)..." << std::endl;
    using cuda_dl::core::DeviceTensor;
    using cuda_dl::core::DType;
    using cuda_dl::core::Variable;
    using cuda_dl::core::SGD;

    // We will fit parameter w to approach 2.0 and b to approach 1.0.
    // Loss = (w - 2.0)^2 + (b - 1.0)^2
    // dw = 2 * (w - 2.0), db = 2 * (b - 1.0)
    auto w = Variable::create(DeviceTensor({1}, DType::Float32));
    auto b = Variable::create(DeviceTensor({1}, DType::Float32));
    w->data().copy_from_host(std::vector<float>{0.0F}.data(), 1);
    b->data().copy_from_host(std::vector<float>{0.0F}.data(), 1);

    SGD opt({w, b}, 0.1F);

    for (int step = 0; step < 50; ++step) {
        opt.zero_grad();
        
        std::vector<float> w_host = copy_to_host(w->data());
        std::vector<float> b_host = copy_to_host(b->data());

        // Set gradients mock representation of loss = (w-2)^2 + (b-1)^2
        std::vector<float> gw{ 2.0F * (w_host[0] - 2.0F) };
        std::vector<float> gb{ 2.0F * (b_host[0] - 1.0F) };

        w->grad().copy_from_host(gw.data(), 1);
        b->grad().copy_from_host(gb.data(), 1);

        opt.step();
    }

    std::vector<float> w_final = copy_to_host(w->data());
    std::vector<float> b_final = copy_to_host(b->data());

    std::cout << "  SGD Final Weight: " << w_final[0] << " (Expected: 2.0)" << std::endl;
    std::cout << "  SGD Final Bias: " << b_final[0] << " (Expected: 1.0)" << std::endl;

    expect_close(w_final, {2.0F}, "Convergence of weight w failed", 1.0e-3F);
    expect_close(b_final, {1.0F}, "Convergence of bias b failed", 1.0e-3F);
}

void test_simple_fit_adam()
{
    std::cout << "Testing optimization loop convergence (Adam)..." << std::endl;
    using cuda_dl::core::DeviceTensor;
    using cuda_dl::core::DType;
    using cuda_dl::core::Variable;
    using cuda_dl::core::Adam;

    auto w = Variable::create(DeviceTensor({1}, DType::Float32));
    auto b = Variable::create(DeviceTensor({1}, DType::Float32));
    w->data().copy_from_host(std::vector<float>{0.0F}.data(), 1);
    b->data().copy_from_host(std::vector<float>{0.0F}.data(), 1);

    Adam opt({w, b}, 0.1F);

    for (int step = 0; step < 200; ++step) {
        opt.zero_grad();
        
        std::vector<float> w_host = copy_to_host(w->data());
        std::vector<float> b_host = copy_to_host(b->data());

        std::vector<float> gw{ 2.0F * (w_host[0] - 2.0F) };
        std::vector<float> gb{ 2.0F * (b_host[0] - 1.0F) };

        w->grad().copy_from_host(gw.data(), 1);
        b->grad().copy_from_host(gb.data(), 1);

        opt.step();
    }

    std::vector<float> w_final = copy_to_host(w->data());
    std::vector<float> b_final = copy_to_host(b->data());

    std::cout << "  Adam Final Weight: " << w_final[0] << " (Expected: 2.0)" << std::endl;
    std::cout << "  Adam Final Bias: " << b_final[0] << " (Expected: 1.0)" << std::endl;

    expect_close(w_final, {2.0F}, "Adam convergence of weight w failed", 1.0e-3F);
    expect_close(b_final, {1.0F}, "Adam convergence of bias b failed", 1.0e-3F);
}

} // namespace

int main()
{
    try {
        test_sgd();
        test_sgd_momentum();
        test_rmsprop();
        test_adam();
        test_adamw();
        test_simple_fit();
        test_simple_fit_adam();

        std::cout << "All Optimizer verification checks passed successfully." << std::endl;
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Verification failed: " << error.what() << std::endl;
        return EXIT_FAILURE;
    }
}
