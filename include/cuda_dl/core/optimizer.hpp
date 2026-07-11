#pragma once

#include <cuda_dl/core/autograd.hpp>
#include <cuda_dl/core/device_tensor.cuh>
#include <cuda_dl/ops/optimizers.cuh>

#include <cmath>
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

namespace cuda_dl::core {

class Optimizer {
public:
    explicit Optimizer(std::vector<std::shared_ptr<Variable>> params)
        : params_(std::move(params)) {}

    virtual ~Optimizer() = default;

    // Non-copyable, movable
    Optimizer(const Optimizer&) = delete;
    Optimizer& operator=(const Optimizer&) = delete;
    Optimizer(Optimizer&&) noexcept = default;
    Optimizer& operator=(Optimizer&&) noexcept = default;

    virtual void step() = 0;

    void zero_grad()
    {
        for (auto& param : params_) {
            if (param) {
                param->zero_grad();
            }
        }
    }

    const std::vector<std::shared_ptr<Variable>>& params() const noexcept
    {
        return params_;
    }

protected:
    std::vector<std::shared_ptr<Variable>> params_;
};

class SGD : public Optimizer {
public:
    explicit SGD(
        std::vector<std::shared_ptr<Variable>> params,
        const float lr,
        const float momentum = 0.0F,
        const float weight_decay = 0.0F)
        : Optimizer(std::move(params))
        , lr_(lr)
        , momentum_(momentum)
        , weight_decay_(weight_decay)
    {
        if (momentum_ > 0.0F) {
            for (const auto& param : params_) {
                DeviceTensor velocity(param->data().shape(), param->data().dtype());
                velocity.zero();
                velocities_.push_back(std::move(velocity));
            }
        }
    }

    void step() override
    {
        for (std::size_t i = 0; i < params_.size(); ++i) {
            auto& param = params_[i];
            if (!param->requires_grad()) {
                continue;
            }

            if (momentum_ > 0.0F) {
                cuda_dl::ops::sgd_momentum_step(
                    param->data(),
                    param->grad(),
                    velocities_[i],
                    lr_,
                    momentum_,
                    weight_decay_
                );
            } else {
                cuda_dl::ops::sgd_step(
                    param->data(),
                    param->grad(),
                    lr_,
                    weight_decay_
                );
            }
        }
    }

    float lr() const noexcept { return lr_; }
    void set_lr(const float lr) noexcept { lr_ = lr; }

    float momentum() const noexcept { return momentum_; }
    float weight_decay() const noexcept { return weight_decay_; }

private:
    float lr_;
    float momentum_;
    float weight_decay_;
    std::vector<DeviceTensor> velocities_;
};

class RMSProp : public Optimizer {
public:
    explicit RMSProp(
        std::vector<std::shared_ptr<Variable>> params,
        const float lr,
        const float alpha = 0.99F,
        const float eps = 1.0e-8F,
        const float weight_decay = 0.0F)
        : Optimizer(std::move(params))
        , lr_(lr)
        , alpha_(alpha)
        , eps_(eps)
        , weight_decay_(weight_decay)
    {
        for (const auto& param : params_) {
            DeviceTensor square_avg(param->data().shape(), param->data().dtype());
            square_avg.zero();
            square_averages_.push_back(std::move(square_avg));
        }
    }

    void step() override
    {
        for (std::size_t i = 0; i < params_.size(); ++i) {
            auto& param = params_[i];
            if (!param->requires_grad()) {
                continue;
            }

            cuda_dl::ops::rmsprop_step(
                param->data(),
                param->grad(),
                square_averages_[i],
                lr_,
                alpha_,
                eps_,
                weight_decay_
            );
        }
    }

    float lr() const noexcept { return lr_; }
    void set_lr(const float lr) noexcept { lr_ = lr; }

    float alpha() const noexcept { return alpha_; }
    float eps() const noexcept { return eps_; }
    float weight_decay() const noexcept { return weight_decay_; }

private:
    float lr_;
    float alpha_;
    float eps_;
    float weight_decay_;
    std::vector<DeviceTensor> square_averages_;
};

class Adam : public Optimizer {
public:
    explicit Adam(
        std::vector<std::shared_ptr<Variable>> params,
        const float lr = 1.0e-3F,
        const float beta1 = 0.9F,
        const float beta2 = 0.999F,
        const float eps = 1.0e-8F,
        const float weight_decay = 0.0F)
        : Optimizer(std::move(params))
        , lr_(lr)
        , beta1_(beta1)
        , beta2_(beta2)
        , eps_(eps)
        , weight_decay_(weight_decay)
        , step_count_(0)
    {
        for (const auto& param : params_) {
            DeviceTensor m_tensor(param->data().shape(), param->data().dtype());
            DeviceTensor v_tensor(param->data().shape(), param->data().dtype());
            m_tensor.zero();
            v_tensor.zero();
            m_.push_back(std::move(m_tensor));
            v_.push_back(std::move(v_tensor));
        }
    }

    void step() override
    {
        step_impl(false);
    }

    float lr() const noexcept { return lr_; }
    void set_lr(const float lr) noexcept { lr_ = lr; }

    float beta1() const noexcept { return beta1_; }
    float beta2() const noexcept { return beta2_; }
    float eps() const noexcept { return eps_; }
    float weight_decay() const noexcept { return weight_decay_; }
    std::size_t step_count() const noexcept { return step_count_; }

protected:
    void step_impl(const bool decoupled_decay)
    {
        step_count_++;
        const float bias_correction1 = 1.0F - std::pow(beta1_, static_cast<float>(step_count_));
        const float bias_correction2 = 1.0F - std::pow(beta2_, static_cast<float>(step_count_));

        for (std::size_t i = 0; i < params_.size(); ++i) {
            auto& param = params_[i];
            if (!param->requires_grad()) {
                continue;
            }

            cuda_dl::ops::adam_step(
                param->data(),
                param->grad(),
                m_[i],
                v_[i],
                lr_,
                beta1_,
                beta2_,
                eps_,
                bias_correction1,
                bias_correction2,
                weight_decay_,
                decoupled_decay
            );
        }
    }

    float lr_;
    float beta1_;
    float beta2_;
    float eps_;
    float weight_decay_;
    std::size_t step_count_;
    std::vector<DeviceTensor> m_;
    std::vector<DeviceTensor> v_;
};

class AdamW : public Adam {
public:
    explicit AdamW(
        std::vector<std::shared_ptr<Variable>> params,
        const float lr = 1.0e-3F,
        const float beta1 = 0.9F,
        const float beta2 = 0.999F,
        const float eps = 1.0e-8F,
        const float weight_decay = 0.01F)
        : Adam(std::move(params), lr, beta1, beta2, eps, weight_decay) {}

    void step() override
    {
        step_impl(true);
    }
};

} // namespace cuda_dl::core
