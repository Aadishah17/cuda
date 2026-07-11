#pragma once

#include <cuda_dl/core/device_tensor.cuh>
#include <cuda_dl/ops/activations.cuh>
#include <cuda_dl/ops/conv2d.cuh>
#include <cuda_dl/ops/linear.cuh>
#include <cuda_dl/ops/loss.cuh>
#include <cuda_dl/ops/pooling.cuh>
#include <cuda_dl/ops/tensor_ops.cuh>

#include <cstddef>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace cuda_dl::core {

class Variable;

namespace detail {

void build_topo(
    const std::shared_ptr<Variable>& node,
    std::vector<std::shared_ptr<Variable>>& topo,
    std::unordered_set<Variable*>& visited);

} // namespace detail


class Variable : public std::enable_shared_from_this<Variable> {
public:
    Variable(DeviceTensor data, const bool requires_grad = true)
        : data_(std::move(data))
        , grad_(data_.shape(), data_.dtype())
        , requires_grad_(requires_grad)
    {
        grad_.zero();
    }

    Variable(const Variable&) = delete;
    Variable& operator=(const Variable&) = delete;

    Variable(Variable&&) noexcept = default;
    Variable& operator=(Variable&&) noexcept = default;

    virtual ~Variable() = default;

    // Factory method
    static std::shared_ptr<Variable> create(DeviceTensor data, const bool requires_grad = true)
    {
        return std::make_shared<Variable>(std::move(data), requires_grad);
    }

    // Accessors
    DeviceTensor& data() noexcept { return data_; }
    const DeviceTensor& data() const noexcept { return data_; }

    DeviceTensor& grad() noexcept { return grad_; }
    const DeviceTensor& grad() const noexcept { return grad_; }

    const std::vector<std::shared_ptr<Variable>>& parents() const noexcept { return parents_; }
    const std::string& op() const noexcept { return op_; }
    bool requires_grad() const noexcept { return requires_grad_; }
    bool is_leaf() const noexcept { return parents_.empty(); }

    // Graph Construction Helpers
    void set_parents(std::vector<std::shared_ptr<Variable>> parents)
    {
        parents_ = std::move(parents);
    }

    void set_op(std::string op)
    {
        op_ = std::move(op);
    }

    void set_backward(std::function<void(cudaStream_t)> backward_fn)
    {
        backward_fn_ = std::move(backward_fn);
    }

    void set_backward(std::function<void()> backward_fn)
    {
        backward_fn_ = [fn = std::move(backward_fn)](cudaStream_t) { fn(); };
    }

    void zero_grad()
    {
        grad_.zero();
    }

    void accumulate_grad(const DeviceTensor& contribution, cudaStream_t stream = nullptr)
    {
        if (!requires_grad_) {
            return;
         }

         if (data_.dtype() != contribution.dtype()
             || data_.shape().dimensions() != contribution.shape().dimensions()) {
             throw std::invalid_argument("autograd gradient shape or dtype mismatch");
         }

         // This keeps accumulation on the device.
         grad_ = cuda_dl::ops::add(grad_, contribution, stream);
    }

    // Backpropagation trigger
    void backward(cudaStream_t stream = nullptr)
    {
        if (!requires_grad_) {
            return;
        }

        // 1. Build topological sort starting from this node
        std::vector<std::shared_ptr<Variable>> topo;
        std::unordered_set<Variable*> visited;
        detail::build_topo(shared_from_this(), topo, visited);

        // Intermediate gradients are per-backward-pass state. Leaf gradients
        // deliberately persist so callers can accumulate across microbatches.
        for (const auto& node : topo) {
            if (!node->is_leaf()) {
                node->zero_grad();
            }
        }

        // The default vector-Jacobian product uses an all-ones upstream vector.
        // It is equivalent to differentiating the sum of the output tensor.
        set_grad_to_ones(stream);

        // Traverse children before parents so every parent sees its complete
        // upstream gradient when its callback runs.
        for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
            const auto& node = *it;
            if (node->requires_grad_ && node->backward_fn_) {
                node->backward_fn_(stream);
            }
        }
    }

private:
    void set_grad_to_ones(cudaStream_t stream = nullptr)
    {
        if (grad_.element_count() == 0) {
            throw std::invalid_argument("backward requires a non-empty output tensor");
        }

        if (stream != nullptr) {
            CUDADL_CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        std::vector<float> ones(grad_.element_count(), 1.0F);
        grad_.copy_from_host(ones.data(), ones.size());
    }

    DeviceTensor data_;
    DeviceTensor grad_;
    std::vector<std::shared_ptr<Variable>> parents_;
    std::string op_{"none"};
    bool requires_grad_{true};
    std::function<void(cudaStream_t)> backward_fn_;
};

namespace detail {

inline void build_topo(
    const std::shared_ptr<Variable>& node,
    std::vector<std::shared_ptr<Variable>>& topo,
    std::unordered_set<Variable*>& visited)
{
    if (visited.count(node.get()) == 0) {
        visited.insert(node.get());
        for (const auto& parent : node->parents()) {
            if (parent) {
                build_topo(parent, topo, visited);
            }
        }
        topo.push_back(node);
    }
}

} // namespace detail

} // namespace cuda_dl::core

namespace cuda_dl::autograd {

using VariablePtr = std::shared_ptr<cuda_dl::core::Variable>;

namespace detail {

inline void require_non_null(const VariablePtr& input)
{
    if (!input) {
        throw std::invalid_argument("autograd operations require non-null variables");
    }
}

inline void require_same_shape_and_dtype(const VariablePtr& lhs, const VariablePtr& rhs)
{
    require_non_null(lhs);
    require_non_null(rhs);

    if (lhs->data().dtype() != rhs->data().dtype()
        || lhs->data().shape().dimensions() != rhs->data().shape().dimensions()) {
        throw std::invalid_argument("autograd elementwise operations require matching shapes and dtypes");
    }
}

} // namespace detail

inline VariablePtr relu(const VariablePtr& input, cudaStream_t stream = nullptr)
{
    detail::require_non_null(input);

    auto output = cuda_dl::core::Variable::create(
        cuda_dl::ops::relu_forward(input->data(), stream),
        input->requires_grad());
    output->set_op("relu");

    if (!output->requires_grad()) {
        return output;
    }

    output->set_parents({input});

    const std::weak_ptr<cuda_dl::core::Variable> output_weak = output;
    const std::weak_ptr<cuda_dl::core::Variable> input_weak = input;
    output->set_backward([output_weak, input_weak](cudaStream_t strm) {
        const auto output_node = output_weak.lock();
        const auto input_node = input_weak.lock();
        if (!output_node || !input_node) {
            throw std::logic_error("autograd graph node expired before relu backward");
        }

        input_node->accumulate_grad(
            cuda_dl::ops::relu_backward(input_node->data(), output_node->grad(), strm),
            strm);
    });

    return output;
}

inline VariablePtr linear(
    const VariablePtr& input,
    const VariablePtr& weight,
    const VariablePtr& bias,
    cudaStream_t stream = nullptr)
{
    detail::require_non_null(input);
    detail::require_non_null(weight);
    detail::require_non_null(bias);

    if (input->data().dtype() != weight->data().dtype()
        || input->data().dtype() != bias->data().dtype()) {
        throw std::invalid_argument("autograd linear requires matching dtypes");
    }

    const bool requires_grad = input->requires_grad()
        || weight->requires_grad()
        || bias->requires_grad();
    auto output = cuda_dl::core::Variable::create(
        cuda_dl::ops::linear_forward(input->data(), weight->data(), bias->data(), stream),
        requires_grad);
    output->set_op("linear");

    if (!requires_grad) {
        return output;
    }

    output->set_parents({input, weight, bias});

    const std::weak_ptr<cuda_dl::core::Variable> output_weak = output;
    const std::weak_ptr<cuda_dl::core::Variable> input_weak = input;
    const std::weak_ptr<cuda_dl::core::Variable> weight_weak = weight;
    const std::weak_ptr<cuda_dl::core::Variable> bias_weak = bias;
    output->set_backward([output_weak, input_weak, weight_weak, bias_weak](cudaStream_t strm) {
        const auto output_node = output_weak.lock();
        const auto input_node = input_weak.lock();
        const auto weight_node = weight_weak.lock();
        const auto bias_node = bias_weak.lock();
        if (!output_node || !input_node || !weight_node || !bias_node) {
            throw std::logic_error("autograd graph node expired before linear backward");
        }

        auto gradients = cuda_dl::ops::linear_backward(
            input_node->data(),
            weight_node->data(),
            output_node->grad(),
            strm);
        input_node->accumulate_grad(gradients.input_grad, strm);
        weight_node->accumulate_grad(gradients.weight_grad, strm);
        bias_node->accumulate_grad(gradients.bias_grad, strm);
    });

    return output;
}

inline VariablePtr conv2d(
    const VariablePtr& input,
    const VariablePtr& weight,
    const VariablePtr& bias,
    const std::size_t padding = 0,
    const std::size_t stride = 1,
    cudaStream_t stream = nullptr)
{
    detail::require_non_null(input);
    detail::require_non_null(weight);
    detail::require_non_null(bias);

    if (input->data().dtype() != weight->data().dtype()
        || input->data().dtype() != bias->data().dtype()) {
        throw std::invalid_argument("autograd conv2d requires matching dtypes");
    }

    const bool requires_grad = input->requires_grad()
        || weight->requires_grad()
        || bias->requires_grad();
    auto output = cuda_dl::core::Variable::create(
        cuda_dl::ops::conv2d_forward(input->data(), weight->data(), bias->data(), padding, stride, stream),
        requires_grad);
    output->set_op("conv2d");

    if (!requires_grad) {
        return output;
    }

    output->set_parents({input, weight, bias});

    const std::weak_ptr<cuda_dl::core::Variable> output_weak = output;
    const std::weak_ptr<cuda_dl::core::Variable> input_weak = input;
    const std::weak_ptr<cuda_dl::core::Variable> weight_weak = weight;
    const std::weak_ptr<cuda_dl::core::Variable> bias_weak = bias;
    output->set_backward([output_weak, input_weak, weight_weak, bias_weak, padding, stride](cudaStream_t strm) {
        const auto output_node = output_weak.lock();
        const auto input_node = input_weak.lock();
        const auto weight_node = weight_weak.lock();
        const auto bias_node = bias_weak.lock();
        if (!output_node || !input_node || !weight_node || !bias_node) {
            throw std::logic_error("autograd graph node expired before conv2d backward");
        }

        auto gradients = cuda_dl::ops::conv2d_backward(
            input_node->data(),
            weight_node->data(),
            output_node->grad(),
            padding,
            stride,
            strm);
        input_node->accumulate_grad(gradients.input_grad, strm);
        weight_node->accumulate_grad(gradients.weight_grad, strm);
        bias_node->accumulate_grad(gradients.bias_grad, strm);
    });

    return output;
}

inline VariablePtr maxpool2d(
    const VariablePtr& input,
    const std::size_t pool_height,
    const std::size_t pool_width,
    const std::size_t padding = 0,
    const std::size_t stride = 2,
    cudaStream_t stream = nullptr)
{
    detail::require_non_null(input);

    auto forward = cuda_dl::ops::maxpool2d_forward(
        input->data(),
        pool_height,
        pool_width,
        padding,
        stride,
        stream);
    auto output = cuda_dl::core::Variable::create(
        std::move(forward.output),
        input->requires_grad());
    output->set_op("maxpool2d");

    if (!output->requires_grad()) {
        return output;
    }

    auto argmax = std::make_shared<cuda_dl::core::DeviceBuffer<int>>(std::move(forward.argmax));
    output->set_parents({input});

    const std::weak_ptr<cuda_dl::core::Variable> output_weak = output;
    const std::weak_ptr<cuda_dl::core::Variable> input_weak = input;
    output->set_backward([output_weak, input_weak, argmax](cudaStream_t strm) {
        const auto output_node = output_weak.lock();
        const auto input_node = input_weak.lock();
        if (!output_node || !input_node) {
            throw std::logic_error("autograd graph node expired before maxpool2d backward");
        }

        input_node->accumulate_grad(
            cuda_dl::ops::maxpool2d_backward(input_node->data(), output_node->grad(), *argmax, strm),
            strm);
    });

    return output;
}

inline VariablePtr softmax_cross_entropy(
    const VariablePtr& logits,
    const cuda_dl::core::DeviceBuffer<int>& targets,
    cudaStream_t stream = nullptr)
{
    detail::require_non_null(logits);

    auto forward = cuda_dl::ops::softmax_cross_entropy_forward(logits->data(), targets, stream);
    auto output = cuda_dl::core::Variable::create(
        std::move(forward.mean_loss),
        logits->requires_grad());
    output->set_op("softmax_cross_entropy");

    if (!output->requires_grad()) {
        return output;
    }

    auto probabilities = std::make_shared<cuda_dl::core::DeviceTensor>(std::move(forward.probabilities));
    auto saved_targets = std::make_shared<cuda_dl::core::DeviceBuffer<int>>(targets.clone());
    output->set_parents({logits});

    const std::weak_ptr<cuda_dl::core::Variable> output_weak = output;
    const std::weak_ptr<cuda_dl::core::Variable> logits_weak = logits;
    output->set_backward([output_weak, logits_weak, probabilities, saved_targets](cudaStream_t strm) {
        const auto output_node = output_weak.lock();
        const auto logits_node = logits_weak.lock();
        if (!output_node || !logits_node) {
            throw std::logic_error("autograd graph node expired before softmax cross-entropy backward");
        }

        auto logits_grad = cuda_dl::ops::softmax_cross_entropy_backward(*probabilities, *saved_targets, strm);
        logits_node->accumulate_grad(cuda_dl::ops::multiply(logits_grad, output_node->grad(), strm), strm);
    });

    return output;
}

inline VariablePtr add(const VariablePtr& lhs, const VariablePtr& rhs, cudaStream_t stream = nullptr)
{
    detail::require_same_shape_and_dtype(lhs, rhs);

    const bool requires_grad = lhs->requires_grad() || rhs->requires_grad();
    auto output = cuda_dl::core::Variable::create(
        cuda_dl::ops::add(lhs->data(), rhs->data(), stream),
        requires_grad);
    output->set_op("add");

    if (!requires_grad) {
        return output;
    }

    output->set_parents({lhs, rhs});

    const std::weak_ptr<cuda_dl::core::Variable> output_weak = output;
    const std::weak_ptr<cuda_dl::core::Variable> lhs_weak = lhs;
    const std::weak_ptr<cuda_dl::core::Variable> rhs_weak = rhs;
    output->set_backward([output_weak, lhs_weak, rhs_weak](cudaStream_t strm) {
        const auto output_node = output_weak.lock();
        const auto lhs_node = lhs_weak.lock();
        const auto rhs_node = rhs_weak.lock();
        if (!output_node || !lhs_node || !rhs_node) {
            throw std::logic_error("autograd graph node expired before add backward");
        }

        lhs_node->accumulate_grad(output_node->grad(), strm);
        rhs_node->accumulate_grad(output_node->grad(), strm);
    });

    return output;
}

inline VariablePtr multiply(const VariablePtr& lhs, const VariablePtr& rhs, cudaStream_t stream = nullptr)
{
    detail::require_same_shape_and_dtype(lhs, rhs);

    const bool requires_grad = lhs->requires_grad() || rhs->requires_grad();
    auto output = cuda_dl::core::Variable::create(
        cuda_dl::ops::multiply(lhs->data(), rhs->data(), stream),
        requires_grad);
    output->set_op("multiply");

    if (!requires_grad) {
        return output;
    }

    output->set_parents({lhs, rhs});

    const std::weak_ptr<cuda_dl::core::Variable> output_weak = output;
    const std::weak_ptr<cuda_dl::core::Variable> lhs_weak = lhs;
    const std::weak_ptr<cuda_dl::core::Variable> rhs_weak = rhs;
    output->set_backward([output_weak, lhs_weak, rhs_weak](cudaStream_t strm) {
        const auto output_node = output_weak.lock();
        const auto lhs_node = lhs_weak.lock();
        const auto rhs_node = rhs_weak.lock();
        if (!output_node || !lhs_node || !rhs_node) {
            throw std::logic_error("autograd graph node expired before multiply backward");
        }

        lhs_node->accumulate_grad(cuda_dl::ops::multiply(output_node->grad(), rhs_node->data(), strm), strm);
        rhs_node->accumulate_grad(cuda_dl::ops::multiply(output_node->grad(), lhs_node->data(), strm), strm);
    });

    return output;
}

inline VariablePtr reshape(const VariablePtr& input, const cuda_dl::core::TensorShape& new_shape, cudaStream_t /*stream*/ = nullptr)
{
    detail::require_non_null(input);

    if (input->data().element_count() != new_shape.element_count()) {
        throw std::invalid_argument("autograd reshape: total element count mismatch");
    }

    auto reshaped_data = input->data().clone();
    reshaped_data.reshape(new_shape);

    auto output = cuda_dl::core::Variable::create(
        std::move(reshaped_data),
        input->requires_grad());
    output->set_op("reshape");

    if (!output->requires_grad()) {
        return output;
    }

    output->set_parents({input});

    const std::weak_ptr<cuda_dl::core::Variable> output_weak = output;
    const std::weak_ptr<cuda_dl::core::Variable> input_weak = input;
    output->set_backward([output_weak, input_weak](cudaStream_t strm) {
        const auto output_node = output_weak.lock();
        const auto input_node = input_weak.lock();
        if (!output_node || !input_node) {
            throw std::logic_error("autograd graph node expired before reshape backward");
        }

        auto input_grad_contrib = output_node->grad().clone();
        input_grad_contrib.reshape(input_node->data().shape());
        input_node->accumulate_grad(input_grad_contrib, strm);
    });

    return output;
}

} // namespace cuda_dl::autograd

