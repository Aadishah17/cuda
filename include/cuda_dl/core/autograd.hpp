#pragma once

#include <cuda_dl/core/device_tensor.cuh>

#include <cstddef>
#include <functional>
#include <memory>
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

    // Graph Construction Helpers
    void set_parents(std::vector<std::shared_ptr<Variable>> parents)
    {
        parents_ = std::move(parents);
    }

    void set_op(std::string op)
    {
        op_ = std::move(op);
    }

    void set_backward(std::function<void()> backward_fn)
    {
        backward_fn_ = std::move(backward_fn);
    }

    // Backpropagation trigger
    void backward()
    {
        if (!requires_grad_) {
            return;
        }

        // 1. Build topological sort starting from this node
        std::vector<std::shared_ptr<Variable>> topo;
        std::unordered_set<Variable*> visited;
        detail::build_topo(shared_from_this(), topo, visited);

        // 2. Initialize the gradient of the root node to 1.0F
        std::vector<float> ones(grad_.element_count(), 1.0F);
        grad_.copy_from_host(ones.data(), ones.size());

        // 3. Traverse the nodes in reverse topological order (children before parents)
        for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
            const auto& node = *it;
            if (node->backward_fn_) {
                node->backward_fn_();
            }
        }
    }

private:
    DeviceTensor data_;
    DeviceTensor grad_;
    std::vector<std::shared_ptr<Variable>> parents_;
    std::string op_{"none"};
    bool requires_grad_{true};
    std::function<void()> backward_fn_;
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
