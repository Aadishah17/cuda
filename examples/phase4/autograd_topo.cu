#include <cuda_dl/core/autograd.hpp>

#include <cstddef>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void expect(const bool condition, const char* const message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

} // namespace

int main()
{
    try {
        using cuda_dl::core::DeviceTensor;
        using cuda_dl::core::DType;
        using cuda_dl::core::Variable;

        // Create leaf variables
        auto var_a = Variable::create(DeviceTensor({1}, DType::Float32));
        auto var_b = Variable::create(DeviceTensor({1}, DType::Float32));

        // Create intermediate variable C depending on A and B
        auto var_c = Variable::create(DeviceTensor({1}, DType::Float32));
        var_c->set_parents({var_a, var_b});
        var_c->set_op("dummy_add");

        // Create output variable D depending on C
        auto var_d = Variable::create(DeviceTensor({1}, DType::Float32));
        var_d->set_parents({var_c});
        var_d->set_op("dummy_relu");

        // We will log the execution order of backward calls
        std::vector<std::string> execution_order;

        var_d->set_backward([&execution_order]() {
            execution_order.push_back("D");
        });

        var_c->set_backward([&execution_order]() {
            execution_order.push_back("C");
        });

        var_a->set_backward([&execution_order]() {
            execution_order.push_back("A");
        });

        var_b->set_backward([&execution_order]() {
            execution_order.push_back("B");
        });

        // Trigger backward pass
        std::cout << "Running topological sort backward pass..." << std::endl;
        var_d->backward();

        // Print execution order
        std::cout << "Backward execution order: ";
        for (const auto& name : execution_order) {
            std::cout << name << " ";
        }
        std::cout << std::endl;

        // Verify topological order properties:
        // D must run first, then C, then A and B.
        expect(execution_order.size() == 4, "Execution count mismatch");
        expect(execution_order[0] == "D", "D did not execute first");
        expect(execution_order[1] == "C", "C did not execute second");
        
        // A and B must execute at positions 2 and 3
        const bool last_two_ok = (execution_order[2] == "A" && execution_order[3] == "B") ||
                                 (execution_order[2] == "B" && execution_order[3] == "A");
        expect(last_two_ok, "A and B did not execute last");

        std::cout << "Autograd topological sort and node representation verified successfully." << std::endl;
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Verification failed: " << error.what() << std::endl;
        return EXIT_FAILURE;
    }
}
