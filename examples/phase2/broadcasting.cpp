#include <cuda_dl/core/tensor.hpp>

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

void expect_shape(
    const cuda_dl::core::TensorShape& shape,
    const std::vector<std::size_t>& expected_dimensions)
{
    expect(shape.dimensions() == expected_dimensions, "broadcast output shape mismatch");
}

} // namespace

int main()
{
    using cuda_dl::core::TensorShape;

    expect_shape(
        cuda_dl::core::broadcast_shapes(TensorShape({2, 3, 4}), TensorShape({1, 4})),
        {2, 3, 4});

    expect_shape(
        cuda_dl::core::broadcast_shapes(TensorShape({5, 1}), TensorShape({1, 7})),
        {5, 7});

    expect_shape(
        cuda_dl::core::broadcast_shapes(TensorShape(), TensorShape({8, 16})),
        {8, 16});

    expect(
        cuda_dl::core::are_broadcast_compatible(TensorShape({3, 1}), TensorShape({1, 4})),
        "compatible shapes were rejected");

    expect(
        !cuda_dl::core::are_broadcast_compatible(TensorShape({2, 3}), TensorShape({4, 3})),
        "incompatible shapes were accepted");

    std::cout << "Broadcasting verified: scalar, rank expansion, singleton expansion, and rejection cases"
              << std::endl;

    return 0;
}
