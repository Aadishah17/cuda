#include <cuda_dl/core/tensor.hpp>

#include <cstdlib>
#include <iostream>
#include <stdexcept>

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
        const cuda_dl::core::Tensor matrix({2, 3, 4}, cuda_dl::core::DType::Float32);

        expect(matrix.rank() == 3, "rank mismatch");
        expect(matrix.element_count() == 24, "element count mismatch");
        expect(matrix.bytes() == 24 * sizeof(float), "byte count mismatch");
        expect(matrix.shape().stride(0) == 12, "stride 0 mismatch");
        expect(matrix.shape().stride(1) == 4, "stride 1 mismatch");
        expect(matrix.shape().stride(2) == 1, "stride 2 mismatch");

        const cuda_dl::core::Tensor scalar({}, cuda_dl::core::DType::Float32);
        expect(scalar.shape().is_scalar(), "scalar rank mismatch");
        expect(scalar.element_count() == 1, "scalar element count mismatch");
        expect(scalar.bytes() == sizeof(float), "scalar byte count mismatch");

        std::cout << matrix.description() << '\n';
        std::cout << "Rank: " << matrix.rank()
                  << ", elements: " << matrix.element_count()
                  << ", bytes: " << matrix.bytes() << '\n';
        std::cout << "Contiguous row-major strides: ["
                  << matrix.shape().stride(0) << ", "
                  << matrix.shape().stride(1) << ", "
                  << matrix.shape().stride(2) << "]\n";
        std::cout << "Tensor metadata validation completed successfully.\n";

        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
