#pragma once

#include <algorithm>
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cuda_dl::core {

enum class DType {
    Float32,
};

inline std::size_t dtype_size_bytes(const DType dtype)
{
    switch (dtype) {
    case DType::Float32:
        return sizeof(float);
    }

    throw std::invalid_argument("unsupported tensor dtype");
}

inline const char* dtype_name(const DType dtype)
{
    switch (dtype) {
    case DType::Float32:
        return "float32";
    }

    return "unknown";
}

class TensorShape {
public:
    TensorShape() = default;

    TensorShape(std::initializer_list<std::size_t> dimensions)
        : dimensions_(dimensions)
    {
        compute_metadata();
    }

    explicit TensorShape(std::vector<std::size_t> dimensions)
        : dimensions_(std::move(dimensions))
    {
        compute_metadata();
    }

    std::size_t rank() const noexcept
    {
        return dimensions_.size();
    }

    const std::vector<std::size_t>& dimensions() const noexcept
    {
        return dimensions_;
    }

    const std::vector<std::size_t>& strides() const noexcept
    {
        return strides_;
    }

    std::size_t dimension(const std::size_t axis) const
    {
        if (axis >= dimensions_.size()) {
            throw std::out_of_range("tensor dimension axis out of range");
        }

        return dimensions_[axis];
    }

    std::size_t stride(const std::size_t axis) const
    {
        if (axis >= strides_.size()) {
            throw std::out_of_range("tensor stride axis out of range");
        }

        return strides_[axis];
    }

    std::size_t element_count() const noexcept
    {
        return element_count_;
    }

    bool is_scalar() const noexcept
    {
        return dimensions_.empty();
    }

private:
    static std::size_t checked_multiply(
        const std::size_t lhs,
        const std::size_t rhs,
        const char* const message)
    {
        if ((rhs != 0) && (lhs > (std::numeric_limits<std::size_t>::max() / rhs))) {
            throw std::overflow_error(message);
        }

        return lhs * rhs;
    }

    void compute_metadata()
    {
        strides_.assign(dimensions_.size(), 1);

        if (dimensions_.empty()) {
            element_count_ = 1;
            return;
        }

        for (std::size_t i = dimensions_.size() - 1; i > 0; --i) {
            strides_[i - 1] = checked_multiply(
                strides_[i],
                dimensions_[i],
                "tensor stride overflow");
        }

        element_count_ = 1;
        for (const std::size_t dimension : dimensions_) {
            element_count_ = checked_multiply(
                element_count_,
                dimension,
                "tensor element count overflow");
        }
    }

    std::vector<std::size_t> dimensions_;
    std::vector<std::size_t> strides_;
    std::size_t element_count_{1};
};

inline TensorShape broadcast_shapes(const TensorShape& lhs, const TensorShape& rhs)
{
    const auto& lhs_dimensions = lhs.dimensions();
    const auto& rhs_dimensions = rhs.dimensions();
    const std::size_t output_rank = std::max(lhs_dimensions.size(), rhs_dimensions.size());

    std::vector<std::size_t> output_dimensions(output_rank, 1);

    for (std::size_t offset = 0; offset < output_rank; ++offset) {
        const std::size_t lhs_axis_from_end = lhs_dimensions.size() > offset
            ? lhs_dimensions[lhs_dimensions.size() - 1 - offset]
            : 1;
        const std::size_t rhs_axis_from_end = rhs_dimensions.size() > offset
            ? rhs_dimensions[rhs_dimensions.size() - 1 - offset]
            : 1;

        if (lhs_axis_from_end != rhs_axis_from_end && lhs_axis_from_end != 1 && rhs_axis_from_end != 1) {
            throw std::invalid_argument("tensor shapes are not broadcast-compatible");
        }

        output_dimensions[output_rank - 1 - offset] = std::max(lhs_axis_from_end, rhs_axis_from_end);
    }

    return TensorShape(std::move(output_dimensions));
}

inline bool are_broadcast_compatible(const TensorShape& lhs, const TensorShape& rhs) noexcept
{
    try {
        static_cast<void>(broadcast_shapes(lhs, rhs));
        return true;
    } catch (...) {
        return false;
    }
}

class Tensor {
public:
    Tensor(TensorShape shape, const DType dtype)
        : shape_(std::move(shape))
        , dtype_(dtype)
        , size_bytes_(checked_size_bytes(shape_.element_count(), dtype_))
    {
    }

    Tensor(std::initializer_list<std::size_t> dimensions, const DType dtype)
        : Tensor(TensorShape(dimensions), dtype)
    {
    }

    const TensorShape& shape() const noexcept
    {
        return shape_;
    }

    DType dtype() const noexcept
    {
        return dtype_;
    }

    std::size_t rank() const noexcept
    {
        return shape_.rank();
    }

    std::size_t element_count() const noexcept
    {
        return shape_.element_count();
    }

    std::size_t bytes() const noexcept
    {
        return size_bytes_;
    }

    std::string description() const
    {
        std::string result = "Tensor(shape=[";
        const auto& dimensions = shape_.dimensions();

        for (std::size_t i = 0; i < dimensions.size(); ++i) {
            if (i != 0) {
                result += ", ";
            }
            result += std::to_string(dimensions[i]);
        }

        result += "], dtype=";
        result += dtype_name(dtype_);
        result += ")";

        return result;
    }

private:
    static std::size_t checked_size_bytes(
        const std::size_t element_count,
        const DType dtype)
    {
        const std::size_t element_size = dtype_size_bytes(dtype);
        if ((element_size != 0)
            && (element_count > (std::numeric_limits<std::size_t>::max() / element_size))) {
            throw std::overflow_error("tensor byte size overflow");
        }

        return element_count * element_size;
    }

    TensorShape shape_;
    DType dtype_;
    std::size_t size_bytes_{0};
};

} // namespace cuda_dl::core
