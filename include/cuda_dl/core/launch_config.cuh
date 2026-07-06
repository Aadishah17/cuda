#pragma once

#include <cstddef>
#include <limits>
#include <stdexcept>

namespace cuda_dl::core {

struct LaunchConfig1D {
    int blocks_per_grid{0};
    int threads_per_block{0};
};

inline LaunchConfig1D make_1d_launch_config(
    const std::size_t element_count,
    const int threads_per_block = 256)
{
    if (threads_per_block <= 0) {
        throw std::invalid_argument("threads_per_block must be positive");
    }

    const std::size_t threads = static_cast<std::size_t>(threads_per_block);
    const std::size_t blocks = element_count == 0 ? 0 : 1 + ((element_count - 1) / threads);

    if (blocks > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error("1D CUDA launch requires too many blocks");
    }

    return LaunchConfig1D{static_cast<int>(blocks), threads_per_block};
}

} // namespace cuda_dl::core
