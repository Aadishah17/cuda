#pragma once

#include <cstddef>
#include <limits>
#include <stdexcept>

namespace cuda_dl::core {

struct LaunchConfig1D {
    int blocks_per_grid{0};
    int threads_per_block{0};
};

struct LaunchConfig2D {
    int blocks_x{0};
    int blocks_y{0};
    int threads_x{0};
    int threads_y{0};
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

inline LaunchConfig2D make_2d_launch_config(
    const std::size_t rows,
    const std::size_t columns,
    const int threads_x = 16,
    const int threads_y = 16)
{
    if (threads_x <= 0 || threads_y <= 0) {
        throw std::invalid_argument("2D launch thread dimensions must be positive");
    }

    const std::size_t block_columns = static_cast<std::size_t>(threads_x);
    const std::size_t block_rows = static_cast<std::size_t>(threads_y);
    const std::size_t blocks_x = columns == 0 ? 0 : 1 + ((columns - 1) / block_columns);
    const std::size_t blocks_y = rows == 0 ? 0 : 1 + ((rows - 1) / block_rows);

    if (blocks_x > static_cast<std::size_t>(std::numeric_limits<int>::max())
        || blocks_y > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error("2D CUDA launch requires too many blocks");
    }

    return LaunchConfig2D{
        static_cast<int>(blocks_x),
        static_cast<int>(blocks_y),
        threads_x,
        threads_y};
}

} // namespace cuda_dl::core
