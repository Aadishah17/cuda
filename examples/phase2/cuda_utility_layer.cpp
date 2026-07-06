#include <cuda_dl/core/launch_config.cuh>

#include <cstddef>
#include <iostream>
#include <limits>
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
    const cuda_dl::core::LaunchConfig1D exact = cuda_dl::core::make_1d_launch_config(1024, 256);
    expect(exact.blocks_per_grid == 4, "exact launch block count mismatch");
    expect(exact.threads_per_block == 256, "exact launch thread count mismatch");

    const cuda_dl::core::LaunchConfig1D rounded = cuda_dl::core::make_1d_launch_config(1025, 256);
    expect(rounded.blocks_per_grid == 5, "rounded launch block count mismatch");

    const cuda_dl::core::LaunchConfig1D empty = cuda_dl::core::make_1d_launch_config(0, 256);
    expect(empty.blocks_per_grid == 0, "empty launch block count mismatch");

    const cuda_dl::core::LaunchConfig2D launch_2d = cuda_dl::core::make_2d_launch_config(33, 17, 16, 16);
    expect(launch_2d.blocks_x == 2, "2D launch x block count mismatch");
    expect(launch_2d.blocks_y == 3, "2D launch y block count mismatch");

    bool invalid_threads_rejected = false;
    try {
        static_cast<void>(cuda_dl::core::make_1d_launch_config(1, 0));
    } catch (const std::invalid_argument&) {
        invalid_threads_rejected = true;
    }
    expect(invalid_threads_rejected, "invalid thread count was accepted");

    bool overflow_rejected = false;
    try {
        const std::size_t too_many_blocks = static_cast<std::size_t>(std::numeric_limits<int>::max()) + 1;
        static_cast<void>(cuda_dl::core::make_1d_launch_config(too_many_blocks, 1));
    } catch (const std::overflow_error&) {
        overflow_rejected = true;
    }
    expect(overflow_rejected, "oversized launch was accepted");

    std::cout << "CUDA utility layer verified: 1D/2D launch config math and guards" << std::endl;

    return 0;
}
