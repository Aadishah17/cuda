#pragma once

#include <cuda_dl/core/dataset.hpp>
#include <cuda_dl/core/device_buffer.cuh>
#include <cuda_dl/core/device_tensor.cuh>

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <random>
#include <vector>

namespace cuda_dl::core {

class DataLoader {
public:
    DataLoader(const MNISTDataset& dataset, const std::size_t batch_size, const bool shuffle = true)
        : dataset_(dataset)
        , batch_size_(batch_size)
        , shuffle_(shuffle)
        , current_index_(0)
        , rng_(1337) // Seed for reproducible batches
    {
        indices_.resize(dataset_.size());
        std::iota(indices_.begin(), indices_.end(), 0);
        reset();
    }

    void reset()
    {
        current_index_ = 0;
        if (shuffle_) {
            std::shuffle(indices_.begin(), indices_.end(), rng_);
        }
    }

    bool next_batch(DeviceTensor& batch_images, DeviceBuffer<int>& batch_labels)
    {
        if (current_index_ + batch_size_ > dataset_.size()) {
            return false;
        }

        const std::size_t H = dataset_.num_rows();
        const std::size_t W = dataset_.num_cols();
        const std::size_t img_size = H * W;

        // Allocate temporary host vectors
        std::vector<float> host_images(batch_size_ * img_size);
        std::vector<int> host_labels(batch_size_);

        for (std::size_t i = 0; i < batch_size_; ++i) {
            const std::size_t sample_idx = indices_[current_index_ + i];
            const auto& img = dataset_.get_image(sample_idx);
            std::copy(img.begin(), img.end(), host_images.begin() + i * img_size);
            host_labels[i] = dataset_.get_label(sample_idx);
        }

        current_index_ += batch_size_;

        // Copy directly from host to GPU device memory
        batch_images.copy_from_host(host_images.data(), host_images.size());
        batch_labels.copy_from_host(host_labels.data(), host_labels.size());

        return true;
    }

    std::size_t num_batches() const noexcept
    {
        return dataset_.size() / batch_size_;
    }

    std::size_t batch_size() const noexcept
    {
        return batch_size_;
    }

private:
    const MNISTDataset& dataset_;
    std::size_t batch_size_;
    bool shuffle_;
    std::size_t current_index_;
    std::vector<std::size_t> indices_;
    std::mt19937 rng_;
};

} // namespace cuda_dl::core
