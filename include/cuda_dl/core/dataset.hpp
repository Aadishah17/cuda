#pragma once

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>

namespace cuda_dl::core {

class MNISTDataset {
public:
    MNISTDataset(const std::string& images_path, const std::string& labels_path)
    {
        load_images(images_path);
        load_labels(labels_path);
        if (images_.size() != labels_.size()) {
            throw std::runtime_error("MNISTDataset: number of images does not match number of labels");
        }
    }

    std::size_t size() const noexcept { return images_.size(); }
    const std::vector<float>& images() const noexcept { return images_flat_; }
    const std::vector<int>& labels() const noexcept { return labels_; }

    std::size_t num_images() const noexcept { return images_.size(); }
    std::size_t num_rows() const noexcept { return rows_; }
    std::size_t num_cols() const noexcept { return cols_; }

    const std::vector<float>& get_image(const std::size_t idx) const
    {
        return images_[idx];
    }

    int get_label(const std::size_t idx) const
    {
        return labels_[idx];
    }

private:
    uint32_t read_big_endian_uint32(std::ifstream& stream)
    {
        uint32_t val = 0;
        stream.read(reinterpret_cast<char*>(&val), 4);
        // Swap bytes from big-endian to host order (assuming little-endian host)
        return ((val >> 24) & 0xff) |
               ((val >> 8) & 0xff00) |
               ((val << 8) & 0xff0000) |
               ((val << 24) & 0xff000000);
    }

    void load_images(const std::string& path)
    {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open MNIST images file: " + path);
        }

        uint32_t magic = read_big_endian_uint32(file);
        if (magic != 2051) {
            throw std::runtime_error("Invalid MNIST images magic number: expected 2051, got " + std::to_string(magic));
        }

        uint32_t num_items = read_big_endian_uint32(file);
        rows_ = read_big_endian_uint32(file);
        cols_ = read_big_endian_uint32(file);

        std::size_t img_size = rows_ * cols_;
        images_.resize(num_items, std::vector<float>(img_size));
        images_flat_.resize(num_items * img_size);

        std::vector<unsigned char> buffer(img_size);
        for (uint32_t i = 0; i < num_items; ++i) {
            file.read(reinterpret_cast<char*>(buffer.data()), img_size);
            for (std::size_t j = 0; j < img_size; ++j) {
                float val = static_cast<float>(buffer[j]) / 255.0F;
                images_[i][j] = val;
                images_flat_[i * img_size + j] = val;
            }
        }
    }

    void load_labels(const std::string& path)
    {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open MNIST labels file: " + path);
        }

        uint32_t magic = read_big_endian_uint32(file);
        if (magic != 2049) {
            throw std::runtime_error("Invalid MNIST labels magic number: expected 2049, got " + std::to_string(magic));
        }

        uint32_t num_items = read_big_endian_uint32(file);
        labels_.resize(num_items);

        std::vector<unsigned char> buffer(num_items);
        file.read(reinterpret_cast<char*>(buffer.data()), num_items);
        for (uint32_t i = 0; i < num_items; ++i) {
            labels_[i] = static_cast<int>(buffer[i]);
        }
    }

    std::size_t rows_{0};
    std::size_t cols_{0};
    std::vector<std::vector<float>> images_;
    std::vector<float> images_flat_;
    std::vector<int> labels_;
};

} // namespace cuda_dl::core
