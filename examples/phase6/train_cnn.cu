#include <cuda_dl/core/autograd.hpp>
#include <cuda_dl/core/dataset.hpp>
#include <cuda_dl/core/dataloader.hpp>
#include <cuda_dl/core/optimizer.hpp>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace {

// Helper to initialize weights using He Normal initialization
void initialize_he_normal(
    cuda_dl::core::DeviceTensor& tensor,
    const std::size_t fan_in,
    std::mt19937& rng)
{
    const float stddev = std::sqrt(2.0F / static_cast<float>(fan_in));
    std::normal_distribution<float> dist(0.0F, stddev);

    std::vector<float> host_vals(tensor.element_count());
    for (std::size_t i = 0; i < host_vals.size(); ++i) {
        host_vals[i] = dist(rng);
    }
    tensor.copy_from_host(host_vals.data(), host_vals.size());
}

// Helper to count correct predictions
std::size_t count_correct(const std::vector<float>& logits, const std::vector<int>& targets, const std::size_t batch_size, const std::size_t num_classes)
{
    std::size_t correct = 0;
    for (std::size_t b = 0; b < batch_size; ++b) {
        float max_val = logits[b * num_classes];
        int pred_class = 0;
        for (std::size_t c = 1; c < num_classes; ++c) {
            float val = logits[b * num_classes + c];
            if (val > max_val) {
                max_val = val;
                pred_class = static_cast<int>(c);
            }
        }
        if (pred_class == targets[b]) {
            correct++;
        }
    }
    return correct;
}

} // namespace

int main()
{
    try {
        std::cout << "=== Phase 6: CNN Training on MNIST ===" << std::endl;

        // 1. Load Dataset
        const std::string data_dir = "data/mnist/";
        std::cout << "Loading MNIST dataset..." << std::endl;
        cuda_dl::core::MNISTDataset train_dataset(
            data_dir + "train-images-idx3-ubyte",
            data_dir + "train-labels-idx1-ubyte"
        );
        cuda_dl::core::MNISTDataset test_dataset(
            data_dir + "t10k-images-idx3-ubyte",
            data_dir + "t10k-labels-idx1-ubyte"
        );

        std::size_t batch_size = 128;
        cuda_dl::core::DataLoader train_loader(train_dataset, batch_size, true);
        cuda_dl::core::DataLoader test_loader(test_dataset, batch_size, false);

        std::cout << "Train set size: " << train_dataset.size() << " (" << train_loader.num_batches() << " batches)" << std::endl;
        std::cout << "Test set size:  " << test_dataset.size() << " (" << test_loader.num_batches() << " batches)" << std::endl;

        // 2. Initialize Model Parameters
        std::mt19937 rng(42);
        using cuda_dl::core::DeviceTensor;
        using cuda_dl::core::DType;
        using cuda_dl::core::Variable;

        // Conv2D weights: [out_channels, in_channels, kernel_height, kernel_width]
        // 8 filters of size 1x3x3
        auto w_conv = Variable::create(DeviceTensor({8, 1, 3, 3}, DType::Float32));
        auto b_conv = Variable::create(DeviceTensor({8}, DType::Float32));
        initialize_he_normal(w_conv->data(), 1 * 3 * 3, rng);
        b_conv->data().zero();

        // Linear weights: [out_features, in_features]
        // Conv2D with padding=1, stride=1: size stays 28x28
        // MaxPool2D with stride=2: size becomes 14x14
        // Flattened features: 8 channels * 14 * 14 = 1568
        auto w_linear = Variable::create(DeviceTensor({10, 1568}, DType::Float32));
        auto b_linear = Variable::create(DeviceTensor({10}, DType::Float32));
        initialize_he_normal(w_linear->data(), 1568, rng);
        b_linear->data().zero();

        // 3. Define Optimizer
        const float lr = 0.005F;
        std::cout << "Using Adam optimizer with learning rate = " << lr << std::endl;
        cuda_dl::core::Adam opt({w_conv, b_conv, w_linear, b_linear}, lr);

        // Preallocate GPU buffers for batch data
        DeviceTensor x_batch({batch_size, 1, 28, 28}, DType::Float32);
        cuda_dl::core::DeviceBuffer<int> y_batch(batch_size);

        // 4. Training Loop (1 Epoch for quick convergence verification)
        const int epochs = 1;
        std::cout << "Starting training for " << epochs << " epoch(s)..." << std::endl;

        auto start_time = std::chrono::high_resolution_clock::now();

        for (int epoch = 0; epoch < epochs; ++epoch) {
            train_loader.reset();
            std::size_t batch_idx = 0;
            float running_loss = 0.0F;
            std::size_t running_correct = 0;
            std::size_t total_samples = 0;

            while (train_loader.next_batch(x_batch, y_batch)) {
                opt.zero_grad();

                // Construct autograd Variables for current batch
                auto x_var = Variable::create(x_batch.clone(), false);
                
                // Forward Pass
                auto conv_out = cuda_dl::autograd::conv2d(x_var, w_conv, b_conv, 1, 1);
                auto relu_out = cuda_dl::autograd::relu(conv_out);
                auto pool_out = cuda_dl::autograd::maxpool2d(relu_out, 2, 2, 0, 2);
                auto flat_out = cuda_dl::autograd::reshape(pool_out, {batch_size, 1568});
                auto logits = cuda_dl::autograd::linear(flat_out, w_linear, b_linear);
                auto loss = cuda_dl::autograd::softmax_cross_entropy(logits, y_batch);

                // Backward Pass
                loss->backward();

                // Optimizer Step
                opt.step();

                // Metrics
                float batch_loss = 0.0F;
                loss->data().copy_to_host(&batch_loss, 1);
                running_loss += batch_loss;

                std::vector<float> logits_host(logits->data().element_count());
                logits->data().copy_to_host(logits_host.data(), logits_host.size());
                std::vector<int> targets_host(batch_size);
                y_batch.copy_to_host(targets_host.data(), targets_host.size());

                std::size_t batch_correct = count_correct(logits_host, targets_host, batch_size, 10);
                running_correct += batch_correct;
                total_samples += batch_size;

                batch_idx++;
                if (batch_idx % 50 == 0) {
                    float avg_loss = running_loss / 50.0F;
                    float avg_acc = static_cast<float>(running_correct) / static_cast<float>(total_samples) * 100.0F;
                    std::cout << "Epoch [" << (epoch + 1) << "/" << epochs << "], Batch ["
                              << batch_idx << "/" << train_loader.num_batches()
                              << "], Loss: " << avg_loss
                              << ", Train Acc: " << avg_acc << "%" << std::endl;
                    running_loss = 0.0F;
                    running_correct = 0;
                    total_samples = 0;
                }
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end_time - start_time;
        std::cout << "Training completed in " << duration.count() << " seconds." << std::endl;

        // 5. Evaluation on Test Set
        std::cout << "Evaluating on test set..." << std::endl;
        test_loader.reset();
        std::size_t test_correct = 0;
        std::size_t test_total = 0;

        // Run evaluation on 2048 test images to keep it fast
        const std::size_t eval_batches = 16; 
        std::size_t evaluated = 0;

        while (test_loader.next_batch(x_batch, y_batch) && evaluated < eval_batches) {
            auto x_var = Variable::create(x_batch.clone(), false);
            
            auto conv_out = cuda_dl::autograd::conv2d(x_var, w_conv, b_conv, 1, 1);
            auto relu_out = cuda_dl::autograd::relu(conv_out);
            auto pool_out = cuda_dl::autograd::maxpool2d(relu_out, 2, 2, 0, 2);
            auto flat_out = cuda_dl::autograd::reshape(pool_out, {batch_size, 1568});
            auto logits = cuda_dl::autograd::linear(flat_out, w_linear, b_linear);

            std::vector<float> logits_host(logits->data().element_count());
            logits->data().copy_to_host(logits_host.data(), logits_host.size());
            std::vector<int> targets_host(batch_size);
            y_batch.copy_to_host(targets_host.data(), targets_host.size());

            test_correct += count_correct(logits_host, targets_host, batch_size, 10);
            test_total += batch_size;
            evaluated++;
        }

        float test_accuracy = static_cast<float>(test_correct) / static_cast<float>(test_total) * 100.0F;
        std::cout << "Test Set Accuracy (on " << test_total << " samples): " << test_accuracy << "%" << std::endl;

        if (test_accuracy < 90.0F) {
            throw std::runtime_error("CNN training failed to converge: Test accuracy was under 90%.");
        }

        std::cout << "CNN MNIST training convergence verified successfully!" << std::endl;
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Training failed: " << error.what() << std::endl;
        return EXIT_FAILURE;
    }
}
