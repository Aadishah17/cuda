#pragma once

#include <cuda_dl/core/cuda_error.cuh>

#include <cuda_runtime.h>

#include <utility>

namespace cuda_dl::core {

class CudaEvent {
public:
    explicit CudaEvent(const unsigned int flags = cudaEventDefault)
    {
        CUDADL_CUDA_CHECK(cudaEventCreateWithFlags(&event_, flags));
    }

    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;

    CudaEvent(CudaEvent&& other) noexcept
        : event_(std::exchange(other.event_, nullptr))
    {
    }

    CudaEvent& operator=(CudaEvent&& other) noexcept
    {
        if (this != &other) {
            release();
            event_ = std::exchange(other.event_, nullptr);
        }

        return *this;
    }

    ~CudaEvent() noexcept
    {
        release();
    }

    void record(const cudaStream_t stream = nullptr)
    {
        CUDADL_CUDA_CHECK(cudaEventRecord(event_, stream));
    }

    void synchronize()
    {
        CUDADL_CUDA_CHECK(cudaEventSynchronize(event_));
    }

    cudaEvent_t get() const noexcept
    {
        return event_;
    }

private:
    void release() noexcept
    {
        if (event_ != nullptr) {
            static_cast<void>(cudaEventDestroy(event_));
            event_ = nullptr;
        }
    }

    cudaEvent_t event_{nullptr};
};

inline float elapsed_milliseconds(const CudaEvent& start, const CudaEvent& stop)
{
    float milliseconds = 0.0F;
    CUDADL_CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start.get(), stop.get()));
    return milliseconds;
}

} // namespace cuda_dl::core
