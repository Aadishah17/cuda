# CUDADL: Production-Grade CUDA Deep Learning Framework from Scratch

CUDADL is a high-performance, modular Deep Learning Framework built from the ground up using **Modern C++17** and **NVIDIA CUDA 13.x**. The framework is engineered for maximum performance on NVIDIA GPUs, featuring custom memory managers, logical/physical tensor structures, and optimized mathematical operations.

---

## Key Features

### Core Infrastructure & Memory Management

* **RAII GPU Memory Manager**: The `DeviceBuffer` class encapsulates raw `cudaMalloc` and `cudaFree` calls into an exception-safe, leak-free C++ RAII resource wrapper. Copy operations are deleted to prevent double-free issues, while Move operations transfer pointer ownership efficiently.
* **Logical/Physical Decoupled Tensors**: The logical layout (`TensorShape` and `Tensor` metadata) is separated from the physical memory storage (`DeviceBuffer`). The logical layer handles shape, rank, and contiguous row-major strides calculation with integer overflow safety.
* **Decoupled Device Tensor**: `DeviceTensor` binds logical shape metadata with physical device memory storage.

### Mathematical Kernels & Operations

* **Vector & Matrix Addition**: Highly coalesced element-wise 1D and 2D tensor addition kernels.
* **Naive & Shared Memory Matrix Multiplication**:
  * *Naive*: Global memory streaming matrix product.
  * *Tiled Shared Memory*: Cooperatively stages sub-matrix tiles into fast, on-chip Shared Memory, minimizing DRAM global memory bandwidth pressure. Supports arbitrary matrix sizes using boundary padding.
* **Dynamic Shape Broadcasting**: Element-wise binary operations (`add`, `subtract`, `multiply`) automatically align ranks and stretch singleton dimensions (size 1) at runtime, resolving offsets on the GPU via dynamic index unflattening.

### Diagnostics & Benchmarking

* **Robust Error Handling**: Macros (`CUDADL_CUDA_CHECK`, `CUDADL_CUDA_CHECK_LAST_KERNEL`, `CUDADL_CUDA_SYNCHRONIZE`) wrap all host/device APIs, converting runtime and asynchronous kernel errors into detailed C++ exceptions containing source file names and line numbers.
* **Precision Event Timer**: Wrap CUDA events into a high-precision RAII timer (`CudaEvent` and `elapsed_milliseconds`) to measure steady-state GPU execution times, bypassing CPU-host latency.
* **GPU Environment Verification**: Diagnostic utility to query device compute capabilities (e.g. RTX 5070 compute 12.0 Blackwell), core and memory clock rates, shared memory limits, and peak theoretical bandwidth.

---

## System Requirements

* **OS**: Windows 10/11
* **CUDA Toolkit**: 13.x (compiled and verified on CUDA 13.2+)
* **Compiler**: C++17 compatible host compiler (MSVC 19.51+ or GCC 9+)
* **Build System**: CMake 3.20+

---

## Build Instructions

To configure and compile the framework targets, run from the repository root:

```powershell
# Configure using Developer Command Prompt variables and NMake
cmake -B build -G "NMake Makefiles"

# Compile all targets
cmake --build build
```

---

## Running Verification Examples

After building, executable binaries are located in the `build/` folder.

### Environment & Diagnostics

```powershell
# Run environment diagnostics query
.\build\env_verify.exe
```

### Phase 1: CUDA Foundations

```powershell
# Run vector addition
.\build\vector_add.exe

# Run tiled shared-memory matrix multiplication
.\build\matmul_tiled.exe

# Run GPU precision benchmark
.\build\benchmarking.exe
```

### Phase 2: Tensor Infrastructure

```powershell
# Run tensor metadata verification
.\build\tensor_metadata.exe

# Run shape broadcasting test
.\build\broadcasting.exe

# Run element-wise operations with dynamic broadcasting
.\build\tensor_binary_ops.exe

# Run DeviceTensor matrix multiplication
.\build\tensor_matmul.exe
```

---

## Project Roadmap

* **Phase 1**: CUDA Foundations, Memory Coalescing, Tiled Matmul, GPU Benchmarking (Completed)
* **Phase 2**: RAII Memory Management, Shape Broadcasting, Tensor Abstractions, CUDA Launch Utilities (Completed)
* **Phase 3**: Neural Network Primitives (Linear, Conv2D, Pooling, Activations) (Completed)
* **Phase 4**: Automatic Differentiation (Computational Graph, Backpropagation)
* **Phase 5**: Optimizers (SGD, Adam, AdamW)
* **Phase 6**: DataLoader and CNN Training on MNIST
* **Phase 7**: Advanced CUDA Optimizations (Tensor Cores, WMMA, mixed-precision)
