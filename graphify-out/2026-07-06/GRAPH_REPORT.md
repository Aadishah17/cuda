# Graph Report - .  (2026-07-06)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 158 nodes · 271 edges · 11 communities
- Extraction: 94% EXTRACTED · 6% INFERRED · 0% AMBIGUOUS · INFERRED: 17 edges (avg confidence: 0.8)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `24302ae4`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_TensorShape|TensorShape]]
- [[_COMMUNITY_cuda_error.cuh|cuda_error.cuh]]
- [[_COMMUNITY_cuda_example_utils.cuh|cuda_example_utils.cuh]]
- [[_COMMUNITY_DeviceTensor|DeviceTensor]]
- [[_COMMUNITY_DeviceBuffer|DeviceBuffer]]
- [[_COMMUNITY_CudaEvent|CudaEvent]]
- [[_COMMUNITY_CudaException|CudaException]]
- [[_COMMUNITY_launch_config.cuh|launch_config.cuh]]
- [[_COMMUNITY_matmul_tiled.cu|matmul_tiled.cu]]
- [[_COMMUNITY_expect_shape|expect_shape]]

## God Nodes (most connected - your core abstractions)
1. `TensorShape` - 20 edges
2. `Tensor` - 18 edges
3. `DeviceBuffer` - 17 edges
4. `DeviceTensor` - 15 edges
5. `CudaException` - 10 edges
6. `CudaEvent` - 9 edges
7. `expect_shape()` - 6 edges
8. `main()` - 6 edges
9. `main()` - 6 edges
10. `main()` - 6 edges

## Surprising Connections (you probably didn't know these)
- `main()` --calls--> `shape_`  [INFERRED]
  examples/phase2/tensor_metadata.cpp → include/cuda_dl/core/tensor.hpp
- `expect_shape()` --references--> `TensorShape`  [EXTRACTED]
  examples/phase2/broadcasting.cpp → include/cuda_dl/core/tensor.hpp
- `main()` --calls--> `TensorShape`  [EXTRACTED]
  examples/phase2/broadcasting.cpp → include/cuda_dl/core/tensor.hpp
- `main()` --calls--> `metadata_`  [INFERRED]
  examples/phase2/tensor_add.cu → include/cuda_dl/core/device_tensor.cuh
- `main()` --calls--> `metadata_`  [INFERRED]
  examples/phase2/tensor_allocation.cu → include/cuda_dl/core/device_tensor.cuh

## Import Cycles
- None detected.

## Communities (11 total, 0 thin omitted)

### Community 0 - "TensorShape"
Cohesion: 0.13
Nodes (16): expect(), main(), are_broadcast_compatible(), broadcast_shapes(), dtype_name(), dtype_size_bytes(), DType, initializer_list (+8 more)

### Community 1 - "cuda_error.cuh"
Cohesion: 0.11
Nodes (13): __global__, no_op_kernel(), add_constant_kernel(), __global__, size_t, expect(), affine_kernel(), __global__ (+5 more)

### Community 2 - "cuda_example_utils.cuh"
Cohesion: 0.11
Nodes (11): __global__, hello_kernel(), compute_cpu_reference(), __global__, vector, main(), matmul_naive_kernel(), __global__ (+3 more)

### Community 3 - "DeviceTensor"
Cohesion: 0.22
Nodes (9): main(), main(), DType, initializer_list, size_t, DeviceTensor, metadata_, storage_ (+1 more)

### Community 4 - "DeviceBuffer"
Cohesion: 0.20
Nodes (8): size_t, DeviceBuffer, data_, element_count_, size_bytes_, validate_copy(), Pointer, T

### Community 5 - "CudaEvent"
Cohesion: 0.21
Nodes (7): cudaEvent_t, __global__, main(), vector_add_kernel(), CudaEvent, event_, elapsed_milliseconds()

### Community 6 - "CudaException"
Cohesion: 0.27
Nodes (7): cudaError_t, check_cuda(), CudaException, file_, operation_, string, runtime_error

### Community 7 - "launch_config.cuh"
Cohesion: 0.28
Nodes (7): expect(), main(), size_t, LaunchConfig1D, blocks_per_grid, threads_per_block, make_1d_launch_config()

### Community 8 - "matmul_tiled.cu"
Cohesion: 0.40
Nodes (5): compute_cpu_reference(), __global__, vector, main(), matmul_tiled_kernel()

### Community 9 - "expect_shape"
Cohesion: 0.53
Nodes (5): size_t, vector, expect(), expect_shape(), main()

## Knowledge Gaps
- **12 isolated node(s):** `operation_`, `file_`, `event_`, `data_`, `element_count_` (+7 more)
  These have ≤1 connection - possible missing edges or undocumented components.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `DeviceBuffer` connect `DeviceBuffer` to `cuda_error.cuh`, `DeviceTensor`?**
  _High betweenness centrality (0.179) - this node is a cross-community bridge._
- **Why does `Tensor` connect `TensorShape` to `expect_shape`, `DeviceTensor`, `cuda_error.cuh`?**
  _High betweenness centrality (0.179) - this node is a cross-community bridge._
- **Why does `TensorShape` connect `TensorShape` to `expect_shape`, `DeviceTensor`?**
  _High betweenness centrality (0.174) - this node is a cross-community bridge._
- **What connects `operation_`, `file_`, `event_` to the rest of the system?**
  _12 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `TensorShape` be split into smaller, more focused modules?**
  _Cohesion score 0.13068181818181818 - nodes in this community are weakly interconnected._
- **Should `cuda_error.cuh` be split into smaller, more focused modules?**
  _Cohesion score 0.1067193675889328 - nodes in this community are weakly interconnected._
- **Should `cuda_example_utils.cuh` be split into smaller, more focused modules?**
  _Cohesion score 0.1111111111111111 - nodes in this community are weakly interconnected._