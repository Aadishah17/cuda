# Graph Report - .  (2026-07-06)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 178 nodes · 318 edges · 9 communities
- Extraction: 93% EXTRACTED · 7% INFERRED · 0% AMBIGUOUS · INFERRED: 22 edges (avg confidence: 0.8)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `e7cf00fc`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_vector|vector]]
- [[_COMMUNITY_TensorShape|TensorShape]]
- [[_COMMUNITY_DeviceTensor|DeviceTensor]]
- [[_COMMUNITY_CudaException|CudaException]]
- [[_COMMUNITY_DeviceBuffer|DeviceBuffer]]
- [[_COMMUNITY_BroadcastConfig|BroadcastConfig]]
- [[_COMMUNITY_CudaEvent|CudaEvent]]
- [[_COMMUNITY_launch_config.cuh|launch_config.cuh]]

## God Nodes (most connected - your core abstractions)
1. `TensorShape` - 22 edges
2. `DeviceTensor` - 20 edges
3. `Tensor` - 18 edges
4. `DeviceBuffer` - 17 edges
5. `CudaException` - 10 edges
6. `BroadcastConfig` - 10 edges
7. `CudaEvent` - 9 edges
8. `BroadcastOperand` - 7 edges
9. `make_broadcast_config()` - 7 edges
10. `expect_shape()` - 6 edges

## Surprising Connections (you probably didn't know these)
- `main()` --calls--> `metadata_`  [INFERRED]
  examples/phase2/tensor_allocation.cu → include/cuda_dl/core/device_tensor.cuh
- `main()` --calls--> `shape_`  [INFERRED]
  examples/phase2/tensor_metadata.cpp → include/cuda_dl/core/tensor.hpp
- `expect_shape()` --references--> `TensorShape`  [EXTRACTED]
  examples/phase2/broadcasting.cpp → include/cuda_dl/core/tensor.hpp
- `main()` --calls--> `TensorShape`  [EXTRACTED]
  examples/phase2/broadcasting.cpp → include/cuda_dl/core/tensor.hpp
- `main()` --references--> `DeviceTensor`  [INFERRED]
  examples/phase2/tensor_add.cu → include/cuda_dl/core/device_tensor.cuh

## Import Cycles
- None detected.

## Communities (9 total, 0 thin omitted)

### Community 0 - "vector"
Cohesion: 0.07
Nodes (25): __global__, hello_kernel(), compute_cpu_reference(), __global__, vector, main(), matmul_naive_kernel(), compute_cpu_reference() (+17 more)

### Community 1 - "TensorShape"
Cohesion: 0.13
Nodes (15): expect(), main(), are_broadcast_compatible(), broadcast_shapes(), dtype_name(), dtype_size_bytes(), DType, initializer_list (+7 more)

### Community 2 - "DeviceTensor"
Cohesion: 0.14
Nodes (17): expect(), main(), main(), vector, expect(), main(), max_error_against(), DType (+9 more)

### Community 3 - "CudaException"
Cohesion: 0.12
Nodes (13): cudaError_t, __global__, no_op_kernel(), add_constant_kernel(), __global__, size_t, check_cuda(), CudaException (+5 more)

### Community 4 - "DeviceBuffer"
Cohesion: 0.20
Nodes (8): size_t, DeviceBuffer, data_, element_count_, size_bytes_, validate_copy(), Pointer, T

### Community 5 - "BroadcastConfig"
Cohesion: 0.20
Nodes (15): __device__, binary_broadcast_kernel(), BroadcastConfig, lhs, output_dimensions, rank, rhs, BroadcastOperand (+7 more)

### Community 6 - "CudaEvent"
Cohesion: 0.21
Nodes (7): cudaEvent_t, __global__, main(), vector_add_kernel(), CudaEvent, event_, elapsed_milliseconds()

### Community 7 - "launch_config.cuh"
Cohesion: 0.28
Nodes (7): expect(), main(), size_t, LaunchConfig1D, blocks_per_grid, threads_per_block, make_1d_launch_config()

## Knowledge Gaps
- **17 isolated node(s):** `operation_`, `file_`, `event_`, `data_`, `element_count_` (+12 more)
  These have ≤1 connection - possible missing edges or undocumented components.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `TensorShape` connect `TensorShape` to `vector`, `DeviceTensor`, `BroadcastConfig`?**
  _High betweenness centrality (0.185) - this node is a cross-community bridge._
- **Why does `DeviceTensor` connect `DeviceTensor` to `TensorShape`, `DeviceBuffer`?**
  _High betweenness centrality (0.175) - this node is a cross-community bridge._
- **Why does `DeviceBuffer` connect `DeviceBuffer` to `DeviceTensor`, `CudaException`?**
  _High betweenness centrality (0.160) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `DeviceTensor` (e.g. with `main()` and `max_error_against()`) actually correct?**
  _`DeviceTensor` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `operation_`, `file_`, `event_` to the rest of the system?**
  _17 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `vector` be split into smaller, more focused modules?**
  _Cohesion score 0.06825396825396825 - nodes in this community are weakly interconnected._
- **Should `TensorShape` be split into smaller, more focused modules?**
  _Cohesion score 0.13306451612903225 - nodes in this community are weakly interconnected._