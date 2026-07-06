---
type: community
cohesion: 0.14
members: 30
---

# DeviceTensor

**Cohesion:** 0.14 - loosely connected
**Members:** 30 nodes

## Members
- [[.DeviceTensor()]] - code - include/cuda_dl/core/device_tensor.cuh
- [[.bytes()_1]] - code - include/cuda_dl/core/device_tensor.cuh
- [[.copy_from_host()_1]] - code - include/cuda_dl/core/device_tensor.cuh
- [[.copy_to_host()_1]] - code - include/cuda_dl/core/device_tensor.cuh
- [[.data()]] - code - include/cuda_dl/core/device_tensor.cuh
- [[.dtype()]] - code - include/cuda_dl/core/device_tensor.cuh
- [[.element_count()]] - code - include/cuda_dl/core/device_tensor.cuh
- [[.rank()]] - code - include/cuda_dl/core/device_tensor.cuh
- [[.zero()_1]] - code - include/cuda_dl/core/device_tensor.cuh
- [[DType]] - code
- [[DeviceTensor]] - code - include/cuda_dl/core/device_tensor.cuh
- [[add()]] - code - include/cuda_dl/ops/tensor_ops.cuh
- [[binary_elementwise()]] - code - include/cuda_dl/ops/tensor_ops.cuh
- [[device_tensor.cuh]] - code - include/cuda_dl/core/device_tensor.cuh
- [[expect()_2]] - code - examples/phase2/tensor_add.cu
- [[expect()_3]] - code - examples/phase2/tensor_binary_ops.cu
- [[initializer_list]] - code
- [[main()_10]] - code - examples/phase2/tensor_add.cu
- [[main()_11]] - code - examples/phase2/tensor_allocation.cu
- [[main()_12]] - code - examples/phase2/tensor_binary_ops.cu
- [[max_error_against()]] - code - examples/phase2/tensor_binary_ops.cu
- [[metadata_]] - code - include/cuda_dl/core/device_tensor.cuh
- [[multiply()]] - code - include/cuda_dl/ops/tensor_ops.cuh
- [[size_t_4]] - code
- [[storage_]] - code - include/cuda_dl/core/device_tensor.cuh
- [[subtract()]] - code - include/cuda_dl/ops/tensor_ops.cuh
- [[tensor_add.cu]] - code - examples/phase2/tensor_add.cu
- [[tensor_binary_ops.cu]] - code - examples/phase2/tensor_binary_ops.cu
- [[tensor_ops.cuh]] - code - include/cuda_dl/ops/tensor_ops.cuh
- [[vector_3]] - code

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/DeviceTensor
SORT file.name ASC
```

## Connections to other communities
- 7 edges to [[_COMMUNITY_BroadcastConfig]]
- 5 edges to [[_COMMUNITY_TensorShape]]
- 4 edges to [[_COMMUNITY_vector]]
- 2 edges to [[_COMMUNITY_CudaException]]
- 1 edge to [[_COMMUNITY_DeviceBuffer]]
- 1 edge to [[_COMMUNITY_launch_config.cuh]]

## Top bridge nodes
- [[tensor_ops.cuh]] - degree 15, connects to 3 communities
- [[device_tensor.cuh]] - degree 8, connects to 3 communities
- [[DeviceTensor]] - degree 20, connects to 2 communities
- [[.DeviceTensor()]] - degree 7, connects to 1 community
- [[main()_11]] - degree 6, connects to 1 community