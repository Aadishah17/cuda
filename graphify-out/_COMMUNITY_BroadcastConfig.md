---
type: community
cohesion: 0.20
members: 15
---

# BroadcastConfig

**Cohesion:** 0.20 - loosely connected
**Members:** 15 nodes

## Members
- [[BroadcastConfig]] - code - include/cuda_dl/ops/tensor_ops.cuh
- [[BroadcastOperand]] - code - include/cuda_dl/ops/tensor_ops.cuh
- [[__device__]] - code
- [[__global___9]] - code
- [[binary_broadcast_kernel()]] - code - include/cuda_dl/ops/tensor_ops.cuh
- [[dimensions]] - code - include/cuda_dl/ops/tensor_ops.cuh
- [[lhs]] - code - include/cuda_dl/ops/tensor_ops.cuh
- [[make_broadcast_config()]] - code - include/cuda_dl/ops/tensor_ops.cuh
- [[make_broadcast_operand()]] - code - include/cuda_dl/ops/tensor_ops.cuh
- [[output_dimensions]] - code - include/cuda_dl/ops/tensor_ops.cuh
- [[rank]] - code - include/cuda_dl/ops/tensor_ops.cuh
- [[rhs]] - code - include/cuda_dl/ops/tensor_ops.cuh
- [[size_t_7]] - code
- [[stdsize_t broadcast_offset()]] - code - include/cuda_dl/ops/tensor_ops.cuh
- [[strides]] - code - include/cuda_dl/ops/tensor_ops.cuh

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/BroadcastConfig
SORT file.name ASC
```

## Connections to other communities
- 7 edges to [[_COMMUNITY_DeviceTensor]]
- 3 edges to [[_COMMUNITY_TensorShape]]

## Top bridge nodes
- [[make_broadcast_config()]] - degree 7, connects to 2 communities
- [[make_broadcast_operand()]] - degree 5, connects to 2 communities
- [[BroadcastConfig]] - degree 10, connects to 1 community
- [[BroadcastOperand]] - degree 7, connects to 1 community
- [[stdsize_t broadcast_offset()]] - degree 5, connects to 1 community