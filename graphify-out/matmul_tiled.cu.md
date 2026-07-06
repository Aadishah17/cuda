---
source_file: "examples/phase1/matmul_tiled.cu"
type: "code"
community: "vector"
location: "L1"
tags:
  - graphify/code
  - graphify/EXTRACTED
  - community/vector
---

# matmul_tiled.cu

## Connections
- [[compute_cpu_reference()_1]] - `contains` [EXTRACTED]
- [[cuda_example_utils.cuh]] - `imports` [EXTRACTED]
- [[main()_4]] - `contains` [EXTRACTED]
- [[matmul_tiled_kernel()]] - `contains` [EXTRACTED]
- [[vector_4]] - `imports` [EXTRACTED]

#graphify/code #graphify/EXTRACTED #community/vector