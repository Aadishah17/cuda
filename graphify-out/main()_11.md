---
source_file: "examples/phase2/tensor_allocation.cu"
type: "code"
community: "DeviceTensor"
location: "L24"
tags:
  - graphify/code
  - graphify/INFERRED
  - community/DeviceTensor
---

# main()

## Connections
- [[.copy_from_host()_1]] - `calls` [INFERRED]
- [[.copy_to_host()_1]] - `calls` [INFERRED]
- [[.data()]] - `calls` [INFERRED]
- [[.element_count()]] - `calls` [INFERRED]
- [[metadata_]] - `calls` [INFERRED]
- [[tensor_allocation.cu]] - `contains` [EXTRACTED]

#graphify/code #graphify/INFERRED #community/DeviceTensor