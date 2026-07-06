---
source_file: "examples/phase2/tensor_add.cu"
type: "code"
community: "DeviceTensor"
location: "L21"
tags:
  - graphify/code
  - graphify/INFERRED
  - community/DeviceTensor
---

# main()

## Connections
- [[.copy_from_host()_1]] - `calls` [INFERRED]
- [[.copy_to_host()_1]] - `calls` [INFERRED]
- [[.element_count()]] - `calls` [INFERRED]
- [[DeviceTensor]] - `references` [INFERRED]
- [[expect()_2]] - `calls` [EXTRACTED]
- [[tensor_add.cu]] - `contains` [EXTRACTED]

#graphify/code #graphify/INFERRED #community/DeviceTensor