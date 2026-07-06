---
source_file: "examples/phase2/tensor_binary_ops.cu"
type: "code"
community: "DeviceTensor"
location: "L47"
tags:
  - graphify/code
  - graphify/EXTRACTED
  - community/DeviceTensor
---

# main()

## Connections
- [[.copy_from_host()_1]] - `calls` [INFERRED]
- [[.copy_to_host()_1]] - `calls` [INFERRED]
- [[.element_count()]] - `calls` [INFERRED]
- [[expect()_3]] - `calls` [EXTRACTED]
- [[max_error_against()]] - `calls` [EXTRACTED]
- [[tensor_binary_ops.cu]] - `contains` [EXTRACTED]

#graphify/code #graphify/EXTRACTED #community/DeviceTensor