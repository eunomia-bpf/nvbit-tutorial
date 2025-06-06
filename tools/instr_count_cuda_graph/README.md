# instr_count_cuda_graph

This variant of the instruction counter supports CUDA Graphs. Every launched kernel, whether part of a graph capture or a normal launch, has its own counter so graph executions can be profiled accurately.

## Source Files
- `instr_count_cuda_graph.cu` – host code that handles graph related callbacks and manages per-kernel counters.
- `inject_funcs.cu` – device function `count_instrs` used by all kernels.

## Instrumentation Flow
1. **Per-kernel counters** – A map associates each `CUfunction` with an index in the managed array `kernel_counter`. During capture or graph node creation the kernel is instrumented and the pointer to its counter is passed to the injected call.
2. **Handling graph APIs** – The CUDA callback intercepts `cudaGraphAddKernelNode`, stream capture launches and graph launches. Counters are printed after a graph completes.
3. **Standard launches** – Regular kernel launches are instrumented in the same way as `instr_count` and synchronized at exit.

## Building
Run `make` here or `make -C tools` if you wish to build all examples.

## Running
Use LD_PRELOAD with an application that uses CUDA graphs:

```bash
LD_PRELOAD=./tools/instr_count_cuda_graph/instr_count_cuda_graph.so ./app
```

## Interpreting Results
Each line identifies the kernel ID and total number of dynamic instructions executed. When graphs launch multiple kernels simultaneously each counter is printed separately so you can understand the cost of each node.
