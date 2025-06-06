# instr_count_cuda_graph

Extension of the instruction counter that supports kernels launched
through CUDA graphs.  Each kernel receives its own counter so graphs
containing multiple launches can be tracked accurately.

## How it works

`instr_count_cuda_graph.cu` instruments kernels similarly to
`instr_count` but maintains a map of kernels to counters and handles
callbacks for CUDA graph APIs.  When graph nodes are launched the tool
prints per-kernel instruction counts using data collected in managed
memory.

## Building

Compile with `make` in this directory or from the top `tools` folder.

## Running

Preload `instr_count_cuda_graph.so` before executing a CUDA application
that uses graphs:

```bash
LD_PRELOAD=./tools/instr_count_cuda_graph/instr_count_cuda_graph.so ./app
```

The output lists each kernel ID along with its dynamic instruction count
just like the original `instr_count` tool.
