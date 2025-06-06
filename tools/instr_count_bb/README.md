# instr_count_bb

This tool counts dynamic instructions but touches only the first instruction of each basic block. It demonstrates how to use NVBit's control flow graph to reduce instrumentation overhead compared to `instr_count`.

## Source Files
- `instr_count.cu` – host component that builds the CFG and inserts calls.
- `inject_funcs.cu` – device functions `count_instrs` and `count_pred_off` that update counters.

## Instrumentation Flow
1. **CFG inspection** – During `instrument_function_if_needed` the static CFG for each function is retrieved using `nvbit_get_CFG`. The first instruction of every basic block receives a call to `count_instrs` with the block size. Predicated instructions can additionally call `count_pred_off`.
2. **Launch Handling** – The CUDA callback enables or disables instrumentation per kernel and prints counts after `cudaDeviceSynchronize` when the kernel finishes.
3. **Counters** – Two counters are kept: one for total instructions and one for instructions whose predicate evaluated to false (optional).

## Building
Run `make` in this directory or execute `make -C tools` from the repository root.

## Running
Launch your application with the library preloaded:

```bash
LD_PRELOAD=./tools/instr_count_bb/instr_count_bb.so ./test-apps/vectoradd/vectoradd
```

Environment variables mirror those of `instr_count` for choosing instruction and kernel ranges.

## Interpreting Results
The printed lines match the format of `instr_count`. Because only one call per basic block is injected the slowdown is usually much smaller while the total instruction count remains the same.
