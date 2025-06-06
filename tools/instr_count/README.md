# instr_count

This tool measures how many assembly instructions each CUDA kernel executes. It is a good starting point for learning how to inject device code with NVBit.

## Source Files
- `instr_count.cu` – host side tool that registers callbacks for CUDA launches and inserts instrumentation.
- `inject_funcs.cu` – device function `count_instrs` used to update the counter.

## Instrumentation Flow
1. **Initialization** – `nvbit_at_init` reads environment variables such as `INSTR_BEGIN`/`INSTR_END` to choose which instructions to instrument and `START_GRID_NUM`/`END_GRID_NUM` to limit which kernels are measured.
2. **Kernel Launch** – When a kernel is launched the tool locks a mutex, instruments the function if it has not been seen before and enables the instrumented version.
3. **Instruction Injection** – For every SASS instruction between the selected indexes a call to `count_instrs` is inserted. The device function increments `counter` either per warp or per thread.
4. **Completion** – After the launch the host synchronizes, prints the number of instructions for the kernel and updates a running total.

## Building
Run `make` inside this directory (or `make -C tools`). This requires `nvcc` and the CUDA toolkit.

## Running
Preload the shared library before running any CUDA program:

```bash
LD_PRELOAD=./tools/instr_count/instr_count.so ./test-apps/vectoradd/vectoradd
```

Environment variables control the behaviour. For example `COUNT_WARP_LEVEL=0` counts individual threads while `EXCLUDE_PRED_OFF=1` ignores instructions where the predicate is false.

## Interpreting Results
The tool prints a line per kernel in the form:

```
kernel 0 - foo(...) - #thread-blocks N,  kernel instructions M, total instructions T
```

`M` is the number of dynamic instructions executed by that kernel while `T` accumulates counts across all launches.
