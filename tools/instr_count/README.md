# instr_count

This example counts dynamic instructions executed by each CUDA kernel.
It instruments every SASS instruction and calls a device function that
updates a global counter.  The tool can optionally count at the warp or
thread granularity and can exclude predicated off instructions.

## How it works

`instr_count.cu` registers callbacks for CUDA launches. When a kernel
is first seen it is instrumented via `nvbit_insert_call` to call the
`count_instrs` device function defined in `inject_funcs.cu`.  The device
function uses `atomicAdd` to increment a counter stored in managed
memory.  When the kernel completes the host prints the number of
instructions executed along with a running total.

Important environment variables include `INSTR_BEGIN/END` to restrict
instrumentation to a range of instructions and `START_GRID_NUM` and
`END_GRID_NUM` to limit which kernels are instrumented.

## Building

Run `make` inside the `tools/instr_count` directory or from the top
`tools` directory to build all examples.

## Running

Preload the resulting `instr_count.so` before launching your CUDA
application:

```bash
LD_PRELOAD=./tools/instr_count/instr_count.so ./test-apps/vectoradd/vectoradd
```

The tool prints a banner describing the active options followed by a line
per kernel showing the number of executed instructions.
