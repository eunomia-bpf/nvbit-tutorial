# mov_replace

This small tool demonstrates how to modify the SASS of a kernel. Every `MOV` instruction is removed and replaced with a call to our own device function that performs the move using NVBit register access intrinsics.

## Source Files
- `mov_replace.cu` – host side logic that scans instructions and performs the replacement.
- `inject_funcs.cu` – device function `mov_replace` that reads from the source operand and writes to the destination register.

## Instrumentation Flow
1. During instrumentation each function is scanned. If the opcode prefix is `MOV` the original instruction is removed via `nvbit_remove_orig`.
2. A call to `mov_replace` is inserted in its place. The tool passes the destination register index, the source (either a register or an immediate value) and a flag indicating the operand type. Predication is handled via `nvbit_add_call_arg_guard_pred_val`.
3. At runtime the injected function performs the move using `nvbit_read_reg`, `nvbit_read_ureg` and `nvbit_write_reg` so program behaviour remains unchanged.

## Building
Run `make` in this directory (or `make -C tools`).

## Running
Preload the library when executing your program:

```bash
LD_PRELOAD=./tools/mov_replace/mov_replace.so ./test-apps/vectoradd/vectoradd
```

## Interpreting Results
The output of the CUDA program should be identical to the native run. Inspect the SASS with `nvdisasm` to verify that all `MOV` instructions have been replaced by calls to `mov_replace`. This approach can be adapted to substitute other instructions with custom behaviour.
