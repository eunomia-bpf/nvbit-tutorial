# opcode_hist

`opcode_hist` builds a histogram of executed opcodes for each kernel launch. It shows how you can use NVBit to collect per-opcode statistics on the GPU.

## Source Files
- `opcode_hist.cu` – host side code that inserts calls before every instruction and prints the histogram on kernel completion.
- `inject_funcs.cu` – device function `count_instrs` that updates the histogram in managed memory.

## Instrumentation Flow
1. All instructions in the selected range have a call to `count_instrs` inserted before them. The host assigns each opcode a small integer ID which is passed as an argument.
2. On the device an array `histogram` is indexed by that ID. Each warp (or each thread depending on `COUNT_WARP_LEVEL`) performs an atomic add to increment its opcode slot.
3. After the kernel finishes the host prints the total dynamic instruction count followed by the non-zero entries of the histogram.

## Building
Run `make` in this directory or `make -C tools`.

## Running
Example invocation:

```bash
LD_PRELOAD=./tools/opcode_hist/opcode_hist.so ./test-apps/vectoradd/vectoradd
```

## Interpreting Results
The output lists, for each kernel, the number of times each opcode executed. This can be useful for coarse-grained profiling or spotting unexpected instructions.
