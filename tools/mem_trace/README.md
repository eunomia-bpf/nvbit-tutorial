# mem_trace

`mem_trace` records the memory addresses accessed by each instruction. The addresses for all 32 lanes of a warp are collected on the device and streamed back to the host so they can be printed or analysed.

## Source Files
- `mem_trace.cu` – host side implementation handling CUDA callbacks and receiving channel messages.
- `inject_funcs.cu` – device function `instrument_mem` gathers addresses across the warp and sends a `mem_access_t` structure.
- `common.h` – definition of the structure used to transfer one warp's memory access.

## Instrumentation Flow
1. During instrumentation every memory reference instruction (excluding constant space) is given a call to `instrument_mem`. The opcode is mapped to a small integer ID so the host can decode it.
2. `instrument_mem` uses `__shfl_sync` to read the address from each lane and stores them in an array along with CTA and warp information. The first active lane pushes the structure onto the channel.
3. A dedicated host thread receives the structures and prints a human readable line containing the grid launch ID, CTA coordinates, warp ID, opcode name and the 32 collected addresses.

## Building
Run `make` in this directory or `make -C tools` to compile all tools. Ensure that the CUDA toolkit is installed.

## Running
Launch your program with:

```bash
LD_PRELOAD=./tools/mem_trace/mem_trace.so ./test-apps/vectoradd/vectoradd
```

## Interpreting Results
Each output line begins with the context and launch ID followed by CTA and warp identifiers. It then prints the opcode and the 32 addresses touched by that warp for that instruction. This data can be post-processed to analyse memory access patterns.
