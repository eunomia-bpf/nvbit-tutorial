# mem_trace

Traces memory accesses performed by each kernel.  Addresses for each
warp are sent from the GPU to the host via an NVBit channel and printed
on the CPU.

## How it works

The tool instruments every memory reference instruction (except constant
and non-memory operations).  `instrument_mem` in `inject_funcs.cu`
collects the addresses touched by all threads in a warp and pushes the
data to a channel.  A host thread receives the data and formats a line of
text for each memory instruction executed.

## Building

Run `make` in this directory or from the main `tools` folder.

## Running

```bash
LD_PRELOAD=./tools/mem_trace/mem_trace.so ./test-apps/vectoradd/vectoradd
```

Each reported line includes the CTA, warp ID, opcode and the 32 addresses
accessed by the warp.
