# opcode_hist

Generates a histogram of executed instruction opcodes.  For each kernel
launch the tool prints the total number of instructions followed by a
count for every opcode encountered.

## How it works

`opcode_hist.cu` instruments every instruction and passes an opcode ID to
the `count_instrs` device function.  A per-opcode counter stored in
managed memory is incremented from the device.  When a kernel finishes
the host code prints the histogram and accumulates a global instruction
count.

## Building

Run `make` here or from the `tools` directory.

## Running

```bash
LD_PRELOAD=./tools/opcode_hist/opcode_hist.so ./test-apps/vectoradd/vectoradd
```

The output lists the instruction counts for each opcode executed within
the kernel.
