# instr_count_bb

A variation of `instr_count` that instruments only the first instruction
of each basic block.  This greatly reduces overhead while still
producing the same instruction counts.

## How it works

`instr_count_bb.cu` uses the static control flow graph provided by NVBit
to identify basic blocks.  The tool injects a call to `count_instrs`
(defined in `inject_funcs.cu`) before the first instruction in every
block and passes the block size as an argument.  An additional optional
instrumentation counts threads where the predicate is off.

## Building

Run `make` in this directory or from the top level `tools` folder.

## Running

Use the shared library as with other tools:

```bash
LD_PRELOAD=./tools/instr_count_bb/instr_count_bb.so ./test-apps/vectoradd/vectoradd
```

The output matches `instr_count` but instrumentation overhead is lower.
